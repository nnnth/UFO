# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig, reduce_mean, InstanceList, OptInstanceList)
from ..utils import multi_apply, get_uncertain_point_coords_with_randomness
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
import pycocotools.mask as maskUtils
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from .beam_search import beam_search
import cv2
from ..layers import mask_matrix_nms
from mmcv.ops import point_sample
from .dataset_labels import dataset_labels


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


@MODELS.register_module()
class UFOViTInsSegHead(BaseModule):
    r"""Instance Segmentation head for UFO. It's a non-parametric head for 
        UFO decoding and post-processing in instance segmentation task.
    """
    def __init__(self,
            train_cfg: ConfigType = None,
            test_cfg: ConfigType = None,
            init_cfg: OptMultiConfig = None,
            task_prompt: str='instance_segmentation',
            ignore_index=-100,
            max_length=30,
            beam_num=1,
            temperature=1.0,
            alpha=1.0,
            mask_token_id=30524,
            mask_loss_weight=1.,
            cls_loss_weight=1.,
            loss_dice: ConfigType = dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                naive_dice=True,
                loss_weight=1.0),
            nms=None,
            repeat_times=3,
            sample_prob=0.0,
            ) -> None:
        super().__init__(init_cfg=init_cfg)
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('UFO do not build sampler.')
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.task_prompt = task_prompt
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.beam_num = beam_num
        self.temperature = temperature
        self.alpha = alpha
        self._init_layers()
        self.mask_token_id = mask_token_id
        self.mask_loss_weight = mask_loss_weight
        self.loss_dice = MODELS.build(loss_dice)
        self.cls_loss_weight = cls_loss_weight
        self.nms = nms
        self.repeat_times = repeat_times
        self.sample_prob = sample_prob
        self.dataset_labels = dataset_labels
        if train_cfg:
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)
        
    def _init_layers(self) -> None:
        pass

    def init_weights(self) -> None:
        pass

    def reset_hyparameter(self, cfgs):
        for k in list(cfgs.keys()):
            setattr(self, k, cfgs[k])
        self.loss_reg = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.ignore_index)

    def get_prompt(self, batch_data_samples):
        dataset_name = batch_data_samples[0].metainfo['dataset_name']
        classes = list(self.dataset_labels[dataset_name])
        num_classes = batch_data_samples[0].metainfo['head_cfg']['num_classes']
        classes_text = ','.join(classes)
        prompt = f'Task: {self.task_prompt}; Class: {classes_text}.'
        return [prompt for img in batch_data_samples]
    
    def get_prompt_sample(self, batch_data_samples, target_tokens):
        if torch.rand(1)[0] > self.sample_prob:
            return self.get_prompt(batch_data_samples)
        dataset_name = batch_data_samples[0].metainfo['dataset_name']
        classes = list(self.dataset_labels[dataset_name])
        num_classes = batch_data_samples[0].metainfo['head_cfg']['num_classes']
        prompts = []
        for target_token in target_tokens:
            # remove background
            unique_classes = target_token[:,0].unique()
            if unique_classes[-1] == num_classes:
                pos_classes = unique_classes[:-1]
            else:
                pos_classes = unique_classes
            neg_mask = torch.ones(num_classes).to(target_token.device)
            neg_mask[pos_classes] = 0
            neg_classes = neg_mask.nonzero().squeeze(-1)
            if len(neg_classes) > 0:
                sample_neg_classes = neg_classes[torch.randperm(len(neg_classes))[:len(pos_classes)*5]]
                sample_classes = torch.cat([pos_classes,sample_neg_classes])
            else:
                print("Ins seg: no negative classes")
                print(pos_classes)
                sample_classes = pos_classes
            sample_classes = sample_classes[torch.randperm(len(sample_classes))]
            sample_classes_list =[classes[i] for i in sample_classes]
            classes_text = ','.join(sample_classes_list)
            prompt = f'Task: {self.task_prompt}; Class: {classes_text}.'
            prompts.append(prompt)
        return prompts
    
    def translate_text(self,targets_tokens_tensor, batch_img_metas):
        dataset_name = batch_img_metas[0]['dataset_name']
        classes = list(self.dataset_labels[dataset_name]) + ['background']
        use_vocab_list = classes
        num_classes = batch_img_metas[0]['head_cfg']['num_classes']
        
        target_texts_list = []
        for targets_tokens in targets_tokens_tensor:
            target_texts = []
            for tokens in targets_tokens:
                if use_vocab_list[tokens[0]] == 'background':
                    text = 'background'
                else:
                    text = use_vocab_list[tokens[0]] + ','+'<MASK>'*16
                target_texts.append(text)
            target_texts_list.extend(target_texts)  

        return target_texts_list
    
    def get_targets_based_on_reference(self, 
                reference_preds_list: List[Tensor],
                batch_gt_instances: InstanceList,
                batch_img_metas: List[dict]) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            reference_preds_list (list[Tensor]): Grid positions for each image, 
                with normalized coordinate (cx, cy) and shape [num_queries, 2].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - input_tokens_tensor (Tensor): Input tokens of each image for training.
              has shape (bs, num_queries, 4).
            - targets_tokens_tensor (Tensor): GT tokens of each image (bs, num_queries, 5).
            - tokens_weights_tensor (Tensor): GT tokens weights of each image, 
              has shape (bs, num_queries, 5).
        """
        (input_tokens_list, targets_tokens_list, tokens_weights_list,
        gt_inds_list) = multi_apply(self._get_targets_single_based_on_reference,
                                      reference_preds_list,
                                      batch_gt_instances, batch_img_metas)

        # only support parallel training, means query_num of each image is equal
        return (torch.stack(input_tokens_list), torch.stack(targets_tokens_list), 
                torch.stack(tokens_weights_list), torch.stack(gt_inds_list))
    
    def _get_targets_single_based_on_reference(self, reference_pred: Tensor,
                gt_instances: InstanceData,
                img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Assign targets based on distance between boxes and grids.

        Args:
            reference_pred (Tensor): Grid positions for one image, 
                with normalized coordinate (cx, cy) and shape [num_queries, 2]
                or normalized coordinate (cx, cy, w, h) and shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - input_tokens (Tensor): Input tokens of each image for training.
            - targets_tokens (Tensor): GT tokens of each image.
            - tokens_weights (Tensor]): GT tokens weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        if reference_pred.shape[-1] == 2:
            # cx, cy
            factor = reference_pred.new_tensor([img_w, img_h]).unsqueeze(0)
            # convert reference_pred from normalized to unnormalized
            reference_pred = reference_pred * factor
        elif reference_pred.shape[-1] == 4:
            # cx, cy, w, h
            factor = reference_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            # convert reference_pred from normalized to unnormalized
            reference_pred = bbox_cxcywh_to_xyxy(reference_pred) * factor
        else:
            raise NotImplementedError
        num_bboxes = reference_pred.size(0)
        reference_pred_instances = InstanceData(points=reference_pred)
        # repeat gt instances
        unrepeat_num = len(gt_instances.labels)
        repeat_labels = gt_instances.labels.unsqueeze(0).repeat(self.repeat_times,1).flatten(0,1)
        repeat_bboxes = gt_instances.bboxes.unsqueeze(0).repeat(self.repeat_times,1,1).flatten(0,1)
        repeat_masks = torch.from_numpy(gt_instances.masks.masks).to(repeat_labels.device).unsqueeze(0).repeat(self.repeat_times,1,1,1).flatten(0,1)
        gt_instances = InstanceData(
            labels=repeat_labels,
            bboxes=repeat_bboxes,
            masks=repeat_masks,
        )
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=reference_pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        
        targets_tokens = labels[:, None]
        tokens_weights = label_weights[:, None]

        # input tokens for parallel training
        input_tokens = targets_tokens.clone()
        
        # reverse repeat, as gt instances outside is not repeated
        assign_result.gt_inds[pos_inds] = (assign_result.gt_inds[pos_inds] -1) % unrepeat_num + 1

        return (input_tokens, targets_tokens, tokens_weights, assign_result.gt_inds)
    
    def loss(self, all_layer_pred_seq_logits: Tensor,
                   all_layer_target_tokens: List[Tensor],
                   all_layer_token_weights: List[Tensor],
                   image_features,
                   seq_embed,
                   assign_inds,
                   batch_gt_instances,
                   batch_img_metas) -> Dict[str, Tensor]:

        loss_inputs = (all_layer_pred_seq_logits,
                       all_layer_target_tokens,
                       all_layer_token_weights,
                       [image_features],
                       [seq_embed],
                       [assign_inds],
                       batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
    
    def loss_by_feat(self, all_layer_pred_seq_logits: Tensor,
                           all_layer_target_tokens: List[Tensor],
                           all_layer_token_weights: List[Tensor],
                           image_features,
                           seq_embed,
                           assign_inds,
                           batch_gt_instances: InstanceList,
                           batch_img_metas: List[dict],
                           batch_gt_instances_ignore: OptInstanceList = None) -> Dict[str, Tensor]:
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_layer_pred_seq_logits (Tensor): Outputs from the
                autoregressive head, has shape (num_decoder_layers, bs,
                num_queries, max_token_len, vocab_size).
            all_layer_target_tokens (Tensor): GT targets for
                autoregressive head, has shape (num_decoder_layers, bs,
                num_queries, max_token_len).
            all_layer_token_weights (Tensor): GT weights of 
                each token, has shape (num_decoder_layers, bs, num_queries, 
                max_token_len).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'
        losses_cls,losses_mask, losses_dice = multi_apply(
            self.loss_by_feat_single,
            all_layer_pred_seq_logits,
            all_layer_target_tokens,
            all_layer_token_weights,
            image_features,
            seq_embed,
            assign_inds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
    
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i,loss_mask_i,loss_dice_i in zip(losses_cls[:-1],losses_mask[:-1],losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i[0]
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i[0]
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def loss_by_feat_single(self, pred_seq_logits: Tensor, 
                                  targets_tokens_tensor: Tensor,
                                  tokens_weights_tensor: Tensor,
                                  image_features,
                                  seq_embed,
                                  assign_inds,
                                  batch_gt_instances: InstanceList,
                                  batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            pred_seq_logits (Tensor): Outputs from the autoregressive head, 
                has shape (bs, num_queries, max_token_len, vocab_size).
            targets_tokens_tensor (Tensor): GT targets for autoregressive head, 
                has shape (bs, num_queries, max_token_len).
            tokens_weights_tensor (Tensor): GT weights of each token, has shape 
                (bs, num_queries, max_token_len).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`
        """
        num_imgs, num_queries = pred_seq_logits.shape[:2]

        # split classification and regression logits
        logits = pred_seq_logits.reshape(-1, pred_seq_logits.shape[-1])

        # construct weighted avg_factor 
        avg_factor = tokens_weights_tensor.sum()
        avg_factor = reduce_mean(
            pred_seq_logits.new_tensor([avg_factor]))
        avg_factor = max(avg_factor, 1)

        # ignore negative queries regression
        tokens_weights_tensor = tokens_weights_tensor.contiguous().view(-1)
        flat_targets_tokens_tensor = targets_tokens_tensor.contiguous().view(-1)
        ignore_token_ids = torch.nonzero((tokens_weights_tensor==0.)).squeeze(-1)
        flat_targets_tokens_tensor[ignore_token_ids] = self.ignore_index
        
        loss_cls = self.loss_reg(logits, flat_targets_tokens_tensor) / avg_factor.to(torch.float32) * self.cls_loss_weight
        
        # binary mask
        seq_embed = seq_embed.view(num_imgs,num_queries, *seq_embed.shape[-2:])
        gt_masks = []
        pred_masks = []
        for k, gt_instance in enumerate(batch_gt_instances):
            single_assign_ids = assign_inds[k]
            single_seq_embed = seq_embed[k]
            single_image_features = image_features[k]
            single_targets_tokens = targets_tokens_tensor[k]
            
            gt_mask = torch.from_numpy(gt_instance.masks.masks).to(assign_inds.device)
            gt_assign_ids = single_assign_ids[single_assign_ids>0] -1
            gt_mask = gt_mask[gt_assign_ids]
            gt_masks.append(gt_mask)
            
            pos_seq_embed = single_seq_embed[single_assign_ids>0]
            pos_targets_tokens = single_targets_tokens[single_assign_ids>0]
            mask_features = pos_seq_embed[pos_targets_tokens==self.mask_token_id]

            pred_mask = mask_features @ single_image_features.flatten(0,1).permute(1,0) / math.sqrt(mask_features.shape[-1])
            height,width = single_image_features.shape[:2]
            pred_mask = pred_mask.view(mask_features.shape[0], height, width)
            pred_mask = pred_mask.view(gt_mask.shape[0], 4, 4, height, width).permute(0, 3, 1, 4, 2).flatten(1,2).flatten(2,3)
            pred_masks.append(pred_mask)
            

        pred_masks = torch.cat(pred_masks,dim=0)
        gt_masks = torch.cat(gt_masks,dim=0)
        
        num_total_masks = reduce_mean(pred_seq_logits.new_tensor([len(pred_masks)]))
        num_total_masks = max(num_total_masks, 1)
        
        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                pred_masks.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            pred_masks.unsqueeze(1), points_coords).squeeze(1)
        
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)
        
        # mask loss
        h, w = pred_masks.shape[-2:]
        loss_mask = sigmoid_focal_loss(
            mask_point_preds, mask_point_targets, num_total_masks) * self.mask_loss_weight
        
        return (loss_cls, loss_mask, loss_dice)
    
    def decoder_inference(self, backbone, patch_embed: Tensor, patch_mask: Tensor, text_embed: Tensor, text_mask: Tensor, 
            grid_pos_embed: Tensor, grid_mask: Tensor, references: Tensor, 
            tokenizer, grid_interpolate: bool=True, global_only_image: bool=True) -> Dict:
        """AutoRegressive decoding target tokens.
        
        Args:
            layers_module (torch module): transformer module with parameter.
            patch_embed (Tensor): image patch embedding has (bs, patch_H, patch_W, C).
            patch_mask (Tensor): image patch mask has (bs, patch_H, patch_W).
            text_embed (Tensor): text input embedding. Default is None.
            text_mask (Tensor): text input mask. Default is None.
            grid_pos_embed (Tensor): grid_pos_embed has (bs, sampled_query_num, C).
                task identifier + position embedding.
            grid_mask (Tensor): grid mask has (bs, grid_H, grid_W)
            references (Tensor): normalized grid position (bs, num_queries, 2).
            bert_embed_func (Callable): bert embedding function.
            task_embedding (Tensor): task identifier embedding for each task with shape (C)
            vocabulary_embed (Tensor): dynamic vocabulary for this task with (vocabulary_num, C)
            grid_interpolate (bool): if use grid interpolation for local information. Default is True.
            global_only_image (bool): if global layer only process image. Default is True.

        Returns:
            dict: The dictionary of decoding outputs.
        """
        patch_resolution = patch_embed.shape[1:3]
        grid_resolution_perwin = [grid_mask.shape[i+1] // (patch_resolution[i] \
                    // backbone.layers[0].window_size) for i in range(2)]
        batch_size, query_num = references.shape[:2]
        references = references[:, :, :2]
        grid_mask = grid_mask.flatten(1)

        # compute observation interaction (e.g., image, text, and local feature token)
        _, pre_kv_list = backbone.text_forward(text_embed, key_padding_mask=text_mask)
        patch_embed, pre_kv_list = backbone.img_forward(patch_embed, pre_kv_list, grid_interpolate, references, text_mask)
        _, pre_kv_list = backbone.grid_forward(grid_pos_embed, pre_kv_list, grid_interpolate, references, False, False, global_only_image)
 
        input_ids = torch.full((batch_size*query_num*self.beam_num, 1), tokenizer.cls_token_id, device=patch_embed.device)
    
        past_len = 0
        if self.beam_num > 1:
            outputs_ids, outputs_logits, outputs_feats = beam_search(backbone, input_ids, pre_kv_list, batch_size*query_num, self.beam_num, past_len, self.max_length, global_only_image,
                                                    self.temperature, self.alpha,
                                         tokenizer.bos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)
        else:
            outputs_ids = []
            outputs_logits = []
            outputs_feats = []
            end_mask = torch.zeros_like(input_ids).bool()
            for i in range(self.max_length):
                token_feat, logits, pre_kv_list = backbone.decode_forward(input_ids, pre_kv_list, past_len+i, global_only_image)
                softmax_logits = logits.softmax(dim=-1)
                logits, input_ids = softmax_logits.max(dim=-1)
                end_mask = (input_ids == tokenizer.sep_token_id) | end_mask
                input_ids[end_mask] = tokenizer.sep_token_id
                outputs_ids.append(input_ids)
                outputs_logits.append(logits)
                outputs_feats.append(token_feat)
                if end_mask.all():
                    break
            outputs_ids = torch.cat(outputs_ids,dim=1)
            outputs_logits = torch.cat(outputs_logits, dim=1)
            outputs_feats = torch.cat(outputs_feats,dim=1)

        outputs_ids = outputs_ids.view(batch_size, -1, outputs_ids.shape[-1])
        outputs_logits = outputs_logits.view(batch_size, -1, outputs_logits.shape[-1])
        outputs_feats = outputs_feats.view(batch_size, -1, *outputs_feats.shape[-2:])
        outputs_references = references.view(batch_size, -1, references.shape[-1])
        
        output_dict = {'outputs_logits': outputs_logits, 'outputs_ids': outputs_ids, 'references':outputs_references, 'image_feats':patch_embed, 'outputs_feats':outputs_feats}
            
        return output_dict
    
    def predict(self, outputs_logits: Tensor, outputs_ids: Tensor, references: Tensor,
                image_feats, outputs_feats,
                batch_data_samples: SampleList, rescale: bool = True, tokenizer=None) -> InstanceList:
        """Perform inference of instance segmentation head.

        Args:
            outputs_classes (Tensor): Classification scores of the last layer, 
                has shape (bs, num_queries, cls_out_channels).
            outputs_coords (Tensor): Regression outputs of the last layers. 
                Each is a 3D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (bs, num_queries, 4).
            outputs_polygons (Tensor): normalized polygons format 
                (d_1, d_2, ..., d_raynum), has shape (num_decoder_layers, bs, 
                num_queries, polygon_num), it normalized by half hypotenuse of image shape.
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                 Samples. It usually includes information such as
                 `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): If `True`, return boxes in original image space. 

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            logits = outputs_logits[img_id]
            ids = outputs_ids[img_id]
            img_meta = batch_img_metas[img_id]
            refer = references[img_id]
            img_feat = image_feats[img_id]
            output_feat = outputs_feats[img_id]
            results = self._predict_single(logits, ids, refer, img_feat, output_feat, img_meta, rescale, tokenizer)
            result_list.append(results)
        return result_list

    def _predict_single(self, logits: Tensor, ids: Tensor, refer: Tensor, img_feat, output_feat,
                                img_meta: dict, rescale: bool = True, tokenizer = None) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox and polygon predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Argmax outputs from the last layer for each image, 
                with coordinate format (cx, cy, w, h) and shape [num_queries, 4].
            polar_dis_pred (Tensor): normalized polygons format 
                (d_1, d_2, ..., d_raynum), has shape (num_decoder_layers, 
                bs, num_queries, ray_num), it normalized by half hypotenuse 
                of image shape.
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            :obj:`InstanceData`: Instance Segmentation results of each image after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 
                  arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, ori_h, ori_w).
        """
        max_per_img = self.test_cfg.get('max_per_img', len(logits))
        # NOTE: assume that all the images are in the same scale 
        img_shape = img_meta['img_shape'] # or img_meta['batch_input_shape']
        
        scores, indexes = logits[...,0].reshape(-1).topk(len(logits))

        topk_ids = ids[indexes]
        topk_refer = refer[indexes]
        topk_feat = output_feat[indexes]
        
        topk_texts = tokenizer.batch_decode(topk_ids,skip_special_tokens=True)

        dataset_name = img_meta['dataset_name']
        class_names = list(self.dataset_labels[dataset_name])
        # translate to lower
        class_names = [name.lower() for name in class_names]
        class_names_dict = {name:i for i,name in enumerate(class_names)}
        det_labels = []
        det_scores = []
        mask_feat = []
        for k, text in enumerate(topk_texts):
            splits = text.split(',')
            if splits[0] != 'background' and splits[0] in class_names and '<MASK>' in text:
                if len(topk_feat[k][topk_ids[k]==self.mask_token_id]) != 16:
                    continue
                det_labels.append(class_names_dict[splits[0]])
                det_scores.append(scores[k])
                # NOTE: only 16
                mask_feat.append(topk_feat[k][topk_ids[k]==self.mask_token_id][:16])
        if len(mask_feat) > 0:
            mask_feat = torch.stack(mask_feat,dim=0).flatten(0,1)
            assert mask_feat.shape[0] % 16 == 0

            pred_mask = mask_feat @ img_feat.flatten(0,1).permute(1,0) / math.sqrt(mask_feat.shape[-1])
            height,width = img_feat.shape[:2]
            pred_mask = pred_mask.view(mask_feat.shape[0], height, width)
            pred_mask = pred_mask.view(mask_feat.shape[0]//16, 4, 4, height, width).permute(0, 3, 1, 4, 2).flatten(1,2).flatten(2,3)
            

            pred_mask = F.interpolate(
                pred_mask.unsqueeze(1),
                img_meta['ori_shape'],
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            pred_mask = pred_mask.sigmoid()
            det_labels = torch.LongTensor(det_labels).to(topk_refer.device)
            det_scores = torch.FloatTensor(det_scores).to(topk_refer.device)
            
            # filter mask too small
            nonzero_mask = ((pred_mask.flatten(1,2)>0.5).int().sum(-1)>0)
            pred_mask = pred_mask[nonzero_mask]
            det_labels = det_labels[nonzero_mask]
            det_scores = det_scores[nonzero_mask]
            
            # recompute score
            for k in range(len(det_scores)):
                mask = pred_mask[k]
                mask_score = mask[mask>0.5].mean()
                det_scores[k] = det_scores[k] * mask_score
            pred_mask = pred_mask > 0.5
    
            # nms
            if self.nms is not None:
                det_scores, det_labels, _, keep_inds = mask_matrix_nms(
                    pred_mask,
                    det_labels,
                    det_scores,
                    kernel=self.nms.kernel,
                    sigma=self.nms.sigma,
                    filter_thr=self.nms.filter_thr)
                pred_mask = pred_mask[keep_inds]
        else: 
            det_labels = torch.LongTensor([]).to(topk_refer.device)
            det_scores = torch.FloatTensor([]).to(topk_refer.device)
            pred_mask = torch.BoolTensor([]).to(topk_refer.device)
        
        results = InstanceData()
        results.scores = det_scores
        results.labels = det_labels
        results.masks = pred_mask
        
        # dummy
        results.bboxes = torch.zeros(len(det_scores),4).to(topk_refer.device)
        return results