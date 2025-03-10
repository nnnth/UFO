# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig, reduce_mean, InstanceList, OptInstanceList)
from ..utils import multi_apply
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, get_box_tensor
from mmcv.ops import batched_nms
from .dataset_labels import dataset_labels

@MODELS.register_module()
class UFOInternVLDetHead(BaseModule):
    r"""Detection head for UFO. It's a non-parametric head for 
        UFO decoding and post-processing in detection task.
    """
    def __init__(self,
            train_cfg: ConfigType = None,
            test_cfg: ConfigType = None,
            init_cfg: OptMultiConfig = None,
            task_prompt: str='detection',
            ignore_index=-100,
            max_length=30,
            beam_num=1,
            temperature=0.3,
            alpha=10.0,
            nms=None,
            repeat_times=3,
            sample_prob=0.0,) -> None:
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
        self.nms = nms
        self.repeat_times = repeat_times
        self.sample_prob = sample_prob
        self.dataset_labels = dataset_labels

        self._init_layers()

    def _init_layers(self) -> None:
        pass

    def init_weights(self) -> None:
        pass

    def reset_hyparameter(self, cfgs):
        for k in list(cfgs.keys()):
            setattr(self, k, cfgs[k])
        self.loss_reg = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)
    
    def get_prompt(self, batch_data_samples):
        dataset_name = batch_data_samples[0].metainfo['dataset_name']
        classes = list(self.dataset_labels[dataset_name])

        num_classes = batch_data_samples[0].metainfo['head_cfg']['num_classes']
        num_bins = batch_data_samples[0].metainfo['head_cfg']['num_bins']
        classes_text = ','.join(classes)
        prompt = f'<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user\nDetect objects in the image that belong to the following categories: {classes_text}. The coordinate range is 0-{num_bins}.\n<image><|im_end|><|im_start|>assistant\n'
        return [prompt for img in batch_data_samples]
    
    def get_prompt_sample(self, batch_data_samples, target_tokens):
        if torch.rand(1)[0] > self.sample_prob:
            return self.get_prompt(batch_data_samples)
        dataset_name = batch_data_samples[0].metainfo['dataset_name']
        classes = list(self.dataset_labels[dataset_name])

        num_classes = batch_data_samples[0].metainfo['head_cfg']['num_classes']
        num_bins = batch_data_samples[0].metainfo['head_cfg']['num_bins']
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
                sample_neg_classes = neg_classes[torch.randperm(len(neg_classes))[:len(pos_classes)*3]]
                sample_classes = torch.cat([pos_classes,sample_neg_classes])
            else:
                print("Detection: no negative class")
                print(pos_classes)
                sample_classes = pos_classes
            sample_classes = sample_classes[torch.randperm(len(sample_classes))]
            sample_classes_list =[classes[i] for i in sample_classes]
            classes_text = ','.join(sample_classes_list)
            prompt = f'<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user\nDetect objects with the following categories in the image: {classes_text}. The coordinate range is 0-{num_bins}.\n<image><|im_end|><|im_start|>assistant\n'
            prompts.append(prompt)
        return prompts
    
    def translate_text(self,targets_tokens_tensor, batch_img_metas):
        dataset_name = batch_img_metas[0]['dataset_name']
        classes = list(self.dataset_labels[dataset_name]) + ['background']

        num_classes = batch_img_metas[0]['head_cfg']['num_classes']
        num_bins = batch_img_metas[0]['head_cfg']['num_bins']
        use_vocab_list = classes + [str(i) for i in range(num_bins+1)]

        target_texts_list = []
        for targets_tokens in targets_tokens_tensor:
            target_texts = []
            for tokens in targets_tokens:
                if use_vocab_list[tokens[0]] == 'background':
                    text = 'background'
                else:
                    label_text = use_vocab_list[tokens[0]]
                    box_text = ','.join([use_vocab_list[idx] for idx in tokens[1:]])
                    text = f'{label_text},<box>{box_text}</box>'
                target_texts.append(text)
            target_texts_list.extend(target_texts)  
        return target_texts_list
    
    def get_targets_based_on_reference(self, 
                reference_preds_list: List[Tensor],
                batch_gt_instances: InstanceList,
                batch_img_metas: List[dict]) -> tuple:
        """Compute regression and classification targets for a batch image.

        Assign targets based on distance between boxes and grids.

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
        ) = multi_apply(self._get_targets_single_based_on_reference,
                                      reference_preds_list,
                                      batch_gt_instances, batch_img_metas)

        # only support parallel training, means query_num of each image is equal
        return (torch.stack(input_tokens_list), torch.stack(targets_tokens_list), 
                torch.stack(tokens_weights_list))
    
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
        # repeat gt, one gt for multi grids
        unrepeat_num = len(gt_instances.labels)
        repeat_labels = gt_instances.labels.unsqueeze(0).repeat(self.repeat_times,1).flatten(0,1)
        repeat_bboxes = gt_instances.bboxes.unsqueeze(0).repeat(self.repeat_times,1,1).flatten(0,1)
        if 'mask' in gt_instances:
            repeat_masks = torch.from_numpy(gt_instances.masks.masks).to(repeat_labels.device).unsqueeze(0).repeat(self.repeat_times,1,1,1).flatten(0,1)
            gt_instances = InstanceData(
                labels=repeat_labels,
                bboxes=repeat_bboxes,
                masks=repeat_masks,
            )
        else:
            gt_instances = InstanceData(
                labels=repeat_labels,
                bboxes=repeat_bboxes,
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
        # get postive gt
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]
        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bboxes = gt_bboxes.new_full((reference_pred.shape[0], 4), self.num_vocal - 1, dtype=torch.float)
        bbox_weights = gt_bboxes.new_zeros((reference_pred.shape[0], 4))
        bbox_weights[pos_inds] = 1.0

        # UFO regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / reference_pred.new_tensor([
                                    img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)

        # center prediction is converted to center residual
        pos_reference_pred_normalized = \
                reference_pred[pos_inds] / reference_pred.new_tensor([
                                            img_w, img_h]).unsqueeze(0)
        pos_center_residual = pos_gt_bboxes_targets[:, :2] - pos_reference_pred_normalized
        # assume pos_center_residual range [-0.5, 0.5]
        pos_gt_bboxes_targets[:, :2] = pos_center_residual + 0.5
                
        bboxes[pos_inds] = pos_gt_bboxes_targets # [0., 1.]

        # convert to gt to tokens
        # coord and scale token [0, self.num_bins]
        bboxes_tokens = (bboxes * self.num_bins).round().long().clamp(min=0, max=self.num_bins)

        # classification token
        bboxes_tokens = bboxes_tokens + self.num_classes + 1
        
        # ignore 
        bboxes_tokens[neg_inds] = self.num_vocal - 1
        
        targets_tokens = torch.cat([labels[:, None], bboxes_tokens], dim=1)
        tokens_weights = torch.cat([label_weights[:, None], bbox_weights], dim=1)

        # input tokens for parallel training
        input_tokens = targets_tokens[:, :-1]

        return (input_tokens, targets_tokens, tokens_weights)
    
    def loss(self, all_layer_pred_seq_logits: Tensor,
                   all_layer_target_tokens: List[Tensor],
                   all_layer_token_weights: List[Tensor],
                   batch_gt_instances,
                   batch_img_metas) -> Dict[str, Tensor]:

        loss_inputs = (all_layer_pred_seq_logits,
                       all_layer_target_tokens,
                       all_layer_token_weights,
                       batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
    
    def loss_by_feat(self, all_layer_pred_seq_logits: Tensor,
                           all_layer_target_tokens: List[Tensor],
                           all_layer_token_weights: List[Tensor],
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
        losses_cls = multi_apply(
            self.loss_by_feat_single,
            all_layer_pred_seq_logits,
            all_layer_target_tokens,
            all_layer_token_weights,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
    
        loss_dict['loss_cls'] = losses_cls[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i in losses_cls[:-1]:
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i[0]
            num_dec_layer += 1
        return loss_dict

    def loss_by_feat_single(self, pred_seq_logits: Tensor, 
                                  targets_tokens_tensor: Tensor,
                                  tokens_weights_tensor: Tensor,
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
            Tuple[Tensor]: A tuple including `loss_cls` and `loss_reg`
        """
        num_imgs, num_queries, seq_len, vocab_size = pred_seq_logits.shape

        pred_seq_logits = pred_seq_logits.view(-1, vocab_size)

        tokens_weights_tensor = tokens_weights_tensor.contiguous().view(-1)
        targets_tokens_tensor = targets_tokens_tensor.contiguous().view(-1)
        ignore_token_ids = torch.nonzero((tokens_weights_tensor==0.)).squeeze(-1)
        targets_tokens_tensor[ignore_token_ids] = self.ignore_index

        loss_cls = self.loss_reg(pred_seq_logits, targets_tokens_tensor)
        return (loss_cls,)
    
    def predict(self, outputs_logits: Tensor, outputs_ids: Tensor, references: Tensor,
            batch_data_samples: SampleList, rescale: bool = True, tokenizer=None) -> InstanceList:
        """Perform inference of detection head.

        Args:
            outputs_classes (Tensor): Classification scores of the last layer, 
                has shape (bs, num_queries, cls_out_channels).
            outputs_coords (Tensor): Regression outputs of the last layers. 
                Each is a 3D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (bs, num_queries, 4).
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
            results = self._predict_single(logits, ids, refer, img_meta, rescale, tokenizer)
            result_list.append(results)
        return result_list

    def _predict_single(self, logits: Tensor, ids: Tensor, refer: Tensor,
                    img_meta: dict, rescale: bool = True, tokenizer = None) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Argmax outputs from the last layer for each image, 
                with coordinate format (cx, cy, w, h) and shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image space. 

        Returns:
            :obj:`InstanceData`: Detection results of each image after the post process.
            Each item usually contains following keys.
                - scores (Tensor): Classification scores, has a shape (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4), the last dimension 4 
                  arrange as (x1, y1, x2, y2).
        """
        
        max_per_img = self.test_cfg.get('max_per_img', len(logits))
        # NOTE: assume that all the images are in the same scale 
        img_shape = img_meta['img_shape'] # or img_meta['batch_input_shape']
        
        scores, indexes = logits[...,0].reshape(-1).topk(len(logits))
        
        topk_ids = ids[indexes]
        topk_refer = refer[indexes]
        
        topk_texts = tokenizer.batch_decode(topk_ids)

        num_bins = img_meta['head_cfg']['num_bins']
        dataset_name = img_meta['dataset_name']
        class_names = list(self.dataset_labels[dataset_name])
        # translate to lower
        class_names = [name.lower() for name in class_names]
        class_names_dict = {name:i for i,name in enumerate(class_names)}
        det_labels = []
        det_bboxes = []
        det_scores = []
        for k, text in enumerate(topk_texts):
            text = text.split('<|im_end|>')[0]
            splits = text.split(',')
            pred_cls = splits[0].strip()
            if pred_cls != 'background' and pred_cls in class_names:
                det_labels.append(class_names_dict[pred_cls])
                det_scores.append(scores[k])
                try: 
                    box_texts = text.split('</box>')[0].split('<box>')[1].strip().split(',')
                    bboxes = torch.FloatTensor([int(num.replace(' ',''))/num_bins for num in box_texts]).to(topk_refer.device)
                    bboxes[:2] = bboxes[:2] - 0.5 + topk_refer[k]
                    if len(bboxes) != 4:
                        del det_labels[-1]
                        del det_scores[-1]
                    else:
                        det_bboxes.append(bboxes)
                except:
                    # print("parse fail",text)
                    del det_labels[-1]
                    del det_scores[-1]
            if len(det_labels) == max_per_img:
                break
        if len(det_labels) > 0:
            det_labels = torch.LongTensor(det_labels).to(topk_refer.device)
            det_scores = torch.FloatTensor(det_scores).to(topk_refer.device)
            det_bboxes = torch.stack(det_bboxes,dim=0)
            
            det_bboxes = bbox_cxcywh_to_xyxy(det_bboxes)
            det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
            det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
            det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
            det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
            if rescale:
                assert img_meta.get('scale_factor') is not None
                det_bboxes /= det_bboxes.new_tensor(
                    img_meta['scale_factor']).repeat((1, 2))
        else:
            det_labels = torch.LongTensor([]).to(topk_refer.device)
            det_scores = torch.FloatTensor([]).to(topk_refer.device)
            det_bboxes = torch.FloatTensor([]).to(topk_refer.device).view(-1,4)

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = det_scores
        results.labels = det_labels
        # use nms
        if len(det_labels)>0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                                results.labels, self.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
        return results

