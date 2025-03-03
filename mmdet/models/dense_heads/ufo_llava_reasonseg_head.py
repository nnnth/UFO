# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.model import BaseModule
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig, reduce_mean, InstanceList, OptInstanceList)
from ..utils import multi_apply
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from torch.nn.utils.rnn import pad_sequence
import random 
import math 
from ..utils import get_uncertain_point_coords_with_randomness
from mmcv.ops import point_sample
from .mask_postprocess import keep_largest_n_components

SHORT_QUESTION_LIST = [
    "Can you segment the {class_name} in this image?",
    "Please segment the {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    "{sent} Please respond with segmentation mask.",
    "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]
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
class UFOLLaVAReasonSegHead(BaseModule):
    r"""Visual Grounding head for UFO. It's a non-parametric head for 
        UFO decoding and post-processing in visual grounding task.
    """
    def __init__(self,
            train_cfg: ConfigType = None,
            test_cfg: ConfigType = None,
            init_cfg: OptMultiConfig = None,
            task_prompt: str='refer_segmentation',
            ignore_index=-100,
            max_length=20,
            beam_num=1,
            temperature=1.0,
            alpha=1.0,
            mask_token_id=151655,
            mask_loss_weight=1.,
            cls_loss_weight=1.,
            loss_dice: ConfigType = dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                naive_dice=True,
                loss_weight=1.0),
            mask_threshold=0.5,
            ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.task_prompt = task_prompt
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.beam_num = beam_num
        self.temperature = temperature
        self.alpha = alpha
        self.long_question_list = LONG_QUESTION_LIST
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self._init_layers()
        self.mask_token_id = mask_token_id
        # self.loss_mask = MODELS.build(loss_mask)
        self.mask_loss_weight = mask_loss_weight
        self.loss_dice = MODELS.build(loss_dice)
        self.cls_loss_weight = cls_loss_weight
        self.train_cfg = {}
        self.num_points = self.train_cfg.get('num_points', 12544)
        self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
        self.importance_sample_ratio = self.train_cfg.get(
            'importance_sample_ratio', 0.75)
        self.mask_threshold = mask_threshold
        print("mask threshold",self.mask_threshold)
    def _init_layers(self) -> None:
        pass

    def init_weights(self) -> None:
        pass

    def reset_hyparameter(self, cfgs):
        for k in list(cfgs.keys()):
            setattr(self, k, cfgs[k])
        self.loss_reg = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)
    
    def get_conversations(self, batch_data_samples, training):
        target_texts_list = []
        prompt_template = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. User: {question}\n<image> ASSISTANT: "
        for ds in batch_data_samples:
            is_sentence = ds.metainfo['is_sentence']
            if is_sentence:
                if training:
                    question_template = random.choice(self.long_question_list)
                else:
                    question_template = self.long_question_list[-1]
                question = question_template.format(sent=ds.text)
            else:
                if training:
                    question_template = random.choice(self.short_question_list)
                else:
                    question_template = self.short_question_list[-1]
                question = question_template.format(class_name=ds.text.lower())
            prompt = prompt_template.format(question=question)
            if training:
                answer = random.choice(self.answer_list)
                answer = answer.replace('[SEG]', '<MASK>'*16, 1)
                target_texts_list.append(prompt + answer + '</s>')  
            else:
                target_texts_list.append(prompt)
        return target_texts_list
    
    def loss(self, all_layer_pred_seq_logits: Tensor,
                   all_layer_target_tokens: List[Tensor],
                   all_layer_token_weights: List[Tensor],
                   image_features,
                   seq_embed,
                   batch_gt_instances,
                   batch_img_metas) -> Dict[str, Tensor]:

        loss_inputs = (all_layer_pred_seq_logits,
                       all_layer_target_tokens,
                       all_layer_token_weights,
                       [image_features],
                       [seq_embed],
                       batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
    
    def loss_by_feat(self, all_layer_pred_seq_logits: Tensor,
                           all_layer_target_tokens: List[Tensor],
                           all_layer_token_weights: List[Tensor],
                           image_features,
                           seq_embed,
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
            num_total_pos (List[int]): Number of positive samples in all images.
            num_total_neg (List[int]): Number of negative samples in all images.
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
            num_total_pos (int): Number of positive samples in all images.
            num_total_neg (int): Number of negative samples in all images.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_reg`
        """
        num_imgs, num_queries = pred_seq_logits.shape[:2]

        # classification loss
        pred_seq_reg_logits = pred_seq_logits.reshape(-1, pred_seq_logits.shape[-1])
        # construct weighted avg_factor 

        # ignore negative queries regression
        tokens_weights_tensor = tokens_weights_tensor.contiguous().view(-1)
        flat_targets_tokens_tensor = targets_tokens_tensor.contiguous().view(-1)
        ignore_token_ids = torch.nonzero((tokens_weights_tensor==0.)).squeeze(-1)
        flat_targets_tokens_tensor[ignore_token_ids] = self.ignore_index

        # calculate loss
        loss_cls = self.loss_reg(pred_seq_reg_logits, flat_targets_tokens_tensor)

        gt_masks = []
        pred_masks = []
        for k, gt_instance in enumerate(batch_gt_instances):
            single_seq_embed = seq_embed[k]
            single_image_features = image_features[k]
            single_targets_tokens = targets_tokens_tensor[k]
            
            gt_mask = torch.from_numpy(gt_instance.masks).to(seq_embed.device)
            gt_masks.append(gt_mask)
            
            pos_seq_embed = single_seq_embed
            pos_targets_tokens = single_targets_tokens
            mask_features = pos_seq_embed[pos_targets_tokens==self.mask_token_id]

            pred_mask = mask_features @ single_image_features.flatten(0,1).permute(1,0) / math.sqrt(mask_features.shape[-1])
            height,width = single_image_features.shape[:2]
            pred_mask = pred_mask.view(mask_features.shape[0], height, width)
            pred_mask = pred_mask.view(gt_mask.shape[0], 4, 4, height, width).permute(0, 3, 1, 4, 2).flatten(1,2).flatten(2,3)
            pred_mask = F.interpolate(
                pred_mask.unsqueeze(1),
                gt_mask.shape[-2:],
                mode='bilinear',
                align_corners=False).squeeze(1)
            pred_masks.append(pred_mask)
        
        pred_masks = torch.cat(pred_masks,dim=0)
        gt_masks = torch.cat(gt_masks,dim=0)
        
        num_total_masks = reduce_mean(pred_seq_logits.new_tensor([len(pred_masks)]))
        num_total_masks = max(num_total_masks, 1)
        
        # dice loss
        loss_dice = self.loss_dice(
            pred_masks, gt_masks, avg_factor=num_total_masks)
        
        # mask loss
        h, w = pred_masks.shape[-2:]
        loss_mask = sigmoid_focal_loss(
            pred_masks.view(-1,1), gt_masks.view(-1,1).float(), num_total_masks*h*w) * self.mask_loss_weight
        
        return (loss_cls, loss_mask, loss_dice)
    
    def predict(self, outputs_logits: Tensor, outputs_ids: Tensor, 
            image_feats, outputs_feats,
            batch_data_samples: SampleList, rescale: bool = True, tokenizer=None) -> InstanceList:
        """Perform inference of visual grounding head.

        Args:
            outputs_coords (Tensor): Regression outputs of the last layers. 
                Each is a 3D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (bs, num_queries, 4). 
                Default num_queries is 1.
            batch_data_samples (list[:obj:`DataSample`]): The Data
                Samples. It usually includes information such as
                `gt_bboxes`, `text` and so on.
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
            img_feat = image_feats[img_id]
            output_feat = outputs_feats[img_id]
            results = self._predict_single(logits, ids, img_feat, output_feat, img_meta, rescale, tokenizer)
            result_list.append(results)
            
        return result_list

    def _predict_single(self, logits: Tensor, ids: Tensor, img_feat, output_feat,
                    img_meta: dict, rescale: bool = True, tokenizer = None) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            bbox_pred (Tensor): Argmax outputs from the last layer for each image, 
                with coordinate format (cx, cy, w, h) and shape [num_queries, 4].
                Default num_queries is 1
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            results (Tensor): grounding results of each image after the 
                post process, has a shape (1, 4), the last dimension 4 
                arrange as (x1, y1, x2, y2)
        """
        # NOTE: assume that all the images are in the same scale 
        img_shape = img_meta['img_shape'] # or img_meta['batch_input_shape']
        texts = tokenizer.batch_decode(ids,skip_special_tokens=True)
 
        mask_feat = []
        for k, text in enumerate(texts):
            # NOTE: only 16
            mask_feat.append(output_feat[k][ids[k]==self.mask_token_id][:16])
        if len(mask_feat) > 0:
            mask_feat = torch.stack(mask_feat,dim=0).flatten(0,1)
            if (mask_feat.shape[0] % 16 != 0):
                print(texts)
                pred_mask = torch.zeros(1,*img_meta['ori_shape'], device=ids.device).bool()
                return pred_mask
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
            
            pred_mask = pred_mask > self.mask_threshold
        else: 
            pred_mask = torch.BoolTensor([]).to(topk_refer.device)
        if pred_mask.shape[0] == 0:
            print(texts)
            print(mask_feat)
            print(len(mask_feat))
            pred_mask = torch.zeros(1,*img_meta['ori_shape'], device=ids.device).bool()
        pred_mask = torch.from_numpy(keep_largest_n_components(pred_mask.cpu().numpy()))
        return pred_mask 
    
    def add_pred_to_datasample(self, data_samples, seg_pred):
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size = len(seg_pred)
        for i in range(batch_size):
            data_samples[i].set_data({
                'pred_sem_seg':
                PixelData(**{'data': seg_pred[i].to(torch.long)})
            })

        return data_samples