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
from mmdet.structures import SampleList, DataSample
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig, reduce_mean, InstanceList, OptInstanceList)
from ..utils import multi_apply
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
@MODELS.register_module()
class UFOInternVLCaptionHead(BaseModule):
    r"""Caption head for UFO. It's a non-parametric head for 
        UFO decoding and post-processing in image caption task.
    """
    def __init__(self,
            init_cfg: OptMultiConfig = None,
            task_prompt: str='caption',
            ignore_index=-100,
            max_length=20,
            beam_num=2,
            temperature: float=0.7,
            alpha: float=0.75,
            ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.task_prompt = task_prompt
        self.ignore_index = ignore_index
        self.max_length = max_length
        self.beam_num = beam_num
        self.temperature = temperature
        self.alpha = alpha
        self._init_layers()

    def _init_layers(self) -> None:
        pass

    def init_weights(self) -> None:
        pass
    
    def reset_hyparameter(self, cfgs):
        for k in list(cfgs.keys()):
            setattr(self, k, cfgs[k])
        self.loss_cls = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)
        
    def get_conversations(self, batch_data_samples, training):
        prompt_template = '<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user\nProvide a one-sentence caption for the provided image.\n<image><|im_end|><|im_start|>assistant\n'
        target_texts_list = []
        for ds in batch_data_samples:
            if training:
                target_text = prompt_template + ds.gt_caption + '<|im_end|>'
            else:
                target_text = prompt_template
            target_texts_list.append(target_text)
        return target_texts_list
    
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
            target_tokens_tensor (Tensor): GT targets for autoregressive head, 
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
            Tuple[Tensor]: A tuple including `loss_cls`
        """
        # not pred at token
        pred_seq_cls_logits = pred_seq_logits.reshape(-1, pred_seq_logits.shape[-1])

        # shift target to next token
        tokens_weights_tensor = tokens_weights_tensor.contiguous().view(-1)
        targets_tokens_tensor = targets_tokens_tensor.contiguous().view(-1)
        ignore_token_ids = torch.nonzero((tokens_weights_tensor==0.)).squeeze(-1)
        targets_tokens_tensor[ignore_token_ids] = self.ignore_index

        loss_cls = self.loss_cls(pred_seq_cls_logits, targets_tokens_tensor)
        return (loss_cls,)
    
    def predict(self, outputs_ids: Tensor, outputs_logits: Tensor, batch_data_samples: SampleList, rescale: bool = True, tokenizer=None) -> List:
        
        outputs_ids = outputs_ids.squeeze(1)
        outputs_texts = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        return outputs_texts
