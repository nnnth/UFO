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
from .beam_search import beam_search

@MODELS.register_module()
class UFOViTCaptionHead(BaseModule):
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
        self.loss_cls = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.ignore_index)
    
    def get_prompt(self, batch_data_samples):
        prompt = f'Task: {self.task_prompt}; Give a caption for this image:'
        return [prompt for img in batch_data_samples]

    def get_targets_based_on_reference(self, 
                reference_preds_list: List[Tensor],
                batch_gt_instances: InstanceList,
                batch_img_metas: List[dict],
                tokenizer: None) -> tuple:
        """Prepare next token targets for caption.

        Args:
            reference_preds_list (list[Tensor]): Grid positions for each image, 
                with normalized coordinate (cx, cy) and shape [num_queries, 2].
                Dummy for caption.
            batch_gt_instances (list[str]): Batch of
                gt_instance. It includes raw caption text.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            tokenizer: tokenizer class used in all tasks.
        Returns:
            tuple: a tuple containing the following targets.

            - input_tokens_tensor (Tensor): Input tokens of each image for training.
              has shape (bs, num_queries, dec_length).
            - target_tokens_tensor (Tensor): GT tokens of each image (bs, num_queries, dec_length).
            - tokens_weights_tensor (Tensor): GT tokens weights of each image, 
              has shape (bs, num_queries, dec_length).
        """
        batch_size = len(reference_preds_list)
        (input_tokens_list, target_tokens_list, tokens_weights_list, pos_inds_list, 
         neg_inds_list) = multi_apply(self._get_targets_single_based_on_reference,
                                reference_preds_list, batch_gt_instances, 
                                batch_img_metas, [tokenizer for _ in range(batch_size)])

        # only support parallel training, means query_num of each image is equal
        return (torch.stack(input_tokens_list), torch.stack(target_tokens_list), 
                torch.stack(tokens_weights_list))
        
    def _get_targets_single_based_on_reference(self, reference_pred: Tensor,
                gt_instances: InstanceData,
                img_meta: dict,
                tokenizer: None) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            reference_pred (Tensor): Grid positions for each image, 
                with normalized coordinate (cx, cy) and shape [num_queries, 2] 
                or normalized coordinate (cx, cy, w, h) and shape [num_queries, 4].
                Dummy for caption.
            gt_instances (:obj:`str`): Ground truth of caption text
            img_meta (dict): Meta information for one image.
            tokenizer: tokenizer class used in all tasks.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - input_tokens (Tensor): Input tokens of each image for training.
            - target_tokens (Tensor): GT tokens of each image.
            - tokens_weights (Tensor]): GT tokens weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # tokenizer raw text
        text = tokenizer(
            gt_instances,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(reference_pred.device)
        # text.input_ids[:, 0] = tokenizer.bos_token_id
        input_tokens = text.input_ids
        tokens_weights = text.attention_mask
        # not pred at end token
        tokens_weights[0,tokens_weights.sum()-1] = 0

        # prepare targets, ignore pad token and start token
        target_tokens = input_tokens.masked_fill(
            input_tokens == tokenizer.pad_token_id, self.ignore_index)
        target_tokens[...,:1] = self.ignore_index

        return (input_tokens, target_tokens, tokens_weights)
    
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
                            target_tokens_tensor: Tensor,
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
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`
        """
        # not pred at token
        pred_seq_cls_logits = pred_seq_logits[:, :, :-1,:].reshape(-1, self.num_vocal)
        # construct weighted avg_factor 
        cls_avg_factor = tokens_weights_tensor.sum()
        cls_avg_factor = reduce_mean(
            pred_seq_logits.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # shift target to next token
        target_tokens_tensor = target_tokens_tensor[:,:,1:].contiguous()
    
        loss_cls = self.loss_cls(pred_seq_cls_logits, target_tokens_tensor.view(-1)) / cls_avg_factor

        return (loss_cls,)
    
    def decoder_inference(self, backbone, patch_embed: Tensor, patch_mask: Tensor, text_embed: Tensor, text_mask: Tensor, 
            grid_pos_embed: Tensor, grid_mask: Tensor, references: Tensor, 
            tokenizer, grid_interpolate: bool=True, global_only_image: bool=False) -> Dict:
        """AutoRegressive decoding target tokens.
        
        Args:
            layers_module (torch module): transformer module with parameter.
            patch_embed (Tensor): image patch embedding has (bs, patch_H, patch_W, C).
            patch_mask (Tensor): image patch mask has (bs, patch_H, patch_W).
            text_embed (Tensor): text input embedding. Default is None.
            text_mask (Tensor): text input mask. Default is None.
            grid_pos_embed (Tensor): grid_pos_embed has (bs, sampled_query_num, C).
                task identifier + position embedding.
            references (Tensor): normalized grid position (bs, num_queries, 2).
            bert_embed_func (Callable): bert embedding function.
            task_embedding (Tensor): task identifier embedding for each task with shape (C)
            vocabulary_embed (Tensor): dynamic vocabulary for this task with (vocabulary_num, C)
            grid_interpolate (bool): if use grid interpolation for local information. Default is True.
            global_only_image (bool): if global layer only process image. Default is True.

        Returns:
            dict: The dictionary of decoding outputs.
        """
        batch_size, query_num = references.shape[:2]
        references = references[:, :, :2]

        # compute observation interaction (e.g., image, text, and local feature token)
        _, pre_kv_list = backbone.text_forward(text_embed, key_padding_mask=text_mask)
        _, pre_kv_list = backbone.img_forward(patch_embed, pre_kv_list, grid_interpolate, references, text_mask)
 
        input_ids = torch.full((batch_size*query_num*self.beam_num, 1), tokenizer.cls_token_id, device=patch_embed.device)
        
        past_len = 0
        if self.beam_num > 1:
            outputs_ids, outputs_logits,_ = beam_search(backbone, input_ids, pre_kv_list, batch_size*query_num, self.beam_num, past_len, self.max_length, global_only_image,
                                                    self.temperature, self.alpha,
                                         tokenizer.bos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)
        else:
            outputs_ids = []
            outputs_logits = []
            end_mask = torch.zeros_like(input_ids).bool()
            for i in range(self.max_length):
                _, logits, pre_kv_list = backbone.decode_forward(input_ids, pre_kv_list, past_len+i, global_only_image)
                softmax_logits = logits.softmax(dim=-1)
                logits, input_ids = softmax_logits.max(dim=-1)
                end_mask = (input_ids == tokenizer.sep_token_id) | end_mask
                input_ids[end_mask] = tokenizer.sep_token_id
                outputs_ids.append(input_ids)
                outputs_logits.append(logits)
                if end_mask.all():
                    break
            outputs_ids = torch.cat(outputs_ids,dim=1)
            outputs_logits = torch.cat(outputs_logits, dim=1)
 
        outputs_ids = outputs_ids.view(batch_size, -1, outputs_ids.shape[-1])
        outputs_logits = outputs_logits.view(batch_size, -1, outputs_logits.shape[-1])
        outputs_references = references.view(batch_size, -1, references.shape[-1])
        output_dict = {'outputs_ids': outputs_ids}
            
        return output_dict
    
    def predict(self, outputs_ids: Tensor, batch_data_samples: SampleList, rescale: bool = True, tokenizer=None) -> List:
        
        outputs_ids = outputs_ids.squeeze(1)
        outputs_texts = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        return outputs_texts
