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
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from .beam_search import beam_search
from torch.nn.utils.rnn import pad_sequence

@MODELS.register_module()
class UFOViTGroundHead(BaseModule):
    r"""Visual Grounding head for UFO. It's a non-parametric head for 
        UFO decoding and post-processing in visual grounding task.
    """
    def __init__(self,
            train_cfg: ConfigType = None,
            test_cfg: ConfigType = None,
            init_cfg: OptMultiConfig = None,
            task_prompt: str='grounding',
            ignore_index=-100,
            max_length=20,
            beam_num=1,
            temperature=1.0,
            alpha=1.0,) -> None:
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
        self.loss_reg = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.ignore_index)
    
    def get_prompt(self, batch_data_samples):
        return [f'Task: {self.task_prompt}; Description: {ds.text}' for ds in batch_data_samples]
    
    def translate_text(self,targets_tokens_tensor, batch_img_metas):
        num_bins = batch_img_metas[0]['head_cfg']['num_bins']
        use_vocab_list = [str(i) for i in range(num_bins+1)]
        
        target_texts_list = []
        for targets_tokens in targets_tokens_tensor:
            target_texts = []

            for tokens in targets_tokens:
                text = ','.join([use_vocab_list[idx] for idx in tokens[:4]])
                # for multi box
                for k in range(1,len(tokens)//4):
                    if tokens[4*k] != -1:
                        text = text + ';' + ','.join([use_vocab_list[idx] for idx in tokens[4*k:4*(k+1)]])
                    else:
                        break
                target_texts.append(text)
            target_texts_list.extend(target_texts)  
        return target_texts_list
    
    def get_targets_based_on_reference(self, 
                reference_preds_list: List[Tensor],
                batch_gt_instances: InstanceList,
                batch_img_metas: List[dict]) -> tuple:
        """Compute regression targets for a batch image.

        Args:
            reference_preds_list (list[Tensor]): Grid positions for one image, 
                with normalized coordinate (cx, cy) and shape [num_queries, 2]
                or normalized coordinate (cx, cy, w, h) and shape [num_queries, 4].
            batch_gt_instances (list[:obj:`ndarray`]): Batch of
                gt_instance. It usually includes gt box for grounding.
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
        (input_tokens_list, targets_tokens_list, tokens_weights_list
        ) = multi_apply(self._get_targets_single_based_on_reference,
                                      reference_preds_list,
                                      batch_gt_instances, batch_img_metas)
        input_tokens_tensor = pad_sequence(input_tokens_list,batch_first=True,padding_value=-1)
        target_tokens_tensor = pad_sequence(targets_tokens_list,batch_first=True,padding_value=-1)
        tokens_weights_tensor = pad_sequence(tokens_weights_list,batch_first=True,padding_value=-1)
        # only support parallel training, means query_num of each image is equal
        return (input_tokens_tensor.unsqueeze(1), target_tokens_tensor.unsqueeze(1), 
                tokens_weights_tensor.unsqueeze(1))
    
    def _get_targets_single_based_on_reference(self, reference_pred: Tensor,
                gt_instances: InstanceData,
                img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

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
        num_bboxes = gt_instances.shape[0]
        pos_inds = torch.arange(num_bboxes,dtype=torch.long,device=reference_pred.device)
        neg_inds = torch.tensor([],dtype=torch.long,device=reference_pred.device)

        pos_gt_bboxes = torch.from_numpy(gt_instances).to(reference_pred.device)

        # bbox targets
        bbox_targets = torch.ones((num_bboxes, 4), 
                                  device=factor.device) * (self.num_vocal - 1)
        bbox_weights = torch.zeros((num_bboxes, 4), device=factor.device)
        bbox_weights[pos_inds] = 1.0

        # UFO regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / reference_pred.new_tensor([
                                    img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)

        # center prediction is converted to center residual
        pos_reference_pred_normalized = \
                reference_pred / reference_pred.new_tensor([
                                            img_w, img_h]).unsqueeze(0)
        pos_center_residual = pos_gt_bboxes_targets[:, :2] - pos_reference_pred_normalized
        # assume pos_center_residual range [-0.5, 0.5]
        pos_gt_bboxes_targets[:, :2] = pos_center_residual + 0.5
                
        bbox_targets[pos_inds] = pos_gt_bboxes_targets # [0., 1.]

        # convert to gt to tokens
        # coord and scale token [0, self.num_bins]
        targets_tokens = (bbox_targets * self.num_bins).round().long().clamp(min=0, max=self.num_bins)

        # flatten target tokens for multi box
        targets_tokens = targets_tokens.view(-1)
        bbox_weights = bbox_weights.view(-1)
        # input tokens for parallel training
        input_tokens = targets_tokens[:-1]

        return (input_tokens, targets_tokens, bbox_weights)
    
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
        losses_reg = multi_apply(
            self.loss_by_feat_single,
            all_layer_pred_seq_logits,
            all_layer_target_tokens,
            all_layer_token_weights,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
    
        loss_dict['loss_reg'] = losses_reg[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_reg_i in zip(losses_reg[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_reg'] = loss_reg_i[0]
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
            Tuple[Tensor]: A tuple including `loss_reg`
        """
        num_imgs, num_queries = pred_seq_logits.shape[:2]

        # classification loss
        pred_seq_reg_logits = pred_seq_logits.reshape(-1, pred_seq_logits.shape[-1])
        # construct weighted avg_factor 
        avg_factor = tokens_weights_tensor.sum()
        avg_factor = reduce_mean(
            pred_seq_logits.new_tensor([avg_factor]))
        avg_factor = max(avg_factor, 1)

        # ignore negative queries regression
        tokens_weights_tensor = tokens_weights_tensor.contiguous().view(-1)
        targets_tokens_tensor = targets_tokens_tensor.contiguous().view(-1)
        ignore_token_ids = torch.nonzero((tokens_weights_tensor==0.)).squeeze(-1)
        targets_tokens_tensor[ignore_token_ids] = self.ignore_index

        # calculate loss
        loss_reg = self.loss_reg(pred_seq_reg_logits, targets_tokens_tensor) / avg_factor

        return (loss_reg,)
    
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
        batch_size, query_num = references.shape[:2]
        references = references[:, :, :2]

        # compute observation interaction (e.g., image, text, and local feature token)
        _,pre_kv_list = backbone.text_forward(text_embed, key_padding_mask=text_mask)
        _,pre_kv_list = backbone.img_forward(patch_embed, pre_kv_list, grid_interpolate, references, text_mask)
 
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

        output_dict = {'outputs_logits': outputs_logits, 'outputs_ids': outputs_ids, 'references':outputs_references}
            
        return output_dict
    
    def predict(self, outputs_logits: Tensor, outputs_ids: Tensor, references: Tensor,
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
            refer = references[img_id]
            results = self._predict_single(logits, ids, refer, img_meta, rescale, tokenizer)
            result_list.append(results)
            
        return result_list

    def _predict_single(self, logits: Tensor, ids: Tensor, refer: Tensor,
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

        num_bins = img_meta['head_cfg']['num_bins']
        det_bboxes = []
        for k, text in enumerate(texts):
            splits = text.split(',')
            if len(splits) >= 4:
                try: 
                    bboxes = torch.FloatTensor([int(num.replace(' ',''))/num_bins for num in splits[:4]]).to(refer.device)
                    bboxes[:2] = bboxes[:2] - 0.5 + refer[k]
                    det_bboxes.append(bboxes)
                except:
                    # print("parse fail",text)
                    det_bboxes.append(torch.zeros(4).to(refer.device))
            else:
                det_bboxes.append(torch.zeros(4).to(refer.device))

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
        
        results = det_bboxes
        return results

