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
from torch.nn.utils.rnn import pad_sequence
import random 

QUESTION_LIST = [
     "Please help me locate {class_name} in the image.",
     " Let’s find {class_name} within the image.",
     "Do you know where {class_name} is within the image?",
     "Where can we locate {class_name} in the image?",
]
 
@MODELS.register_module()
class UFOInternVLGroundHead(BaseModule):
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
        self.question_list = QUESTION_LIST

    def _init_layers(self) -> None:
        pass

    def init_weights(self) -> None:
        pass

    def reset_hyparameter(self, cfgs):
        for k in list(cfgs.keys()):
            setattr(self, k, cfgs[k])
        self.loss_reg = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)
    
    def get_conversations(self, batch_data_samples, training):
        num_bins = batch_data_samples[0].metainfo['head_cfg']['num_bins']
        prompt_template = '<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|><|im_start|>user\n{question}\n<image><|im_end|><|im_start|>assistant\n'
        target_texts_list = []
        for ds in batch_data_samples:
            if training:
                question_template = random.choice(self.question_list)
            else:
                question_template = self.question_list[-1]
            question = question_template.format(class_name=ds.text.lower())
            prompt = prompt_template.format(question=question)

            # translate box to bins
            if training:
                gt_bboxes = ds.gt_bboxes
                img_h, img_w = ds.metainfo['img_shape']
                factor = np.array([[img_w,img_h,img_w,img_h]])
                gt_bboxes = gt_bboxes / factor
                gt_bboxes = bbox_xyxy_to_cxcywh(torch.from_numpy(gt_bboxes))
                gt_bboxes = (gt_bboxes*num_bins).round().long().clamp(min=0,max=num_bins)
                box_str_list = []
                for gt_box in gt_bboxes:
                    box_str = f'<box>{gt_box[0]},{gt_box[1]},{gt_box[2]},{gt_box[3]}</box>'
                    box_str_list.append(box_str)
                answer = ';'.join(box_str_list)
                target_texts_list.append(prompt + answer + '<|im_end|>')  
            else:
                target_texts_list.append(prompt)
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

        # ignore negative queries regression
        tokens_weights_tensor = tokens_weights_tensor.contiguous().view(-1)
        targets_tokens_tensor = targets_tokens_tensor.contiguous().view(-1)
        ignore_token_ids = torch.nonzero((tokens_weights_tensor==0.)).squeeze(-1)
        targets_tokens_tensor[ignore_token_ids] = self.ignore_index

        # calculate loss
        loss_reg = self.loss_reg(pred_seq_reg_logits, targets_tokens_tensor)

        return (loss_reg,)
    
    def predict(self, outputs_logits: Tensor, outputs_ids: Tensor, 
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
            results = self._predict_single(logits, ids, img_meta, rescale, tokenizer)
            result_list.append(results)
            
        return result_list

    def _predict_single(self, logits: Tensor, ids: Tensor, 
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
        texts = tokenizer.batch_decode(ids)

        num_bins = img_meta['head_cfg']['num_bins']
        det_bboxes = []
        for k, text in enumerate(texts):
            try: 
                text = text.split('</box>')[0].split('<box>')[1].strip()
                text = text.replace('.','')
                splits = text.split(',')
                if len(splits) >= 4:
                    bboxes = torch.FloatTensor([int(num.replace(' ',''))/num_bins for num in splits[:4]]).to(logits.device)
                    bboxes[:2] = bboxes[:2]
                else:
                    bboxes = torch.zeros(4).to(logits.device)
                det_bboxes.append(bboxes)
            except:
                # print("parse fail",text)
                det_bboxes.append(torch.zeros(4).to(logits.device))

        det_bboxes = torch.stack(det_bboxes,dim=0)

        try:
            det_bboxes = bbox_cxcywh_to_xyxy(det_bboxes)
        except:
            print("bbox_cxcywh_to_xyxy fail",texts, det_bboxes)
        
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

