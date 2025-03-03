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

import transformers 
from . import llava_conversation as conversation_lib
from .llava_constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from .llava_mm_utils import tokenizer_image_token

IGNORE_TOKEN_ID = -100       

def preprocess_llava(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        template_name,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN * num_image_token_list[i]}{DEFAULT_IM_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "  # <|im_end|><|im_start|>assistant\n

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            # assert len(parts) == 2, (len(parts), rou)
            if len(parts) != 2:
                print(len(parts), rou)
                parts.append('')
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_masks,
    )

@MODELS.register_module()
class UFOLLaVAVQAHead(BaseModule):
    r"""Caption head for UFO. It's a non-parametric head for 
        UFO decoding and post-processing in image caption task.
    """
    def __init__(self,
            init_cfg: OptMultiConfig = None,
            task_prompt: str='vqa',
            template_name: str='Hermes-2',
            ignore_index=-100,
            max_length=20,
            beam_num=2,
            temperature: float=0.7,
            alpha: float=0.75,
            ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.task_prompt = task_prompt
        self.template_name = template_name
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
        
    def get_conversations(self, batch_data_samples,  training, tokenizer, num_image_token=256):
        conversations_list = []
        num_image_token_list = []
        for data_sample in batch_data_samples:
            conversations_list.append(data_sample.conversations)
            num_image_token_list.append(num_image_token)
        ret_dict = preprocess_llava(conversations_list, tokenizer, num_image_token_list, self.template_name)

        return ret_dict

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
