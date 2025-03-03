# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, Tuple, List, Union
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import Linear
from mmengine.model import xavier_init
from torch import Tensor, nn
from torch.nn.init import normal_
import torch.utils.checkpoint as checkpoint

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList, DetDataSample, \
                             SegOptSampleList, SegSampleList, SegDataSample, DataSample
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.utils import resize_pos_embed
from mmdet.registry import TOKENIZER
from .base import BaseDetector
from .internvl_utils import split_image
from PIL import Image
from transformers.cache_utils import DynamicCache

IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>' 
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
GRID_TOKEN = '<s>'

@MODELS.register_module()
class UFO_InternVL(BaseDetector, metaclass=ABCMeta):
    r"""
    Args:
        use_checkpoints (bool): whether use torch.utils.checkpoints to save 
            cuda memory.
        support_tasks (List): support task names in UFO.
    """
    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 backbone: ConfigType = None, 
                 head_list: OptConfigType = None,
                 use_checkpoints: bool = False,
                 lora_r=128,
                 lora_alpha=2*128,
                 lora_dropout=0.05,
                 train_llm=True,
                 train_mlp=False,
                 train_vit=False,
                 support_tasks: List = ['detection', 'semantic_segmentation', 'instance_segmentation', 'caption','grounding', 'refer_segmentation', 'vqa', 'refer_caption'],
                 tokenizer: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.tokenizer_cfg = tokenizer

        self.tokenizer = TOKENIZER.build(self.tokenizer_cfg)
        self.backbone = MODELS.build(backbone)

        # Add <MASK> token
        extra_tokens = ['<MASK>']
        self.tokenizer.add_tokens(extra_tokens)
        num_new_tokens = len(extra_tokens)
        self.backbone.language_model.resize_token_embeddings(len(self.tokenizer))
        output_embeddings = self.backbone.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        self.backbone.config.llm_config.vocab_size = len(self.tokenizer)
        self.backbone.language_model.config.vocab_size = len(self.tokenizer)

        # set parameter requires_grad
        # train lora (LLM), LLM input embed, LLM output embed, LLM output norm by default
        self.backbone.wrap_llm_lora(r=lora_r, lora_alpha=lora_alpha,lora_dropout=lora_dropout)
        for name,param in self.backbone.named_parameters():
            param.requires_grad = False
        if train_llm:
            self.set_llm()
        if train_mlp:
            self.set_mlp()
        if train_vit:
            self.set_vit()
        for name,param in self.backbone.named_parameters():
            if param.requires_grad:
                print(name)
        for name,param in self.backbone.named_parameters():
            if not param.requires_grad:
                print("not require",name)

        # bulid non parametric task-specific heads for label assignment, loss computation and  post-processing
        self.head_list = head_list
        self.task_specific_heads = dict()
        for head_name in list(self.head_list.keys()):
            head_cfg = self.head_list[head_name]
            self.task_specific_heads[head_name] = MODELS.build(head_cfg)

        self.use_checkpoints = use_checkpoints # checkpoints for saving CUDA memory
        self.support_tasks = support_tasks
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
    
    def set_llm(self):
        # NOTE: use lora for LLM by default
        for name,param in self.backbone.named_parameters():
            if 'lora' in name or 'tok_embeddings' in name or 'language_model.base_model.model.output' in name or 'language_model.base_model.model.model.norm' in name:
                param.requires_grad = True
    
    def set_mlp(self):
        for name,param in self.backbone.named_parameters():
            if 'mlp1' in name:
                param.requires_grad = True
    
    def set_vit(self):
        for name,param in self.backbone.named_parameters():
            if 'vision' in name or 'patch_embedding' in name or 'position_embedding' in name:
                param.requires_grad = True

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.
        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_bboxes` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
            head_inputs_dict = self.forward_visual_modeling(batch_inputs, batch_data_samples)
            losses = self.task_specific_heads[self.mode+'_head'].loss(
                **head_inputs_dict)
        
        # add task name as prefix
        task_losses = dict()
        for k in list(losses.keys()):
            task_losses[self.mode+'_'+k] = losses[k]

        return task_losses
    
    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with 
           task-specific post-processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Perception results of the input images.
            Each DataSample (eg. DetDataSample, SefDataSample, and so on) 
            usually contain pred_result. And the pred_result usually contains 
            various keys with different tasks.

            Detection: 
              - pred_instances:
                  - scores (Tensor): Classification scores
                  - labels (Tensor): Labels of bboxes
                  - bboxes (Tensor): the last dimension 4 arrange as (x1, y1, x2, y2).
            Instance Segmentation: 
              - pred_instances:
                  - scores (Tensor): Classification scores
                  - labels (Tensor): Labels of instances
                  - masks (Tensor): Masks of instances
                  - bboxes (Tensor): the last dimension 4 arrange as (x1, y1, x2, y2).
            Semantic Segmentation: 
              - pred_sem_seg:
                  - data: (Tensor):
              - seg_logits:
                  - data: (Tensor):
            Caption:
              - pred_caption: text of caption.
            Visual Grounding:
              - pred_bboxes (Tensor): Has a shape (1, 4)

        """
        # multi-layer transformer forward passing with specific non-parametric heads
        with torch.amp.autocast(device_type='cuda', dtype=self.dtype):
            head_inputs_dict = self.forward_visual_modeling(batch_inputs, batch_data_samples)
            # post-processing of various tasks
            results_list = self.task_specific_heads[self.mode+'_head'].predict(
                **head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples, tokenizer=self.tokenizer)
        # generate evaluation samples with different formats
        if self.mode in ['detection', 'instance_segmentation']:
            batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        elif self.mode in ['semantic_segmentation', 'refer_segmentation', 'reason_segmentation']:
            batch_data_samples = self.task_specific_heads[self.mode+'_head'].add_pred_to_datasample(
                                                            batch_data_samples, results_list)
        elif self.mode in ['caption', 'refer_caption']:
            for sample, results in zip(batch_data_samples, results_list):
                sample.pred_caption = results
        elif self.mode == 'grounding':
            for sample, results in zip(batch_data_samples,results_list):
                if sample.get('gt_bboxes') is not None:
                    gt_bboxes = torch.Tensor(sample.get('gt_bboxes'))
                    scale_factor = torch.Tensor(sample.metainfo.get('scale_factor'))
                    gt_bboxes /= scale_factor.repeat((1, 2))
                    sample.gt_bboxes = gt_bboxes
                sample.pred_bboxes = results
        else:
            raise NotImplementedError
        return batch_data_samples
    
    def forward_visual_modeling(self, batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Dict:
        """
        Args:
            batch_inputs (Tensor]:  images of each batch with (bs, 3, h, w).
            batch_data_samples (list[:obj:`DetDataSample` or `SegDataSample], 
                optional): The batch data samples. It usually includes 
                information such as `gt_instance` or `gt_panoptic_seg` 
                or `gt_sem_seg`. Defaults to None.

        Returns:
            dict: The dictionary of specific head function inputs.
        """
        # any resolution: split image to blocks
        device = batch_inputs.device
        self.global_batch_size = batch_inputs.shape[0]
        numpy_batch_inputs = batch_inputs.permute(0,2,3,1).cpu().numpy()
        batch_patch_imgs = []
        for numpy_img in numpy_batch_inputs:
            pil_img = Image.fromarray(numpy_img.astype(np.uint8))
            # NOTE: assume all image in batch share same shape
            patch_imgs, window_block_shape = split_image(pil_img)
            batch_patch_imgs.append(patch_imgs)

        batch_inputs = torch.cat(batch_patch_imgs).to(device)
        self.patch_batch_size = batch_inputs.shape[0]

        patch_embed = self.backbone.extract_feature(batch_inputs)
        self.patch_resolution = (int(math.sqrt(patch_embed.shape[1])),int(math.sqrt(patch_embed.shape[1])))
        self.window_size = self.patch_resolution[0]
        self.window_block_shape = window_block_shape

        # construct input dict of multi-layer transformer and non-parametric post-processing 
        transformer_inputs_dict, head_inputs_dict = self.pre_transformer(
            patch_embed, self.patch_resolution, batch_data_samples)
        transformer_inputs_dict['batch_data_samples'] = batch_data_samples

        self.grid_interpolate = self.multi_tasks_cfgs['grid_interpolate']
        if self.grid_interpolate:
            # for multi-prediction tasks: e.g. object detection, semantic segmentation
            transformer_inputs_dict = self.prepare_grid(patch_embed, transformer_inputs_dict)

        # multi-layer transformer forward passing based on the task-specific rules
        # rules are pre-defined in task-specific heads
        with torch.backends.cuda.sdp_kernel(enable_math=False,enable_flash=False,enable_mem_efficient=True):
            if self.grid_interpolate:
                transformer_outputs_dict = self.grid_forward_transformer(**transformer_inputs_dict)
            else:
                transformer_outputs_dict = self.forward_transformer(**transformer_inputs_dict)

        # update post-processing input dict
        head_inputs_dict.update(transformer_outputs_dict)

        return head_inputs_dict

    def pre_transformer(self, patch_embed: Tensor, patch_resolution: Tuple,
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.
        Args:
            patch_embed (Tensor): Image patch embedding, which has
                shape (bs, patch_H, patch_W, C).
            patch_resolution (Tuple): Resolution of the image feature map.
            batch_data_samples (list[:obj:`DetDataSample` or `SegDataSample`], 
                optional): The batch data samples. It usually includes 
                information such as `gt_instance` or `gt_panoptic_seg` or 
                `gt_sem_seg`. Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of multi-layer 
                transformer and the second dict contains the inputs of 
                post-processing with various task head for.
            - transformer_inputs_dict (dict): The keyword args dictionary of
              `self.forward_transformer()`'.
            - head_inputs_dict (dict): The keyword args dictionary of
              `self.task_specific_heads`.
        """
        # tasks in each batch are same, which means each iter only samples one task
        assert len(set([batch_data_samples[b].task_name for b in \
            range(len(batch_data_samples))])) == 1, 'tasks of the batch must be same.'

        self.mode = batch_data_samples[0].task_name
        self.multi_tasks_cfgs = batch_data_samples[0].git_cfg
        self.num_classes = batch_data_samples[0].head_cfg['num_classes']
        # init head hyparameter of current samples
        # here assume that all samples have the same hyperparameter (the same source)
        self.task_specific_heads[self.mode+'_head'].reset_hyparameter(batch_data_samples[0].head_cfg)

        transformer_inputs_dict = dict(
            patch_embed=patch_embed, # bs, patch_H, patch_W, C
            patch_resolution=patch_resolution, # (patch_H, patch_W)
        ) 
        head_inputs_dict = {}

        return transformer_inputs_dict, head_inputs_dict

    def forward_transformer(self, patch_embed: Tensor, patch_resolution: Tuple,
                        batch_data_samples: SampleList) -> Dict:
        """Forward with Multi-Layer Transformer.
        Args:
            patch_embed (Tensor): patch embedding has (bs, patch_H, patch_W, C).
            patch_mask (Tensor): patch masks has (bs, img_h, img_w).
            patch_resolution (Tuple): patch masks has (patch_H, patch_W).
            grid_mask (Tensor): grid_mask has (bs, grid_H, grid_W).
            grid_int_position (Tensor): grid_int_position has (bs, num_queries, 2).
            grid_reference (Tensor): (bs, num_queries, 2).
            batch_data_samples (list[:obj:`DetDataSample` or `SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_bboxes` and `gt_sem_seg`.

        Returns:
            dict: The dictionary of decoder outputs.
        """
        self.num_image_token = 256 * (self.patch_batch_size // self.global_batch_size)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.grid_token_id = self.tokenizer.convert_tokens_to_ids(GRID_TOKEN)
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token + IMG_END_TOKEN
        # when inference, using padding left to avoid pad tokens in middle
        self.tokenizer.padding_side = 'right' if self.training else 'left'
        self.end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # get text conversations
        if self.mode == 'vqa':
            conversations = self.task_specific_heads[self.mode+'_head'].get_conversations(batch_data_samples, self.training, self.tokenizer)
        else:
            raw_conversations = self.task_specific_heads[self.mode+'_head'].get_conversations(batch_data_samples, self.training)
            new_conversations = []
            for conversation in raw_conversations:
                conversation = conversation.replace('<image>', image_tokens, 1)
                new_conversations.append(conversation)
            raw_conversations = new_conversations

            conversations = self.tokenizer(raw_conversations, return_tensors='pt', padding=True).to(patch_embed.device)

        conversation_ids = conversations['input_ids'].to(patch_embed.device)
        mm_embed = self.backbone.language_model.get_input_embeddings()(conversation_ids).clone()
        mm_mask = ~conversations['attention_mask'].bool().to(patch_embed.device)
        mm_mask[conversation_ids==self.tokenizer.pad_token_id] = True
        
        token_weights = conversations['attention_mask']

        # replace img context embedding with patch embed
        B,N,C = mm_embed.shape
        conversation_ids = conversation_ids.reshape(B*N)
        selected = (conversation_ids == self.img_context_token_id) 
        mm_embed = mm_embed.reshape(-1,C)
        mm_embed[selected] = mm_embed[selected] * 0.0 + patch_embed.reshape(-1,C)
        mm_embed = mm_embed.reshape(B,N,C)
        selected = selected.reshape(B,N)
        conversation_ids = conversation_ids.reshape(B,N)
        token_weights[selected] = 0

        # mask input prompt from loss compuation
        # NOTE: Assume single-turn conversations in other modes
        if self.mode != 'vqa':
            # use 'assistant' to split
            assistant_pos = torch.nonzero(conversation_ids == self.tokenizer.convert_tokens_to_ids('istant'))
            min_pos = 1e5
            if len(assistant_pos) != len(token_weights):
                # use </img> to split instead
                img_end_token = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
                assistant_pos = torch.nonzero(conversation_ids == img_end_token)
            if len(assistant_pos) == len(token_weights):
                for k, assist_pos in enumerate(assistant_pos):
                    token_weights[k][:assist_pos[1]+1] = 0
                    min_pos = min(min_pos, assist_pos[1])
            else:
                print("fail locate </img>", raw_conversations, assistant_pos)
                min_pos = 0
        else:
            labels = conversations['labels']
            token_weights[labels==-100] = 0
        
        if self.training:
            all_layer_pred_seq_logits = []
            all_layer_target_tokens = []
            all_layer_token_weights = []

            batch_gt_instances, batch_img_metas = [], []
            for data_sample in batch_data_samples:
                batch_img_metas.append(data_sample.metainfo)
                if self.mode == 'caption':
                    batch_gt_instances.append(data_sample.gt_caption)
                elif self.mode in ['grounding']:
                    batch_gt_instances.append(data_sample.gt_bboxes)
                elif self.mode in ['refer_segmentation', 'reason_segmentation']:
                    batch_gt_instances.append(data_sample.gt_masks)
                elif self.mode == 'vqa':
                    batch_gt_instances.append(data_sample.conversations)
                elif self.mode == 'refer_caption':
                    batch_gt_instances.append(data_sample.text)
                else:
                    raise NotImplementedError

            input_embed = mm_embed
            input_mask = mm_mask

            attn_mask = self.get_vl_attn_mask(input_embed, input_mask, selected)
            position_ids = None
            for layer_id, layer in enumerate(self.backbone.language_model.base_model.model.model.layers): 
                input_embed = checkpoint.checkpoint(layer.__call__, input_embed, attn_mask, position_ids=position_ids, use_reentrant=False)[0]

            input_embed = self.backbone.language_model.base_model.model.model.norm(input_embed)
            if self.mode in ['refer_segmentation', 'reason_segmentation']:
                patch_embed = input_embed[selected].view(*patch_embed.shape)
                patch_embed = patch_embed.view(patch_embed.shape[0], *self.patch_resolution, -1)

            if self.mode == 'vqa':
                # only compute target tokens to save memory
                shift_token_weights = token_weights[:, 1:]
                token_weights_mask = shift_token_weights > 0
                shift_input_embed = input_embed[:, :-1, :][token_weights_mask]
                shift_target_tokens = conversation_ids[:, 1:][token_weights_mask]
                pred_seq_logits = self.backbone.language_model.output(shift_input_embed)
                all_layer_pred_seq_logits.append(pred_seq_logits)
                all_layer_target_tokens.append(shift_target_tokens)
                all_layer_token_weights.append(shift_token_weights[token_weights_mask])
            else:
                # use min pos to split responses
                input_embed = input_embed[:, min_pos:]
                conversation_ids = conversation_ids[:, min_pos:]
                token_weights = token_weights[:, min_pos:]

                pred_seq_logits = self.backbone.language_model.output(input_embed)
                all_layer_pred_seq_logits.append(pred_seq_logits[:, :-1, :])
                all_layer_target_tokens.append(conversation_ids[:, 1:])
                all_layer_token_weights.append(token_weights[:, 1:])
            all_layer_pred_seq_logits = torch.stack(all_layer_pred_seq_logits)
            output_dict = {'all_layer_pred_seq_logits': all_layer_pred_seq_logits,
                           'all_layer_target_tokens': all_layer_target_tokens,
                           'all_layer_token_weights': all_layer_token_weights,
                            'batch_gt_instances':batch_gt_instances,
                           'batch_img_metas':batch_img_metas}
            if self.mode in ['refer_segmentation', 'reason_segmentation']:
                output_dict['image_features'] = self.merge_patch(patch_embed)
                output_dict['seq_embed'] = input_embed[:, :-1, :]
            return output_dict
        else:
            embed_len = mm_embed.shape[1]
            input_embed = mm_embed
            input_mask = mm_mask

            # position ids for padding left
            pad_offset = mm_mask.int().sum(dim=1)
            position_ids = torch.arange(embed_len)[None, :].expand(input_embed.shape[0], -1).to(input_embed.device)
            position_ids = position_ids - pad_offset.unsqueeze(-1)
            pad_ids = position_ids[position_ids<0]
            pad_ids += embed_len
            position_ids[position_ids<0] = pad_ids

            attn_mask = self.get_vl_attn_mask(input_embed, input_mask, selected)
            # forward prompt and image
            kv_caches = {}
            for layer_id, layer in enumerate(self.backbone.language_model.base_model.model.model.layers): 
                input_embed, past_key_value = layer(input_embed, attn_mask, position_ids=position_ids, use_cache=True)
                kv_caches[layer_id] = past_key_value

            input_embed = self.backbone.language_model.base_model.model.model.norm(input_embed)
            if self.mode in ['refer_segmentation', 'reason_segmentation']:
                patch_embed = input_embed[selected].view(*patch_embed.shape)
                patch_embed = patch_embed.view(patch_embed.shape[0], *self.patch_resolution, -1)

            # decode first token
            logits = self.backbone.language_model.output(input_embed[:,-1:])
            softmax_logits = logits.softmax(-1)
            logits, input_ids = softmax_logits.max(dim=-1)

            input_embed = self.backbone.language_model.get_input_embeddings()(input_ids)
            outputs_ids = [input_ids]
            outputs_logits = [logits]
            outputs_feats = [input_embed[:,-1:]]
            end_mask = torch.zeros_like(input_ids).bool()
            # autoregressive decode
            for token_idx in range(29):
                for layer_id, layer in enumerate(self.backbone.language_model.base_model.model.model.layers): 
                    position_ids = torch.arange(embed_len+token_idx, embed_len+token_idx+1).to(input_embed.device)[None, None, :].expand(input_embed.shape[0], 1, -1).flatten(1,2)
                    position_ids = position_ids - pad_offset.unsqueeze(-1)
                    attn_mask = torch.cat([mm_mask[:, None, :], torch.zeros(1, token_idx+1,device=input_embed.device)[None, :, :].expand(input_embed.shape[0], -1, -1)], dim=-1)
                    attn_mask = attn_mask.float().unsqueeze(1)
                    attn_mask[attn_mask>0] = -float('inf')

                    input_embed, past_key_value = layer(input_embed, attn_mask, position_ids=position_ids, past_key_value=kv_caches[layer_id], use_cache=True)
                    kv_caches[layer_id] = past_key_value

                seq_embed = self.backbone.language_model.base_model.model.model.norm(input_embed)
                logits = self.backbone.language_model.output(seq_embed)
                softmax_logits = logits.softmax(dim=-1)
                logits, input_ids = softmax_logits.max(dim=-1)
                end_mask = (input_ids == self.end_token_id) | end_mask
                input_ids[end_mask] = self.end_token_id

                outputs_ids.append(input_ids.flatten(0,1).unsqueeze(1))
                outputs_logits.append(logits.flatten(0,1).unsqueeze(1))
                outputs_feats.append(seq_embed.flatten(0,1).unsqueeze(1))

                if end_mask.all():
                    break
                input_embed = self.backbone.language_model.get_input_embeddings()(input_ids)
 
            outputs_ids = torch.cat(outputs_ids,dim=1)
            outputs_logits = torch.cat(outputs_logits, dim=1)
            outputs_feats = torch.cat(outputs_feats, dim=1)

            batch_size = self.global_batch_size
            outputs_ids = outputs_ids.view(batch_size, -1, outputs_ids.shape[-1])
            outputs_logits = outputs_logits.view(batch_size, -1, outputs_logits.shape[-1])
            outputs_feats = outputs_feats.view(batch_size, -1, *outputs_feats.shape[-2:])

            output_dict = {'outputs_ids': outputs_ids,'outputs_logits':outputs_logits}
            if self.mode in ['refer_segmentation', 'reason_segmentation']:
                output_dict['outputs_feats'] = outputs_feats
                output_dict['image_feats'] = self.merge_patch(patch_embed)
            
            return output_dict
        
    def prepare_grid(self, patch_embed, transformer_inputs_dict):
        self.grid_resolution_perwin = self.multi_tasks_cfgs['grid_resolution_perwin']
        self.samples_grids_eachwin = self.multi_tasks_cfgs['samples_grids_eachwin'] \
            if self.multi_tasks_cfgs['samples_grids_eachwin'] != -1 \
            else  self.grid_resolution_perwin[0] * self.grid_resolution_perwin[1]
        assert self.samples_grids_eachwin <= self.grid_resolution_perwin[0] * self.grid_resolution_perwin[1], \
               'grid sampled in each window should not be greater than original grids'

        ## generate sampled grids in a window (block)
        current_device = patch_embed.device
        grid_H, grid_W = self.grid_resolution_perwin

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, grid_H - 1, grid_H, dtype=torch.float32, device=current_device),
            torch.linspace(0, grid_W - 1, grid_W, dtype=torch.float32, device=current_device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        
        # normalize grids
        grid_scale = grid.new_zeros((self.patch_batch_size, 1, 1, 2))
        grid_scale[:, :, :, 0] = grid_W
        grid_scale[:, :, :, 1] = grid_H
        grid = (grid.unsqueeze(0).expand(self.patch_batch_size, -1, -1, -1) + 0.5) / grid_scale
        grid_reference = grid.view(self.patch_batch_size, -1, 2).detach() # bs, grid_num, 2

        # global grid reference for target assignment, grid interpolation and window sample
        block_W, block_H = self.window_block_shape
        block_y, block_x = torch.meshgrid(
            torch.linspace(0, block_H - 1, block_H, dtype=torch.float32, device=current_device),
            torch.linspace(0, block_W - 1, block_W, dtype=torch.float32, device=current_device))
        block = torch.cat([block_x.unsqueeze(-1), block_y.unsqueeze(-1)], -1).flatten(0,1)
        grid_num = grid_reference.shape[1]
        global_grid_reference = grid_reference.view(self.global_batch_size, -1, grid_num, 2)
        global_grid_reference = global_grid_reference + block.unsqueeze(0).unsqueeze(2)
        global_grid_reference = global_grid_reference.flatten(1,2)
        global_grid_reference[:,:,0] = global_grid_reference[:,:,0] / block_W 
        global_grid_reference[:,:,1] = global_grid_reference[:,:,1] / block_H

        global_grid_scale = global_grid_reference.new_zeros((self.global_batch_size, 1, 2))
        global_grid_scale[:, :, 0] = grid_W * block_W
        global_grid_scale[:, :, 1] = grid_H * block_H
        global_grid_int_position = (global_grid_reference * global_grid_scale) - 0.5
        global_grid_int_position = global_grid_int_position[0]

        transformer_inputs_dict.update(dict(
            global_grid_int_position=global_grid_int_position,
            global_grid_reference=global_grid_reference,
        ))
        return transformer_inputs_dict
    
    def grid_forward_transformer(self, patch_embed: Tensor, patch_resolution: Tuple,
                        global_grid_int_position, global_grid_reference: Tensor, 
                        batch_data_samples: SampleList) -> Dict:
        """Forward with Multi-Layer Transformer.
        Args:
            patch_embed (Tensor): patch embedding has (bs, patch_H, patch_W, C).
            patch_resolution (Tuple): patch masks has (patch_H, patch_W).
            grid_int_position (Tensor): grid_int_position has (bs, num_queries, 2).
            grid_reference (Tensor): (bs, num_queries, 2).
            batch_data_samples (list[:obj:`DetDataSample` or `SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_bboxes` and `gt_sem_seg`.

        Returns:
            dict: The dictionary of decoder outputs.
        """
        self.patch_num = self.patch_batch_size // self.global_batch_size
        self.num_image_token = 256 * self.patch_num
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.grid_token_id = self.tokenizer.convert_tokens_to_ids(GRID_TOKEN)
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token + IMG_END_TOKEN
        # dense prediction always right
        self.tokenizer.padding_side = 'right'
        self.end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # get text prompts
        if self.training:
            all_layer_pred_seq_logits = []
            all_layer_target_tokens = []
            all_layer_token_weights = []

            batch_gt_instances, batch_img_metas = [], []
            for data_sample in batch_data_samples:
                batch_img_metas.append(data_sample.metainfo)
                if self.mode in ['detection','instance_segmentation','phrase_detection']:
                    batch_gt_instances.append(data_sample.gt_instances)
                elif self.mode == 'semantic_segmentation':
                    batch_gt_instances.append(data_sample.gt_sem_seg)
                else:
                    raise NotImplementedError

            batch_size = len(batch_gt_instances)
            # assign label based on reference points
            reference_preds = [r for r in global_grid_reference]
            if self.mode in ['instance_segmentation','semantic_segmentation']:
                # translate semantic segmentation to instance segmentation
                if self.mode == 'semantic_segmentation':
                    batch_gt_instances = self.task_specific_heads[self.mode+'_head']._seg_data_to_instance_data(batch_gt_instances)
                input_tokens, target_tokens, token_weights, assign_inds = self.task_specific_heads[self.mode+'_head'].get_targets_based_on_reference(
                                reference_preds, batch_gt_instances, batch_img_metas)
            else:
                input_tokens, target_tokens, token_weights = self.task_specific_heads[self.mode+'_head'].get_targets_based_on_reference(
                                reference_preds, batch_gt_instances, batch_img_metas)
 
            text_prompts = self.task_specific_heads[self.mode+'_head'].get_prompt_sample(batch_data_samples, target_tokens)
            new_text_prompts = []
            for prompt in text_prompts:
                prompt = prompt.replace('<image>', image_tokens, 1)
                new_text_prompts.append(prompt)
            text_prompts = new_text_prompts

            prompt = self.tokenizer(text_prompts, padding=True, return_tensors='pt',).to(patch_embed.device)

            prompt_ids = prompt.input_ids

            text_embed = self.backbone.language_model.get_input_embeddings()(prompt_ids).clone()
            text_mask = ~prompt.attention_mask.bool()
            text_mask[prompt_ids==self.tokenizer.pad_token_id] = True

            # replace image context tokens with patch embedding
            B,N,C = text_embed.shape
            prompt_ids = prompt_ids.reshape(B*N)
            selected = (prompt_ids == self.img_context_token_id) 
            text_embed = text_embed.reshape(-1,C)
            text_embed[selected] = text_embed[selected] * 0.0 + patch_embed.reshape(-1,C)
            text_embed = text_embed.reshape(B,N,C)
            selected = selected.reshape(B,N)
            
            # random sample grids in window
            batch_select_index = self.window_grid_sample(input_tokens, global_grid_int_position, self.window_block_shape)
            
            # sample by index
            select_input_tokens = torch.gather(input_tokens, 1, batch_select_index[:, 
                                        :, None].repeat(1, 1, input_tokens.shape[-1]))
            select_target_tokens = torch.gather(target_tokens, 1, batch_select_index[:, 
                                        :, None].repeat(1, 1, target_tokens.shape[-1]))
            select_token_weights = torch.gather(token_weights, 1, batch_select_index[:, 
                                        :, None].repeat(1, 1, token_weights.shape[-1]))
            select_grid_reference = torch.gather(global_grid_reference, 1, batch_select_index[:, 
                                        :, None].repeat(1, 1, global_grid_reference.shape[-1]))
            select_query_num = select_grid_reference.shape[1]
            
            if self.mode in ['instance_segmentation','semantic_segmentation']:
                select_assign_inds = torch.gather(assign_inds, 1, batch_select_index)
            
            # translate targets to texts
            batch_raw_texts = self.task_specific_heads[self.mode+'_head'].translate_text(select_target_tokens, batch_img_metas)
            # add end token for each sub-prediction
            new_batch_raw_texts = []
            for raw_text in batch_raw_texts:
                raw_text = raw_text + '<|im_end|>'
                new_batch_raw_texts.append(raw_text)
            batch_raw_texts = new_batch_raw_texts
            
            target_texts = self.tokenizer(batch_raw_texts, padding='longest', truncation=True, max_length=512, return_tensors='pt',).to(patch_embed.device)

            select_target_tokens = target_texts.input_ids.view(batch_size, select_query_num,-1)
            select_input_tokens = target_texts.input_ids.view(batch_size,select_query_num,-1)
            select_token_weights = target_texts.attention_mask.view(batch_size,select_query_num,-1)
            # in semseg, background is ignored as pad_token
            # not pred pad token
            select_token_weights[:, :, :-1][select_target_tokens[:, :, 1:]==self.tokenizer.pad_token_id] = 0
            # pad token not pred as well
            select_token_weights[select_input_tokens==self.tokenizer.pad_token_id] = 0
            
            # prepare input sequence feature
            # sub-predictions are concatenated to a long sequence
            input_seq = select_input_tokens.flatten(1,2).clone()
            
            # get feature by index
            seq_embed = self.backbone.language_model.get_input_embeddings()(input_seq).clone()
            seq_mask =  ~select_token_weights.flatten(1,2).bool()

            # get grid embedd by interpolating image features
            grid_embed = self.get_grid_embed(patch_embed, select_grid_reference, 0).flatten(0, 2)
            grid_selected = (input_seq == self.grid_token_id)
            seq_embed[grid_selected] = seq_embed[grid_selected] * 0.0 + grid_embed.to(seq_embed.dtype)
            
            input_embed = torch.cat([text_embed, seq_embed],dim=1)
            input_mask = torch.cat([text_mask, seq_mask],dim=1)

            # mask each sub-prediction from seeing each other
            attn_mask = self.get_dp_attn_mask(input_embed, input_mask, selected)
            attn_mask, position_ids = self.prepare_mask_and_position(attn_mask, text_mask.shape[1], *select_input_tokens.shape[1:3])
            for layer_id, layer in enumerate(self.backbone.language_model.base_model.model.model.layers): 
                pre_len = text_mask.shape[1]
                if layer_id != 0:
                    # readd feature interpolated from patch embed
                    text_embed, seq_embed = input_embed[:, :pre_len], input_embed[:, pre_len:]
                    patch_embed = text_embed[selected].view(*patch_embed.shape)
                    grid_embed = self.get_grid_embed(patch_embed, select_grid_reference, layer_id).flatten(0, 2)
                    seq_embed[grid_selected] += grid_embed
                    input_embed = torch.cat([text_embed, seq_embed], dim=1)
                input_embed = checkpoint.checkpoint(layer.__call__, input_embed, attn_mask, position_ids=position_ids, use_reentrant=False)[0]

            # extract predictions
            input_embed = self.backbone.language_model.base_model.model.model.norm(input_embed)
            output_embed = input_embed[:, -seq_embed.shape[1]:]
            output_embed = output_embed.view(*select_target_tokens.shape, -1)
            output_embed = output_embed[:, :, :-1, :]
            select_target_tokens = select_target_tokens[:, :, 1:]
            select_token_weights = select_token_weights[:, :, 1:]
        

            pred_seq_logits = self.backbone.language_model.output(output_embed)
            pred_seq_logits = pred_seq_logits.view(batch_size, -1, pred_seq_logits.shape[-2], pred_seq_logits.shape[-1])
            all_layer_pred_seq_logits.append(pred_seq_logits)
            all_layer_target_tokens.append(select_target_tokens)
            all_layer_token_weights.append(select_token_weights)
                    
            all_layer_pred_seq_logits = torch.stack(all_layer_pred_seq_logits)
            output_dict = {'all_layer_pred_seq_logits': all_layer_pred_seq_logits,
                           'all_layer_target_tokens': all_layer_target_tokens,
                           'all_layer_token_weights': all_layer_token_weights,
                           'batch_gt_instances':batch_gt_instances,
                           'batch_img_metas':batch_img_metas}
            if self.mode in ['instance_segmentation','semantic_segmentation']:
                text_embed, seq_embed = input_embed[:, :pre_len], input_embed[:, pre_len:]
                patch_embed = text_embed[selected].view(*patch_embed.shape)
                patch_embed = patch_embed.view(patch_embed.shape[0], *self.patch_resolution, -1)

                output_dict['image_features'] = self.merge_patch(patch_embed)
                output_dict['seq_embed'] = output_embed
                output_dict['assign_inds'] = select_assign_inds
            return output_dict
        else:
            text_prompts = self.task_specific_heads[self.mode+'_head'].get_prompt(batch_data_samples)
            new_text_prompts = []
            for prompt in text_prompts:
                prompt = prompt.replace('<image>', image_tokens, 1)
                new_text_prompts.append(prompt)
            text_prompts = new_text_prompts

            prompt = self.tokenizer(text_prompts, return_tensors='pt', padding=True).to(patch_embed.device)
            
            prompt_ids = prompt.input_ids
            text_embed = self.backbone.language_model.get_input_embeddings()(prompt_ids).clone()
            text_mask = ~prompt.attention_mask.bool()
            text_mask[prompt_ids==self.tokenizer.pad_token_id] = True

            # replace img context embedding with patch embed
            B,N,C = text_embed.shape
            prompt_ids = prompt_ids.reshape(B*N)
            selected = (prompt_ids == self.img_context_token_id) 
            text_embed = text_embed.reshape(-1,C)
            text_embed[selected] = text_embed[selected] * 0.0 + patch_embed.reshape(-1,C)
            text_embed = text_embed.reshape(B,N,C)
            selected = selected.reshape(B,N)

            embed_len = text_embed.shape[1]

            # prepare input, prompt + image + grid
            grid_embed = self.get_grid_embed(patch_embed, global_grid_reference, 0).flatten(1, 2)
            input_embed = torch.cat([text_embed, grid_embed], dim=1)

            grid_mask = torch.zeros(*grid_embed.shape[:2]).to(grid_embed.device)
            input_mask = torch.cat([text_mask, grid_mask], dim=1).bool()
            attn_mask = self.get_dp_attn_mask(input_embed, input_mask, selected)

            grid_num = grid_embed.shape[1]
            grid_mask = torch.eye(grid_num).to(attn_mask.device).float()
            grid_mask[grid_mask<1] = -float('inf')
            grid_mask[grid_mask>0] = 0
            attn_mask[:, :, -grid_num:, -grid_num:] = grid_mask
            position_ids = torch.arange(embed_len).to(attn_mask.device)[None, :].repeat(attn_mask.shape[0], 1)
            grid_ids = torch.arange(embed_len,embed_len+1).to(attn_mask.device)[None, :].repeat(*grid_embed.shape[:2])
            position_ids = torch.cat([position_ids, grid_ids], dim=1)

            kv_caches = {}
            for layer_id, layer in enumerate(self.backbone.language_model.base_model.model.model.layers): 
                if layer_id != 0:
                    text_embed, seq_embed = input_embed[:, :embed_len], input_embed[:, embed_len:]
                    patch_embed = text_embed[selected].view(*patch_embed.shape)
                    grid_embed = self.get_grid_embed(patch_embed, global_grid_reference, layer_id).flatten(1, 2)
                    seq_embed += grid_embed
                    input_embed = torch.cat([text_embed, seq_embed], dim=1)
                input_embed, past_key_value = layer(input_embed, attn_mask, position_ids=position_ids, use_cache=True)
                kv_caches[layer_id] = past_key_value
            
            input_embed = self.backbone.language_model.base_model.model.model.norm(input_embed)
            if self.mode in ['instance_segmentation','semantic_segmentation']:
                text_embed, seq_embed = input_embed[:, :embed_len], input_embed[:, embed_len:]
                patch_embed = text_embed[selected].view(*patch_embed.shape)
                patch_embed = patch_embed.view(patch_embed.shape[0], *self.patch_resolution, -1)

            # get first token for all sub-predictions
            logits = self.backbone.language_model.output(input_embed[:,-grid_num:])
            softmax_logits = logits.softmax(-1)
            logits, input_ids = softmax_logits.max(dim=-1)

            input_embed = self.backbone.language_model.get_input_embeddings()(input_ids)
            outputs_ids = [input_ids.flatten(0,1).unsqueeze(1)]
            outputs_logits = [logits.flatten(0,1).unsqueeze(1)]
            outputs_feats = [input_embed[:,-grid_num:].flatten(0,1).unsqueeze(1)]
            end_mask = torch.zeros_like(input_ids).bool()
            # each time the model forward decodes a token for all subtasks
            for token_idx in range(29):
                for layer_id, layer in enumerate(self.backbone.language_model.base_model.model.model.layers): 
                    position_ids = torch.arange(embed_len+token_idx+1, embed_len+token_idx+2).to(input_embed.device)[None, None, :].expand(*input_embed.shape[:2], -1).flatten(1,2)
                    attn_mask = torch.zeros(grid_num, embed_len+(token_idx+2)*grid_num,device=input_embed.device).bool()
                    for k in range(token_idx+2):
                        start = embed_len + k*grid_num
                        end = embed_len + (k+1)*grid_num
                        attn_mask[:, start:end] = ~torch.eye(grid_num).to(attn_mask.device).bool()
                    attn_mask = attn_mask.float()
                    attn_mask[attn_mask>0] = -float('inf')
                    attn_mask = attn_mask[None, None, :, :].expand(input_embed.shape[0], 1, -1, -1)
                    input_embed, past_key_value = layer(input_embed, attn_mask, position_ids=position_ids, past_key_value=kv_caches[layer_id], use_cache=True)
                    kv_caches[layer_id] = past_key_value

                seq_embed = self.backbone.language_model.base_model.model.model.norm(input_embed)
                logits = self.backbone.language_model.output(seq_embed)
                softmax_logits = logits.softmax(dim=-1)
                logits, input_ids = softmax_logits.max(dim=-1)
                end_mask = (input_ids == self.end_token_id) | end_mask
                input_ids[end_mask] = self.end_token_id

                outputs_ids.append(input_ids.flatten(0,1).unsqueeze(1))
                outputs_logits.append(logits.flatten(0,1).unsqueeze(1))
                outputs_feats.append(seq_embed.flatten(0,1).unsqueeze(1))

                if end_mask.all():
                    break
                input_embed = self.backbone.language_model.get_input_embeddings()(input_ids)
 
            outputs_ids = torch.cat(outputs_ids,dim=1)
            outputs_logits = torch.cat(outputs_logits, dim=1)
            outputs_feats = torch.cat(outputs_feats, dim=1)

            batch_size = self.global_batch_size
            outputs_ids = outputs_ids.view(batch_size, -1, outputs_ids.shape[-1])
            outputs_logits = outputs_logits.view(batch_size, -1, outputs_logits.shape[-1])
            outputs_references = global_grid_reference.view(batch_size, -1, global_grid_reference.shape[-1])
            outputs_feats = outputs_feats.view(batch_size, -1, *outputs_feats.shape[-2:])

            output_dict = {'outputs_ids': outputs_ids,'outputs_logits':outputs_logits,'references':outputs_references}
            if self.mode in ['instance_segmentation','semantic_segmentation']:
                output_dict['outputs_feats'] = outputs_feats
                output_dict['image_feats'] = self.merge_patch(patch_embed)
            
            return output_dict

    
    def get_grid_embed(self, patch_embed, grid_reference, layer_id):
        patch_embed = patch_embed.view(patch_embed.shape[0], *self.patch_resolution, -1)
        patch_embed = self.merge_patch(patch_embed)
        if self.grid_interpolate:
            memory = patch_embed.permute(0, 3, 1, 2) 
            grid_position = grid_reference[:, :, :2].unsqueeze(2) * 2 - 1 
            interp_feat = F.grid_sample(memory, grid_position, align_corners=False)
            interp_feat = interp_feat.permute(0, 2, 3, 1) 
            interp_feat = interp_feat
        return interp_feat
    
    def get_vl_attn_mask(self, input_embed, text_mask, selected):
        B, N  = input_embed.shape[:2]
        casual_mask = torch.triu(torch.ones(N, N, device=input_embed.device), diagonal=1).bool()
        text_mask = text_mask[:,None,None,:].expand(-1,1,N,-1)
        attn_mask = text_mask | casual_mask[None, None,:,:].expand(B,1,-1,-1)
        # patch embed bidirectional
        img_selected = selected[:, None, :, None].expand(-1, 1, -1, N)
        img_attn_mask = attn_mask[img_selected]
        bidr_mask = selected[:, None, None, :].expand(-1, 1, self.num_image_token, -1)
        attn_mask[img_selected] = img_attn_mask & (~bidr_mask.contiguous().view(-1))

        # each token can see itself, orelse will cause nan in attention
        if not self.training:
            diag_mask = (~torch.eye(N, device=input_embed.device)[None, None, :, :].expand(B, 1, -1, -1).bool())
            attn_mask = diag_mask & attn_mask

        attn_mask = attn_mask.float()
        attn_mask[attn_mask>0] = -float('inf')
        return attn_mask
    
    def get_dp_attn_mask(self, input_embed, text_mask, selected):
        B, N  = input_embed.shape[:2]
        casual_mask = torch.triu(torch.ones(N, N, device=input_embed.device), diagonal=1).bool()
        text_mask = text_mask[:,None,None,:].expand(-1,1,N,-1)
        attn_mask = text_mask | casual_mask[None, None,:,:].expand(B,1,-1,-1)
        # patch embed bidirectional
        M = selected.shape[1]
        
        observe_mask = attn_mask[:, :, :M, :M]
        img_selected = selected[:, None, :, None].expand(-1, 1, -1, M)
        img_attn_mask = observe_mask[img_selected]
        bidr_mask = selected[:, None, None, :].expand(-1, 1, self.num_image_token, -1)
        observe_mask[img_selected] = img_attn_mask & (~bidr_mask.contiguous().view(-1))
        attn_mask[:, :, :M, :M] = observe_mask

        attn_mask = attn_mask.float()
        attn_mask[attn_mask>0] = -float('inf')
        return attn_mask
    
    def prepare_mask_and_position(self, attn_mask, pre_len, seq_num, seq_len):
        seq_mask = attn_mask[:, :, pre_len:, pre_len:]
        # sub-predictions cannot see each other
        seq_mask[:] = 1

        B = attn_mask.shape[0]
        position_ids = torch.arange(pre_len)[None, :].repeat(B, 1)
        seq_ids = torch.arange(pre_len, pre_len+seq_len)[None, :].repeat(B, 1)
        for k in range(seq_num):
            start = k * seq_len 
            end = (k+1) * seq_len 
            # each sub-prediction uses a unidirectional attention internally
            seq_mask[:, :, start:end, start:end] = torch.triu(torch.ones(seq_len, seq_len, device=attn_mask.device), diagonal=1)[None, None, :, :]
            position_ids = torch.cat([position_ids, seq_ids], dim=1)

        seq_mask[seq_mask>0] = -float('inf')
        attn_mask[:, :, pre_len:, pre_len:] += seq_mask

        return attn_mask, position_ids.to(attn_mask.device)

    def window_grid_sample(self, input_tokens, grid_int_position, window_shape):
        # generate window id for each grid
        batch_size = input_tokens.shape[0]
        win_coord_W = grid_int_position[:, 0] // self.grid_resolution_perwin[0]
        win_coord_H = grid_int_position[:, 1] // self.grid_resolution_perwin[1]
        win_inds_eachgrid = win_coord_H * window_shape[1] + win_coord_W
        win_inds_eachgrid = win_inds_eachgrid.view(-1).int()
        # serial computing for each sample
        batch_select_index = [[] for _ in range(batch_size)]
        for bs in range(batch_size):
            for win_h in range(window_shape[0]):
                for win_w in range(window_shape[1]):
                    win_id = win_h * window_shape[1] + win_w
                    # get grids in current window
                    grid_index = torch.nonzero(win_inds_eachgrid == win_id).squeeze(-1)
                    # NOTE: dummy in cap and ground
                    pos_index = torch.nonzero(input_tokens[bs, grid_index, 0] \
                                              != (self.num_classes)).squeeze(-1)
                    neg_index = torch.nonzero(input_tokens[bs, grid_index, 0] \
                                              == (self.num_classes)).squeeze(-1)
                    pos_mapping_index = grid_index[pos_index]
                    neg_mapping_index = grid_index[neg_index]
                    # prioritize filling with positive samples
                    if pos_mapping_index.shape[0] <= self.samples_grids_eachwin:
                        # fill all postive samples then random select negative samples
                        select_index_per_win = pos_mapping_index
                        neg_sampled_num = self.samples_grids_eachwin-select_index_per_win.shape[0]
                        random_neg_index = neg_mapping_index[torch.randperm(neg_mapping_index.size(0))[:neg_sampled_num]]
                        random_neg_index = random_neg_index.to(select_index_per_win.device).long()
                        select_index_per_win = torch.cat([select_index_per_win, random_neg_index])
                    else:
                        # random select positive samples
                        select_index_per_win = pos_mapping_index[torch.randperm(pos_mapping_index.size(0))[:self.samples_grids_eachwin]]
                    batch_select_index[bs].append(select_index_per_win)
            batch_select_index[bs] = torch.cat(batch_select_index[bs])
        # bs, win_num * samples_grids_eachwin
        batch_select_index = torch.stack(batch_select_index) 
        return batch_select_index 
    
    def merge_patch(self, patch_embed):
        # reverse any resolution split
        patch_num = self.patch_batch_size // self.global_batch_size
        if patch_num == 1:
            return patch_embed
        win_h, win_w = self.window_block_shape
        patch_h, patch_w = self.patch_resolution
        global_h, global_w = win_h*patch_h, win_w*patch_w
        all_global_patch_embed = []
        for i in range(self.global_batch_size):
            window_patch_embeds = patch_embed[i*patch_num:(i+1)*patch_num]
            
            global_patch_embed = torch.zeros(global_h, global_w, patch_embed.shape[-1], device=patch_embed.device)
            for j in range(win_h):
                start_h, end_h = j*patch_h, (j+1)*patch_h
                for k in range(win_w):
                    start_w, end_w = k*patch_w, (k+1)*patch_w
                    idx = j*win_w + k
                    global_patch_embed[start_h:end_h, start_w:end_w] = window_patch_embeds[idx]
            all_global_patch_embed.append(global_patch_embed)
        merge_patch_embed = torch.stack(all_global_patch_embed, dim=0)

        return merge_patch_embed

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        pass
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        pass
