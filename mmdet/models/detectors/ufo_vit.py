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

@MODELS.register_module()
class UFO_ViT(BaseDetector, metaclass=ABCMeta):
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
                 mean_output: bool = True,
                 mean_layes: List = [12,13,14,15,16,17],
                 support_tasks: List = ['detection', 'semantic_segmentation', 'instance_segmentation', 'caption','grounding'],
                 tokenizer: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.tokenizer_cfg = tokenizer

        # build multi-layer transformer with image patch embedding
        self.backbone = MODELS.build(backbone)

        # bulid non parametric task-specific heads for label assignment and post-processing
        self.head_list = head_list
        self.task_specific_heads = dict()
        for head_name in list(self.head_list.keys()):
            head_cfg = self.head_list[head_name]
            self.task_specific_heads[head_name] = MODELS.build(head_cfg)

        self.use_checkpoints = use_checkpoints # checkpoints for saving CUDA memory
        self.mean_output = mean_output
        self.mean_layes = mean_layes
        self.support_tasks = support_tasks
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.tokenizer = TOKENIZER.build(self.tokenizer_cfg)
        self.tokenizer.add_tokens(['<MASK>'])

        last_layer = len(self.backbone.layers)-1
        for name,param in self.named_parameters():
            if f'backbone.layers.{last_layer}.attn.rel_pos' in name:
                param.requires_grad = False
        if torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            # NOTE: float16 inference in V100 cause numerial overflow
            self.dtype = torch.float32
        
    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        self.backbone.embed.load_state_dict(torch.load(self.backbone.text_cfg['pretrain_path'], 
                                        map_location=self.backbone.pos_embed.device))
    
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
        elif self.mode == 'semantic_segmentation':
            batch_data_samples = self.task_specific_heads[self.mode+'_head'].add_pred_to_datasample(
                                                            batch_data_samples, results_list)
        elif self.mode == 'caption':
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
        batch_size = batch_inputs.shape[0]
        # patch embedding
        patch_embed, patch_resolution = self.backbone.patch_embed(batch_inputs)
        patch_embed = patch_embed.view(batch_size, *patch_resolution, patch_embed.shape[-1])
        img_pos_embed = resize_pos_embed(self.backbone.pos_embed.flatten(1, 2),
                        self.backbone.patch_resolution, patch_resolution,
                        mode=self.backbone.interpolate_mode, num_extra_tokens=0)
        patch_embed = patch_embed + img_pos_embed.view(1, *patch_resolution, self.backbone.embed_dims)
        patch_embed = self.backbone.drop_after_pos(patch_embed)

        # construct input dict of multi-layer transformer and non-parametric post-processing 
        transformer_inputs_dict, head_inputs_dict = self.pre_transformer(
            patch_embed, patch_resolution, batch_data_samples)
        transformer_inputs_dict['batch_data_samples'] = batch_data_samples

        # multi-layer transformer forward passing based on the task-specific rules
        # rules are pre-defined in task-specific heads
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
        batch_size = patch_embed.size(0)
        current_device = patch_embed.device
        self.mode = batch_data_samples[0].task_name
        self.multi_tasks_cfgs = batch_data_samples[0].git_cfg
        # tasks in each batch are same, which means each iter only samples one task
        assert len(set([batch_data_samples[b].task_name for b in \
            range(len(batch_data_samples))])) == 1, 'tasks of the batch must be same.'

        # init visual modeling hyparameter of current samples
        self.grid_resolution_perwin = self.multi_tasks_cfgs['grid_resolution_perwin']
        self.grid_interpolate = self.multi_tasks_cfgs['grid_interpolate']
        self.num_classes = batch_data_samples[0].head_cfg['num_classes']
        self.global_only_image = self.multi_tasks_cfgs['global_only_image']
        self.samples_grids_eachwin = self.multi_tasks_cfgs['samples_grids_eachwin'] \
            if self.multi_tasks_cfgs['samples_grids_eachwin'] != -1 \
            else  self.grid_resolution_perwin[0] * self.grid_resolution_perwin[1]
        assert self.samples_grids_eachwin <= self.grid_resolution_perwin[0] * self.grid_resolution_perwin[1], \
               'grid sampled in each window should not be greater than original grids'
        # init head hyparameter of current samples
        # here assume that all samples have the same hyperparameter (the same source)
        self.task_specific_heads[self.mode+'_head'].reset_hyparameter(batch_data_samples[0].head_cfg)

        if isinstance(batch_data_samples[0], DetDataSample):
            batch_input_shape = batch_data_samples[0].batch_input_shape
        elif isinstance(batch_data_samples[0], SegDataSample):
            batch_input_shape = batch_data_samples[0].pad_shape
        elif isinstance(batch_data_samples[0], DataSample):
            batch_input_shape = batch_data_samples[0].img_shape[:2]
        else:
            raise NotImplementedError
        ## generate sampled grids
        grid_H_win, grid_W_win = self.grid_resolution_perwin
        window_size = self.backbone.layers[0].window_size
        # patch resolution of the whole image
        patch_H, patch_W = patch_resolution 
        assert patch_H % window_size == 0 and patch_W % window_size == 0, "padding inner \
        window is not implemented, patch scale must be a multiple of window size"
        win_H = patch_H // window_size
        win_W = patch_W // window_size
        grid_H, grid_W = grid_H_win * win_H, grid_W_win * win_W

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, grid_H - 1, grid_H, dtype=torch.float32, device=current_device),
            torch.linspace(0, grid_W - 1, grid_W, dtype=torch.float32, device=current_device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        grid_int_position = grid.clone()
        
        # normalize grids and construct start task identifier tokens
        grid_scale = grid.new_zeros((batch_size, 1, 1, 2))
        grid_scale[:, :, :, 0] = grid_W
        grid_scale[:, :, :, 1] = grid_H
        grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) / grid_scale
        grid_reference = grid.view(batch_size, -1, 2).detach() # bs, grid_num, 2

        # generate position embedding of all grids
        grid_pos_embed = resize_pos_embed(self.backbone.pos_embed.flatten(1, 2),
                self.backbone.patch_resolution, [grid_H, grid_W],
                mode=self.backbone.interpolate_mode, num_extra_tokens=0) # bs, grid_H * grid_W, C
        grid_start_embed = grid_pos_embed.repeat(batch_size,1,1)

        ## generate image mask, patch mask, grid_mask
        img_shape_list = [sample.img_shape[:2] for sample in batch_data_samples]
        input_img_h, input_img_w = batch_input_shape
        masks = patch_embed.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0
        patch_mask = F.interpolate(masks[None],
                 size=patch_resolution).to(torch.bool).squeeze(0)
        grid_mask = F.interpolate(patch_mask[None].to(masks.dtype),
                size=[grid_H, grid_W]).to(torch.bool).squeeze(0)

        transformer_inputs_dict = dict(
            patch_embed=patch_embed, # bs, patch_H, patch_W, C
            patch_mask=patch_mask, # bs, img_h, img_w
            patch_resolution=patch_resolution, # (patch_H, patch_W)
            grid_start_embed=grid_start_embed,  # (bs, query_num, C)
            grid_mask=grid_mask, # bs, grid_H, grid_W
            grid_int_position=grid_int_position, # (grid_H, gird_W, 2)
            grid_reference=grid_reference) # (bs, query_num, 4)
        head_inputs_dict = {}

        return transformer_inputs_dict, head_inputs_dict

    def forward_transformer(self, patch_embed: Tensor, patch_mask: Tensor, patch_resolution: Tuple,
                        grid_start_embed: Tensor, grid_mask: Tensor, grid_int_position: Tensor, 
                        grid_reference: Tensor, batch_data_samples: SampleList) -> Dict:
        """Forward with Multi-Layer Transformer.
        Args:
            patch_embed (Tensor): patch embedding has (bs, patch_H, patch_W, C).
            patch_mask (Tensor): patch masks has (bs, img_h, img_w).
            patch_resolution (Tuple): patch masks has (patch_H, patch_W).
            grid_start_embed (Tensor): grid_start_embed has (bs, sampled_query_num, C).
            grid_mask (Tensor): grid_mask has (bs, grid_H, grid_W).
            grid_int_position (Tensor): grid_int_position has (bs, num_queries, 2).
            grid_reference (Tensor): (bs, num_queries, 2).
            batch_data_samples (list[:obj:`DetDataSample` or `SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_bboxes` and `gt_sem_seg`.

        Returns:
            dict: The dictionary of decoder outputs.
        """
        # get text prompts
        text_prompts = self.task_specific_heads[self.mode+'_head'].get_prompt(batch_data_samples)
        
        prompt = self.tokenizer(text_prompts, padding='longest', truncation=True, 
                max_length=512, return_tensors='pt',).to(patch_embed.device)
        
        # remove end token
        prompt_ids = prompt.input_ids[:, :-1]
        text_embed = self.backbone.embed(prompt_ids)
        text_mask = ~prompt.attention_mask.bool()[:, :-1]
        text_mask[prompt_ids==self.tokenizer.sep_token_id] = True
        
        if self.training:
            all_layer_pred_seq_logits = []
            all_layer_target_tokens = []
            all_layer_token_weights = []

            batch_gt_instances, batch_img_metas = [], []
            for data_sample in batch_data_samples:
                batch_img_metas.append(data_sample.metainfo)
                if self.mode in ['detection','instance_segmentation']:
                    batch_gt_instances.append(data_sample.gt_instances)
                elif self.mode == 'semantic_segmentation':
                    batch_gt_instances.append(data_sample.gt_sem_seg)
                elif self.mode == 'caption':
                    batch_gt_instances.append(data_sample.gt_caption)
                elif self.mode == 'grounding':
                    batch_gt_instances.append(data_sample.gt_bboxes)
                else:
                    raise NotImplementedError

            batch_size = len(batch_gt_instances)
            grid_token = grid_start_embed
            # assign label based on reference points
            reference_preds = [r for r in grid_reference]
            if self.mode == 'caption':
                input_tokens, target_tokens, token_weights = self.task_specific_heads[self.mode+'_head'].get_targets_based_on_reference(
                                reference_preds, batch_gt_instances, batch_img_metas, self.tokenizer)
            elif self.mode in ['instance_segmentation','semantic_segmentation']:
                # translate semantic segmentation to instance segmentation
                if self.mode == 'semantic_segmentation':
                    batch_gt_instances = self.task_specific_heads[self.mode+'_head']._seg_data_to_instance_data(batch_gt_instances)
                input_tokens, target_tokens, token_weights, assign_inds = self.task_specific_heads[self.mode+'_head'].get_targets_based_on_reference(
                                reference_preds, batch_gt_instances, batch_img_metas)
            elif self.mode in ['detection', 'grounding']:
                input_tokens, target_tokens, token_weights = self.task_specific_heads[self.mode+'_head'].get_targets_based_on_reference(
                                reference_preds, batch_gt_instances, batch_img_metas)
            else:
                raise NotImplementedError
 
            # random sample grids in window
            grid_H, grid_W = grid_mask.shape[1:]
            window_shape = (grid_H // self.grid_resolution_perwin[0], grid_W // self.grid_resolution_perwin[1])
            batch_select_index = self.window_grid_sample(input_tokens, grid_int_position, window_shape)
            
            # sample by index
            select_input_tokens = torch.gather(input_tokens, 1, batch_select_index[:, 
                                        :, None].repeat(1, 1, input_tokens.shape[-1]))
            select_target_tokens = torch.gather(target_tokens, 1, batch_select_index[:, 
                                        :, None].repeat(1, 1, target_tokens.shape[-1]))
            select_token_weights = torch.gather(token_weights, 1, batch_select_index[:, 
                                        :, None].repeat(1, 1, token_weights.shape[-1]))
            select_grid_token = torch.gather(grid_token, 1, batch_select_index[:, 
                                        :, None].repeat(1, 1, grid_token.shape[-1]))
            select_grid_reference = torch.gather(grid_reference, 1, batch_select_index[:, 
                                        :, None].repeat(1, 1, grid_reference.shape[-1]))
            if self.mode in ['instance_segmentation','semantic_segmentation']:
                select_assign_inds = torch.gather(assign_inds, 1, batch_select_index)

            select_grid_start_embed = select_grid_token.clone().unsqueeze(2)
            select_query_num = select_grid_token.shape[1]
            
            if self.mode != 'caption':
                # translate ids to raw text
                batch_raw_texts = self.task_specific_heads[self.mode+'_head'].translate_text(select_target_tokens, batch_img_metas)
                
                target_texts = self.tokenizer(batch_raw_texts, padding='longest', truncation=True, max_length=512, return_tensors='pt',).to(patch_embed.device)

                select_target_tokens = target_texts.input_ids[:,1:].view(batch_size, select_query_num,-1)
                select_input_tokens = target_texts.input_ids[:,:-1].view(batch_size,select_query_num,-1)
                select_token_weights = target_texts.attention_mask[:,:-1].view(batch_size,select_query_num,-1)
                # in semseg, background is ignored as pad_token
                select_token_weights[select_target_tokens==self.tokenizer.pad_token_id] = 0
                # pad token not pred as well
                select_token_weights[select_input_tokens==self.tokenizer.pad_token_id] = 0

            # prepare input sequence feature
            input_seq = select_input_tokens.view(-1, select_input_tokens.shape[-1]).clone()
            # get feature by index
            seq_embed = self.backbone.embed(input_seq)

            grid_embed = select_grid_token.unsqueeze(2)
            # output features of different layers
            aux_feat = []
            aux_image_feat = []
            for layer_id, layer in enumerate(self.backbone.layers): 
                # prepend grid feature and generate attention mask                            
                grid_embed = self.get_grid_embed(patch_embed, select_grid_reference, select_grid_start_embed, grid_embed,layer_id)
                grid_attn_mask, seq_attn_mask = self.get_attn_mask(layer.window_size, patch_embed, grid_embed,seq_embed)
                img_text_key_padding_mask = self.get_key_padding_mask(layer.window_size, patch_embed, text_mask)

                # caption and grounding only use global attention            
                disable_window = self.mode == 'caption' or self.mode == 'grounding'
                if self.use_checkpoints:
                    text_embed, inter_kv = checkpoint.checkpoint(layer.seq_forward, text_embed, key_padding_mask=text_mask, use_reentrant=False)
                    patch_embed, inter_kv = checkpoint.checkpoint(layer.img_forward, patch_embed, key_padding_mask=img_text_key_padding_mask,
                                                                  pre_kv=inter_kv, use_reentrant=False)
                    if self.global_only_image and layer.window_size <= 0:
                        continue
                    if self.grid_interpolate:
                        grid_embed, inter_kv = checkpoint.checkpoint(layer.token_forward, grid_embed, select_grid_reference,
                                                        grid_attn_mask, inter_kv, disable_window,use_reentrant=False)
                    seq_embed, _  = checkpoint.checkpoint(layer.seq_forward, seq_embed, seq_attn_mask, pre_kv=inter_kv, use_reentrant=False)
                    if self.mean_output and layer_id in self.mean_layes:
                        aux_feat.append(seq_embed)
                        aux_image_feat.append(patch_embed)
            if self.mean_output:
                seq_embed = torch.stack(aux_feat, dim=-1).mean(dim=-1)
                patch_embed = torch.stack(aux_image_feat, dim=-1).mean(dim=-1)
            pred_seq_logits = seq_embed @ self.backbone.embed.word_embeddings.weight.transpose(0,1)
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
                output_dict['image_features'] = patch_embed
                output_dict['seq_embed'] = seq_embed
                output_dict['assign_inds'] = select_assign_inds
            return output_dict
        else:
            output_dict = self.task_specific_heads[self.mode+'_head'].decoder_inference(self.backbone, 
                patch_embed, patch_mask, text_embed, text_mask, grid_start_embed, grid_mask, grid_reference, self.tokenizer, self.grid_interpolate, self.global_only_image)
            
            return output_dict
    
    def get_grid_embed(self, patch_embed, grid_reference, grid_start_embed, grid_forward, layer_id):
        grid_feature = grid_forward
        # add grid start embed when layer_id > 0
        if layer_id != 0:
            grid_feature = grid_feature + grid_start_embed
        # interpolate image feature by grid position
        if self.grid_interpolate:
            memory = patch_embed.permute(0, 3, 1, 2) 
            grid_position = grid_reference[:, :, :2].unsqueeze(2) * 2 - 1 
            interp_feat = F.grid_sample(memory, grid_position, align_corners=False)
            interp_feat = interp_feat.permute(0, 2, 3, 1) 
            grid_feature = grid_feature + interp_feat
        else:
            grid_feature = grid_feature
        return grid_feature
    
    def get_attn_mask(self, window_size, patch_embed, grid_embed, seq_embed):
        patch_resolution = patch_embed.shape[1:3]
        embed_len = seq_embed.shape[1]
        
        if window_size > 0:
            # window attention
            window_patch_len = window_size * window_size 
            grid_num = self.samples_grids_eachwin
        else:
            # global attention
            window_patch_len = patch_resolution[0] * patch_resolution[1]
            grid_num = grid_embed.shape[1]

        grid_attn_mask = torch.zeros(grid_num, window_patch_len+grid_num, device=patch_embed.device)
        grid_attn_mask[:, -grid_num:] = 1-torch.eye(grid_num, device=patch_embed.device)
        
        if self.grid_interpolate:
            seq_pre_len = 1
        else:
            seq_pre_len = window_patch_len
        
        seq_attn_mask = torch.triu(torch.ones(embed_len,embed_len,device=patch_embed.device), diagonal=1)
        seq_attn_mask = torch.cat([
            torch.zeros(embed_len, seq_pre_len, device=patch_embed.device), seq_attn_mask
        ], dim=1)
        
        grid_attn_mask = grid_attn_mask.bool()
        seq_attn_mask = seq_attn_mask.bool()
          
        return grid_attn_mask, seq_attn_mask
    
    def get_key_padding_mask(self, window_size, patch_embed, text_mask):
        patch_resolution = patch_embed.shape[1:3]
        if window_size > 0:
            # window attention
            window_patch_len = window_size * window_size 
            num_window = (patch_resolution[0] // window_size) * (patch_resolution[1] // window_size)
            text_mask = text_mask.unsqueeze(1).repeat(1, num_window, 1).flatten(0, 1)
        else:
            # global attention
            window_patch_len = patch_resolution[0] * patch_resolution[1]
        patch_mask = torch.zeros(text_mask.shape[0], window_patch_len,device=patch_embed.device)
        key_padding_mask = torch.cat([text_mask, patch_mask], dim=1).bool()
        return key_padding_mask

    def window_grid_sample(self, input_tokens, grid_int_position, window_shape):
        # generate window id for each grid
        batch_size = input_tokens.shape[0]
        win_coord_H = grid_int_position[:, :, 0] // self.grid_resolution_perwin[0]
        win_coord_W = grid_int_position[:, :, 1] // self.grid_resolution_perwin[1]
        win_inds_eachgrid = win_coord_W * window_shape[0] + win_coord_H
        win_inds_eachgrid = win_inds_eachgrid.view(-1).int()
        # serial computing for each sample
        batch_select_index = [[] for _ in range(batch_size)]
        for bs in range(batch_size):
            for win_h in range(window_shape[0]):
                for win_w in range(window_shape[1]):
                    win_id = win_h * window_shape[1] + win_w
                    # get grids in current window
                    grid_index = torch.nonzero(win_inds_eachgrid == win_id).squeeze(-1)
                    # NOTE: dummy in seg, cap and ground
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

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        pass
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        pass
