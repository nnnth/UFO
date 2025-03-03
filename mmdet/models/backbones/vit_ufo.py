from typing import Optional, Sequence, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmdet.registry import MODELS
from mmdet.models.utils import to_2tuple, LayerNorm2d, build_norm_layer, resize_pos_embed
from mmdet.models.backbones.base_backbone import BaseBackbone

from abc import ABCMeta, abstractmethod
from ..utils.text_embedding import BertEmbeddings

def window_partition(x: torch.Tensor,
                     window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        x (torch.Tensor): Input tokens with [B, H, W, C].
        window_size (int): Window size.

    Returns:
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

        - ``windows``: Windows after partition with
        [B * num_windows, window_size, window_size, C].
        - ``(Hp, Wp)``: Padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int,
                       pad_hw: Tuple[int, int],
                       hw: Tuple[int, int]) -> torch.Tensor:
    """Window unpartition into original sequences and removing padding.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        x (torch.Tensor): Input tokens with
            [B * num_windows, window_size, window_size, C].
        window_size (int): Window size.
        pad_hw (tuple): Padded height and width (Hp, Wp).
        hw (tuple): Original height and width (H, W) before padding.

    Returns:
        torch.Tensor: Unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def grid_window_partition(x: torch.Tensor, grid_position: torch.Tensor,
        image_pad_hw: Tuple, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition into non-overlapping windows with padding if needed.

    Args:
        x (torch.Tensor): Input grid tokens with [B, grid_num, len_per_grid, C].
        grid_position (torch.Tensor): Input norm grid position with [B, grid_num, 2],
        image_pad_hw (Tuple): Pad image hw.
        its has been window padded.
        window_size (int): Window size.

    Returns:
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]

        - ``windows_grid``: Windows after partition with
        [B * num_windows, grid_per_win * len_per_grid, C].
        - ``batch_win_inds``: window indices of each grid 
        with [B, grid_num * len_per_grid]
        - ``grid_inner_win_inds``: inner window indices of each grid
        with [B, grid_num * len_per_grid]
    """
    B, N, L, C = x.shape
    # computer window coord
    num_win_x, num_win_y = (image_pad_hw[1] // window_size, 
                                    image_pad_hw[0] // window_size)
    num_wins = num_win_x * num_win_y
    win_coors_x = grid_position[:, :, 0] * image_pad_hw[1] // window_size # bs, sampled_grid_num
    win_coors_y = grid_position[:, :, 1] * image_pad_hw[0] // window_size # bs, sampled_grid_num
    batch_win_inds = win_coors_y * num_win_x + win_coors_x 
    batch_win_inds = batch_win_inds.long() # bs, grid_num
    batch_win_inds = batch_win_inds.unsqueeze(-1).repeat(1, 1, L) # bs, grid_num, len_per_grid
    batch_win_inds = batch_win_inds.view(B, N*L) # bs, grid_num * len_per_grid

    # assume that batch win inds for each batch is the same
    assert (batch_win_inds[0] == batch_win_inds[-1]).all(), "batch win inds are different"
    sorted, indices = torch.sort(batch_win_inds[0], stable=True)
    # assume the grid num of each window is the same
    token_per_win = sorted.shape[0] // num_wins
    token_inner_win_inds = torch.zeros_like(batch_win_inds[0], dtype=torch.int64)
    token_inner_win_inds[indices] = torch.arange(token_per_win, device=x.device).repeat(num_wins)
    token_inner_win_inds = token_inner_win_inds[None, :].repeat(B, 1)

    scatter_indices = batch_win_inds * token_per_win + token_inner_win_inds # bs, grid_num * len_per_grid
    scatter_indices = scatter_indices.unsqueeze(-1).repeat(1, 1, C).long() # bs, grid_num * len_per_grid, C
    x = x.view(B, N*L, C)
    windows_grid = torch.zeros((B, num_wins * token_per_win, C), 
            device=x.device, dtype=x.dtype).scatter_(1, scatter_indices, x)
    windows_grid = windows_grid.view(B*num_wins, token_per_win, C)
    return windows_grid, scatter_indices

def grid_window_unpartition(windows_grid: torch.Tensor, 
                            scatter_indices: torch.Tensor) -> torch.Tensor:
    """Window unpartition into original sequences and removing padding.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        windows_grid (Tensor): Windows after partition with
        [B * num_windows, grid_per_win * len_per_grid, C].
        scatter_indices (Tensor): Scatter_indices generated by grid 
        window partition with shape [B, grid_num * len_per_grid, C]

    Returns:
        torch.Tensor: Unpartitioned sequences with [B, grid_per_win * len_per_grid, C].
    """
    B, _, C = scatter_indices.shape
    _, rescatter_indices = torch.sort(scatter_indices[0, :, 0])
    windows_grid = windows_grid.reshape(B, -1, C)
    rescatter_indices = rescatter_indices[None, :, None].repeat(B, 1, C)
    x = torch.zeros((B, windows_grid.shape[1], C), 
        device=windows_grid.device, dtype=windows_grid.dtype).scatter_(1, rescatter_indices, windows_grid)
    
    return x

def get_rel_pos(q_size: int, k_size: int,
                rel_pos: torch.Tensor) -> torch.Tensor:
    """Get relative positional embeddings according to the relative positions
    of query and key sizes.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        q_size (int): Size of query q.
        k_size (int): Size of key k.
        rel_pos (torch.Tensor): Relative position embeddings (L, C).

    Returns:
        torch.Tensor: Extracted positional embeddings according to relative
        positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode='linear',
        )
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                  max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords -
                       k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """Borrowed from https://github.com/facebookresearch/segment-anything/

    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

    Args:
        attn (torch.Tensor): Attention map.
        q (torch.Tensor): Query q in the attention layer with shape
            (B, q_h * q_w, C).
        rel_pos_h (torch.Tensor): Relative position embeddings (Lh, C) for
            height axis.
        rel_pos_w (torch.Tensor): Relative position embeddings (Lw, C) for
            width axis.
        q_size (tuple): Spatial sequence size of query q with (q_h, q_w).
        k_size (tuple): Spatial sequence size of key k with (k_h, k_w).

    Returns:
        torch.Tensor: Attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)

    rel_pos = (rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).contiguous().view(B, q_h * q_w, k_h * k_w)

    return rel_pos

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings.

    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        embed_dims (int): The embedding dimension, patch_embed and grid_token
        num_heads (int): Parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        use_rel_pos (bool):Whether to use relative position embedding.
            Defaults to False.
        input_size (int, optional): Input resolution for calculating the
            relative positional parameter size. Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = head_embed_dims**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (input_size is not None), \
                'Input size must be provided if using relative position embed.'
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_embed_dims))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_embed_dims))
    
    def forward(self, x: torch.Tensor, attention_scope=None, attn_mask=None, key_padding_mask=None, pre_kv=None,) -> torch.Tensor:
        B, N, _ = x.shape
        # qkv with shape (3, B, nHead, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, N, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)
        if pre_kv is not None:
            k = torch.cat([pre_kv[0], k], dim=1) 
            v = torch.cat([pre_kv[1], v], dim=1)
            
        # combine attn_mask and key_padding mask
        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool, 'key_padding_mask must be a torch.bool'
            key_padding_mask = key_padding_mask.view(B,1,1,k.shape[1]).expand(-1,self.num_heads,-1,-1).reshape(B*self.num_heads,1,k.shape[1])
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(0) | key_padding_mask
            else:
                attn_mask = key_padding_mask.expand(-1,q.shape[1],-1)
        # add attn_mask
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool, 'attn_mask must be a torch.bool'
            attn_mask_bool = attn_mask.clone()
            attn_mask = attn_mask.to(q.dtype).masked_fill(attn_mask_bool, -float('inf'))
        if self.use_rel_pos and attention_scope is not None: # only operate the image patch
            H, W = attention_scope
            rel_pos = add_decomposed_rel_pos(q[:, -(H*W):, :], self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
            if attn_mask is not None:
                if attn_mask.ndim == 2:
                    attn_mask = attn_mask.unsqueeze(0).repeat(rel_pos.shape[0],1,1)
                attn_mask[:,-(H*W):, -(H*W):] = attn_mask[:,-(H*W):, -(H*W):] + rel_pos
            else:
                attn_mask = rel_pos

        q_1 = q.view(B,self.num_heads,q.shape[1],-1)
        k_1 = k.view(B,self.num_heads,k.shape[1],-1)
        v_1 = v.view(B,self.num_heads,v.shape[1],-1)
        if attn_mask is not None and attn_mask.dim() > 2:
            attn_mask = attn_mask.view(B,self.num_heads,N,-1)

        x = F.scaled_dot_product_attention(q_1,k_1,v_1,attn_mask=attn_mask)
        
        x = x.view(B, self.num_heads, N,
                            -1).permute(0, 2, 1, 3).reshape(B, N, -1)
        x = self.proj(x)
        
        pre_kv_update = torch.stack([k, v])

        return x, pre_kv_update


class TransformerLayer(BaseModule):
    """Transformer layer with window attention in our UFO.
    It's the same with transformer layer used in ViT and SAM
    Borrowed from https://github.com/facebookresearch/segment-anything/

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        act_cfg (dict): The activation config for FFNs.
            Defaluts to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        use_rel_pos (bool):Whether to use relative position embedding.
            Defaults to False.
        window_size (int): Window size for window attention. Defaults to 0.
        input_size (int, optional): Input resolution for calculating the
            relative positional parameter size. Defaults to None.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 use_rel_pos: bool = False,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.window_size = window_size

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)
        
    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self):
        return 
    
    def img_forward(self, x, attn_mask=None, key_padding_mask=None, pre_kv=None,):
        # Window partition
        batch_size = x.shape[0]
        attention_scope = x.shape[1:3]
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            attention_scope = x.shape[1:3]
            self.img_pad_hw = pad_hw
            if pre_kv is not None:
                new_pre_kv = pre_kv.view(2, batch_size, self.attn.num_heads, -1, pre_kv.shape[-1])
                new_pre_kv = new_pre_kv.unsqueeze(2).repeat(1, 1, x.shape[0]//batch_size, 1, 1, 1)
                pre_kv = new_pre_kv.flatten(1, 3)
        if pre_kv is not None:
            old_kv_len = pre_kv.shape[2]
        else:
            old_kv_len = 0
        # generate input and key padding mask
        x = x.view(x.shape[0], -1, x.shape[3]) 
        
        # block forward
        shortcut = x
        x = self.ln1(x)
        with torch.backends.cuda.sdp_kernel(enable_math=False,enable_flash=False,enable_mem_efficient=True):
            x, inter_kv = self.attn.forward(x, attention_scope=attention_scope, attn_mask=attn_mask, key_padding_mask=key_padding_mask, pre_kv=pre_kv)

        x = shortcut + x
        x = self.ffn(self.ln2(x), identity=x)
 
        x = x.view(x.shape[0], attention_scope[0], attention_scope[1], x.shape[2])
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        inter_kv = inter_kv[:, :, old_kv_len:]
        return x, inter_kv
    
    def token_forward(self, grid_token, grid_position, attn_mask=None, pre_kv=None, disabled_window=False):
        B, G, N, C = grid_token.shape
        # Window partition
        if self.window_size > 0 and not disabled_window:
            flatten_grid_token, scatter_indices = \
                grid_window_partition(grid_token, grid_position, self.img_pad_hw, self.window_size)
        else:
            flatten_grid_token = grid_token.view(B, -1, C) # B, grid_num*1, C
        x = flatten_grid_token
        # block forward
        shortcut = x
        x = self.ln1(x)
        with torch.backends.cuda.sdp_kernel(enable_math=False,enable_flash=False,enable_mem_efficient=True):
            x, pre_kv_update = self.attn.forward(x, attn_mask=attn_mask, pre_kv=pre_kv)

        x = shortcut + x
        x = self.ffn(self.ln2(x), identity=x)
        # Reverse window partition
        if self.window_size > 0 and not disabled_window:
            batch_size, grid_per_win, len_per_grid = grid_token.shape[:3]
            grid_token = grid_window_unpartition(x, scatter_indices)
            grid_token = grid_token.view(batch_size, grid_per_win, len_per_grid, -1)
        else:
            flatten_grid_token = x
            grid_token = flatten_grid_token.view(B, grid_token.shape[1], grid_token.shape[2], C)
        if pre_kv is not None:
            if self.window_size > 0 and not disabled_window:
                new_pre_kv = pre_kv_update[:, :, -x.shape[1]:]
                new_pre_kv = new_pre_kv.view(2, x.shape[0], self.attn.num_heads, -1, new_pre_kv.shape[-1])
                new_pre_kv = new_pre_kv.permute(1, 3, 0, 2, 4).flatten(2,4)
                kv_scatter = scatter_indices.unsqueeze(2).repeat(1,1,2,1).flatten(2,3)
                new_pre_kv = grid_window_unpartition(new_pre_kv, kv_scatter)
                new_pre_kv = new_pre_kv.flatten(0,1).view(-1,1,2, self.attn.num_heads,pre_kv_update.shape[-1]).permute(2,0,3,1,4).flatten(1,2)
            else:
                new_pre_kv = pre_kv_update[:, :, -x.shape[1]:]
                new_pre_kv = new_pre_kv.view(2, x.shape[0], self.attn.num_heads, -1, new_pre_kv.shape[-1])
                new_pre_kv = new_pre_kv.permute(0, 1, 3, 2, 4).flatten(1, 3).unsqueeze(2)
            return grid_token, new_pre_kv
        return grid_token

    def seq_forward(self, x, attn_mask=None, key_padding_mask=None, pre_kv=None):
        shortcut = x
        x = self.ln1(x)
        with torch.backends.cuda.sdp_kernel(enable_math=False,enable_flash=False,enable_mem_efficient=True):
            x, pre_kv_update = self.attn.forward(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, pre_kv=pre_kv)

        x = shortcut + x
        x = self.ffn(self.ln2(x), identity=x)

        return x, pre_kv_update


@MODELS.register_module()
class ViTUFO(BaseBackbone):
    """Vision Transformer as image encoder used in UFO-ViT.

    A PyTorch implement of backbone: `Segment Anything
    <https://arxiv.org/abs/2304.02643>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'base', 'large', 'huge'. If use dict, it should have
            below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.
            - **global_attn_indexes** (int): The index of layers with global
              attention.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_channels (int): The num of output channels, if equal to 0, the
            channel reduction layer is disabled. Defaults to 256.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        out_type (str): The type of output features. Please choose from

            - ``"raw"`` or ``"featmap"``: The feature map tensor from the
              patch tokens with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).

            Defaults to ``"raw"``.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        use_abs_pos (bool): Whether to use absolute position embedding.
            Defaults to True.
        use_rel_pos (bool):Whether to use relative position embedding.
            Defaults to True.
        window_size (int): Window size for window attention. Defaults to 14.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072,
                'global_attn_indexes': [2, 5, 8, 11]
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096,
                'global_attn_indexes': [5, 11, 17, 23]
            }),
        **dict.fromkeys(
            ['h', 'huge'], {
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120,
                'global_attn_indexes': [7, 15, 23, 31]
            }),
    }
    OUT_TYPES = {'raw', 'featmap', 'avg_featmap'}

    def __init__(self,
                 arch: str = 'base',
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 out_channels: int = 256,
                 out_indices: int = -1,
                 out_type: str = 'raw',
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 qkv_bias: bool = True,
                 use_abs_pos: bool = True,
                 use_rel_pos: bool = True,
                 use_checkpoints: bool = False,
                 new_more_layers: int = 0,
                 window_size: int = 14,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 frozen_stages: int = -1,
                 interpolate_mode: str = 'bicubic',
                 patch_cfg: dict = dict(),
                 text_cfg: dict = dict(),
                 layer_cfgs: dict = dict(),
                 init_cfg: Optional[dict] = None,
                 mean_layers=[12,13,14,15,16,17]):
        super().__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.global_attn_indexes = self.arch_settings['global_attn_indexes']
        self.img_size = to_2tuple(img_size)

        # add new global layer
        if isinstance(new_more_layers, int):
            new_global_index = [i for i in range(self.num_layers, self.num_layers+new_more_layers)]
            self.num_layers += new_more_layers
            self.global_attn_indexes += new_global_index
        elif isinstance(new_more_layers, list):
            new_global_index = []
            for i, att_type in enumerate(new_more_layers):
                assert att_type in ['win', 'global'], 'layer type is wrong, only win or global'
                if att_type == 'global':
                    new_global_index.append(i+self.num_layers)
            self.global_attn_indexes += new_global_index
            self.num_layers += len(new_more_layers)
        else:
            raise NotImplementedError

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        # Set text embedding
        _text_cfg = {'hidden_dropout_prob': 0.1,
                                'hidden_size': text_cfg['hidden_size'],
                                'layer_norm_eps': 1e-12,
                                'max_position_embeddings': 512,
                                'pad_token_id': 0,
                                'add_type_embeddings': False,
                                'vocab_size': 30524}
        _text_cfg.update(text_cfg)
        text_embedding_type = _text_cfg['type']
        assert text_embedding_type in ['bert-base', 'bert-large', 'bert-huge'], 'Embedding type is not support.'
        self.embed = BertEmbeddings(_text_cfg)
        self.text_cfg = _text_cfg

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        self.use_abs_pos = use_abs_pos
        self.use_checkpoints = use_checkpoints
        self.interpolate_mode = interpolate_mode
        if use_abs_pos:
            # Set position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, *self.patch_resolution, self.embed_dims))
            self.drop_after_pos = nn.Dropout(p=drop_rate)
            self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        if use_rel_pos:
            self._register_load_state_dict_pre_hook(
                self._prepare_relative_position)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.
                arch_settings['feedforward_channels'],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                window_size=window_size
                if i not in self.global_attn_indexes else 0,
                input_size=self.patch_resolution,
                use_rel_pos=use_rel_pos,
                norm_cfg=norm_cfg)
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerLayer(**_layer_cfg))

        self.out_channels = out_channels
        if self.out_channels > 0:
            self.channel_reduction = nn.Sequential(
                nn.Conv2d(
                    self.embed_dims,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(out_channels, eps=1e-6),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(out_channels, eps=1e-6),
            )

        # freeze stages only when self.frozen_stages > 0
        self.frozen_stages = frozen_stages
        if self.frozen_stages > 0:
            self._freeze_stages()
        self.mean_layers = mean_layers

    def init_weights(self):
        super().init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        # freeze channel_reduction module
        if self.frozen_stages == self.num_layers and self.out_channels > 0:
            m = self.channel_reduction
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    def forward(self):
        return 
    
    def img_forward(self, patch_embed, pre_kv_list=None, grid_interpolate=False, references=None, text_mask=None):
        grid_interpolate_feats = [] 
        patch_resolution = patch_embed.shape[1:3]
        window_size = self.layers[0].window_size
        self.num_window = (patch_embed.shape[1] // window_size) * (patch_embed.shape[2] // window_size)
        aux_ids = self.mean_layers
        aux_feats = []
        for i, layer in enumerate(self.layers):
            if grid_interpolate:
                # compute tokens of local image feature
                input_img_patch = patch_embed.permute(0, 3, 1, 2) 
                grid_position = references[:, :, :2].unsqueeze(2) * 2 - 1 
                grid_local_feat = F.grid_sample(input_img_patch, grid_position, align_corners=False)
                grid_interpolate_feats.append(grid_local_feat.squeeze(-1).permute(0, 2, 1))
            window_patch_num = layer.window_size ** 2 if layer.window_size > 0 else np.prod(patch_resolution)
            if layer.window_size > 0:
                pre_text_mask = text_mask.unsqueeze(1).repeat(1, self.num_window, 1).flatten(0, 1)
            else:
                pre_text_mask = text_mask

            img_mask = torch.zeros(pre_text_mask.shape[0], window_patch_num, device=patch_embed.device)
            img_text_key_padding_mask = torch.cat([pre_text_mask, img_mask], dim=1).bool()
            patch_embed, inter_kv = layer.img_forward(patch_embed, key_padding_mask=img_text_key_padding_mask, pre_kv=pre_kv_list[i])
            pre_kv_list[i] = inter_kv
            if i in aux_ids:
                aux_feats.append(patch_embed)
        patch_embed = torch.stack(aux_feats,dim=-1).mean(dim=-1)
        self.grid_interpolate_feats = grid_interpolate_feats
        self.patch_feat = patch_embed
        
        return patch_embed, pre_kv_list
    
    def grid_forward(self, grid_pos_embed, pre_kv_list, grid_interpolate=False, grid_position=None, disable_window=True, return_logits=False, global_only_image=True):
        batch_size, query_num = grid_pos_embed.shape[:2]
        patch_resolution = self.patch_feat.shape[1:3]
        grid_pos_embed = grid_pos_embed.view(batch_size,query_num, 1, -1)
        x = grid_pos_embed.clone()
        for i, layer in enumerate(self.layers):
            if global_only_image and layer.window_size <=0:
                pre_kv_list[i] = []
                continue
            if grid_interpolate:
                x += self.grid_interpolate_feats[i].unsqueeze(2)
            if i > 0:
                x += grid_pos_embed
            
            # generate attn masks
            window_patch_num = layer.window_size ** 2 if layer.window_size > 0 else np.prod(patch_resolution)
            unit_grid_num = query_num // self.num_window if layer.window_size > 0 else query_num
            attn_mask = torch.zeros(unit_grid_num, window_patch_num, device=x.device)
            iter_pad_masks = (1. - torch.eye(unit_grid_num).to(attn_mask.device))
            attn_mask = torch.cat([attn_mask, iter_pad_masks], dim=1).bool()
            
            x, pre_kv_update = layer.token_forward(grid_token=x, grid_position=grid_position, 
                    attn_mask=attn_mask, pre_kv=pre_kv_list[i], disabled_window=disable_window)

            pre_kv_list[i] = pre_kv_update
        if return_logits:
            logits = x @ self.embed.word_embeddings.weight.transpose(0, 1)
            return x, logits, pre_kv_list
        else:
            return x, pre_kv_list
          
    def text_forward(self, text_embed, attn_mask=None, key_padding_mask=None):
        x = text_embed
        new_pre_kv_list = []
        for i, layer in enumerate(self.layers):
            x, pre_kv_update = layer.seq_forward(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            new_pre_kv_list.append(pre_kv_update)
        return x, new_pre_kv_list
    
    def decode_forward(self, input_ids, pre_kv_list, past_len, global_only_image=True):
        x = self.embed(input_ids=input_ids, past_key_values_length=past_len)
        aux_ids = self.mean_layers
        aux_feats = []
        for i, layer in enumerate(self.layers):
            if global_only_image and layer.window_size <=0:
                pre_kv_list[i] = []
                continue
            x, pre_kv_update = layer.seq_forward(x, pre_kv=pre_kv_list[i])
            if i in aux_ids:
                aux_feats.append(x)
            pre_kv_list[i] = pre_kv_update
        x = torch.stack(aux_feats,dim=-1).mean(dim=-1)
        logits = x @ self.embed.word_embeddings.weight.transpose(0, 1)
        return x, logits, pre_kv_list

    def reorder_cache(self, beam_idx, pre_kv_list):
        for i, pre_kv in enumerate(pre_kv_list):
            if len(pre_kv) == 0:
                continue
            num_heads = self.arch_settings['num_heads']
            pre_len, dim = pre_kv.shape[-2:]
            new_kv = pre_kv.view(2, -1, num_heads, pre_len, dim)
            new_kv = torch.gather(new_kv, 1, beam_idx.view(1, -1, 1, 1, 1).repeat(2, 1, num_heads, pre_len ,dim)).flatten(1, 2)
            pre_kv_list[i] = new_kv
        return pre_kv_list
    
    def repeat_cache(self, beam_num, pre_kv_list):
        for i, pre_kv in enumerate(pre_kv_list):
            if len(pre_kv) == 0:
                continue
            num_heads = self.arch_settings['num_heads']
            pre_len, dim = pre_kv.shape[-2:]
            new_kv = pre_kv.view(2, -1, num_heads, pre_len, dim)
            new_kv = new_kv.unsqueeze(2).repeat(1, 1, beam_num, 1, 1, 1).flatten(1, 3)
            pre_kv_list[i] = new_kv
        return pre_kv_list
    
    def _format_output(self, x) -> torch.Tensor:
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        if self.out_type == 'raw' or self.out_type == 'featmap':
            return x
        elif self.out_type == 'avg_featmap':
            # (B, C, H, W) -> (B, C, N) -> (B, N, C)
            x = x.flatten(2).permute(0, 2, 1)
            return x.mean(dim=1)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = ckpt_pos_embed_shape[1:3]
            pos_embed_shape = self.patch_embed.init_out_size

            flattened_pos_embed = state_dict[name].flatten(1, 2)
            resized_pos_embed = resize_pos_embed(flattened_pos_embed,
                                                 ckpt_pos_embed_shape,
                                                 pos_embed_shape,
                                                 self.interpolate_mode, 0)
            state_dict[name] = resized_pos_embed.view(1, *pos_embed_shape,
                                                      self.embed_dims)

    def _prepare_relative_position(self, state_dict, prefix, *args, **kwargs):
        state_dict_model = self.state_dict()
        all_keys = list(state_dict_model.keys())
        for key in all_keys:
            if 'rel_pos_' in key:
                ckpt_key = prefix + key
                if ckpt_key not in state_dict:
                    continue
                relative_position_pretrained = state_dict[ckpt_key]
                relative_position_current = state_dict_model[key]
                L1, _ = relative_position_pretrained.size()
                L2, _ = relative_position_current.size()
                if L1 != L2:
                    new_rel_pos = F.interpolate(
                        relative_position_pretrained.reshape(1, L1,
                                                             -1).permute(
                                                                 0, 2, 1),
                        size=L2,
                        mode='linear',
                    )
                    new_rel_pos = new_rel_pos.reshape(-1, L2).permute(1, 0)
                    from mmengine.logging import MMLogger
                    logger = MMLogger.get_current_instance()
                    logger.info(f'Resize the {ckpt_key} from '
                                f'{state_dict[ckpt_key].shape} to '
                                f'{new_rel_pos.shape}')
                    state_dict[ckpt_key] = new_rel_pos

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.num_layers + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name in ('cls_token', 'pos_embed'):
            layer_depth = 0
        elif param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('layers'):
            layer_id = int(param_name.split('.')[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
