"""
EVA ViT (Exploiting Vision Augmentation ViT) 视觉模型实现。
包含完整的 ViT 架构组件（arXiv:2302.04870）：
  - LayerNormWithForceFP32: 强制 FP32 精度的 LayerNorm
  - DropPath: 随机深度正则化
  - Mlp: Transformer MLP 块
  - Attention: 多头自注意力（含相对位置偏置/窗口注意力/解耦相对位置偏置）
  - Block: Transformer 编码器块 (Pre-norm / Post-norm)
  - PatchEmbed: 图像分块嵌入层
  - RelativePositionBias: 窗口内相对位置偏置
  - DecoupledRelativePositionBias: 解耦相对位置偏置（高宽分离）
  - EVAViTForImageClassification: EVA ViT 图像分类器
"""
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import nn


import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from transformers.modeling_outputs import ImageClassifierOutput


import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.parameter import Parameter
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from torch import Tensor, Size
from typing import Union, List
import numbers


_shape_t = Union[int, List[int], Size]


class LayerNormWithForceFP32(nn.Module):
    """
    强制使用 FP32 计算的 LayerNorm。
    与标准 nn.LayerNorm 不同，该实现强制将输入和权重/偏置转为 float32 计算后再转回原始 dtype，
    以避免 FP16/BF16 下数值精度不足导致的训练不稳定。
    推理时不可少，因为混合精度下若保持 FP16 做 normalization 可能产生 NaN。
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: _shape_t
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNormWithForceFP32, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        # 强制将所有张量转为 float32 计算，结果转回输入的原始 dtype
        return F.layer_norm(
            input.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(input)

    def extra_repr(self) -> Tensor:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class DropPath(nn.Module):
    """
    随机深度 (Stochastic Depth) 正则化。
    在残差块的主路径中按概率随机丢弃整个样本（将其输出置零），
    用于训练深层 ViT 时的正则化，防止过拟合并提高泛化能力。
    参考: "Deep Networks with Stochastic Depth" (ECCV 2016).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    """
    Transformer MLP 块。
    结构: fc1 (Linear) → GELU 激活 → fc2 (Linear) → Dropout。
    标准的两层全连接前馈网络，隐藏层维度通常为输入维度的 mlp_ratio 倍。
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    多头自注意力 (Multi-Head Self-Attention) 模块。
    支持三种相对位置偏置形式:
        - 无偏置: 标准缩放点积注意力
        - 常规相对位置偏置 (RelativePositionBias): 窗口内 token 对的完整相对位置编码
        - 解耦相对位置偏置 (DecoupledRelativePositionBias): 将高和宽的位置编码分离，降低参数量

    可选窗口注意力: 当 window_size 不为 None 时，计算窗口内的相对位置索引表。
    """

    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rel_pos_bias = None
        self.qk_float = True

        self.window_size = None
        self.relative_position_bias_table = None

        if window_size:
            # 使用解耦相对位置偏置: 高和宽的位置编码独立计算并相加
            if use_decoupled_rel_pos_bias:
                self.rel_pos_bias = DecoupledRelativePositionBias(
                    window_size=window_size, num_heads=num_heads)
            else:
                # 常规窗口相对位置偏置: 构建完整的相对位置索引表
                self.window_size = window_size
                self.num_relative_distance = (
                    2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3    # (2*14-1) * (2*14-1) + 3
                # 相对位置偏置表: 每个 (相对距离, 注意力头) 对应一个可学习的偏置值
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
                # cls to token & token 2 cls & cls to cls

                # ---- 构建窗口内 token 对的相对位置坐标网格 ----
                # 生成高度和宽度方向的坐标轴
                coords_h = torch.arange(window_size[0])
                coords_w = torch.arange(window_size[1])
                # meshgrid 生成 2D 坐标网格: shape [2, Wh, Ww], 第0维为行坐标, 第1维为列坐标
                coords = torch.stack(torch.meshgrid(
                    [coords_h, coords_w]))  # 2, Wh, Ww
                # 展平为 [2, Wh*Ww]: 每个 token 的 (行, 列) 坐标
                coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
                # 广播相减得到 pairwise 相对坐标: [2, Wh*Ww, Wh*Ww]
                # relative_coords[d, i, j] = coords_flatten[d, i] - coords_flatten[d, j]
                relative_coords = coords_flatten[:, :,
                                                 None] - coords_flatten[:, None, :]
                relative_coords = relative_coords.permute(
                    1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
                # 将坐标偏移到从 0 开始: 原范围为 [-(W-1), W-1], 加 W-1 后变为 [0, 2W-2]
                relative_coords[:, :, 0] += window_size[0] - \
                    1  # shift to start from 0
                relative_coords[:, :, 1] += window_size[1] - 1
                # 将 2D 相对坐标压缩为 1D 索引: row_idx * (2*Ww-1) + col_idx
                relative_coords[:, :, 0] *= 2 * window_size[1] - 1
                # 构建包含 CLS token 的完整索引矩阵 (Wh*Ww+1, Wh*Ww+1)
                relative_position_index = \
                    torch.zeros(
                        size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
                # Wh*Ww, Wh*Ww: 填充 patch token 之间的相对位置
                relative_position_index[1:, 1:] = relative_coords.sum(-1)
                # CLS → token: 索引 num_relative_distance - 3
                relative_position_index[0, 0:] = self.num_relative_distance - 3
                # token → CLS: 索引 num_relative_distance - 2
                relative_position_index[0:, 0] = self.num_relative_distance - 2
                # CLS → CLS: 索引 num_relative_distance - 1
                relative_position_index[0, 0] = self.num_relative_distance - 1

                # 注册为 buffer（不参与梯度更新，但随模型保存/加载）
                self.register_buffer(
                    "relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            # 拼接 Q/K/V 偏置: Q 偏置 + 零填充 K 偏置 + V 偏置
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(
                self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Q 乘以缩放因子 1/sqrt(d_k)，稳定梯度
        q = q * self.scale
        # 使用 float32 计算注意力分数以提高数值精度
        if self.qk_float:
            attn = (q.float() @ k.float().transpose(-2, -1))
        else:
            attn = (q @ k.transpose(-2, -1))

        # ---- 添加窗口相对位置偏置 ----
        if self.relative_position_bias_table is not None:
            # 从 relative_position_bias_table 中查找相对位置索引对应的偏置值
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

        # ---- 添加解耦相对位置偏置 ----
        if self.rel_pos_bias is not None:
            attn = attn + self.rel_pos_bias().type_as(attn)

        # ---- 添加外部传入的相对位置偏置（共享偏置场景） ----
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias.type_as(attn)
        if attn_mask is not None:
            attn_mask = attn_mask.bool()
            attn = attn.masked_fill(
                ~attn_mask[:, None, None, :], float("-inf"))
        # softmax 归一化后转回原始 dtype
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        # 加权求和 + 重塑 + 输出投影
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer 编码器块。
    支持两种归一化模式:
        - Pre-norm (默认): Norm → Attention/MLP → 残差连接 (主流做法, 训练稳定)
        - Post-norm: Attention/MLP → Norm → 残差连接 (原 Transformer 设计)

    支持 LayerScale (init_values): 为每个残差分支添加可学习的缩放因子 gamma，
    用于稳定深层 ViT 的训练。参考 "CaiT: Going deeper with Image Transformers".

    残差路径使用 DropPath (随机深度) 代替传统的 Dropout。
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False,
                 postnorm=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size,
            use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic num_layers, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # LayerScale 可学习缩放因子
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            # 无 LayerScale: 标准残差连接
            if self.postnorm:
                # Post-norm: 先做 attn/mlp，再做 norm
                x = x + self.drop_path(
                    self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                # Pre-norm: 先做 norm，再做 attn/mlp
                x = x + self.drop_path(
                    self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # 使用 LayerScale: gamma 缩放残差分支输出
            if self.postnorm:
                x = x + self.drop_path(
                    self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(
                    self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    图像分块嵌入层 (Patch Embedding)。
    使用 Conv2d 将输入图像 (B, C, H, W) 切分为固定大小的 patch 并投影到嵌入维度。
    例如: 224x224 图像, patch_size=16 → 14x14=196 个 patch。
    """
    """ Image to Patch Embedding
    """

    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * \
            (image_size[0] // patch_size[0])
        self.patch_shape = (
            image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Conv2d 投影: kernel_size=stride=patch_size, 等效于无重叠的分块嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.image_size[0] and W == self.image_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        # Conv2d → flatten 空间维度 → transpose: (B, C, H, W) → (B, embed_dim, H', W') → (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):
    """
    相对位置偏置（窗口内 token 对的位置编码）。
    为窗口内每对 token (包括 CLS token) 学习一个偏置值，添加到注意力分数中。

    偏置表大小: (2*Wh-1)*(2*Ww-1)+3，其中 +3 为 CLS token 相关的三种关系:
        - CLS → token  (索引 num_relative_distance - 3)
        - token → CLS  (索引 num_relative_distance - 2)
        - CLS → CLS    (索引 num_relative_distance - 1)
    """

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (
            2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # ---- 构建窗口内 token 对的相对位置坐标网格 ----
        # 与 Attention 类中相同的相对位置索引构建逻辑
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:,
                                1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index",
                             relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        """返回窗口内 (含 CLS) 的相对位置偏置: [num_heads, num_tokens, num_tokens]"""
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.permute(2, 0, 1).contiguous()


def _maske_1d_rel_pos_index(seq_len):
    """
    构建一维相对位置索引矩阵。
    返回 shape [seq_len, seq_len] 的矩阵，元素 (i, j) = i - j + seq_len - 1，
    即相对偏移从 0 开始的最大绝对偏移为 2*seq_len - 1。
    用于解耦相对位置偏置中单维度（高或宽）的索引。
    """
    index = torch.arange(seq_len)
    return index.view(1, seq_len) - index.view(seq_len, 1) + seq_len - 1


def _add_cls_to_index_matrix(index, num_tokens, offset):
    """
    将 CLS token 加入到已有的相对位置索引矩阵中。

    参数:
        index: [num_tokens, num_tokens] 的 patch token 相对位置索引矩阵。
        num_tokens: patch token 数量 (Wh * Ww)。
        offset: CLS→token 关系索引的偏移值 (= 2*W-1，即最大相对距离)。
    返回:
        [num_tokens+1, num_tokens+1] 的索引矩阵，包含 CLS 的关系:
        - CLS → token: offset
        - token → CLS: offset + 1
        - CLS → CLS: offset + 2
    """
    index = index.contiguous().view(num_tokens, num_tokens)
    new_index = torch.zeros(
        size=(num_tokens + 1, num_tokens + 1), dtype=index.dtype)
    new_index[1:, 1:] = index
    new_index[0, 0:] = offset
    new_index[0:, 0] = offset + 1
    new_index[0, 0] = offset + 2
    return new_index


class DecoupledRelativePositionBias(nn.Module):
    """
    解耦的相对位置偏置（将高和宽的位置编码分离）。

    与 RelativePositionBias 不同，该类将 2D 相对位置分解为高度维度和宽度维度的
    独立一维位置编码，然后相加得到最终的相对位置偏置。这样参数量从 O(W^2*H^2)
    降至 O(W*H)，在大窗口场景下显著减少内存和计算开销。

    偏置表:
        - relative_position_bias_for_high: [2*Wh+2, num_heads]  高度方向位置编码
        - relative_position_bias_for_width: [2*Ww+2, num_heads]  宽度方向位置编码
    """

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (
            2 * window_size[0] + 2, 2 * window_size[1] + 2)

        num_tokens = window_size[0] * window_size[1]

        # 高度和宽度方向各自独立的相对位置偏置表
        self.relative_position_bias_for_high = nn.Parameter(
            torch.zeros(self.num_relative_distance[0], num_heads))
        self.relative_position_bias_for_width = nn.Parameter(
            torch.zeros(self.num_relative_distance[1], num_heads))
        # cls to token & token 2 cls & cls to cls

        # ---- 构建高度方向的相对位置索引 ----
        # _maske_1d_rel_pos_index 生成一维索引, 然后广播扩展到 2D 窗口的所有 token 对
        h_index = _maske_1d_rel_pos_index(window_size[0]).view(
            window_size[0], 1, window_size[0], 1).expand(-1, window_size[1], -1, window_size[1])
        # 将 CLS token 加入索引矩阵
        h_index = _add_cls_to_index_matrix(
            h_index, num_tokens, 2 * window_size[0] - 1)
        self.register_buffer("relative_position_high_index", h_index)

        # ---- 构建宽度方向的相对位置索引 ----
        w_index = _maske_1d_rel_pos_index(window_size[1]).view(
            1, window_size[1], 1, window_size[1]).expand(window_size[0], -1, window_size[0], -1)
        w_index = _add_cls_to_index_matrix(
            w_index, num_tokens, 2 * window_size[1] - 1)

        self.register_buffer("relative_position_width_index", w_index)

    def forward(self):
        """
        返回解耦相对位置偏置: [num_heads, num_tokens, num_tokens]。
        高度方向偏置 + 宽度方向偏置 → 使用 embedding 查表实现。
        """
        relative_position_bias = \
            F.embedding(input=self.relative_position_high_index, weight=self.relative_position_bias_for_high) + \
            F.embedding(input=self.relative_position_width_index,
                        weight=self.relative_position_bias_for_width)
        return relative_position_bias.permute(2, 0, 1).contiguous()


class EVAViTForImageClassification(nn.Module):
    """
    EVA ViT 图像分类器。

    架构: PatchEmbed → [CLS Token + Position Embedding] → N 个 Block (自注意力 + MLP) →
          Norm/FcNorm (mean pooling 或 CLS token) → Linear 分类头。

    核心特性:
        - 支持绝对位置嵌入 (use_abs_pos_emb) 和相对位置偏置 (use_rel_pos_bias)
        - 支持共享/解耦相对位置偏置，减少参数量
        - 支持 mean pooling (推荐) 或 CLS token 两种特征聚合方式
        - fix_init_weight: 按层深度 rescale 权重 (除以 sqrt(2*layer_id))
        - stop_grad_conv1: 可选冻结 patch embedding 的梯度
        - 支持梯度检查点 (use_checkpoint) 以节省显存
    """
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, image_size=224, patch_size=16, in_chans=3, num_labels=1000, embed_dim=768, num_layers=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_abs_pos_emb=True,
                 use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_decoupled_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, use_checkpoint=False, stop_grad_conv1=False):
        super().__init__()
        self.num_labels = num_labels
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            image_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token: 可学习的分类标记，shape [1, 1, embed_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 绝对位置嵌入: 用于保留空间位置信息
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 共享相对位置偏置: 所有 Block 共享同一个 RelativePositionBias 实例
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias
        self.use_checkpoint = use_checkpoint
        self.stop_grad_conv1 = stop_grad_conv1

        # 确定窗口大小: 仅在启用相对位置偏置时需要
        if use_decoupled_rel_pos_bias or use_rel_pos_bias:
            window_size = self.patch_embed.patch_shape
        else:
            window_size = None

        # 随机深度衰减规则 (stochastic depth decay rule):
        # 从 0 线性增加到 drop_path_rate, 深层 Block 有更高的丢弃概率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias)
            for i in range(num_layers)])

        # 特征聚合方式选择:
        # use_mean_pooling: norm = Identity, fc_norm = LayerNorm (对 patch token 做 mean pooling 后 normalize)
        # 否则: norm = LayerNorm, fc_norm = None (对 CLS token 直接输出)
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.classifier = nn.Linear(
            embed_dim, num_labels) if num_labels > 0 else nn.Identity()

        # 权重初始化: pos_embed 和 cls_token 使用 trunc_normal
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.classifier, nn.Linear):
            trunc_normal_(self.classifier.weight, std=.02)
        self.apply(self._init_weights)
        # 按层深度 rescale 权重 (EVA 特有的初始化策略)
        self.fix_init_weight()

        # 分类头权重按 init_scale 缩放（默认 0.001），使初始 logits 接近零
        if isinstance(self.classifier, nn.Linear):
            self.classifier.weight.data.mul_(init_scale)
            self.classifier.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        """
        EVA 特有的权重 rescale 初始化。
        对每个 Block 的注意力投影 (attn.proj) 和 MLP 输出层 (mlp.fc2) 的权重
        除以 sqrt(2 * layer_id)，其中 layer_id 从 1 开始计数。
        这样可以控制残差分支的方差，使得深层 Block 的残差贡献更小，
        从而在训练初期保持接近恒等映射，利于深层模型稳定训练。
        参考: "DeepNet: Scaling Transformers to 1,000 Layers" 的初始化思想。
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """标准 ViT 权重初始化: Linear 用 trunc_normal, LayerNorm 的 bias=0 weight=1"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        """pos_embed 和 cls_token 不参与权重衰减（通常为可学习的位置/标记参数）"""
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_labels, global_pool=''):
        self.num_labels = num_labels
        self.classifier = nn.Linear(
            self.embed_dim, num_labels) if num_labels > 0 else nn.Identity()

    def forward_features(self, x, return_patch_tokens=False):
        """
        ViT 编码器前向传播 (不含分类头)。

        参数:
            x: 输入图像 tensor [B, C, H, W]。
            return_patch_tokens: 是否返回所有 patch token 的特征。
        返回:
            - return_patch_tokens=True → [B, num_patches, embed_dim] 所有 patch 特征
            - use_mean_pooling=True  → [B, embed_dim] 所有 patch 特征的均值 (mean pooling)
            - use_mean_pooling=False → [B, embed_dim] CLS token 特征
        """
        x = self.patch_embed(x)

        # 可选: 阻止梯度通过 patch embedding 卷积层回传
        # 用于 Offsite-Tuning 场景中冻结底层特征提取器（arXiv:2302.04870）
        if self.stop_grad_conv1:
            x = x.detach()

        batch_size, seq_len, _ = x.size()

        # 在序列最前添加 CLS token (stole cls_tokens impl from Phil Wang, thanks)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # 预计算共享的相对位置偏置（所有 Block 复用同一个偏置）
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.use_checkpoint:
                # 梯度检查点: 节省显存，但略微增加计算量（重计算）
                x = checkpoint.checkpoint(blk, x, rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias)

        x = self.norm(x)
        if self.fc_norm is not None:
            # Mean pooling 模式: 对 patch token (除去 CLS) 做均值池化，再经 fc_norm
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            # CLS token 模式: 直接取第一个 token (CLS) 作为全局特征
            if return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, pixel_values, labels=None):
        x = self.forward_features(pixel_values)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            # 单标签分类: 使用交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
        )
