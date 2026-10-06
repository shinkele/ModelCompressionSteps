# ============================================================================
# offsite_tuning/param_efficient.py
# ============================================================================
# 本模块实现了 Offsite-Tuning (arXiv:2302.04870) 所需的三种参数高效微调
# (Parameter-Efficient Fine-Tuning, PEFT) 方法：
#   1. LoRA (Low-Rank Adaptation, arXiv:2106.09685)
#      在权重矩阵旁添加低秩分解矩阵 BA，仅训练 A 和 B。
#   2. Adapter (arXiv:1902.00751)
#      bottleneck FC-ReLU-FC 残差模块，插入到自注意力和 FFN 的输出之后。
#   3. BitFit (arXiv:2106.10199)
#      仅训练偏置参数 (bias terms)，冻结所有权重矩阵。
#
# 在 Offsite-Tuning 的流程中，教师模型侧先使用 PEFT 方法训练适配参数，
# 然后将适配参数与模拟器 (Emulator) 一起发送给学生模型侧进行推理。
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.pytorch_utils import Conv1D


# =========================== LoRA (Low-Rank Adaptation) ===========================

class LoRALayer():
    """
    LoRA 基类，定义低秩分解的公共属性和权重合并/分离的标记。

    核心思想：对于预训练权重 W ∈ R^{d×k}，LoRA 引入低秩分解
        W' = W + BA,  其中 B ∈ R^{d×r}, A ∈ R^{r×k}, r ≪ min(d,k)
    训练时仅更新 A 和 B，大幅减少可训练参数量。

    参数:
        r: 低秩分解的秩 (rank)。
        lora_alpha: 缩放因子，实际缩放比例为 lora_alpha / r。
        lora_dropout: LoRA 分支上的 dropout 概率。
        merge_weights: 是否在 eval 时将 BA 合并到 W 中（加速推理）。
    """
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoRALinear(nn.Linear, LoRALayer):
    """
    将 LoRA 应用于线性层 (nn.Linear)。

    同时继承 nn.Linear 和 LoRALayer，在前向传播中计算：
        output = x @ W^T + (dropout(x) @ A^T @ B^T) * scaling
    其中 scaling = lora_alpha / r。

    支持从 nn.Linear 和 Conv1D (GPT-2 的 1D 卷积层) 初始化。
    """
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            # A: (r, in_features) — 低秩分解的下投影矩阵
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            # B: (out_features, r) — 低秩分解的上投影矩阵
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            # GPT-2 的 Conv1D 权重存储格式为 (fan_in, fan_out)，
            # 而 nn.Linear 为 (fan_out, fan_in)。这里转置以统一存储格式。
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        """
        初始化 LoRA 参数:
          - A 使用 kaiming_uniform_ (与 nn.Linear 默认初始化一致)
          - B 初始化为全零 (使得初始时 BA = 0，等价于原始模型)
        """
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """
        切换到训练模式。
        如果之前处于 merged 状态 (权重已合并)，则先将 BA 从 W 中减去，
        恢复为独立的 W 和 BA，以便独立更新 A 和 B。
        """
        def T(w):
            # 如果权重存储格式为 (fan_in, fan_out)，需要转置后再做运算
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            # 从合并的权重中减去 BA，恢复原始预训练权重
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        """
        切换到评估模式。
        如果 merge_weights=True，将 BA 合并到 W 中 (W = W + BA * scaling)，
        这样推理时只需一次矩阵乘法，无需额外计算 LoRA 分支，提升推理速度。
        """
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            # 将 BA 合并到预训练权重中: W_merged = W + BA * scaling
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        """
        前向传播。

        未合并状态 (训练时):
            output = x @ W^T + dropout(x) @ A^T @ B^T * scaling
        已合并状态 (eval 且 merge_weights=True):
            output = x @ W_merged^T  (BA 已内化到 W 中)
        """
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            # 未合并：分别计算原始权重和 LoRA 分支，然后相加
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                # LoRA 分支：x -> dropout -> @A^T -> @B^T -> *scaling
                result += (self.lora_dropout(x) @ self.lora_A.T @
                           self.lora_B.T) * self.scaling
            return result
        else:
            # 已合并 (或 r=0)：直接使用合并后的权重
            return F.linear(x, T(self.weight), bias=self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        """
        从 nn.Linear 层创建 LoRALinear 层 (用于 OPT 等模型的线性层替换)。
        """
        # Create a LoRALinear layer from a linear layer
        lora_linear = cls(linear.in_features,
                          linear.out_features,
                          r=r,
                          lora_alpha=lora_alpha,
                          lora_dropout=lora_dropout,
                          merge_weights=merge_weights
                          )
        # Copy the weights
        lora_linear.weight.data = linear.weight.data
        if hasattr(linear, 'bias') and linear.bias is not None:
            lora_linear.bias.data = linear.bias.data
        return lora_linear

    @classmethod
    def from_conv1d(cls, conv1d: Conv1D, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        """
        从 Conv1D 层创建 LoRALinear 层 (用于 GPT-2 的 1D 卷积层替换)。

        Conv1D 是 GPT/GPT-2 中使用的"伪卷积"层，本质上是线性层，
        但其权重存储格式为 (fan_in, fan_out)，而 nn.Linear 为 (fan_out, fan_in)。
        因此从 Conv1D 复制权重时需要转置 (.T) 以适配 nn.Linear 的存储格式。
        """
        # 1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
        #
        # Basically works like a linear layer but the weights are transposed.
        # Create a LoRALinear layer from a conv1d layer
        lora_linear = cls(conv1d.weight.size(0),
                          conv1d.weight.size(1),
                          r=r,
                          lora_alpha=lora_alpha,
                          lora_dropout=lora_dropout,
                          merge_weights=merge_weights
                          )
        # Copy the weights
        # Conv1D 权重形状为 (fan_in, fan_out)，需要转置为 nn.Linear 的 (fan_out, fan_in)
        lora_linear.weight.data = conv1d.weight.data.T
        if hasattr(conv1d, 'bias') and conv1d.bias is not None:
            lora_linear.bias.data = conv1d.bias.data
        return lora_linear


# =========================== Adapter 适配器模块 ===========================

class Adapter(nn.Module):
    """
    Bottleneck 适配器模块 (arXiv:1902.00751)。

    结构: Linear_down -> ReLU -> Dropout -> Linear_up -> 残差连接
    即 FC-ReLU-FC 的 bottleneck 形式：
        output = x + W_up(ReLU(W_down(x)))
    其中 W_down: embed_dim -> adapter_size (降维)
          W_up:   adapter_size -> embed_dim (升维)

    适配器插入在自注意力输出投影和 FFN 输出之后，通过 forward hook 实现。
    原始论文表明仅需添加约 3% 的额外参数即可达到接近全参数微调的效果。
    """
    def __init__(self, embed_dim: int, adapter_size: int, dropout: float = 0.1):
        super().__init__()
        # 下投影：embed_dim -> adapter_size (bottleneck 压缩)
        self.fc1 = nn.Linear(embed_dim, adapter_size)
        self.dropout = nn.Dropout(dropout)
        # 上投影：adapter_size -> embed_dim (恢复原始维度)
        self.fc2 = nn.Linear(adapter_size, embed_dim)
        self.act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        前向传播: output = x + W_up(ReLU(W_down(x)))

        残差连接确保初始化时适配器近似恒等映射 (W_up 通常初始化为接近零)，
        不会破坏预训练模型已有的表示能力。
        """
        residual = x
        x = self.fc1(x)        # 下投影
        x = self.act_fn(x)     # 非线性激活
        x = self.dropout(x)    # 正则化
        x = self.fc2(x)        # 上投影
        x = x + residual       # 残差连接
        return x


# =========================== 模型修改工具函数 ===========================

def add_adapter(layer, embed_dim: int, adapter_size: int, dropout: float):
    """
    在单个线性层后添加适配器模块，通过注册 forward hook 实现。

    工作原理：在 layer 的前向传播完成后，自动将输出传入 adapter，
    实现 output = adapter(layer(x)) 的效果，无需修改模型原始 forward 代码。

    参数:
        layer: 目标线性层 (如 self_attn.out_proj, fc2 等)。
        embed_dim: 嵌入维度。
        adapter_size: 适配器 bottleneck 维度。
        dropout: 适配器中的 dropout 概率。
    """
    # add an adapter module after the forward pass of a linear layer
    # register a forward hook to the layer
    def forward_hook(module, input, output):
        return module.adapter(output)
    layer.adapter = Adapter(embed_dim, adapter_size, dropout)
    layer.register_forward_hook(forward_hook)


def use_adapter(layers: nn.ModuleList, adapter_size: int, dropout: float = 0.1):
    """
    将 Adapter 模块注入到 Transformer 的所有 decoder layer 中。

    根据模型架构在每一层的特定位置添加适配器：
      - OPT:  self_attn.out_proj 和 fc2 之后
      - GPT-2: attn.c_proj 和 mlp.c_proj 之后

    添加后冻结所有原始参数，仅训练适配器参数。
    """
    # ===== 按架构在每层插入适配器 =====
    if isinstance(layers[0], OPTDecoderLayer):
        # OPT 架构：在自注意力的输出投影和 FFN 的第二个全连接层后添加适配器
        for layer in layers:
            add_adapter(layer.self_attn.out_proj,
                        layer.embed_dim, adapter_size, dropout)
            add_adapter(layer.fc2, layer.embed_dim, adapter_size, dropout)
    elif isinstance(layers[0], GPT2Block):
        # GPT-2 架构：在注意力的 c_proj 和 MLP 的 c_proj 后添加适配器
        for layer in layers:
            add_adapter(layer.attn.c_proj, layer.attn.embed_dim,
                        adapter_size, dropout)
            add_adapter(layer.mlp.c_proj, layer.attn.embed_dim,
                        adapter_size, dropout)
    else:
        raise NotImplementedError

    # ===== 冻结所有非适配器参数 =====
    # freeze all parameters except the adapter modules
    for param in layers.parameters():
        param.requires_grad = False
    for name, param in layers.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True


def use_lora(layers: nn.ModuleList, r: int, lora_alpha: int, lora_dropout: float = 0.1, merge_weights: bool = False):
    """
    使用 LoRA 替换 Transformer decoder layers 中的所有线性层。

    根据模型架构替换不同的层：
      - OPT:  q_proj, k_proj, v_proj, out_proj, fc1, fc2
      - GPT-2: c_attn, c_proj, c_fc, c_proj (使用 from_conv1d)

    替换后冻结所有原始参数，仅训练 LoRA 参数 (lora_A, lora_B)。
    在 Offsite-Tuning 中，这些 LoRA 参数作为适配参数发送给学生模型。
    """
    # Replace all linear layers with LoRALinear layers
    # ===== 按架构替换线性层为 LoRALinear =====
    if isinstance(layers[0], OPTDecoderLayer):
        # OPT 架构的层映射：
        #   self_attn.{q_proj, k_proj, v_proj, out_proj} — 自注意力中的线性投影
        #   fc1, fc2 — FFN 中的两个全连接层
        for layer in layers:
            layer.self_attn.q_proj = LoRALinear.from_linear(
                layer.self_attn.q_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.self_attn.k_proj = LoRALinear.from_linear(
                layer.self_attn.k_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.self_attn.v_proj = LoRALinear.from_linear(
                layer.self_attn.v_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.self_attn.out_proj = LoRALinear.from_linear(
                layer.self_attn.out_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.fc1 = LoRALinear.from_linear(
                layer.fc1, r, lora_alpha, lora_dropout, merge_weights)
            layer.fc2 = LoRALinear.from_linear(
                layer.fc2, r, lora_alpha, lora_dropout, merge_weights)
    elif isinstance(layers[0], GPT2Block):
        # GPT-2 架构的层映射：
        #   attn.c_attn — 合并的 QKV 投影 (1D 卷积)
        #   attn.c_proj — 注意力输出投影 (1D 卷积)
        #   mlp.c_fc — FFN 第一层 (1D 卷积)
        #   mlp.c_proj — FFN 第二层 (1D 卷积)
        # 注：GPT-2 使用 Conv1D 而非 nn.Linear，因此通过 from_conv1d 转换
        for layer in layers:
            layer.attn.c_attn = LoRALinear.from_conv1d(
                layer.attn.c_attn, r, lora_alpha, lora_dropout, merge_weights)
            layer.attn.c_proj = LoRALinear.from_conv1d(
                layer.attn.c_proj, r, lora_alpha, lora_dropout, merge_weights)
            layer.mlp.c_fc = LoRALinear.from_conv1d(
                layer.mlp.c_fc, r, lora_alpha, lora_dropout, merge_weights)
            layer.mlp.c_proj = LoRALinear.from_conv1d(
                layer.mlp.c_proj, r, lora_alpha, lora_dropout, merge_weights)
    else:
        raise NotImplementedError

    # ===== 冻结所有非 LoRA 参数 =====
    # freeze all parameters except the LoRALinear layers
    for param in layers.parameters():
        param.requires_grad = False
    for name, param in layers.named_parameters():
        if 'lora' in name:
            param.requires_grad = True


def use_bitfit(model: nn.Module):
    """
    应用 BitFit (arXiv:2106.10199) — 仅训练偏置参数。

    冻结模型中的所有参数，然后对维度为 1 的参数 (即偏置项 bias
    和 LayerNorm 中的可学习参数，它们在 PyTorch 中都是 1 维张量)
    重新开启 requires_grad。

    BitFit 是最轻量级的 PEFT 方法之一，可训练参数量极少。
    """
    # freeze all parameters except the bias terms
    # train the bias terms only
    for param in model.parameters():
        param.requires_grad = False
    # 偏置参数和 LayerNorm 参数在 PyTorch 中均为 1 维张量
    for param in model.parameters():
        if param.dim() == 1:
            param.requires_grad = True
