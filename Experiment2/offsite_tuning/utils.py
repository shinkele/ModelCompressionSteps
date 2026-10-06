"""
Offsite-Tuning 核心工具模块
===========================
本模块实现了 Offsite-Tuning (arXiv:2302.04870) 框架的核心功能，
是一种隐私保护的迁移学习框架，允许数据拥有者在无法访问完整大模型的
情况下完成下游任务的微调。

整体架构：
  模型拥有者 (Owner)                      数据拥有者 (Data Owner)
  ┌─────────────────────┐                 ┌──────────────────────┐
  │ 完整大模型 (Teacher) │                 │ 下游数据集 (Private)  │
  │  ↓ 压缩和适配层     │                 │  ↓ 加载                │
  │ 学生模型 (Student)  │  ──── 发送 ──→  │ 学生模型 (Emulator)    │
  │ 适配器 (Adapter)    │                 │ 适配器 (Adapter)      │
  │                     │  ←── 返回适配器 ─│  ↓ 微调                │
  │ 插入全模型          │                 │ 微调后的适配器          │
  └─────────────────────┘                 └──────────────────────┘

核心概念：
  - Teacher (教师模型): 完整大模型的中间层，提供知识蒸馏信号，不离开模型拥有者
  - Student (学生模型): 轻量化的压缩模拟器，从教师层中选出并可选压缩，发送给数据拥有者
  - Adapter (适配层): 大模型的首尾N层，与student一起发送，适配输入/输出分布
  - KD Loss (知识蒸馏损失): 教师输出与学生输出的归一化MSE损失，约束学生模拟教师行为

本模块包含的函数分组：
  1. 辅助层与工具类: MLP, add_prologue, add_epilogue
  2. 层选择策略: uniform_choose_layers
  3. 模型压缩: magnitude_prune, quantize
  4. 参数解析: parse_args
  5. 模型层操作: get_layers, set_layers
  6. 教师-学生构建: setup_teacher_student
  7. 教师/学生切换: to_teacher, to_student
  8. 知识蒸馏: get_kd_loss
  9. 分类头设置: setup_trainable_classification_head
  10. 模型加载/保存: load_adapter, load_student, save_state_dict

参考文献：
  - Offsite-Tuning: Transfer Learning without Full Model (arXiv:2302.04870)
"""

import gc
import os
from copy import deepcopy
import torch
from torch import nn
from accelerate.logging import get_logger
from transformers import (
    SchedulerType,
    MODEL_MAPPING,
    OPTForCausalLM,
    GPT2LMHeadModel,
    BloomForCausalLM,
    ViTForImageClassification,
)
from offsite_tuning.models.clip_vit import CLIPViTForImageClassification
from offsite_tuning.models.eva_vit import EVAViTForImageClassification

import argparse


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)


# ===========================================================================
# 1. 辅助层与工具类 (MLP, Prologue/Epilogue)
# ===========================================================================

class MLP(nn.Module):
    """多层感知机 (Multi-Layer Perceptron) 辅助模块。

    用于在 offsite-tuning 中作为可选的 prologue/epilogue 层，
    帮助学生模型适配教师模型的输入/输出分布。

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        num_layers: 隐藏层数量（默认为1）
        activation: 激活函数类（默认为 nn.ReLU）
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation()

        self.layers = nn.ModuleList()
        # 第一层：input_dim -> hidden_dim
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        # 中间隐藏层：hidden_dim -> hidden_dim
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        # 最后一层：hidden_dim -> output_dim（无激活函数）
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        """前向传播：逐层通过线性变换和激活函数，最后一层不经过激活函数。"""
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


def add_prologue(module, prologue):
    """为模块添加前导层（prologue），通过钩子函数拦截原始 forward 的输入。

    此函数用于教师模型：在教师第一层之前插入 prologue，
    以便捕获教师模型的输入 args/kwargs（存储在 self.input_args 和 self.input_kwargs 中）。
    KD loss 计算时需要通过此机制获取教师的输入。

    Args:
        module: 要添加 prologue 的 nn.Module
        prologue: 前导模块（可为 None，表示仅捕获输入而不做变换）

    Returns:
        修改后的 module（原地修改）
    """
    module.old_forward = module.forward
    module.prologue = prologue

    def new_forward(self):
        def lambda_forward(*args, **kwargs):
            # 缓存输入参数，供 get_kd_loss 使用
            self.input_args = args
            self.input_kwargs = kwargs
            if self.prologue is not None:
                x = self.prologue(args[0])
            else:
                x = args[0]
            args = (x,) + args[1:]
            return self.old_forward(*args, **kwargs)
        return lambda_forward
    module.forward = new_forward(module)
    return module


def add_epilogue(module, epilogue):
    """为模块添加后置层（epilogue），通过钩子函数拦截原始 forward 的输出。

    此函数用于学生模型：在学生最后一层之后插入 epilogue，
    以便缓存学生的隐藏层输出（存储在 self.cached_output 中）。
    KD loss 计算时需要通过此机制获取学生的输出与教师输出进行对比。

    Args:
        module: 要添加 epilogue 的 nn.Module
        epilogue: 后置模块（可为 None，表示仅缓存输出而不做变换）

    Returns:
        修改后的 module（原地修改）
    """
    module.old_forward = module.forward
    module.epilogue = epilogue

    def new_forward(self):
        def lambda_forward(*args, **kwargs):
            output = self.old_forward(*args, **kwargs)
            # 处理 tuple 类型的输出（如 (hidden_states, ...)）
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output

            if self.epilogue is not None:
                x = self.epilogue(x)

            if isinstance(output, tuple):
                output = (x,) + output[1:]
            else:
                output = x

            # 缓存学生输出，供 get_kd_loss 使用
            self.cached_output = x
            return output
        return lambda_forward
    module.forward = new_forward(module)
    return module


# ===========================================================================
# 2. 层选择策略
# ===========================================================================

def uniform_choose_layers(layers: nn.ModuleList, num_student_layers=None):
    """均匀层选择策略：从完整模型的层列表中按等间距选取若干层构成学生模型。

    采样公式: idx = round(i * stride), 其中 stride = (len(layers) - 1) / (num_student_layers - 1)
    这种策略确保首尾层总是被选中，中间层均匀分布，最大程度保留原模型的表示能力。

    Args:
        layers: 完整模型的 transformer 层列表
        num_student_layers: 要选择的学生层数，默认为 None（使用全部层）

    Returns:
        nn.ModuleList: 选中的学生层列表
    """
    if num_student_layers is None:
        num_student_layers = len(layers)

    student = nn.ModuleList()
    # 计算采样步长：确保首尾层被选中，中间层均匀分布
    stride = (len(layers) - 1) / (num_student_layers - 1)

    for i in range(num_student_layers):
        idx = round(i * stride)
        logger.info(f"Adding layer {idx} to student")
        student.append(layers[idx])

    return student


# ===========================================================================
# 3. 模型压缩 (剪枝 & 量化)
# ===========================================================================

@torch.no_grad()
def magnitude_prune(model, ratio):
    """幅值剪枝 (Magnitude Pruning)：按参数绝对值排序，将 ratio 比例的最小幅值参数置零。

    这是一种非结构化剪枝方法，通过将绝对值最小的权重置零来减少模型的有效参数量。
    剪枝后的稀疏结构可以降低模型传输成本。

    Args:
        model: 要剪枝的模型
        ratio: 剪枝比例，范围 [0, 1]，如 0.5 表示将50%的权重置零
    """
    for param in model.parameters():
        # 跳过偏置项（一维参数），只剪枝权重矩阵
        if param.dim() == 1:
            continue
        # 计算需要剪枝的参数数量
        num_prune = int(param.numel() * ratio)
        # 找到第 num_prune 小的绝对值作为阈值
        threshold = param.abs().view(-1).kthvalue(num_prune).values.item()
        # 生成 mask：绝对值 >= 阈值的保留，否则置零
        mask = (param.abs() >= threshold).to(param.dtype)
        param.mul_(mask)


@torch.no_grad()
def quantize(model, bits):
    """均匀量化 (Uniform Quantization)：将32位浮点参数量化到指定位宽。

    使用零点量化 (zero-point quantization) 方案：
      zp = (max + min) / 2          — 零点（zero point）
      scale = (max - min) / (2^bits - 1)  — 量化步长
      quantized = round((w - zp) / scale) * scale + zp

    量化可以减少模型存储和传输的大小，是 offsite-tuning 中压缩模拟器的可选步骤。

    Args:
        model: 要量化的模型
        bits: 量化位宽，如 8 表示 8-bit 量化
    """
    for param in model.parameters():
        # 跳过偏置项（一维参数），只量化权重矩阵
        if param.dim() == 1:
            continue
        min, max = param.min(), param.max()
        # 零点（zero point）：取值范围的中心
        zp = (max + min) / 2
        # 量化步长（scale）：将取值范围均匀映射到 2^bits 个区间
        scale = (max - min) / (2 ** bits - 1)
        # 量化-反量化过程：模拟精度损失
        param.sub_(zp).div_(scale).round_().mul_(scale).add_(zp)


# ===========================================================================
# 4. 命令行参数解析
# ===========================================================================

def parse_args():
    """解析 Offsite-Tuning 训练脚本的所有命令行参数。

    返回的 argparse.Namespace 包含所有训练、模型压缩、评估相关的配置。
    此函数被 run_clm.py 和 run_image_classification.py 共用。

    Returns:
        argparse.Namespace: 解析后的参数字典
    """
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")

    # ---- 数据集参数 ----
    parser.add_argument(
        "--num_choose_train_expamples",
        type=int,
        default=1000,
        help="num_choose_train_expamples",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        type=int,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )

    # ---- 模型与分词器参数 ----
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    # ---- 训练超参数 ----
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamw',
        help='Optimizer to use. Can be adamw or sgd',
        choices=['adamw', 'sgd']
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum to use for sgd optimizer."
    )
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )

    # ---- 输出与日志参数 ----
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=88,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        '--no_save_model',
        action='store_true',
        help='Whether or not to save the model.'
    )

    # ---- 知识蒸馏参数 ----
    parser.add_argument(
        '--kd_weight',
        type=float,
        default=0.0,
        help='Weight of the knowledge distillation loss.'
    )
    parser.add_argument(
        '--lm_weight',
        type=float,
        default=1.0,
        help='Weight of the language modeling loss (task loss).'
    )

    # ---- 预分词数据集路径 ----
    parser.add_argument(
        '--train_tokenized_dataset',
        type=str,
        default=None,
        help='Path to the tokenized training dataset.'
    )
    parser.add_argument(
        '--val_tokenized_dataset',
        type=str,
        default=None,
        help='Path to the tokenized validation dataset.'
    )
    parser.add_argument(
        "--train_num_samples",
        type=int,
        default=None,
        help="The number of samples to use for training set.",
    )
    parser.add_argument(
        "--validation_num_samples",
        type=int,
        default=None,
        help="The number of samples to use for validation set.",
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=200,
        help='Number of training steps between evaluations.'
    )

    # ---- 学生模型（Emulator）参数 ----
    parser.add_argument(
        '--num_student_layers',
        type=int,
        default=None,
        help='Number of layers in the student model. If None, uses all selected layers.'
    )
    parser.add_argument(
        '--load_student',
        type=str,
        default=None,
        help='Path to the pretrained student model checkpoint.'
    )
    parser.add_argument(
        '--student_l_pad',
        type=int,
        default=0,
        help='Number of adapter layers on the left (bottom) side of the student.'
    )
    parser.add_argument(
        '--student_r_pad',
        type=int,
        default=0,
        help='Number of adapter layers on the right (top) side of the student.'
    )
    parser.add_argument(
        '--student_layer_selection_strategy',
        type=str,
        default='uniform',
        help='Layer selection strategy for building the student model.',
        choices=['uniform', 'random', 'changes']
    )
    parser.add_argument(
        '--restart_training',
        action='store_true',
        help='Whether to restart training from scratch (ignore existing checkpoint).'
    )

    # ---- 训练模式参数 ----
    parser.add_argument(
        '--train_module',
        type=str,
        default='student',
        help='Part of the model to train: student (emulator), adapter (top/bottom layers), or all.',
        choices=['student', 'adapter', 'all']
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0,
        help='Max gradient norm for gradient clipping.'
    )

    # ---- 模型压缩参数 ----
    parser.add_argument(
        '--magnitude_pruning_ratio',
        type=float,
        default=0.0,
        help='Magnitude pruning ratio applied to the student model. 0 means no pruning.'
    )
    parser.add_argument(
        '--weight_quantization_bits',
        type=int,
        default=None,
        help='Weight quantization bits applied to the student model. None means no quantization.'
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    # ---- 视觉模型 (ViT) 专用参数 ----
    parser.add_argument("--train_dir", type=str, default=None,
                        help="A folder containing the training data.")
    parser.add_argument("--validation_dir", type=str, default=None,
                        help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
    )

    # ---- Offsite-Tuning 特定参数 ----
    parser.add_argument(
        '--freeze_bottom',
        action='store_true',
        help='Whether to freeze the bottom adapter layers (do not train them).'
    )
    parser.add_argument(
        '--no_teacher',
        action='store_true',
        help='Disable teacher model (no knowledge distillation). Used for baseline comparison.'
    )
    parser.add_argument(
        '--classifier_lr_multiplier',
        type=float,
        default=1.0,
        help='Learning rate multiplier for the classification head in vision models.'
    )
    parser.add_argument(
        '--select_by_kd',
        action='store_true',
        help='Select best checkpoint by KD loss instead of task accuracy.'
    )
    parser.add_argument(
        '--use_pt_imagefolder',
        action='store_true',
        help='Use PyTorch ImageFolder for loading image data instead of HuggingFace datasets.'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=12,
        help='Number of data loading workers.'
    )
    parser.add_argument(
        '--train_lm_head',
        action='store_true',
        help='Whether to also train the language model head during adapter training.'
    )
    parser.add_argument(
        '--save_module',
        type=str,
        default='student',
        choices=['student', 'adapter', 'all'],
        help='Which module(s) to save: student, adapter, or all.'
    )
    parser.add_argument(
        '--load_adapter',
        type=str,
        default=None,
        help='Path to the pretrained adapter checkpoint.'
    )

    # ---- 评估参数 ----
    parser.add_argument(
        '--tasks',
        type=str,
        default='piqa',
        help='Evaluation tasks (comma-separated).',
    )

    # ---- 参数高效微调 (PEFT) 参数 ----
    parser.add_argument(
        '--use_adapter',
        action='store_true',
        help='Enable Adapter-based parameter-efficient fine-tuning.'
    )
    parser.add_argument(
        '--use_lora',
        action='store_true',
        help='Enable LoRA (Low-Rank Adaptation) fine-tuning.'
    )
    parser.add_argument(
        '--use_bitfit',
        action='store_true',
        help='Enable BitFit (bias-only) fine-tuning.'
    )
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=4,
        help='Rank of the LoRA low-rank matrices.',
    )
    parser.add_argument(
        '--lora_alpha',
        type=float,
        default=32,
        help='Alpha scaling factor for LoRA.',
    )
    parser.add_argument(
        '--adapter_size',
        type=int,
        default=64,
        help='Bottleneck size of the Adapter module.',
    )

    # FIXME: 此参数定义不完整，缺少参数名称、类型和帮助文本。可能是未完成的代码或合并冲突残留。
    parser.add_argument
    args = parser.parse_args()

    return args


# ===========================================================================
# 5. 模型层操作 (Get/Set Layers)
# ===========================================================================

def get_layers(model):
    """获取模型的 transformer 层列表。

    根据模型架构类型返回对应的层模块列表。
    支持 OPT、GPT-2、BLOOM（语言模型）以及 ViT、CLIP ViT、EVA ViT（视觉模型）。

    Args:
        model: HuggingFace 预训练模型实例

    Returns:
        nn.ModuleList 或等效列表: 模型的 transformer 层

    Raises:
        NotImplementedError: 不支持的模型架构
    """
    if isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, GPT2LMHeadModel):
        layers = model.transformer.h
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif isinstance(model, ViTForImageClassification):
        layers = model.vit.encoder.layer
    elif isinstance(model, CLIPViTForImageClassification):
        layers = model.vit.encoder.layers
    elif isinstance(model, EVAViTForImageClassification):
        layers = model.blocks
    else:
        raise NotImplementedError
    return layers


def set_layers(model, layers):
    """设置（替换）模型的 transformer 层列表。

    与 get_layers 配对使用，用于在教师模式和学生模式之间切换。

    Args:
        model: HuggingFace 预训练模型实例
        layers: 新的 transformer 层列表

    Raises:
        NotImplementedError: 不支持的模型架构
    """
    if isinstance(model, OPTForCausalLM):
        model.model.decoder.layers = layers
    elif isinstance(model, GPT2LMHeadModel):
        model.transformer.h = layers
    elif isinstance(model, BloomForCausalLM):
        model.transformer.h = layers
    elif isinstance(model, ViTForImageClassification):
        model.vit.encoder.layer = layers
    elif isinstance(model, CLIPViTForImageClassification):
        model.vit.encoder.layers = layers
    elif isinstance(model, EVAViTForImageClassification):
        model.blocks = layers
    else:
        raise NotImplementedError


# ===========================================================================
# 6. 教师-学生模型构建 (核心函数)
# ===========================================================================

def setup_teacher_student(model, args, accelerator):
    """构建 Offsite-Tuning 的教师-学生架构 —— 这是整个框架的核心函数。

    该函数执行以下步骤：
    1. 冻结所有模型参数
    2. 从完整模型中提取层列表
    3. 构建学生模型（emulator）：从中间层选取或加载预训练的学生
    4. 可选地对学生模型进行剪枝和/或量化压缩
    5. 根据 train_module 策略设置哪些参数需要训练
    6. 分离教师模型（中间层副本）、学生模型（压缩后的中间层）、适配器（首尾层）
    7. 在学生模型的首尾添加 prologue/epilogue 钩子，用于 KD loss 计算
    8. 设置 model.trainable_module 指向需要训练的参数

    train_module 策略说明：
      - 'student':  只训练学生（emulator）层，适配器冻结 —— 用于模拟器蒸馏阶段
      - 'adapter':  只训练适配器（首尾层），学生冻结 —— 用于下游任务微调阶段
      - 'all':      同时训练学生和适配器 —— 用于端到端训练

    Args:
        model: 完整预训练模型
        args: 命令行参数（包含 student_l_pad, student_r_pad, num_student_layers 等）
        accelerator: HuggingFace Accelerator 实例

    Returns:
        修改后的 model（原地修改），新增属性：
          - model.student: 学生模型层
          - model.teacher: 教师模型层（fp16，不可训练）
          - model.adapter: 适配器层（首尾层）
          - model.student_l: 学生第一层（含 prologue 钩子）
          - model.student_r: 学生最后一层（含 epilogue 钩子）
          - model.trainable_module: 可训练模块列表
    """
    # 第一步：冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    layers = get_layers(model)

    # 计算学生层区间：l 为左侧 adapter 层数，r 为右侧 adapter 层起始索引
    l, r = args.student_l_pad, len(layers) - args.student_r_pad

    # 构建学生模型：从检查点加载或从当前模型提取
    if args.load_student:
        # 从预训练检查点加载学生模型
        student_state_dict = torch.load(os.path.join(
            args.load_student, 'student.pt'), map_location='cpu')
        student_layers_len = len(
            set([k.split('.')[0] for k in student_state_dict.keys()]))
        logger.info(
            f"Loading student module from {args.load_student} with {student_layers_len} layers.")
        student = deepcopy(layers[:student_layers_len])
        student.load_state_dict(student_state_dict)
    else:
        # 从当前模型的中间层提取学生模型
        student = deepcopy(layers[l:r])

    # 从提取的层中按策略选取指定数量的层
    if args.student_layer_selection_strategy == 'uniform':
        student = uniform_choose_layers(student, args.num_student_layers)
    else:
        raise NotImplementedError

    student = student.to(accelerator.device)

    # ---- 可选的模型压缩 ----
    if args.magnitude_pruning_ratio > 0:
        logger.info(
            f"Pruning student module with magnitude ratio {args.magnitude_pruning_ratio}")
        magnitude_prune(student, args.magnitude_pruning_ratio)

    if args.weight_quantization_bits is not None:
        logger.info(
            f"Quantizing student module with {args.weight_quantization_bits} bits")
        quantize(student, args.weight_quantization_bits)

    # ---- 根据训练策略设置参数的可训练性 ----
    if args.train_module == 'student':
        # 策略1: 只训练学生层 —— 用于模拟器蒸馏
        for param in student.parameters():
            param.data = param.data.float()
            param.requires_grad = True
    elif args.train_module == 'adapter':
        # 策略2: 只训练适配器层（首尾层）—— 用于下游任务微调
        for param in student.parameters():
            param.requires_grad = False
        if not args.freeze_bottom:
            # 训练左侧（底层）adapter 层
            for param in layers[:l].parameters():
                param.data = param.data.float()
                param.requires_grad = True
        # 训练右侧（顶层）adapter 层
        for param in layers[r:].parameters():
            param.data = param.data.float()
            param.requires_grad = True
    elif args.train_module == 'all':
        # 策略3: 同时训练学生和适配器 —— 端到端训练
        for param in student.parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in layers[:l].parameters():
            param.data = param.data.float()
            param.requires_grad = True
        for param in layers[r:].parameters():
            param.data = param.data.float()
            param.requires_grad = True
    else:
        raise NotImplementedError

    # ---- 组装模型组件 ----
    # student: 压缩后的中间层（模拟器/emulator）
    model.student = student
    # teacher: 原始中间层的 fp16 副本（教师模型，不可训练，仅用于 KD）
    model.teacher = layers[l:r].half()
    # adapter: 首尾层（适配器），帮助学生适配输入/输出分布
    model.adapter = layers[:l] + layers[r:]

    for param in model.teacher.parameters():
        param.requires_grad = False

    # 在学生第一层添加 prologue（捕获教师输入）
    add_prologue(model.student[0], None)
    # 在学生最后一层添加 epilogue（缓存学生输出）
    add_epilogue(model.student[-1], None)
    model.student_l = model.student[0]
    model.student_r = model.student[-1]

    num_student_layers = len(model.student)
    logger.info(f"Number of student layers: {num_student_layers}")

    # 设置可训练模块引用
    if args.train_module == 'student':
        model.trainable_module = model.student
    elif args.train_module == 'adapter':
        model.trainable_module = model.adapter
    elif args.train_module == 'all':
        model.trainable_module = model.student + model.adapter
    else:
        raise NotImplementedError

    gc.collect()
    torch.cuda.empty_cache()
    return model


# ===========================================================================
# 7. 教师/学生模式切换
# ===========================================================================

def to_teacher(model, args):
    """切换到教师模式：将教师层（原始完整精度的中间层）放回完整模型。

    用于评估时获取 "plug-in performance"——即完整大模型在微调后的适配器下的性能。
    这是 offsite-tuning 的核心评估指标：教师模式下的性能应该优于学生模式，
    差距 (accuracy gap) 反映了模型压缩带来的信息损失。

    Args:
        model: 包含 model.teacher 属性的模型
        args: 命令行参数（包含 student_l_pad, student_r_pad）
    """
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        model.model.decoder.layers = model.model.decoder.layers[
            :l] + model.teacher + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.teacher + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.teacher + model.transformer.h[r:]
    elif isinstance(model, ViTForImageClassification):
        r = len(model.vit.encoder.layer) - args.student_r_pad
        model.vit.encoder.layer = model.vit.encoder.layer[:l] + \
            model.teacher + model.vit.encoder.layer[r:]
    elif isinstance(model, CLIPViTForImageClassification):
        r = len(model.vit.encoder.layers) - args.student_r_pad
        model.vit.encoder.layers = model.vit.encoder.layers[:l] + \
            model.teacher + model.vit.encoder.layers[r:]
    elif isinstance(model, EVAViTForImageClassification):
        r = len(model.blocks) - args.student_r_pad
        model.blocks = model.blocks[:l] + \
            model.teacher + model.blocks[r:]
    else:
        raise NotImplementedError


def to_student(model, args):
    """切换到学生模式：将学生层（压缩后的模拟器）放回完整模型。

    用于训练和评估学生模型的性能。在训练过程中模型默认处于学生模式，
    通过此函数可以确保模型使用压缩后的模拟器进行计算。

    Args:
        model: 包含 model.student 属性的模型
        args: 命令行参数（包含 student_l_pad, student_r_pad）
    """
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        model.model.decoder.layers = model.model.decoder.layers[
            :l] + model.student + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.student + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.student + model.transformer.h[r:]
    elif isinstance(model, ViTForImageClassification):
        r = len(model.vit.encoder.layer) - args.student_r_pad
        model.vit.encoder.layer = model.vit.encoder.layer[:l] + \
            model.student + model.vit.encoder.layer[r:]
    elif isinstance(model, CLIPViTForImageClassification):
        r = len(model.vit.encoder.layers) - args.student_r_pad
        model.vit.encoder.layers = model.vit.encoder.layers[:l] + \
            model.student + model.vit.encoder.layers[r:]
    elif isinstance(model, EVAViTForImageClassification):
        r = len(model.blocks) - args.student_r_pad
        model.blocks = model.blocks[:l] + \
            model.student + model.blocks[r:]
    else:
        raise NotImplementedError


# ===========================================================================
# 8. 知识蒸馏损失计算
# ===========================================================================

def get_kd_loss(model):
    """计算知识蒸馏损失 (Knowledge Distillation Loss)。

    这是 offsite-tuning 的核心训练信号。其工作原理如下：

    1. 利用之前 add_prologue 缓存的教师输入 (model.student_l.input_args/kwargs)，
       以及 add_epilogue 缓存的学生输出 (model.student_r.cached_output)
    2. 将教师输入通过教师模型（fp16 精度）前向传播，获得教师输出
    3. 将教师输出与学生输出进行比较，计算归一化 MSE 损失

    归一化方式：
      std = sqrt(mean(teacher_output^2))  — 教师输出的 RMS 值
      kd_loss = mean((teacher_output - student_output)^2 / std^2)

    这种归一化保证了不同尺度的特征图之间 KD loss 的可比性。

    注意：所有计算需要处理 fp32/fp16 混合精度。

    Returns:
        torch.Tensor: 标量 KD loss 值
    """
    # 从学生第一层的钩子中获取缓存的教师输入
    kwargs = model.student_l.input_kwargs
    args = model.student_l.input_args
    # 将教师输入转为 fp16，匹配教师模型的数据类型
    output_teacher = args[0].to(torch.float16)
    args = list(args[1:])
    for i, arg in enumerate(args):
        if torch.is_tensor(arg) and arg.dtype == torch.float32:
            args[i] = arg.to(torch.float16)
    args = tuple(args)

    for k, v in kwargs.items():
        if torch.is_tensor(v) and v.dtype == torch.float32:
            kwargs[k] = v.to(torch.float16)

    # 通过教师模型前向传播（不计算梯度）
    with torch.no_grad():
        model.teacher.eval()
        for teacher_layer in model.teacher:
            output_teacher = teacher_layer(output_teacher, *args, **kwargs)
            if isinstance(output_teacher, tuple):
                output_teacher = output_teacher[0]

    # 获取学生缓存的输出
    output_student = model.student_r.cached_output.float()
    output_teacher = output_teacher.float()

    # 归一化 MSE 损失：用教师输出的 RMS 做标准化
    # 这样不同层的 KD loss 量级一致，稳定训练
    std = output_teacher.pow(2).mean().sqrt()
    kd_loss = (output_teacher - output_student).div(std).pow(2).mean()
    return kd_loss


# ===========================================================================
# 9. 分类头设置
# ===========================================================================

def setup_trainable_classification_head(model):
    """设置可训练的分类头（用于视觉模型）。

    在 adapter 或 all 训练模式下，分类头的参数也需要被训练。
    此函数将分类头参数转为 fp32 并开启梯度。

    Args:
        model: 视觉分类模型

    Raises:
        NotImplementedError: 不支持的模型架构
    """
    if isinstance(model, ViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    elif isinstance(model, CLIPViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    elif isinstance(model, EVAViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    else:
        raise NotImplementedError


# ===========================================================================
# 10. 模型加载与保存
# ===========================================================================

def load_adapter(model, adapter_state_dict, args):
    """加载适配器状态字典到模型的 adapter 层（首尾层）。

    用于评估阶段：将微调后的 adapter 参数载入完整模型。

    Args:
        model: 完整预训练模型
        adapter_state_dict: 适配器的状态字典
        args: 命令行参数（包含 student_l_pad, student_r_pad）

    Returns:
        修改后的 model
    """
    l = args.student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        # adapter 层 = 左侧底层 + 右侧顶层
        adapter_layers = model.model.decoder.layers[:l] + model.model.decoder.layers[r:]
        adapter_layers.load_state_dict(adapter_state_dict)
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        adapter_layers = model.transformer.h[:l] + model.transformer.h[r:]
        adapter_layers.load_state_dict(adapter_state_dict)
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        adapter_layers = model.transformer.h[:l] + model.transformer.h[r:]
        adapter_layers.load_state_dict(adapter_state_dict)
    else:
        raise NotImplementedError
    return model


def load_student(model, student_state_dict, args):
    """加载学生状态字典到模型的 student 层，并重新组装模型层列表。

    用于评估阶段：将训练好的学生（模拟器）参数载入完整模型。

    Args:
        model: 完整预训练模型
        student_state_dict: 学生模型的状态字典
        args: 命令行参数（包含 student_l_pad, student_r_pad）

    Returns:
        修改后的 model
    """
    l = args.student_l_pad

    # 从状态字典推断学生层数
    student_layers_len = len(
        set([k.split('.')[0] for k in student_state_dict.keys()]))
    logger.info(f"Loading student module from with {student_layers_len} layers.")
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        student_layers = model.model.decoder.layers[l:l+student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        # 重新组装：adapter_layers + student_layers + adapter_layers
        model.model.decoder.layers = model.model.decoder.layers[:l] + \
            student_layers + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        student_layers = model.transformer.h[l:l+student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.transformer.h = model.transformer.h[:l] + \
            student_layers + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        student_layers = model.transformer.h[l:l+student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.transformer.h = model.transformer.h[:l] + \
            student_layers + model.transformer.h[r:]
    else:
        raise NotImplementedError
    return model


def save_state_dict(state_dict, output_dir, filename):
    """保存状态字典到文件（以 fp16 精度，减少存储空间）。

    Args:
        state_dict: PyTorch 状态字典
        output_dir: 输出目录
        filename: 文件名（如 "student.pt" 或 "adapter.pt"）
    """
    for k in state_dict:
        state_dict[k] = state_dict[k].to(torch.float16).cpu()
    torch.save(state_dict, os.path.join(output_dir, filename))
