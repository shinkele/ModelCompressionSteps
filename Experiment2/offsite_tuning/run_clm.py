#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation

================================================================================
Offsite-Tuning 因果语言模型训练脚本 (Causal Language Modeling)
================================================================================

本脚本是 Offsite-Tuning (arXiv:2302.04870) 框架的核心训练入口之一，用于对因果语言模型
（Causal Language Model, CLM）进行离场微调。

支持的模型架构：
  - OPT (Meta, OPTForCausalLM / OPTModel)
  - GPT-2 (OpenAI, GPT2LMHeadModel / GPT2Model)
  - BLOOM (BigScience, BloomForCausalLM / BloomModel)

训练流程总览 (9个阶段)：
  1. 初始化 (Init)                     — 解析参数、创建 Accelerator、配置日志
  2. 模型与Tokenizer加载                — 加载预训练因果语言模型及分词器
  3. 数据集加载与预处理                  — 支持 text2text 格式与标准 LM 格式两种分支
  4. 教师-学生模型构建                   — 调用 setup_teacher_student() 构建师生结构
  5. 参数高效微调配置 (PEFT)             — 可选 LoRA / Adapter / BitFit
  6. 优化器与学习率调度                   — AdamW + weight decay 分组 + Scheduler
  7. 训练循环                            — LM损失 + 知识蒸馏损失 (KD Loss) 联合优化
  8. 评估与检查点保存                     — 「插接式困惑度」教师/学生互换评估 + 最优模型保存
  9. 结束收尾                            — accelerator.end_training()

核心设计思想：
  - 教师模型 (Teacher): 完整的预训练大模型，参数冻结，提供隐层表示作为蒸馏目标
  - 学生模型/模拟器 (Student/Emulator): 仅包含部分层的轻量模块，实际参与训练
  - 适配器 (Adapter): 连接教师与学生隐层空间的桥接模块
  - to_teacher() / to_student() 互换机制:
    评估时通过 to_teacher() 将学生权重"插接"回教师模型，计算"插接式困惑度"
    (plug-in perplexity)，衡量学生在完整教师上下文中的表现；之后 to_student()
    切回轻量模式继续训练。这一机制无需完整教师参与梯度计算，极大节省显存。
  - 训练损失 = lm_weight * LM损失 + kd_weight * KD蒸馏损失
    其中 KD 损失衡量学生隐层与教师隐层的差异。

典型用法：
  python run_clm.py \
    --model_name_or_path facebook/opt-350m \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --num_train_epochs 10 \
    --block_size 512 \
    --per_device_train_batch_size 4 \
    --train_module student \
    --student_l_pad 2 --student_r_pad 2 \
    --kd_weight 0.5 --lm_weight 0.5 \
    --output_dir ./output
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'


import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import sys
import datasets
import torch
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from copy import deepcopy

import transformers
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTModel
from transformers.models.bloom.modeling_bloom import BloomForCausalLM, BloomModel

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    DataCollatorWithPadding,
    DataCollatorForTokenClassification,
    get_scheduler,
)
from datasets import load_from_disk, DatasetDict
from offsite_tuning.tasks import task_dict
from offsite_tuning.data import get_raw_datasets, get_tokenized_datasets, get_lm_datasets, process_text2text_datasets
from offsite_tuning.utils import (
    MLP,
    add_epilogue,
    add_prologue,
    uniform_choose_layers,
    magnitude_prune,
    quantize,
    parse_args,
    setup_teacher_student,
    get_kd_loss,
    save_state_dict,
    to_student,
    to_teacher
)

from offsite_tuning.param_efficient import (
    use_lora,
    use_bitfit,
    use_adapter
)
import gc

logger = get_logger(__name__)


def main():
    """
    Offsite-Tuning 因果语言模型训练主函数。

    完整训练流程：
      1. 初始化 Accelerator 和日志系统
      2. 加载模型配置、分词器 (Tokenizer) 和预训练因果语言模型
      3. 根据数据集类型 (text2text / 标准LM) 加载并预处理数据
      4. 构建教师-学生模型结构 (setup_teacher_student)
      5. 配置可选的参数高效微调方法 (LoRA / Adapter / BitFit)
      6. 设置优化器 (AdamW, 含 weight decay 分组) 和学习率调度器
      7. 执行训练循环：每个 step 计算 LM Loss + KD Loss 的加权和
      8. 定期评估：通过 to_teacher() / to_student() 互换计算插接困惑度
      9. 保存最优学生权重和/或适配器权重

    关键参数由 parse_args() 统一解析，涵盖模型选择、数据集、训练超参、
    师生结构配置 (student_l_pad / student_r_pad) 等。
    """
    args = parse_args()

    # =========================================================================
    # 1. 初始化 (Init)
    # =========================================================================
    # 初始化 Accelerator: 自动处理设备分配、混合精度、分布式训练。
    # 如果启用了实验追踪 (如 TensorBoard)，也在此处初始化。
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    accelerator_log_kwargs["log_with"] = args.report_to
    # accelerator_log_kwargs["logging_dir"] = args.output_dir
    accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # 创建输出目录 (仅主进程)
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # 配置日志: 同时输出到标准输出和 output_dir/log.txt 文件
    # Make one log on every process with the configuration for debugging.
    # also log to a file in output_dir
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
        ] if accelerator.is_main_process else []
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # 设置随机种子以确保可复现性
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # =========================================================================
    # 2. 模型与Tokenizer加载 (Model & Tokenizer Loading)
    # =========================================================================
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # 加载模型配置 (Config)
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    # 加载分词器 (Tokenizer)
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # 加载预训练因果语言模型 (Causal LM)，使用 float16 以节省显存
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            torch_dtype=torch.float16
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # 当分词器词表大于模型嵌入矩阵时，调整嵌入层大小
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # =========================================================================
    # 3. 数据集加载与预处理 (Dataset Loading & Preprocessing)
    # =========================================================================
    # 数据加载分为两条分支：
    #   (a) Text2Text 分支: 适用于 E2E NLG 等 seq2seq 风格任务，
    #       通过 process_text2text_datasets() 将输入-输出对拼接为因果LM格式
    #   (b) 标准LM分支:   适用于 WikiText-2 等标准语言建模数据集，
    #       通过 get_tokenized_datasets() 分词后由 get_lm_datasets() 分块

    if args.dataset_name in task_dict:  # special case for e2e_nlg dataset
        # ----- Text2Text 分支 -----
        # 适用于 E2E NLG 等任务：原始数据为 (input, output) 对，
        # process_text2text_datasets 负责将其转换为因果语言模型的训练格式
        raw_datasets = get_raw_datasets(args)
        lm_datasets = process_text2text_datasets(
            raw_datasets, args, tokenizer, accelerator)
    else:
        # ----- 标准语言模型分支 -----
        # 支持两种数据来源：(1) 预先分词并保存到磁盘的数据集 (2) 原始文本数据集
        if args.train_tokenized_dataset and args.val_tokenized_dataset:
            # 从磁盘加载已分词的缓存数据集，避免重复分词开销
            tokenized_datasets = load_from_disk(args.train_tokenized_dataset)
            # 兼容单个 Dataset（无 split 结构）：包装成 DatasetDict
            if not isinstance(tokenized_datasets, DatasetDict):
                tokenized_datasets = DatasetDict({"train": tokenized_datasets})
            val_dataset = load_from_disk(args.val_tokenized_dataset)
            if 'validation' in val_dataset:
                tokenized_datasets["validation"] = val_dataset['validation']
            else:
                tokenized_datasets["validation"] = val_dataset['train']
        else:
            # 从 Hub 或本地加载原始文本数据集，并在线分词
            raw_datasets = get_raw_datasets(args)

            tokenized_datasets = get_tokenized_datasets(
                raw_datasets, args, accelerator, tokenizer)

        # 将分词后的数据分块并拼接为固定 block_size 的 LM 训练样本
        lm_datasets = get_lm_datasets(
            tokenized_datasets, args, accelerator, tokenizer)

    # 提取训练集和验证集
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # 可选：限制训练/验证样本数量 (用于快速实验或调试)
    if args.train_num_samples is not None:
        # check if we have enough samples for the training set
        if args.train_num_samples > len(train_dataset):
            args.train_num_samples = len(train_dataset)
        train_dataset = train_dataset.select(
            range(args.train_num_samples))

    if args.validation_num_samples is not None:
        # check if we have enough samples for the validation set
        if args.validation_num_samples > len(eval_dataset):
            args.validation_num_samples = len(eval_dataset)
        eval_dataset = eval_dataset.select(
            range(args.validation_num_samples))

    # 构建 DataLoader
    # 对于因果语言模型，使用 default_data_collator 即可 (动态padding到batch内最长)
    collator = default_data_collator
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collator, batch_size=args.per_device_eval_batch_size
    )

    # =========================================================================
    # 4. 教师-学生模型构建 (Teacher-Student Model Setup)
    # =========================================================================
    # 核心步骤：调用 setup_teacher_student() 将完整预训练模型拆分为：
    #   - Teacher (教师模型): 冻结的原始 Transformer 层，提供蒸馏目标
    #   - Student (学生模型/模拟器): 仅包含部分层 (由 student_l_pad / student_r_pad 决定)，
    #     实际参与训练的可学习模块
    #   - Adapter (适配器): 连接教师与学生隐层的 MLP 桥接模块
    # 详见 offsite_tuning/utils.py 中的 setup_teacher_student() 实现和原论文 Section 3。
    model = setup_teacher_student(model, args, accelerator)

    # --no_teacher 标志处理:
    # 当设置 --no_teacher 时，丢弃教师模型，仅保留学生模块。
    # 此时训练退化为纯粹的学生微调（无知识蒸馏），用于消融实验或极低资源场景。
    # 丢弃教师后显式释放 GPU 缓存。
    if args.no_teacher:
        model.teacher = None
        to_student(model, args)
        gc.collect()
        torch.cuda.empty_cache()

    # 当训练适配器或全部模块时，同时将 lm_head 设为可训练 (float32精度)
    if args.train_module in ['adapter', 'all'] and args.train_lm_head:
        for param in model.lm_head.parameters():
            param.requires_grad = True
            param.data = param.data.float()

    # =========================================================================
    # 5. 参数高效微调配置 (Parameter-Efficient Fine-Tuning, PEFT)
    # =========================================================================
    # 可选的轻量级微调方法，在训练模块 (trainable_module) 上叠加：
    #   - LoRA: 低秩适配 (Low-Rank Adaptation)，在权重矩阵上添加低秩分解的可训练旁路
    #   - Adapter: 在 Transformer 层间插入小型瓶颈适配器模块
    #   - BitFit: 仅微调偏置项 (bias terms)，极大减少可训练参数量

    if args.use_lora:
        use_lora(model.trainable_module, args.lora_rank, args.lora_alpha)

    if args.use_adapter:
        use_adapter(model.trainable_module, args.adapter_size)

    if args.use_bitfit:
        use_bitfit(model.trainable_module)

    # 从检查点恢复训练状态
    if args.load_student and not args.restart_training:
        base_results = json.load(
            open(os.path.join(args.load_student, 'all_results.json'), 'r'))
        starting_epoch = base_results['epoch']
        resume_step = base_results['step'] - \
            starting_epoch * len(train_dataloader)
    else:
        starting_epoch = 0
        resume_step = -1

    # 统计并记录可训练参数数量
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    logger.info(f"Number of trainable parameters: {trainable_params}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(
                f"Trainable parameter: {name} with shape {param.shape} and dtype {param.dtype}")

    # =========================================================================
    # 6. 优化器与学习率调度 (Optimizer & Learning Rate Scheduler)
    # =========================================================================
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    # 将参数分为两组：
    #   (a) 应用 weight decay 的: 权重矩阵 (不包括 bias 和 LayerNorm 权重)
    #   (b) 不应用 weight decay 的: bias 项和 LayerNorm 权重
    # 这是 Transformer 训练的标准做法，避免对归一化层施加正则化。
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    # 计算每个 epoch 的更新步数 (考虑梯度累积)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # 学习率调度器: 支持 linear / cosine / constant 等策略，含 warmup
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # 通过 Accelerator 包装所有组件以实现分布式训练
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # Accelerator 可能在分布式环境下改变 DataLoader 大小，因此重新计算训练步数
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # 初始化实验追踪器 (如 TensorBoard / WandB)
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("offsite_tuning", experiment_config)

    # =========================================================================
    # 7. 训练循环 (Training Loop)
    # =========================================================================
    # 训练循环的核心逻辑：
    #   - 每个 step 计算两个损失:
    #       (1) LM Loss:  因果语言模型的标准交叉熵损失
    #       (2) KD Loss:  学生隐层与教师隐层之间的知识蒸馏损失
    #   - 总损失 = lm_weight * LM Loss + kd_weight * KD Loss
    #     (当 kd_weight=0 时退化为纯 LM 训练)
    #   - 支持梯度累积 (gradient_accumulation_steps)
    #   - 支持从检查点恢复训练 (skip steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def eval_epoch():
        """
        评估函数：在整个验证集上计算学生模型的困惑度 (Perplexity)。

        遍历 eval_dataloader，收集每个 batch 的 LM loss，
        使用 accelerator.gather_for_metrics 聚合多卡结果，
        过滤 NaN 后计算平均损失和困惑度 (PPL = exp(loss))。

        Returns:
            eval_loss (float): 验证集上的平均交叉熵损失
            perplexity (float): 验证集上的困惑度 (PPL = e^{eval_loss})
        """
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            # gather_for_metrics: 在分布式环境下聚合所有进程的损失值
            losses.append(accelerator.gather_for_metrics(
                loss.repeat(args.per_device_eval_batch_size)).cpu())
        losses = torch.cat(losses).flatten()
        # filter out nan
        # 过滤 NaN 值，避免异常样本污染评估结果
        losses = losses[~torch.isnan(losses)]
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        return eval_loss, perplexity

    # =========================================================================
    # 8. 评估与检查点保存 (Evaluation & Checkpoint Saving)
    # =========================================================================
    # 零样本 (zero-shot) 困惑度评估：
    #   训练开始前分别计算教师和学生在验证集上的零样本困惑度，
    #   作为微调效果的基线参考。
    #   - 教师零样本困惑度: 将模型切换到 to_teacher() 模式后评估
    #     (完整教师 + 随机初始化适配器)
    #   - 学生零样本困惑度: 将模型切换到 to_student() 模式后评估
    #     (仅学生 + 适配器)

    # 教师零样本困惑度评估 (teacher zero-shot perplexity)
    # 如果设置了 --no_teacher，则跳过教师评估
    if not args.no_teacher:
        # to_teacher(model.module, args) 
        to_student(accelerator.unwrap_model(model), args) # accelerator.unwrap_model() 是一个同时兼容单卡/多卡的 API
        _, teacher_zero_shot_perplexity = eval_epoch()
        logger.info(
            f"Teacher zero shot perplexity: {teacher_zero_shot_perplexity}")
    else:
        teacher_zero_shot_perplexity = 0

    # 学生零样本困惑度评估 (student zero-shot perplexity)
    # to_student() 将模型切换回学生模式用于后续训练和评估
    # to_student(model.module, args)
    to_student(accelerator.unwrap_model(model), args) # accelerator.unwrap_model() 是一个同时兼容单卡/多卡的 API
    
    # for name, param in model.named_parameters():
    #     logger.info(
    #         f"Parameter: {name} with shape {param.shape}, dtype {param.dtype}, and requires_grad {param.requires_grad}")

    _, student_zero_shot_perplexity = eval_epoch()
    logger.info(
        f"Student zero shot perplexity: {student_zero_shot_perplexity}")
    best_perplexity = float("inf")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)

    completed_steps = 0

    # update the progress_bar if load from checkpoint
    # 从检查点恢复时更新进度条位置
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_lm_loss, total_kd_loss = 0, 0
        interval_lm_loss, interval_kd_loss = 0, 0
        best_lm_loss, best_kd_loss = float("inf"), float("inf")
        skipped_steps = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            # 从检查点恢复时，跳过已完成的前 resume_step 步
            if args.load_student and epoch == starting_epoch and step <= resume_step:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"Skipping step {step} (already completed)")
                completed_steps += 1
                skipped_steps += 1
                continue

            # 梯度累积上下文: 在 accumulate(model) 内，梯度不会在 backward 后立即清零，
            # 而是累积 gradient_accumulation_steps 次后再执行一次优化器更新
            with accelerator.accumulate(model):
                outputs = model(**batch)
                lm_loss = outputs.loss

                # KD Loss (知识蒸馏损失):
                # 通过 get_kd_loss() 计算学生模块隐层输出与教师模块对应隐层输出之间的差异。
                # 当 --no_teacher 时，kd_loss = 0，训练退化为纯 LM 训练。
                if not args.no_teacher:
                    # kd_loss = get_kd_loss(model.module)
                    kd_loss = get_kd_loss(accelerator.unwrap_model(model)) # accelerator.unwrap_model() 是一个同时兼容单卡/多卡的 API
                else:
                    kd_loss = 0

                # 加权组合 LM Loss 和 KD Loss
                # loss = lm_weight * lm_loss + kd_weight * kd_loss
                # 当 kd_weight = 0 时，仅使用 LM Loss (纯语言建模训练)
                loss = args.lm_weight * lm_loss + args.kd_weight * \
                    kd_loss if args.kd_weight != 0 else lm_loss
                progress_bar.set_description(
                    f"Epoch {epoch} - Step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - LM loss: {lm_loss:.4f} - KD loss: {kd_loss:.4f}")

                total_lm_loss += lm_loss.item()
                interval_lm_loss += lm_loss.item()
                best_lm_loss = min(best_lm_loss, lm_loss.item())

                if not args.no_teacher:
                    total_kd_loss += kd_loss.item()
                    interval_kd_loss += kd_loss.item()
                    best_kd_loss = min(best_kd_loss, kd_loss.item())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # end accumulate gradients

            # Checks if the accelerator has performed an optimization step behind the scenes
            # 仅在梯度同步步骤 (完成一次 optimizer.step) 才更新进度条和步数计数
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            else:
                continue

            # 定期评估 (每 eval_steps 步)
            # 评估流程:
            #   1. to_teacher() 切换，计算"插接困惑度" (plug-in perplexity, plug_ppl):
            #      将学生训练好的权重插回完整的教师模型框架中，衡量学生在完整
            #      教师上下文下的表现。这是 Offsite-Tuning 的核心评估指标。
            #   2. to_student() 切换回来，计算学生自身的困惑度 (student_ppl)。
            #   3. 两者的差值 (ppl_gap = student_ppl - plug_ppl) 反映学生模拟教师的能力。
            if completed_steps % args.eval_steps == 0:
                # --- 教师-学生互换评估 (Teacher-Student Swap Evaluation) ---
                # 插接式困惑度: 切换到教师模式，使用教师的前几层和最后几层 +
                # 学生的中间层 + 适配器来计算困惑度
                if not args.no_teacher:
                    # to_teacher(model.module, args)
                    to_teacher(accelerator.unwrap_model(model), args) # accelerator.unwrap_model() 是一个同时兼容单卡/多卡的 API
                    plug_eval_loss, plug_ppl = eval_epoch()
                else:
                    plug_eval_loss, plug_ppl = 0, 0
                # 切回学生模式
                # to_student(model.module, args)
                to_student(accelerator.unwrap_model(model), args) # accelerator.unwrap_model() 是一个同时兼容单卡/多卡的 API
                eval_loss, perplexity = eval_epoch()

                # 计算该评估区间的平均损失
                lm_loss = interval_lm_loss / args.eval_steps
                kd_loss = interval_kd_loss / args.eval_steps
                interval_lm_loss = 0
                interval_kd_loss = 0

                logger.info(
                    f"epoch {epoch} step {completed_steps}: student_ppl: {perplexity:.4f} plug_ppl: {plug_ppl:.4f} lm_loss: {lm_loss:.4f} kd_loss: {kd_loss:.4f}")

                accelerator.log(
                    {
                        "student_ppl": perplexity,
                        "student_eval_loss": eval_loss,
                        "plug_ppl": plug_ppl,
                        "plug_eval_loss": plug_eval_loss,
                        "ppl_gap": perplexity - plug_ppl,
                        "train_lm_loss": lm_loss,
                        "train_kd_loss": kd_loss,
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )
                is_best = perplexity < best_perplexity
                best_perplexity = min(best_perplexity, perplexity)

                # 保存最优模型权重 (仅主进程)
                # 根据 --save_module 参数决定保存内容:
                #   - "student": 仅保存学生模型权重 (student.pt)
                #   - "adapter": 仅保存适配器权重 (adapter.pt)
                #   - "all":     同时保存两者
                # 仅在当前结果优于历史最佳时才保存 (is_best)。
                if not args.no_save_model and is_best and accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    if args.save_module in ["student", "all"]:
                        state_dict = unwrapped_model.student.state_dict()
                        save_state_dict(
                            state_dict, args.output_dir, "student.pt")
                    if args.save_module in ["adapter", "all"]:
                        state_dict = unwrapped_model.adapter.state_dict()
                        save_state_dict(
                            state_dict, args.output_dir, "adapter.pt")

                    gc.collect()
                    torch.cuda.empty_cache()

                # 保存训练结果摘要 (all_results.json)
                # 记录当前最佳困惑度、插接困惑度、零样本基线等关键指标
                if is_best and accelerator.is_main_process:
                    with open(os.path.join(args.output_dir, "all_results.json"), "w+") as f:
                        json.dump({"best_perplexity": best_perplexity,
                                   "plug_perplexity": plug_ppl,
                                   "teacher_zero_shot_perplexity": teacher_zero_shot_perplexity,
                                   "student_zero_shot_perplexity": student_zero_shot_perplexity,
                                   "train_lm_loss": lm_loss,
                                   "train_kd_loss": kd_loss,
                                   "epoch": epoch,
                                   "step": completed_steps,
                                   "trainable_params": trainable_params}, f)

    # 结束 Accelerator 训练 (清理分布式资源)
    accelerator.end_training()


if __name__ == "__main__":
    main()
