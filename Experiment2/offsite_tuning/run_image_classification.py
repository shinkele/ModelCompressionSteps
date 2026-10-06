# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning any 🤗 Transformers model for image classification leveraging 🤗 Accelerate.

================================================================================
Offsite-Tuning 图像分类训练脚本 (Image Classification)
================================================================================

本脚本是 Offsite-Tuning (arXiv:2302.04870) 框架中用于视觉模型的核心训练入口，
对图像分类任务进行离场微调。

支持的模型架构：
  - CLIP ViT  (OpenAI, CLIPVisionModel → CLIPViTForImageClassification)
  - EVA ViT   (EVA, EVAViTForImageClassification, 从本地权重文件加载)
  - 标准 ViT  (HuggingFace, AutoModelForImageClassification)

训练流程总览：
  1. 初始化 Accelerator 与日志系统
  2. 图像预处理与数据增强 (train/val 使用不同的 Transforms)
  3. 数据集加载 (支持 HuggingFace Hub 与 PyTorch ImageFolder 两种方式)
  4. 标签映射 (label ↔ id) 与模型配置
  5. 模型加载 (根据 CLIP / EVA / 标准 ViT 分支选择)
  6. 教师-学生模型构建 (setup_teacher_student) 与分类头设置
  7. DataLoader 构建与 collate_fn 定义
  8. 优化器 (AdamW/SGD, 分类器独立学习率) 与学习率调度器
  9. 训练循环 (Task Loss + KD Loss 联合优化)
 10. 周期性评估 (教师/学生互换 + accuracy 计算) 与检查点保存

核心设计思想与CLM脚本的差异：
  - 图像分类使用准确率 (accuracy) 而非困惑度 (perplexity) 作为评估指标
  - 支持 select_by_kd 模式: 基于 KD Loss 而非准确率选择最优模型
  - 分类头 (classifier) 使用独立的 classifier_lr_multiplier 学习率倍率
  - 评估器闭包 (evaluator closure): 使用函数属性维护评估状态 (interval_task_loss,
    interval_kd_loss, eval_steps, best_acc, best_kd_loss)
  - 教师/学生互换评估: to_teacher() 计算 plug_acc → to_student() 计算 eval_acc
  - ImageFolder 分支与 HuggingFace Dataset 分支使用不同的 collate_fn

典型用法：
  python run_image_classification.py \
    --model_name_or_path openai/clip-vit-base-patch32 \
    --dataset_name cifar100 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --train_module student \
    --student_l_pad 2 --student_r_pad 2 \
    --kd_weight 0.5 --lm_weight 0.5 \
    --output_dir ./output
"""
import argparse
import json
import logging
import math
import os
import sys

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    get_scheduler,
    CLIPVisionConfig,
    CLIPVisionModel,
)


from offsite_tuning.utils import (
    parse_args,
    setup_teacher_student,
    get_kd_loss,
    to_teacher,
    to_student,
    setup_trainable_classification_head
)

from offsite_tuning.models.clip_vit import CLIPViTForImageClassification
from offsite_tuning.models.eva_vit import EVAViTForImageClassification
import gc

logger = get_logger(__name__)


def main():
    """
    Offsite-Tuning 图像分类训练主函数。

    完整训练流程：
      1. 初始化 Accelerator 和日志系统
      2. 配置图像预处理 Transforms (训练/验证使用不同的增强策略)
      3. 加载数据集 (Hub 或本地 ImageFolder)
      4. 根据模型类型 (CLIP / EVA / 标准 ViT) 分支加载模型
      5. 构建教师-学生模型结构并设置可训练分类头
      6. 构建 DataLoader 与 collate_fn
      7. 设置优化器 (分类器使用独立的学习率倍率) 和调度器
      8. 执行训练循环：Task Loss + KD Loss 加权优化
      9. 通过 evaluator 闭包定期评估并保存最优模型

    支持 --select_by_kd 模式：基于知识蒸馏损失而非准确率选择最优检查点，
    适用于无标签验证集的场景。
    """
    args = parse_args()

    # =========================================================================
    # 1. 初始化 Accelerator 与日志系统
    # =========================================================================
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    accelerator_log_kwargs["log_with"] = args.report_to
    accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Handle the repository creation
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    logger.info(accelerator.state)
    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    # =========================================================================
    # 2. 图像预处理与数据增强 (Data Transforms & Preprocessing)
    # =========================================================================
    # 从预训练模型的特征提取器 (Feature Extractor) 获取图像归一化参数和尺寸。
    # 训练和验证使用不同的 Transforms 策略：
    #   - 训练: RandomResizedCrop + RandomHorizontalFlip (数据增强)
    #   - 验证: Resize + CenterCrop (确定性预处理，无随机增强)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name_or_path)

    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.
    # 确定图像尺寸: 支持 "shortest_edge" 和 (height, width) 两种格式
    if "shortest_edge" in feature_extractor.size:
        size = feature_extractor.size["shortest_edge"]
    else:
        size = (feature_extractor.size["height"],
                feature_extractor.size["width"])

    # 归一化: 使用预训练模型的 mean 和 std (如 CLIP 的 ImageNet 统计量)
    normalize = Normalize(mean=feature_extractor.image_mean,
                          std=feature_extractor.image_std)

    # ----- 训练数据增强 (Training Transforms) -----
    # RandomResizedCrop: 随机裁剪 + 缩放，提高空间不变性
    # RandomHorizontalFlip: 随机水平翻转，提高对称不变性
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    # ----- 验证数据预处理 (Validation Transforms) -----
    # Resize + CenterCrop: 确定性缩放和中心裁剪，确保评估一致性
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply _train_transforms across a batch.

        训练批次预处理函数：对批次中的每张图像应用训练 Transforms
        (RandomResizedCrop + RandomHorizontalFlip + Normalize)，
        并转换为 RGB 格式以兼容灰度图/透明图输入。

        Args:
            example_batch: 包含 "image" 字段的 HuggingFace Dataset 批次

        Returns:
            添加了 "pixel_values" 字段的批次字典
        """
        example_batch["pixel_values"] = [train_transforms(
            image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply _val_transforms across a batch.

        验证批次预处理函数：对批次中的每张图像应用验证 Transforms
        (Resize + CenterCrop + Normalize)，确保评估时的确定性。

        Args:
            example_batch: 包含 "image" 字段的 HuggingFace Dataset 批次

        Returns:
            添加了 "pixel_values" 字段的批次字典
        """
        example_batch["pixel_values"] = [val_transforms(
            image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    # =========================================================================
    # 3. 数据集加载 (Dataset Loading)
    # =========================================================================
    # 支持三种数据集加载方式：
    #   (a) HuggingFace Hub 数据集 (--dataset_name)
    #   (b) PyTorch ImageFolder 本地数据集 (--use_pt_imagefolder)
    #   (c) HuggingFace imagefolder 格式的本地数据集 (--train_dir / --validation_dir)
    #
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # ----- 分支 A: 从 HuggingFace Hub 下载数据集 -----
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(args.dataset_name, task="image-classification")
    elif args.use_pt_imagefolder:
        # ----- 分支 B: PyTorch ImageFolder (本地数据集) -----
        # 使用 torchvision.datasets.ImageFolder 直接加载，
        # Transforms 在构造 Dataset 时传入 (而非 with_transform)
        # Load a local dataset using a PyTorch Dataset.
        import torchvision.datasets as pt_datasets
        logging.info("Using PyTorch ImageFolder")
        dataset = {
            "train": pt_datasets.ImageFolder(root=args.train_dir, transform=train_transforms),
            "validation": pt_datasets.ImageFolder(root=args.validation_dir, transform=val_transforms),
        }
    else:
        # ----- 分支 C: HuggingFace imagefolder 格式 (本地文件夹) -----
        # 从本地目录加载图像，自动推断标签。
        # data_files 中的 "**" 通配符匹配所有子目录中的图像文件。
        data_files = {}
        if args.train_dir is not None:
            data_files["train"] = os.path.join(args.train_dir, "**")
        if args.validation_dir is not None:
            data_files["validation"] = os.path.join(args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            task="image-classification",
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder.

    # 如果数据集没有验证集划分，从训练集按比例切分
    # If we don't have a validation split, split off a percentage of train as validation.
    args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # =========================================================================
    # 4. 标签映射与模型加载 (Label Mapping & Model Loading)
    # =========================================================================
    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.

    # 构建 label ↔ id 双向映射
    if args.use_pt_imagefolder:
        labels = dataset["train"].classes
    else:
        labels = dataset["train"].features["labels"].names

    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Load pretrained model and feature extractor
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # ----- 模型加载分支 (CLIP / EVA / 标准 ViT) -----
    # 三种模型使用不同的加载逻辑：
    #   CLIP ViT: 使用 CLIPVisionConfig + CLIPVisionModel，然后包装为 CLIPViTForImageClassification
    #   EVA ViT:  从本地 config.json 和 pytorch_model.bin 直接加载
    #   标准 ViT: 使用 AutoModelForImageClassification 从 HuggingFace Hub 加载

    if 'CLIP' in args.model_name_or_path:
        # ----- CLIP ViT 分支 -----
        # CLIP 的视觉编码器使用 CLIPVisionModel，其分类头由
        # CLIPViTForImageClassification 额外添加。
        # 注意：num_labels 和 label 映射在 config 中设置。
        config = CLIPVisionConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
        )
        model = CLIPVisionModel.from_pretrained(
            args.model_name_or_path,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            torch_dtype=torch.float16
        )
        # 将 CLIPVisionModel 包装为带分类头的图像分类模型
        model = CLIPViTForImageClassification(config, model.vision_model)
    elif 'eva' in args.model_name_or_path:
        # ----- EVA ViT 分支 -----
        # EVA 模型通常以本地文件形式提供 (config.json + pytorch_model.bin)。
        # 标签数量在加载后动态注入 config。
        config = json.load(
            open(os.path.join(args.model_name_or_path, 'config.json')))
        config['num_labels'] = len(labels)
        model = EVAViTForImageClassification(**config)
        state_dict = torch.load(os.path.join(
            args.model_name_or_path, 'pytorch_model.bin'))
        model.load_state_dict(state_dict, strict=False)
    else:
        # ----- 标准 ViT 分支 (HuggingFace Hub) -----
        # 适用于所有 HuggingFace 支持的 Vision Transformer 模型
        # (如 google/vit-base-patch16-224, microsoft/beit-base-patch16-224 等)
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
        )
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            torch_dtype=torch.float16
        )

    # =========================================================================
    # 5. 教师-学生模型构建与分类头设置
    # =========================================================================
    #############################################
    # Teacher-Student model
    # 核心步骤：将完整 ViT 拆分为教师模型 (冻结) 和学生模型/模拟器 (可训练)。
    # setup_teacher_student 返回包含 .teacher, .student, .adapter 属性的模型。
    model = setup_teacher_student(model, args, accelerator)

    # Setup trainable classification heads
    # 当训练适配器或全部模块时，设置可训练的分类头。
    # setup_trainable_classification_head 会创建新的分类器层，
    # 并将原始分类器权重复制过去，确保分类头随训练更新。
    if args.train_module in ['adapter', 'all']:
        setup_trainable_classification_head(model)
    #############################################

    # =========================================================================
    # 6. DataLoader 构建 (DataLoader Construction)
    # =========================================================================
    # 两个分支使用不同的数据集类型，需要不同的 collate_fn：
    #   - PyTorch ImageFolder: 每个样本返回 (image_tensor, label) 元组
    #   - HuggingFace Dataset:   每个样本返回包含 "pixel_values" 和 "labels" 的字典

    if args.use_pt_imagefolder:
        # ----- PyTorch ImageFolder 分支 -----
        # ImageFolder 已在构造时传入 transform，无需 with_transform。
        # 每个样本 (image_tensor, label) → 堆叠为 batch
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        def collate_fn(examples):
            # 从元组格式 (tensor, label) 中提取像素值和标签
            pixel_values = torch.stack([example[0] for example in examples])
            labels = torch.tensor([example[1] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}
    else:
        # ----- HuggingFace Dataset 分支 -----
        # 通过 with_transform 将预处理函数挂载到数据集上，
        # 每次访问时动态调用 preprocess_train / preprocess_val。
        # 主进程先执行变换逻辑，避免多进程竞态。
        with accelerator.main_process_first():
            if args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(
                    seed=args.seed).select(range(args.max_train_samples))
            # Set the training transforms
            # 将训练 Transforms 挂载到数据集，每次 getitem 时调用
            train_dataset = dataset["train"].with_transform(preprocess_train)
            if args.max_eval_samples is not None:
                dataset["validation"] = dataset["validation"].shuffle(
                    seed=args.seed).select(range(args.max_eval_samples))
            # Set the validation transforms
            # 将验证 Transforms 挂载到数据集
            eval_dataset = dataset["validation"].with_transform(preprocess_val)

        def collate_fn(examples):
            # 从字典格式中提取像素值和标签
            pixel_values = torch.stack([example["pixel_values"]
                                        for example in examples])
            labels = torch.tensor([example["labels"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

    # DataLoaders creation:
    # 构建训练和验证 DataLoader，num_workers 控制多进程数据加载

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size, num_workers=args.num_workers
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size, num_workers=args.num_workers
    )

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

    # =========================================================================
    # 7. 优化器与学习率调度 (Optimizer & Scheduler)
    # =========================================================================
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    # 参数分组策略 (与 CLM 脚本类似，但增加了分类头的独立学习率处理):
    #   (a) 非分类器权重 + weight decay:      bias/LayerNorm 以外的参数
    #   (b) 非分类器权重 + 无 weight decay:    bias 和 LayerNorm 权重
    #   (c) 分类器权重 + weight decay:         classifier 中的权重矩阵
    #   (d) 分类器权重 + 无 weight decay:      classifier 中的 bias/LayerNorm
    #
    # 关键设计：分类器使用独立的学习率 = learning_rate * classifier_lr_multiplier
    # 这允许分类头以更快的速度收敛，而骨干网络使用较小的学习率保持预训练知识。
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": 0.0,
        },
        {
            # 分类器权重 (含 weight decay)，使用 classifier_lr_multiplier 倍率的学习率
            "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * args.classifier_lr_multiplier
        },
        {
            # 分类器 bias/LayerNorm (无 weight decay)，使用 classifier_lr_multiplier 倍率的学习率
            "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate * args.classifier_lr_multiplier
        },
    ]

    # 支持 SGD 和 AdamW 两种优化器
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters, lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type {args.optimizer}")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("offsite_tuning", experiment_config)

    # Get the metric function
    # 使用 evaluate 库加载准确率 (accuracy) 指标
    metric = evaluate.load("accuracy")

    # =========================================================================
    # 8. 训练与评估 (Training & Evaluation)
    # =========================================================================
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
        评估函数：在整个验证集上计算图像分类准确率 (Accuracy)。

        遍历 eval_dataloader，对每个 batch 执行前向推理，
        使用 accelerator.gather_for_metrics 聚合多卡预测结果，
        通过 evaluate 库的 accuracy 指标计算最终准确率。

        Returns:
            float: 验证集上的分类准确率 (0.0 ~ 1.0)
        """
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        return eval_metric["accuracy"]

    # 零样本准确率评估 (Zero-shot Accuracy Evaluation)
    # 训练开始前评估教师和学生的零样本分类准确率，作为微调效果基线。
    #
    # select_by_kd 模式:
    #   当设置 --select_by_kd 时，跳过零样本准确率评估 (设为0)，
    #   直接进入学生模式。后续模型选择完全基于 KD Loss 而非准确率，
    #   适用于无法获取验证集标签的场景。

    if args.select_by_kd:
        # select_by_kd 模式: 完全基于知识蒸馏损失选择最优模型
        # 跳过教师/学生零样本准确率评估，直接使用学生模式
        teacher_zero_shot_acc = student_zero_shot_acc = 0
        model = to_student(model, args)
    else:
        # 标准模式: 先评估教师零样本准确率，再评估学生零样本准确率
        # to_teacher() 将学生权重插回完整教师框架进行评估 (插接式准确率)
        model = to_teacher(model, args)
        teacher_zero_shot_acc = eval_epoch()

        # to_student() 切换回学生模式，评估学生的独立准确率
        model = to_student(model, args)
        student_zero_shot_acc = eval_epoch()

    # 统计可训练参数数量
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    logger.info(f"Number of trainable parameters: {trainable_params}")

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(
                f"Trainable parameter: {name} with shape {param.shape} and dtype {param.dtype}")

    logger.info(
        f"Teacher zero shot accuracy: {teacher_zero_shot_acc}")
    logger.info(
        f"Student zero shot accuracy: {student_zero_shot_acc}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # =========================================================================
    # 评估器闭包 (Evaluator Closure)
    # =========================================================================
    # 使用闭包 (closure) 模式实现评估器：
    # evaluator 是一个函数对象，利用 Python 函数属性 (function attributes)
    # 维护跨调用的持久状态，避免使用全局变量或类实例。
    #
    # 状态变量:
    #   evaluator.eval_steps          — 当前评估区间内的步数计数
    #   evaluator.interval_task_loss  — 当前区间的累计任务损失 (Task Loss)
    #   evaluator.interval_kd_loss    — 当前区间的累计知识蒸馏损失 (KD Loss)
    #   evaluator.best_acc            — 历史最佳准确率 (标准模式)
    #   evaluator.best_kd_loss        — 历史最佳 KD Loss (select_by_kd 模式)

    def evaluator(model):
        """
        评估器闭包：在指定步数间隔执行评估并保存最优模型。

        根据 --select_by_kd 参数有两种工作模式:
          (a) select_by_kd 模式: 基于 KD Loss 选择最优模型，不进行教师/学生互换评估
          (b) 标准模式:          基于准确率选择最优模型，执行教师/学生互换评估

        评估流程 (标准模式):
          1. 计算当前 interval 的平均 task_loss 和 kd_loss，并重置计数器
          2. to_teacher() 切换，计算插接准确率 (plug_acc)
          3. to_student() 切换回来，计算学生准确率 (eval_acc)
          4. 记录指标到 accelerator tracker
          5. 如果是最优模型，保存 student.pt 和 all_results.json

        Args:
            model: 当前模型 (经过 accelerator.prepare 包装)
        """
        if evaluator.eval_steps == 0:
            return

        # 计算当前评估区间内的平均损失并重置计数器
        task_loss = evaluator.interval_task_loss / evaluator.eval_steps
        kd_loss = evaluator.interval_kd_loss / evaluator.eval_steps
        evaluator.interval_task_loss = 0
        evaluator.interval_kd_loss = 0
        evaluator.eval_steps = 0

        if args.select_by_kd:
            # select_by_kd 模式: 基于 KD Loss 选择最优模型
            # 跳过教师/学生互换评估以节省计算
            is_best = kd_loss < evaluator.best_kd_loss
            evaluator.best_kd_loss = min(evaluator.best_kd_loss, kd_loss)
            eval_acc = plug_acc = 0
        else:
            # 标准模式: 教师/学生互换评估
            # 1. 切换到教师模式，计算插接准确率 (plug_acc)
            #    衡量学生权重在完整教师框架中的表现
            model = to_teacher(model, args)
            plug_acc = eval_epoch()
            # 2. 切换回学生模式，计算学生独立准确率 (eval_acc)
            model = to_student(model, args)
            eval_acc = eval_epoch()
            # 基于学生准确率判断是否为最优模型
            is_best = eval_acc > evaluator.best_acc
            evaluator.best_acc = max(evaluator.best_acc, eval_acc)

        logger.info(
            f"Epoch {epoch} step {completed_steps}: eval_acc: {eval_acc:.4f} plug_acc: {plug_acc:.4f} task_loss: {task_loss:.4f} kd_loss: {kd_loss:.4f}")

        accelerator.log(
            {
                "eval_acc": eval_acc,
                "plug_acc": plug_acc,
                "acc_gap": plug_acc - eval_acc,
                "train_task_loss": task_loss,
                "train_kd_loss": kd_loss,
                "epoch": epoch,
                "step": completed_steps,
            },
            step=completed_steps,
        )

        # 保存最优模型权重 (仅主进程)
        # 仅保存学生模型 (student.pt)，权重转换为 float16 以减少存储空间
        if not args.no_save_model and is_best and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = unwrapped_model.student.state_dict()
            for k in state_dict:
                state_dict[k] = state_dict[k].to(torch.float16).cpu()
            torch.save(state_dict, os.path.join(
                args.output_dir, "student.pt"))
            gc.collect()
            torch.cuda.empty_cache()

        # 保存训练结果摘要 (all_results.json)
        if is_best and accelerator.is_main_process:
            with open(os.path.join(args.output_dir, "all_results.json"), "w+") as f:
                json.dump({"best_acc": eval_acc,
                           "plug_acc": plug_acc,
                           "teacher_zero_shot_acc": teacher_zero_shot_acc,
                           "student_zero_shot_acc": student_zero_shot_acc,
                           "train_task_loss": task_loss,
                           "train_kd_loss": kd_loss,
                           "epoch": epoch,
                           "step": completed_steps,
                           "trainable_params": trainable_params}, f)

    # 初始化评估器状态变量 (函数属性)
    evaluator.best_acc = student_zero_shot_acc
    evaluator.best_kd_loss = float("inf")
    evaluator.eval_steps = 0
    evaluator.interval_task_loss = 0
    evaluator.interval_kd_loss = 0

    # =========================================================================
    # 训练循环 (Training Loop)
    # =========================================================================
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_task_loss, total_kd_loss = 0, 0
        skipped_steps = 0

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            # 从检查点恢复时跳过已完成的步数
            if args.load_student and epoch == starting_epoch and step <= resume_step:
                progress_bar.update(1)
                progress_bar.set_description(
                    f"Skipping step {step} (already completed)")
                completed_steps += 1
                skipped_steps += 1
                continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                task_loss = outputs.loss

                # KD Loss (知识蒸馏损失): 学生隐层与教师隐层之间的差异
                kd_loss = get_kd_loss(model)

                # 加权组合 Task Loss (分类交叉熵) 和 KD Loss
                # loss = lm_weight * task_loss + kd_weight * kd_loss
                # 注: 参数名为 lm_weight 是为了与 CLM 脚本保持接口一致，
                # 在分类任务中实际为 task_loss 的权重
                loss = args.lm_weight * task_loss + args.kd_weight * \
                    kd_loss if args.kd_weight != 0 else task_loss
                progress_bar.set_description(
                    f"Epoch {epoch} - Step {step} - LR: {optimizer.param_groups[0]['lr']:.2e} - Task loss: {task_loss:.4f} - KD loss: {kd_loss:.4f}")

                total_task_loss += task_loss.item()
                total_kd_loss += kd_loss.item()

                # 累加到评估器的 interval 计数器中
                evaluator.interval_task_loss += task_loss.item()
                evaluator.interval_kd_loss += kd_loss.item()
                evaluator.eval_steps += 1

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            # end accumulate gradients

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            else:
                continue

            # 定期评估 (每 eval_steps 步)
            if completed_steps % args.eval_steps == 0:
                evaluator(model)

        # 每个 epoch 结束时也执行一次评估
        evaluator(model)

    accelerator.end_training()


if __name__ == "__main__":
    main()
