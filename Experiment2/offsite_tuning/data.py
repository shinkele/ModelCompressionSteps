# ============================================================================
# offsite_tuning/data.py
# ============================================================================
# 数据加载与预处理流水线 (Data Loading Pipeline)。
#
# 整体流程 (原始文本 → 分词 → 分组/填充):
#   阶段 1: get_raw_datasets()
#           → 从 HuggingFace Hub 或本地文件加载原始数据集
#   阶段 2: get_tokenized_datasets() / process_text2text_datasets()
#           → 将原始文本转换为 token IDs
#   阶段 3: get_lm_datasets() (仅 CLM/MLM)
#           → 将所有文本拼接后切分为固定长度的块 (chunk)
#   阶段 3': process_text2text_datasets() 中的 padding (仅 text2text 任务)
#           → 将 context-target 拼接并填充到统一长度 (8 的倍数)
#
# 支持两种下游任务范式:
#   - 因果语言模型 (CLM) / 掩码语言模型 (MLM): 标准自监督预训练格式
#   - Text-to-Text 生成任务 (如 PIQA, HellaSwag 等): 条件生成格式
# ============================================================================

from datasets import load_dataset
import logging
from itertools import chain
from offsite_tuning.tasks import task_dict, map_dataset_name_and_config

logger = logging.getLogger(__name__)
 

# ======================== 阶段 1: 获取原始数据集 ========================

def get_raw_datasets(args):
    """
    从 HuggingFace Hub 或本地文件加载原始数据集。

    支持两种加载方式:
      1. Hub 数据集: 指定 args.dataset_name，自动从 HuggingFace Hub 下载。
      2. 本地文件: 指定 args.train_file / args.validation_file，
         支持 CSV/JSON/TXT/JSONL(.zst) 格式。

    如果数据集中没有 validation 划分，则从训练集中按
    args.validation_split_percentage 切分一部分作为验证集。

    返回:
        datasets.DatasetDict: 包含 'train' 和 'validation' 的字典。
    """
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset_name, dataset_config_name = map_dataset_name_and_config(args)
        raw_datasets = load_dataset(
            dataset_name, dataset_config_name, trust_remote_code=True)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                trust_remote_code=True,
            )
            raw_datasets["train"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                trust_remote_code=True,
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        elif extension == 'zst':
            extension = 'json'
        raw_datasets = load_dataset(
            extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    return raw_datasets


# ======================== 阶段 2: 文本分词 (Tokenization) ========================

def get_tokenized_datasets(raw_datasets, args, accelerator, tokenizer, lm_type='clm'):
    """
    对原始文本数据集进行分词，将文本列转换为 input_ids 和 attention_mask。

    参数:
        raw_datasets: 阶段 1 输出的原始数据集。
        args: 命令行参数。
        accelerator: HuggingFace Accelerator 实例，用于分布式同步。
        tokenizer: 分词器。
        lm_type: 'clm' (因果语言模型) 或 'mlm' (掩码语言模型)。
                 MLM 模式会额外返回 special_tokens_mask。

    返回:
        分词后的数据集，包含 input_ids 和 attention_mask 列。
    """
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        if lm_type == 'clm':
            return tokenizer(examples[text_column_name])
        elif lm_type == 'mlm':
            # MLM 需要 special_tokens_mask，用于后续 masking 逻辑
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
        else:
            raise ValueError(f'lm_type {lm_type} not supported')

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    return tokenized_datasets


# ======================== 阶段 3: 文本分组 (CLM/MLM 用) ========================

def _get_block_size(args, tokenizer):
    """
    确定分块大小 (block_size)。

    优先使用 args.block_size；若未指定则使用 tokenizer.model_max_length。
    当 tokenizer.model_max_length 过大时 (>1024)，默认回退到 1024，
    以避免显存溢出。同时 block_size 不能超过 tokenizer 的最大长度。
    """
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)
    return block_size


def get_lm_datasets(tokenized_datasets, args, accelerator, tokenizer, lm_type='clm'):
    """
    将分词后的数据集拼接并切分为固定长度的块 (block_size)。

    处理流程:
      1. 将所有样本的 token 序列展平拼接为一个长序列。
      2. 按 block_size 切分为固定长度的块。
      3. 丢弃不足 block_size 的尾部剩余 (remainder)。
      4. 对于 CLM: labels = input_ids (自回归任务的标签即为输入本身)。

    参数:
        tokenized_datasets: 阶段 2 输出的分词数据集。
        args: 命令行参数。
        accelerator: HuggingFace Accelerator 实例。
        tokenizer: 分词器。
        lm_type: 'clm' 或 'mlm'。

    返回:
        分组后的数据集，每个样本长度为 block_size。
    """
    block_size = _get_block_size(args, tokenizer)
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.

    def group_texts(examples):
        # ===== 将所有文本拼接为一个长序列 =====
        # Concatenate all texts.
        # itertools.chain 将多个列表展平拼接: [tokens_1, tokens_2, ...] -> [所有 tokens]
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        # ===== 截断至 block_size 的整数倍 (丢弃尾部余数) =====
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # ===== 按 block_size 切分为固定长度块 =====
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # CLM: labels = input_ids (自回归语言模型标准做法)
        if lm_type == 'clm':
            result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    return lm_datasets


# ======================== 阶段 2+3 (合并): Text-to-Text 任务处理 ========================

def process_text2text_datasets(raw_datasets, args, tokenizer, accelerator):
    """
    处理 Text-to-Text 格式的数据集 (如 PIQA, HellaSwag, RACE 等)。

    与 CLM/MLM 的三阶段流水线不同，此函数将分词、context-target 拼接、
    label 掩码和填充统一在一个流程中完成。

    处理流程:
      1. 从 task 对象获取 context (输入/问题) 和 target (输出/答案)。
      2. 分别对 context 和 target 进行分词。
      3. 清理 context 末尾和 target 开头的特殊 token (避免重复)。
      4. 拼接 input_ids = context_tokens + target_tokens。
      5. labels 中 context 部分设为 -100 (忽略损失计算)，仅 target 部分参与训练。
      6. 将所有样本填充 (padding) 至统一长度 (8 的倍数)，以支持批处理。

    参数:
        raw_datasets: 原始数据集。
        args: 命令行参数。
        tokenizer: 分词器。
        accelerator: HuggingFace Accelerator 实例。

    返回:
        tokenized_datasets: 包含 input_ids, attention_mask, labels 的数据集。
    """
    task = task_dict[args.dataset_name]

    column_names = raw_datasets["train"].column_names

    def tokenize_function(examples):
        # ===== 步骤 1: 获取 context 和 target 文本 =====
        context = task.get_context(examples)
        target = task.get_target(examples)

        # ===== 步骤 2: 分别分词 =====
        context = tokenizer(context)
        target = tokenizer(target)

        # ===== 步骤 3: 清理边界的特殊 token =====
        # 如果 context 以特殊 token 结尾，移除 (避免与 target 开头重复)
        # if context is ending with special token, remove it
        if len(context['input_ids'][0]) > 0 and context['input_ids'][0][-1] in tokenizer.all_special_ids:
            context['input_ids'] = [i[:-1] for i in context['input_ids']]
            context['attention_mask'] = [a[:-1]
                                         for a in context['attention_mask']]

        # 如果 target 以特殊 token 开头，移除 (避免与 context 结尾重复)
        # if target is starting with special token, remove it
        if len(target['input_ids'][0]) > 0 and target['input_ids'][0][0] in tokenizer.all_special_ids:
            target['input_ids'] = [i[1:] for i in target['input_ids']]
            target['attention_mask'] = [a[1:]
                                        for a in target['attention_mask']]

        # ===== 步骤 4: 拼接 context 和 target =====
        # input_ids = [context tokens] + [target tokens]
        # 模型在推理时读取 context，然后自回归生成 target
        out = {}
        out['input_ids'] = [i1 + i2 for i1,
                            i2 in zip(context['input_ids'], target['input_ids'])]
        out['attention_mask'] = [a1 + a2 for a1,
                                 a2 in zip(context['attention_mask'], target['attention_mask'])]

        # ===== 步骤 5: 构造 labels (context 部分用 -100 掩码) =====
        # set -100 for context tokens
        # -100 是 PyTorch CrossEntropyLoss 的默认 ignore_index，
        # 在 context 位置上的损失不会被计算，模型仅学习生成 target 部分。
        # 格式: labels = [-100, -100, ..., -100, target_token_1, target_token_2, ...]
        out["labels"] = [
            [-100] * len(i1) + i2 for i1, i2 in zip(context['input_ids'], target['input_ids'])]

        return out

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # GPT-2 tokenizer 默认没有 pad_token，这里用 bos_token 作为 pad_token
    if "gpt2" in args.model_name_or_path:
        tokenizer.pad_token = tokenizer.bos_token

    # ===== 步骤 6: 计算最大长度并填充到 8 的倍数 =====
    # pad all instances in lm_datasets to the max length of the dataset
    max_length = -1
    for v in tokenized_datasets.values():
        for x in v:
            max_length = max(max_length, len(x['input_ids']))

    # pad to the multiple of 8
    # 填充到 8 的倍数可以提升 GPU 矩阵运算效率 (Tensor Core 偏好 8 的倍数)
    max_length = (max_length // 8 + 1) * 8

    block_size = _get_block_size(args, tokenizer)
    max_length = min(max_length, block_size)

    def pad_function(examples):
        # ===== 右侧填充到 max_length =====
        # input_ids: 用 pad_token_id 填充
        examples["input_ids"] = [i + [tokenizer.pad_token_id] *
                                 (max_length - len(i)) for i in examples["input_ids"]]
        # attention_mask: 实际 token 为 1，填充部分为 0
        examples["attention_mask"] = [[1] * len(i) + [0] *
                                      (max_length - len(i)) for i in examples["attention_mask"]]
        # labels: 填充部分也用 -100 (忽略损失计算)
        examples["labels"] = [i + [-100] *
                              (max_length - len(i)) for i in examples["labels"]]
        # ===== 截断到 max_length (以防万一) =====
        # truncate to max_length
        examples["input_ids"] = [i[:max_length] for i in examples["input_ids"]]
        examples["attention_mask"] = [a[:max_length]
                                      for a in examples["attention_mask"]]
        examples["labels"] = [l[:max_length] for l in examples["labels"]]
        return examples

    with accelerator.main_process_first():
        tokenized_datasets = tokenized_datasets.map(
            pad_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Padding dataset to max length {max_length}",
        )

    return tokenized_datasets
