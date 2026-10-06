#!/usr/bin/env python
"""
生成预分词验证集 /dataset/opt_tokenized/wikitext-2-raw-v1
供 run_clm.py 的 --val_tokenized_dataset 参数使用。

原理：
  1. 从 HuggingFace 加载原始 WikiText-2 文本
  2. 用 OPT tokenizer 分词（必须与训练时 tokenizer 完全一致）
  3. 通过 save_to_disk 保存为 datasets 格式（load_from_disk 可直接加载）

注意：
  - 这里只做「分词」，不做 block_size 分块。分块由训练脚本里的
    get_lm_datasets() 现场完成，若提前分块会导致二次分块、数据混乱。
  - 保存后的字段为 input_ids / attention_mask，训练时会自动补 labels。
"""
from datasets import load_dataset
from transformers import AutoTokenizer

# 1. 加载原始 WikiText-2 数据集（train/validation/test 三个 split）
raw = load_dataset("wikitext", "wikitext-2-raw-v1")

# 2. 加载 OPT tokenizer —— 必须与训练时的 tokenizer 一致！
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# 3. 分词函数
def tokenize_function(examples):
    return tokenizer(examples["text"])

# 4. 批量分词（num_proc 多进程加速，remove_columns 删掉原始文本列）
tokenized = raw.map(
    tokenize_function,
    batched=True,
    num_proc=10,
    remove_columns=["text"],
)

# 5. 保存到磁盘（load_from_disk 可直接加载）
tokenized.save_to_disk("autodl-tmp/dataset/opt_tokenized/wikitext-2-raw-v1")
print("Saved to autodl-tmp/dataset/opt_tokenized/wikitext-2-raw-v1")