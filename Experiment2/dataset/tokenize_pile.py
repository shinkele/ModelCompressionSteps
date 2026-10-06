#!/usr/bin/env python
"""
生成预分词训练集 /dataset/pile/opt_tokenized/00
供 run_clm.py 的 --train_tokenized_dataset 参数使用。

原理：与 WikiText-2 分词完全一致，只是数据源换成 The Pile，
     输出目录名用分片号「00」区分。

数据源说明：
  - 原 EleutherAI/pile 底层数据指向 the-eye.eu，该站已关停，
    加载时报 503 Service Unavailable，故改用官方去重版
    EleutherAI/the_pile_deduplicated（有 parquet 分片，可正常下载）。
  - 采用 streaming 流式加载，避免一次性下载完整数据。
    「00」只是命名约定 —— 用 take(N) 取前 N 条即可构成「第 0 分片」。

注意：
  - streaming 返回 IterableDataset，它没有 save_to_disk 方法；
    必须先用 Dataset.from_generator 转成普通 Dataset 再落盘。
  - from_generator 会一次性物化到内存，N 不要设太大；
    仅为画示例图，取 10万~100万条即可。
  - the_pile_deduplicated 的文本字段通常为 "text"，若报 KeyError，
    先 print(next(iter(raw))) 检查实际字段名。
  - 同样只做「分词」，不做 block_size 分块。
"""

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# 1. streaming 流式加载（改用官方去重版，原 pile 已失效）
raw = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True) # 数据源可能失效
# EleutherAI/the_pile_deduplicated, 推荐，官方去重版，有 parquet
# monology/pile-uncopyrighted, Pile 的无版权子集
# pietrolesci/pile-deduped, 社区去重镜像


# 只取前 N 条，等价于「第 0 分片」（N 按需调整）
shard = raw.take(200000)

# 2. 加载 OPT tokenizer —— 必须与训练时的 tokenizer 一致！
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

# 3. 生成器：逐条分词（惰性，避免一次性加载全部原始文本）
def gen():
    for example in shard:
        tok = tokenizer(example["text"])
        yield {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}

# 4. 将 IterableDataset 转为普通 Dataset（后者才有 save_to_disk）
tokenized = Dataset.from_generator(gen)

# 5. 保存为第 0 号分片（落地后可用 load_from_disk 反复加载）
tokenized.save_to_disk("autodl-tmp/dataset/pile/opt_tokenized/00")
print("Saved to autodl-tmp/dataset/pile/opt_tokenized/00")