#!/usr/bin/env python
"""
Step C 数据准备：随机挑选 50 个示例并保存为 prompt 格式（JSONL）。

功能：
  1. 从下游数据集（默认 PIQA）随机采样 50 个示例
  2. 用 offsite_tuning/tasks.py 的 PIQA 模板格式化为可直接喂给模型的 prompt
  3. 保存为 JSONL，每行一个 {id, prompt, label, dataset}
"""
import json
import os
from datasets import load_dataset

N_SAMPLES = 100

def sample_and_save_prompts(dataset_name="piqa", dataset_config_name=None,
                            n_samples=N_SAMPLES, seed=42,
                            out_path="sampled_prompts.jsonl"):
    """随机挑选 n 个示例，格式化为 prompt，保存为 JSONL。

    Args:
        dataset_name: 下游数据集名（piqa / wikitext）。
        dataset_config_name: 数据集子配置名（如 wikitext 的 wikitext-2-raw-v1）。
        n_samples: 采样数量，默认 50。
        seed: 随机种子，保证可复现。
        out_path: 输出文件路径（JSONL）。
    """
    # 1. 加载数据集（piqa 含自定义代码，需 trust_remote_code；
    #    wikitext 有多个配置，需指定 wikitext-2-raw-v1）
    raw = load_dataset(dataset_name, dataset_config_name, trust_remote_code=True)

    # 2. 随机采样（用 shuffle + select，保证可复现）
    samples = raw["train"].shuffle(seed=seed).select(range(n_samples))

    # 3. PIQA 模板：复用 tasks.py 的格式
    template = "Question: {}\nAnswer:"

    # 4. 格式化为 prompt 并逐行写入 JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(samples):
            if dataset_name == "piqa":
                # prompt 保持与 run_clm.py 的 context 一致（开放式生成，不含选项）
                prompt = template.format(ex["goal"])

                # 额外保存答案文本，仅作 ground truth 参照（不进 prompt）
                sol1 = ex["sol1"]
                sol2 = ex["sol2"]
                label = int(ex["label"])                       # 0 或 1
                answer = sol1 if label == 0 else sol2          # 正确答案文本
            else:
                # wikitext：语言建模续写任务，方法2 字面对比
                # 将原文按词切分为 prefix（前 80%，喂给模型）+ suffix（后 20%，ground truth）
                text = ex["text"]
                words = text.split()
                n_words = len(words)
                split_idx = int(n_words * 0.8)                  # 前 80% 词作前缀
                prompt = " ".join(words[:split_idx])            # 喂给模型的前缀
                answer = " ".join(words[split_idx:])            # 真实续写（ground truth）
                total_text = text                               # 完整原始文本

            record = {
                "id": i,
                "prompt": prompt,              # 可直接 tokenize 喂给模型
                "label": ex.get("label", ""),  # 原始标签（PIQA 用于对照）
                "dataset": dataset_name,
                "answer": answer,              # ground truth（PIQA=答案，wikitext=续写）
            }

            # PIQA 额外保存两个选项，便于后续与生成结果对比
            if dataset_name == "piqa":
                record["sol1"] = sol1
                record["sol2"] = sol2
            else:
                # wikitext 额外保存完整原始文本
                record["total_text"] = total_text

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {n_samples} prompts to {out_path}")


if __name__ == "__main__":
    #========================= piqa ==============================
    dataset_name = "piqa"
    dataset_config_name = None
    # dataset_name = "wikitext"
    # dataset_config_name = "wikitext-2-raw-v1"

    # 输出目录 + 文件名（补全 .jsonl 后缀）
    out_dir = "sampled_prompts"
    os.makedirs(out_dir, exist_ok=True)           # 确保输出目录存在
    out_path = os.path.join(out_dir, f"{dataset_name}.jsonl")

    sample_and_save_prompts(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        out_path=out_path,
    )

    #========================= wikitexts ==============================
    # dataset_name = "piqa"
    # dataset_config_name = None
    dataset_name = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"

    # 输出目录 + 文件名（补全 .jsonl 后缀）
    out_dir = "sampled_prompts"
    os.makedirs(out_dir, exist_ok=True)           # 确保输出目录存在
    out_path = os.path.join(out_dir, f"{dataset_name}.jsonl")

    sample_and_save_prompts(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        out_path=out_path,
    )
