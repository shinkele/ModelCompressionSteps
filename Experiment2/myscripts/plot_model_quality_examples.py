#!/usr/bin/env python
"""
Step C 打印：读取 inference_results 的 JSONL 结果，在控制台打印模型质量退化对比信息。

功能：
  1. 读取 inference_results/{task}_{num_layers}_{pad}_{pad}.jsonl
  2. 按 id 对齐：同一 prompt 在 3 个压缩率（20/12/2 层）下的 generated_text
  3. 挑选 N 个代表性示例，每个任务打印 prompt / 三个压缩率生成结果 / Ground Truth

运行：
  python plot_model_quality_examples.py
"""
import json
import os
import sys

# Windows 控制台默认使用 GBK，遇到数据中无法编码的字符会抛 UnicodeEncodeError；
# 统一按 UTF-8 输出，避免打印中断或乱码。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============ 配置 ============
TASKS = ["piqa", "wikitext"]
NUM_LAYERS = [20, 12, 2]        # 对应 y = 0.833 / 0.5 / 0.083
TOTAL_LAYERS = 24
PAD = 2
EXAMPLE_LIST = [0, 1]     # 每个任务展示的示例 id 列表（可手动挑选）
RESULTS_DIR = "inference_results"

# 三个压缩率对应的展示标签（与 NUM_LAYERS 一一对应）
LAYER_LABELS = ["y=0.83 (20L)",
                "y=0.50 (12L)",
                "y=0.08 (2L)"]


def load_results(task, num_layers, pad=PAD):
    """读取单个 JSONL，返回 {id: record} 字典。"""
    path = os.path.join(RESULTS_DIR, f"{task}_{num_layers}_{pad}_{pad}.jsonl")
    recs = {}
    if not os.path.exists(path):
        print(f"[警告] 文件不存在: {path}")
        return recs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


def plot_task(task, example_ids=EXAMPLE_LIST):
    """为单个任务打印对比信息到控制台。"""
    # 1. 读取 3 个压缩率的结果，按 id 对齐
    results = {nl: load_results(task, nl) for nl in NUM_LAYERS}

    # 取三个文件里都存在的 id（交集），保证能对齐
    common_ids = set(results[NUM_LAYERS[0]].keys())
    for nl in NUM_LAYERS[1:]:
        common_ids &= set(results[nl].keys())
    # 按 EXAMPLE_LIST 指定的 id 从交集里挑选（不存在的 id 自动跳过）
    ids = [i for i in example_ids if i in common_ids]

    if not ids:
        print(f"[跳过] 任务 {task} 无可用对齐数据")
        return

    # 2. 打印每个示例的对比信息
    print(f"\n{'=' * 60}")
    print(f"Task: {task.upper()}")
    print(f"{'=' * 60}")

    for i, pid in enumerate(ids):
        prompt = results[NUM_LAYERS[0]][pid]["prompt"]
        answer = results[NUM_LAYERS[0]][pid].get("answer", "")

        print(f"\n[Example {i + 1}]  id={pid}")
        print(f"  Prompt:\n    {prompt}")
        for nl, label in zip(NUM_LAYERS, LAYER_LABELS):
            gen = results[nl][pid]["generated_text"]
            print(f"  [{label}]:\n    {gen}")
        print(f"  [Ground Truth]:\n    {answer}")


def main():
    for task in TASKS:
        plot_task(task)


if __name__ == "__main__":
    main()
