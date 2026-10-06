#!/usr/bin/env python
"""
Step C 评估：对 inference_results 里每个 example 的 generated_text 与 answer 计算
BERTScore 与 ROUGE-L，输出到 myscripts/metrics_results/（与原文件一一对应）。

每个输出文件与 inference_results 下的同名文件一一对应；每行保留原 record 的全部
字段，并新增 6 个指标字段：bert_score_p/r/f1、rougeL_p/r/f1。

依赖：
  pip install bert-score rouge-score

运行（在实验三目录）：
  python myscripts/compute_metric.py
"""
import json
import os

import torch
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer


# ============ 配置 ============
RESULTS_DIR = "inference_results"
OUTPUT_DIR = "myscripts/metrics_results"
BERT_MODEL = "roberta-large"   # BERTScore 标准默认模型
ROUND = 4


def compute_rouge_l(rouge, ref, hyp):
    """计算单个 example 的 ROUGE-L (precision/recall/f1)，空文本返回 (None, None, None)。"""
    if not ref or not hyp:
        return None, None, None
    s = rouge.score(ref, hyp)["rougeL"]
    return s.precision, s.recall, s.fmeasure


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith(".jsonl"))
    if not files:
        print(f"[警告] {RESULTS_DIR} 下没有 jsonl 文件")
        return

    for fname in files:
        src_path = os.path.join(RESULTS_DIR, fname)
        with open(src_path, "r", encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]

        hyps = [rec.get("generated_text", "") for rec in recs]
        refs = [rec.get("answer", "") for rec in recs]

        # ---- BERTScore（批量，只对非空文本计算，空文本置 None）----
        valid_idx = [i for i in range(len(hyps)) if hyps[i] and refs[i]]
        bert_p = [None] * len(hyps)
        bert_r = [None] * len(hyps)
        bert_f1 = [None] * len(hyps)
        if valid_idx:
            P, R, F1 = bert_score_fn(
                [hyps[i] for i in valid_idx],
                [refs[i] for i in valid_idx],
                lang="en",
                model_type=BERT_MODEL,
                device=device,
                batch_size=32,
                verbose=True,
            )
            p_list = P.tolist()
            r_list = R.tolist()
            f1_list = F1.tolist()
            for k, i in enumerate(valid_idx):
                bert_p[i] = p_list[k]
                bert_r[i] = r_list[k]
                bert_f1[i] = f1_list[k]

        # ---- 写出：原 record 全部字段 + 6 个指标字段 ----
        out_path = os.path.join(OUTPUT_DIR, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            for i, rec in enumerate(recs):
                rl_p, rl_r, rl_f1 = compute_rouge_l(rouge, refs[i], hyps[i])
                out_rec = dict(rec)
                out_rec.update({
                    "bert_score_p": round(bert_p[i], ROUND) if bert_p[i] is not None else None,
                    "bert_score_r": round(bert_r[i], ROUND) if bert_r[i] is not None else None,
                    "bert_score_f1": round(bert_f1[i], ROUND) if bert_f1[i] is not None else None,
                    "rougeL_p": round(rl_p, ROUND) if rl_p is not None else None,
                    "rougeL_r": round(rl_r, ROUND) if rl_r is not None else None,
                    "rougeL_f1": round(rl_f1, ROUND) if rl_f1 is not None else None,
                })
                f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

        print(f"已保存 {len(recs)} 条 -> {out_path}")

    print("指标计算完成。")


if __name__ == "__main__":
    main()
