#!/usr/bin/env python
"""
Step C 推理：加载 Emulator + Adapter，对 sampled_prompts 做生成推理。

流程：
  1. 加载 OPT-1.3B 模型（fp16）+ tokenizer
  2. 对每个 (压缩率, 任务) 组合：
     - 加载对应的 student.pt（emulator）+ adapter.pt
     - 对 sampled_prompts 里的每个 prompt 做贪心生成
  3. 保存生成结果到 inference_results/ 目录（JSONL）

复用 offsite_tuning.utils 的 load_student / load_adapter，与训练/评估保持一致。

运行：
  python inference_model_quality.py
"""
import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from accelerate import Accelerator

# 必须先初始化 accelerate 状态，offsite_tuning.utils 里的 get_logger 才可用
accelerator = Accelerator()

from offsite_tuning.utils import load_student, load_adapter


# ============ 配置 ============
MODEL_NAME = "facebook/opt-1.3b"
TOTAL_LAYERS = 24          # OPT-1.3B 共 24 层
PAD = 2                    # 适配器层数（左右各 2 层）
NUM_LAYERS_LIST = [20, 12, 2]   # 三个压缩率对应的学生层数

# 每个任务的生成参数
TASK_CONFIG = {
    "piqa":     {"max_new_tokens": 30},
    "wikitext": {"max_new_tokens": 50},
}

# 路径模板
EMULATOR_DIR = "emulators/{model}/{nlayers}_{pad}_{pad}/student.pt"
ADAPTER_DIR = "mylogs/{model}/{task}/ft_emulator/{nlayers}_{pad}_{pad}/adapter.pt"
PROMPT_DIR = "sampled_prompts/{task}.jsonl"
OUTPUT_DIR = "inference_results"


def build_args(pad=PAD):
    """构造一个包含 load_student / load_adapter 所需字段的 args 对象。"""
    return argparse.Namespace(
        student_l_pad=pad,
        student_r_pad=pad,
    )


def load_model_with_checkpoint(model_name, student_path, adapter_path, args, device):
    """加载模型并装载 student（emulator）+ adapter 权重。

    Args:
        model_name: 预训练模型名/路径。
        student_path: student.pt 路径。
        adapter_path: adapter.pt 路径。
        args: 含 student_l_pad / student_r_pad 的 Namespace。
        device: 目标设备。

    Returns:
        装载好 checkpoint 的模型。
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16)

    # 加载 emulator（student）权重
    student_state_dict = torch.load(student_path, map_location="cpu")
    model = load_student(model, student_state_dict, args)

    # 加载 adapter 权重
    adapter_state_dict = torch.load(adapter_path, map_location="cpu")
    model = load_adapter(model, adapter_state_dict, args)

    model = model.to(device)
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_new_tokens, device):
    """对单个 prompt 做贪心生成，返回生成的文本。"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只取新生成的 token（去掉输入部分）
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated


def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"             # 显式属性赋值，确保生效
    tokenizer.pad_token = tokenizer.eos_token   # OPT 无 pad_token，用 eos 代替
    args = build_args(PAD)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for task, task_cfg in TASK_CONFIG.items():
        prompt_path = PROMPT_DIR.format(task=task)
        if not os.path.exists(prompt_path):
            print(f"[跳过] 找不到 prompt 文件: {prompt_path}")
            continue

        # 读取 sampled prompts
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = [json.loads(line) for line in f if line.strip()]

        print(f"任务 {task}: 共 {len(prompts)} 个 prompt")

        for num_layers in NUM_LAYERS_LIST:
            student_path = EMULATOR_DIR.format(
                model=MODEL_NAME, nlayers=num_layers, pad=PAD)
            adapter_path = ADAPTER_DIR.format(
                model=MODEL_NAME, task=task, nlayers=num_layers, pad=PAD)

            if not os.path.exists(student_path):
                print(f"  [跳过] 找不到 student.pt: {student_path}")
                continue
            if not os.path.exists(adapter_path):
                print(f"  [跳过] 找不到 adapter.pt: {adapter_path}")
                continue

            compression_ratio = num_layers / TOTAL_LAYERS
            print(f"  y={compression_ratio:.3f} ({num_layers}层): 加载 checkpoint...")

            model = load_model_with_checkpoint(
                MODEL_NAME, student_path, adapter_path, args, device)

            # 生成并保存结果
            out_path = os.path.join(
                OUTPUT_DIR, f"{task}_{num_layers}_{PAD}_{PAD}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in prompts:
                    prompt = rec["prompt"]
                    generated = generate(
                        model, tokenizer, prompt,
                        task_cfg["max_new_tokens"], device)

                    out_rec = {
                        "id": rec.get("id", ""),
                        "prompt": prompt,
                        "label": rec.get("label", ""),
                        "dataset": task,
                        "num_layers": num_layers,
                        "compression_ratio": round(compression_ratio, 4),
                        "generated_text": generated,
                        # 透传 ground truth 字段（用于对比生成 vs 真实答案）
                        "answer": rec.get("answer", ""),
                    }
                    # PIQA 额外透传两个选项
                    if task == "piqa":
                        out_rec["sol1"] = rec.get("sol1", "")
                        out_rec["sol2"] = rec.get("sol2", "")
                    # WikiText 额外透传完整原文
                    if task == "wikitext":
                        out_rec["total_text"] = rec.get("total_text", "")
                    f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

            print(f"    已保存 {len(prompts)} 条结果 -> {out_path}")

            # 释放显存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("推理完成。")


if __name__ == "__main__":
    run_inference()
