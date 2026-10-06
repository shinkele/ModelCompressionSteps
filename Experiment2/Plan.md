# 完整方案

## 概述

论文 III-D Model Quality 部分的**图 2**：展示同一输入在不同压缩率下，PIQA 和 WikiText-2 的真实生成输出对比。

采用完整 Offsite-Tuning 流程（路径 A）：
1. **蒸馏 Emulator**（KD loss 训练 student）
2. **微调 Adapter**（下游任务上训练 adapter）
3. **加载 Emulator + Adapter → 真实推理 → 绘图**

---

## 整体管线

```
Step A: 蒸馏 3 个 Emulator
  完整模型 24 层 → KD Loss 训练 → student.pt
  3 个压缩率：y=0.83 (20层) / y=0.50 (12层) / y=0.08 (2层)

Step B: 微调 6 组 Adapter
  对每个 Emulator × 每个数据集 (PIQA, WikiText-2) = 6 组
  加载 student.pt，冻结 student，只训练 adapter → adapter.pt

Step C: 加载 Emulator + Adapter → 推理 → 绘图
  对同一输入，对比 3 个压缩率下的生成输出
```

### 时间与硬件估计

| 步骤 | 内容 | 训练量 | 估计时间 |
|------|------|--------|----------|
| Step A | 3 个 emulator 蒸馏 | 3 epoch | ~5 小时 |
| Step B | 6 组 adapter 微调 | 30 epoch | ~3 小时 |
| Step C | 推理+绘图 | 6 次推理 | ~5 分钟 |
| **总计** | | | **~8 小时** |

---

## Step A：蒸馏 Emulator

### 核心思路

修改 `scripts/distill_emulator/opt.sh`，适配单卡运行：
1. 去掉多卡参数（`CUDA_VISIBLE_DEVICES="0,1,...,7" --multi_gpu --num_processes=8`）
2. `--train_tokenized_dataset` 改为普通数据集（`--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1`）
3. 减小 batch size（8卡 bs=18 → 单卡 bs=4）
4. 注释 `--report_to wandb`（如果未配置 wandb）
5. `--num_warmup_steps 2000` → `500`（Wikitext-2 比 Pile 小很多）

### 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| MODEL | facebook/opt-1.3b | 24 层，1.3B 参数 |
| num_student_layers | 20 / 12 / 2 | 对应 y=0.83 / 0.50 / 0.08 |
| pad | 2 | 首尾各 2 层作为 adapter |
| bs | 4 | 单卡 batch size |
| learning_rate | 1e-4 | |
| kd_weight | 30.0 | OPT 蒸馏需较大 KD loss |
| lm_weight | 1.0 | |
| num_train_epochs | 1 | |
| block_size | 512 | |
| eval_steps | 100 | |
| num_warmup_steps | 500 | |
| train_module | student | **只训练 student 层** |

### 命令（3 个压缩率循环）

```bash
for num_student_layers in 20 12 2; do
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --mixed_precision=bf16 \
        offsite_tuning/run_clm.py \
        --model_name_or_path facebook/opt-1.3b \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --learning_rate 1e-4 \
        --num_warmup_steps 500 \
        --lr_scheduler_type cosine \
        --num_train_epochs 1 \
        --lm_weight 1.0 \
        --kd_weight 30.0 \
        --seed 42 \
        --block_size 512 \
        --eval_steps 100 \
        --num_student_layers $num_student_layers \
        --student_l_pad 2 \
        --student_r_pad 2 \
        --train_module student \
        --output_dir emulators/opt-1.3b/${num_student_layers}_2_2
done
```

### 产物

```
emulators/opt-1.3b/
├── 20_2_2/
│   ├── student.pt
│   └── all_results.json
├── 12_2_2/
│   ├── student.pt
│   └── all_results.json
└── 2_2_2/
    ├── student.pt
    └── all_results.json
```

---

## Step B：Adapter 微调

### 核心思路

修改 `scripts/table1/ft_emulator.sh`，去掉多卡参数。

对每个 Step A 产出的 emulator，在两个下游任务上分别微调 adapter。

### 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| MODEL | facebook/opt-1.3b | |
| num_student_layers | 20 / 12 / 2 | 对应 Step A 的产物 |
| pad | 2 | 与 Step A 一致 |
| bs | 4 | 单卡 batch size |
| learning_rate | 5e-5 | Adapter 微调学习率 |
| num_train_epochs | 5 | 微调需更多 epoch |
| train_module | **adapter** | **只训练 adapter 层** |
| kd_weight | 0.0 | 不需要 KD |
| no_teacher | True | 禁用 teacher model |
| restart_training | True | 从零开始训练 |
| load_student | emulators/opt-1.3b/{N}_2_2 | 加载 Step A 的 emulator |
| save_module | all | 保存 student + adapter |

### 数据集配置

| 任务 | dataset_name | dataset_config_name | block_size |
|------|-------------|---------------------|------------|
| PIQA | piqa | — | 512 |
| WikiText-2 | wikitext | wikitext-2-raw-v1 | 512 |

### 命令（6 组循环）

```bash
for num_layers in 20 12 2; do
    for task in piqa wikitext; do
        CONFIG_NAME=""
        if [ "$task" = "wikitext" ]; then
            CONFIG_NAME="--dataset_config_name wikitext-2-raw-v1"
        fi

        CUDA_VISIBLE_DEVICES=0 accelerate launch \
            --mixed_precision=bf16 \
            offsite_tuning/run_clm.py \
            --model_name_or_path facebook/opt-1.3b \
            --dataset_name $task \
            $CONFIG_NAME \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --learning_rate 5e-5 \
            --num_train_epochs 5 \
            --lm_weight 1.0 \
            --kd_weight 0.0 \
            --seed 42 \
            --eval_steps 20 \
            --block_size 512 \
            --num_student_layers $num_layers \
            --student_l_pad 2 \
            --student_r_pad 2 \
            --train_module adapter \
            --save_module all \
            --no_teacher \
            --restart_training \
            --load_student emulators/opt-1.3b/${num_layers}_2_2 \
            --output_dir logs/opt-1.3b/${task}/ft_emulator/${num_layers}_2_2
    done
done
```

### 产物

```
logs/opt-1.3b/
├── piqa/ft_emulator/
│   ├── 20_2_2/
│   │   ├── adapter.pt
│   │   ├── student.pt
│   │   └── all_results.json
│   ├── 12_2_2/
│   └── 2_2_2/
└── wikitext/ft_emulator/
    ├── 20_2_2/
    ├── 12_2_2/
    └── 2_2_2/
```

---

## Step C：采样 + 推理 + 绘图（三个脚本）

Step C 由三个独立脚本组成，按顺序运行：

```
sample_examples.py           → 采样 50 个示例 + 保存 prompt + ground truth
inference_model_quality.py   → 加载 emulator/adapter + 生成 + 透传 ground truth
plot_model_quality_examples.py → 读结果画对比图（含 Ground Truth 列）
```

---

### C1. 采样脚本 `sample_examples.py`

随机采样 50 个示例，保存 prompt 和 ground truth。

**PIQA**（问答任务，有明确答案）：
- prompt = `"Question: {goal}\nAnswer:"`（开放式生成，**不含选项**，与 run_clm.py 的 context 一致）
- ground truth 字段：`answer`（正确答案文本）、`sol1`、`sol2`、`label`

**WikiText**（语言建模，续写任务）：
- 按词切分：前 80% 词作 `prompt`（续写前缀），后 20% 词作 `answer`（真实续写）
- `total_text`：完整原始文本

**输出**：`sampled_prompts/{piqa,wikitext}.jsonl`

---

### C2. 推理脚本 `inference_model_quality.py`

对每个 (压缩率, 任务) 组合，加载 emulator + adapter，对每个 prompt 做贪心生成。

**关键点**：
- 复用 `offsite_tuning.utils` 的 `load_student` / `load_adapter`
- 必须先 `Accelerator()` 再 import utils（否则报 accelerate 初始化错误）
- tokenizer 需 `padding_side="left"` + `pad_token = eos_token`（OPT decoder-only）
- 贪心生成：`do_sample=False`；PIQA `max_new_tokens=30`，WikiText `max_new_tokens=50`
- 透传 ground truth 字段到输出结果

**输出**：`inference_results/{task}_{num_layers}_2_2.jsonl`

---

### C3. 绘图脚本 `plot_model_quality_examples.py`

读取推理结果，按 id 对齐 3 个压缩率，画对比图。

**布局**：5 列
- Input (prompt) | y=0.83 (20L) | y=0.50 (12L) | y=0.08 (2L) | Ground Truth (answer)

**可配置**：`EXAMPLE_LIST = [1, 2, 3, 4]` 手动指定要展示的示例 id

**输出**：`Figures/model_quality_{task}.pdf`

---

### Step C 数据流

```
sampled_prompts/piqa.jsonl          →  inference_results/piqa_20_2_2.jsonl  ┐
sampled_prompts/wikitext.jsonl      →  inference_results/piqa_12_2_2.jsonl  ├─→ Figures/*.pdf
                                        inference_results/piqa_2_2_2.jsonl   │
                                        inference_results/wikitext_*.jsonl  ┘
```

每个 JSONL 记录字段：

| 字段 | 含义 |
|------|------|
| `id` | 示例编号 |
| `prompt` | 模型输入（PIQA=问题，WikiText=前缀） |
| `generated_text` | 模型生成的文本 |
| `answer` | ground truth（PIQA=正确答案，WikiText=真实续写） |
| `sol1`/`sol2` | PIQA 的两个选项 |
| `total_text` | WikiText 完整原文 |
| `num_layers`/`compression_ratio` | 压缩配置 |

---

### 已修复的关键问题（踩坑记录）

| 问题 | 解决 |
|------|------|
| accelerate 初始化错误 | 推理脚本先 `Accelerator()` 再 import utils |
| right-padding 警告 | tokenizer 设 `padding_side="left"` |
| piqa 自定义代码报错 | `load_dataset(..., trust_remote_code=True)` |
| wikitext 需指定配置 | `load_dataset("wikitext", "wikitext-2-raw-v1")` |
| 单 GPU `model.module` 报错 | run_clm.py 5 处改为 `accelerator.unwrap_model(model)` |
| shell 续行符/赋值空格 | 修正 `\` 尾随空格、`task = "piqa"` 等 |

---

## 从实验结果提取的额外数据

图 2 的每个 cell 除了展示 token 输出，还应该标注从 Step B 的 `all_results.json` 中提取的真实 PPL：

| y | PIQA PPL | WikiText-2 PPL |
|---|----------|---------------|
| 0.83 (20层) | 待运行后填入 | 待运行后填入 |
| 0.50 (12层) | 待运行后填入 | 待运行后填入 |
| 0.08 (2层) | 待运行后填入 | 待运行后填入 |
| Teacher baseline (24层) | 35.1 | 31.5 |

这些数值用于图 2 中的质量评级（✓/△/✗）和 caption 引用。

---

## 验证清单

- [ ] Step A 完成后，检查 `all_results.json` 中 PPL 随层数递增而递减
- [ ] Step B 完成后，检查 adapter 微调后的 PPL 是否相比 student zero-shot 有提升
- [ ] Step C 完成后，确认：
  - [ ] `sample_examples.py` 生成 50 条含 ground truth 的 prompt
  - [ ] `inference_model_quality.py` 生成 6 个 JSONL（3 层 × 2 任务）
  - [ ] `plot_model_quality_examples.py` 生成 2 张 PDF，含 Ground Truth 列
  - [ ] y=0.83 的生成接近 ground truth，y=0.08 明显退化

---

## 运行步骤总结（云服务器）

```bash
# 0. 环境准备
pip install torch transformers accelerate datasets evaluate scipy matplotlib
pip install -e .

# 1. Step A: 蒸馏 Emulator（约 5 小时）
bash run_stepA_emulator_distillation.sh

# 2. Step B: 微调 Adapter（约 3 小时）
bash run_stepB_adapter_finetuning.sh

# 3. Step C: 采样 + 推理 + 绘图
python sample_examples.py                 # 采样 50 个示例
python inference_model_quality.py         # 生成推理结果
python plot_model_quality_examples.py     # 画对比图
```

---

## 附录：预分词数据集生成脚本

> 说明：仅在**复现原 opt.sh 的预分词路径**时才需要。若改用现场分词（`--dataset_name wikitext`），
> 可跳过本附录。以下两个脚本生成 `/dataset/opt_tokenized/wikitext-2-raw-v1`（验证集）
> 和 `/dataset/pile/opt_tokenized/00`（训练集，The Pile 第 0 分片）。

### A1. 生成 WikiText-2 验证集分词

保存为 `tokenize_wikitext.py`，在云服务器运行：

```python
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
tokenized.save_to_disk("/dataset/opt_tokenized/wikitext-2-raw-v1")
print("Saved to /dataset/opt_tokenized/wikitext-2-raw-v1")
```

### A2. 生成 The Pile 训练集分词（第 0 分片）

保存为 `tokenize_pile.py`，在云服务器运行（**重型任务，见下方资源说明**）：

```python
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
raw = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

# 只取前 N 条，等价于「第 0 分片」（N 按需调整）
shard = raw.take(1_000_000)

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
tokenized.save_to_disk("/dataset/pile/opt_tokenized/00")
print("Saved to /dataset/pile/opt_tokenized/00")
```

### A3. 资源消耗对比

| 维度 | WikiText-2（A1） | The Pile 单分片（A2, streaming take） |
|------|-----------------|--------------------------------------|
| 原始数据量 | ~2MB | 仅按需读取前 N 条（不落地全部 800GB） |
| 分词后磁盘占用 | ~10MB | 取决于 N（10万条约几 GB） |
| 分词时间 | 几分钟 | 取决于 N，streaming 下显著缩短 |
| 内存需求 | <1GB | 显著低于全量加载 |
| 建议 CPU 核数 | 任意 | streaming 下 map 可不用 num_proc |

### A4. 关键提醒

1. **tokenizer 必须一致**：OPT 的数据必须用 OPT tokenizer 分词，混用会导致 token id 错位。
2. **不要预先分块**：`save_to_disk` 前只分词（得到 input_ids/attention_mask），
   分块由训练脚本的 `get_lm_datasets()` 现场做。
3. **streaming 必须先落地**：`take(N)` 返回的是可迭代流式对象，需先 `save_to_disk`
   才能被训练脚本的 `load_from_disk` 反复加载。
4. **「00」是命名约定**：`take(N)` 取前 N 条即可构成「第 0 分片」，N 按需调整
   （原论文一个分片约 30GB 文本；仅画示例图取 10万~100万条即可）。
5. **原 `EleutherAI/pile` 已失效**：其底层数据指向 the-eye.eu（已关停），
   加载时报 503。改用官方去重版 `EleutherAI/the_pile_deduplicated`。
6. **字段名需确认**：`the_pile_deduplicated` 的文本字段通常为 `text`，
   若报 KeyError，先 `print(next(iter(raw)))` 检查实际字段名。
7. **对图 2 方案的建议**：The Pile 分词仍是重型任务，若仅为画示例图，
   建议改用现场分词（`--dataset_name wikitext`），跳过 A2。
