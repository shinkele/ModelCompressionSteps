# ============================================================================
# offsite_tuning/tasks.py
# ============================================================================
# 下游任务定义与数据集名称映射模块。
#
# 本模块为 Offsite-Tuning (arXiv:2302.04870) 的 text-to-text 评估提供了
# 统一的任务接口。每个任务类实现两个方法:
#   - get_context(examples): 返回输入/问题文本列表。
#   - get_target(examples): 返回目标/答案文本列表。
#
# 支持的下游任务及其格式:
#   ┌────────────────┬──────────────────────────────────────────┐
#   │ 任务            │ 格式                                     │
#   ├────────────────┼──────────────────────────────────────────┤
#   │ PIQA           │ Question: {goal}\nAnswer: {solution}     │
#   │ HellaSwag      │ {activity}: {ctx_a} {ctx_b} {ending}     │
#   │ OpenBookQA     │ Question: {question_stem}\nAnswer: {text}│
#   │ ARC (Easy/Ch.) │ Question: {question}\nAnswer: {choice}   │
#   │ RACE           │ Article: ...  \nQuestion: ...\nAnswer:   │
#   │ SciQ           │ {support}\nQuestion: {q}\nAnswer: {ans}  │
#   │ WebQuestions   │ Question: {q}\nAnswer: {answer}          │
#   └────────────────┴──────────────────────────────────────────┘
#
# task_dict: 将用户指定的任务名称映射到对应的任务类实例。
# map_dataset_name_and_config(): 将简写任务名映射到 HuggingFace Hub
#   上的实际数据集名称和配置名 (如 arc_easy -> ai2_arc/ARC-Easy)。
# LM_EVAL_TASK_NAME_MAPPING: 将本项目的任务名映射到 lm-eval-harness
#   中的标准任务名 (用于外部评估基准测试)。
# ============================================================================

import re


# =========================== 任务类定义 ===========================

class PIQA:
    """
    PIQA (Physical Interaction QA) — 物理常识推理。

    格式: Question: {goal}\nAnswer: {solution}

    goal: 物理常识问题 (如 "How to make a cake?")。
    target: 从 sol1/sol2 中选择正确答案。
    """
    def __init__(self):
        self._template = "Question: {}\nAnswer:"

    def get_context(self, examples):
        ctx = examples['goal']
        return [self._template.format(c) for c in ctx]

    def get_target(self, examples):
        if -1 in examples["label"]:  # test set: 测试集无标签，返回空字符串
            return [""] * len(examples["label"])
        else:
            # 根据 label (0 或 1) 选择 sol1 或 sol2
            gt_tuples = [("sol{}".format(label + 1), idx)
                         for idx, label in enumerate(examples['label'])]
            return [examples[k][i] for k, i in gt_tuples]


class HellaSwag:
    """
    HellaSwag — 常识性自然语言推理 (Commonsense NLI)。

    格式: {activity_label}: {ctx_a} {ctx_b} {ending}

    模型需要从 4 个结尾 (ending) 中选择最合理的那个。
    """
    @classmethod
    def preprocess(cls, text):
        """
        文本预处理:
          - 移除 WikiHow 的 [title] 标记
          - 移除所有方括号标记 [*]
          - 合并多余空格
        """
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def get_context(self, examples):
        # 拼接: activity_label + ctx_a + ctx_b (将 ctx_b 首字母大写)
        ctx_zip = zip(examples["activity_label"],
                      examples["ctx_a"], examples["ctx_b"])
        return [self.preprocess(a + ": " + b + " " + c.capitalize()) for a, b, c in ctx_zip]

    def get_target(self, examples):
        labels = examples["label"]
        endings = examples["endings"]
        targets = []
        for idx, label in enumerate(labels):
            # 测试集 label 为空字符串
            target = '' if label == '' else endings[idx][int(label)]
            targets.append(self.preprocess(target))
        return targets


class OpenBookQA:
    """
    OpenBookQA — 开放域科学问答。

    格式: Question: {question_stem}\nAnswer: {choice_text}

    模型需根据常识和"开放书本"中的科学事实选择正确答案。
    """
    def get_context(self, examples):
        return examples['question_stem']

    def get_target(self, examples):
        choices = examples['choices']
        answers = examples['answerKey']
        targets = []
        for choice, answer in zip(choices, answers):
            # answerKey 格式: 'A', 'B', 'C', 'D' -> 转换为索引
            answer = ord(answer.strip()) - ord('A')
            targets.append(choice['text'][answer])
        return targets


class ARC:
    """
    ARC (AI2 Reasoning Challenge) — 科学推理问答。

    包含两个子集:
      - ARC-Easy (ARC-E): 较简单的问题
      - ARC-Challenge (ARC-C): 需要更深层推理的难题

    格式: Question: {question}\nAnswer: {choice_text}
    """
    def __init__(self):
        self._template = "Question: {}\nAnswer:"

    def get_context(self, examples):
        ctx = examples['question']
        return [self._template.format(c) for c in ctx]

    def get_target(self, examples):
        choices = examples['choices']
        answers = examples['answerKey']
        # ARC 数据集中 answerKey 可能是数字 "1"-"5" 或字母 "A"-"E"
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        for idx, answer in enumerate(answers):
            answer = num_to_letter.get(answer, answer)
            answer = ord(answer) - ord("A")
            answers[idx] = choices[idx]["text"][answer]
        return answers


class RACE:
    """
    RACE (ReAding Comprehension from Examinations) — 阅读理解。

    格式:
      Article: {article}
      Question: {question}
      Answer: {answer_option}

    源自中国初高中英语考试阅读理解题，每个问题有 4 个选项。
    """
    @classmethod
    def doc_to_text(cls, article, question):
        text = "Article: " + article + "\n\n"
        text += "Question: " + question + "\n\n"
        text += "Answer:"
        return text

    def get_context(self, examples):
        return [
            self.doc_to_text(article, question)
            for article, question in zip(examples["article"], examples["question"])
        ]

    def get_target(self, examples):
        answers = examples['answer']
        options = examples['options']
        for idx, answer in enumerate(answers):
            # answer 为 'A', 'B', 'C', 'D' -> 取对应选项文本
            answers[idx] = options[idx][ord(answer) - ord("A")]
        return answers


class SciQ:
    """
    SciQ — 科学问答 (带支持证据)。

    格式: {support}\nQuestion: {question}\nAnswer: {correct_answer}

    每个问题附带一段支持性文本 (support) 作为上下文。
    """
    def __init__(self):
        self._template = "{}\nQuestion: {}\nAnswer:"

    def get_context(self, examples):
        sources = examples['support']
        queries = examples['question']
        return [self._template.format(s, q) for s, q in zip(sources, queries)]

    def get_target(self, examples):
        return examples['correct_answer']


class WebQs:
    """
    WebQuestions — 基于 Freebase 的知识库问答。

    格式: Question: {question}\nAnswer: {answer}

    问题源自 Google Suggest API，答案源自 Freebase。
    每个问题可能有多个答案，这里取第一个。
    """
    def get_context(self, examples):
        return ["Question: " + question + "\nAnswer:" for question in examples["question"]]

    def get_target(self, examples):
        # 每个问题可能有多个答案，取第一个作为 target
        return [" " + answers[0] for answers in examples["answers"]]


# =========================== 任务注册与名称映射 ===========================

# task_dict: 将命令行参数中用户指定的任务名映射到对应的任务类实例。
# 使用方式: task = task_dict[args.dataset_name]; task.get_context(examples)
task_dict = {
    "piqa": PIQA(),
    "hellaswag": HellaSwag(),
    "openbookqa": OpenBookQA(),
    "arc_easy": ARC(),
    "arc_challenge": ARC(),
    "sciq": SciQ(),
    "web_questions": WebQs(),
    "race": RACE(),
}


def map_dataset_name_and_config(args):
    """
    将项目内部的简写任务名映射到 HuggingFace Hub 上的实际数据集名称和配置。

    映射关系:
      arc_easy      -> ai2_arc / ARC-Easy
      arc_challenge -> ai2_arc / ARC-Challenge
      race          -> race / high (高中难度)

    其他数据集名保持不变 (如 piqa, hellaswag, openbookqa 等)。
    返回 (dataset_name, dataset_config_name) 二元组。
    """
    dataset_name = args.dataset_name
    dataset_config_name = args.dataset_config_name
    if args.dataset_name == 'arc_easy':
        dataset_name = 'ai2_arc'
        dataset_config_name = 'ARC-Easy'
    elif args.dataset_name == 'arc_challenge':
        dataset_name = 'ai2_arc'
        dataset_config_name = 'ARC-Challenge'
    elif args.dataset_name == 'race':
        dataset_config_name = 'high'


    return dataset_name, dataset_config_name


# LM_EVAL_TASK_NAME_MAPPING:
# 将本项目内部任务名映射到 lm-evaluation-harness 中的标准任务名。
# lm-eval-harness 是 EleutherAI 维护的标准化 LLM 评估框架，
# 用于在外部基准测试中验证 Offsite-Tuning 的效果。
# 当前映射: web_questions -> "webqs"
LM_EVAL_TASK_NAME_MAPPING = {
    "web_questions": "webqs"
}
