"""
使用 lm-eval-harness 库对 Offsite-Tuning 模型（arXiv:2302.04870）进行零样本/少样本评估。
LMEvalAdaptor 将模型包装为 lm-eval 兼容的接口，支持 perplexity 和生成式任务评估。
"""
import os
from offsite_tuning.utils import parse_args, load_adapter, load_student, get_layers, set_layers, uniform_choose_layers
from offsite_tuning.tasks import LM_EVAL_TASK_NAME_MAPPING
import torch
from lm_eval.base import BaseLM
from lm_eval import evaluator, tasks
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate.logging import get_logger

logger = get_logger(__name__)


class LMEvalAdaptor(BaseLM):
    """
    将 Offsite-Tuning 模型包装为 lm-eval-harness 的 BaseLM 兼容接口。
    实现 _model_call（logits 前向计算）、_model_generate（自回归生成）、
    tok_encode/tok_decode（分词/解码）等核心方法，使模型可被 evaluator 调用。
    参考 lm-eval-harness 的 BaseLM 基类设计。
    """

    def __init__(self, model, tokenizer, batch_size=1):
        super().__init__()

        assert isinstance(batch_size, int)

        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size

    @property
    def eot_token_id(self):
        """返回序列结束 (End-of-Text) token 的 ID"""
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        """返回模型支持的最大序列长度，优先从 config 读取，回退到 2048"""
        if hasattr(self.model.config, 'n_ctx'):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        else:
            return 2048

    @property
    def max_gen_toks(self):
        """生成任务的最大输出 token 数"""
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda"

    def tok_encode(self, string: str):
        """
        将输入字符串编码为 token ID 列表。

        参数:
            string: 待编码的输入文本。
        返回:
            不含特殊 token 的 token ID 列表。
        """
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        """
        将 token ID 序列解码回文本字符串。

        参数:
            tokens: token ID 列表。
        返回:
            解码后的文本字符串。
        """
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        对输入序列执行一次前向传播，返回每个位置的 logits。

        参数:
            inps: shape [batch, sequence] 的 torch tensor，序列长度可随调用变化。
        返回:
            shape [batch, sequence, vocab] 的 logits tensor。
        """
        with torch.no_grad():
            out = self.model(inps)[0]
            return out  # [:, :, :self.tokenizer.vocab_size]

    def _model_generate(self, context, max_length, eos_token_id):
        """
        自回归生成文本续写，使用贪心解码（do_sample=False）。

        参数:
            context: shape [batch, seq] 的上下文 token tensor。
            max_length: 最大生成长度。
            eos_token_id: 结束 token ID，遇到此 token 则停止生成。
        返回:
            续写后的完整 token 序列。
        """
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False
        )


def main():
    """
    Offsite-Tuning 模型评估主函数。

    执行流程:
        1. 解析命令行参数（模型路径、任务列表、输出目录等）。
        2. 以 float16 精度加载预训练语言模型。
        3. （可选）通过 uniform_choose_layers 选择学生层子集，实现 Offsite-Tuning 的层缩减策略。
        4. （可选）加载 Adapter 或 Student 权重状态字典。
        5. 将模型移至 CUDA，加载 tokenizer，包装为 LMEvalAdaptor。
        6. 使用 lm-eval-harness 的 evaluator 对指定任务进行评估。
        7. 打印结果表格，若指定输出目录则保存 JSON 结果。
    """
    args = parse_args()
    # 以 float16 精度加载模型，降低 GPU 显存占用
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float16)

    # Offsite-Tuning: 随机均匀选择 num_student_layers 层作为学生模型
    if args.num_student_layers is not None:
        layers = get_layers(model)
        layers = uniform_choose_layers(layers, args.num_student_layers)
        set_layers(model, layers)

    # 加载 Adapter 模块权重（Offsite-Tuning 的核心组件）
    if args.load_adapter:
        adapter_state_dict = torch.load(args.load_adapter, map_location='cpu')
        model = load_adapter(model, adapter_state_dict, args)

    # 加载完整的学生模型权重（可选的替代方案）
    if args.load_student:
        student_state_dict = torch.load(args.load_student, map_location='cpu')
        model = load_student(model, student_state_dict, args)

    model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    lm_eval_model = LMEvalAdaptor(model, tokenizer)

    # 确定评估任务列表：未指定则使用全量任务，否则按逗号分隔
    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")

    # 将任务名称通过 LM_EVAL_TASK_NAME_MAPPING 映射为 lm-eval-harness 标准任务名
    # 允许用户使用简写或 Offsite-Tuning 内部命名
    task_names = [LM_EVAL_TASK_NAME_MAPPING.get(t, t) for t in task_names]

    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=128,
        no_cache=True,
    )

    print(evaluator.make_table(results))

    # 保存评估结果到 JSON 文件
    if args.output_dir is not None:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        del results["config"]["model"]
        with open(args.output_dir, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
