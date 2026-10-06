"""
CLIP ViT 图像分类器包装模块。
将 HuggingFace 的 CLIPVisionTransformer 包装为图像分类模型（arXiv:2302.04870），
支持回归（num_labels=1）、单标签分类和多标签分类三种任务类型。
"""
import torch
from typing import Dict, List, Optional, Set, Tuple, Union

import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionTransformer

from transformers.modeling_outputs import ImageClassifierOutput


class CLIPViTForImageClassification(nn.Module):
    """
    CLIP ViT 图像分类器。

    架构: CLIPVisionTransformer (ViT 编码器) → pooler_output 特征提取 → nn.Linear 分类头。
    支持三种 problem_type:
        - "regression": 回归任务 (MSE 损失)，num_labels == 1
        - "single_label_classification": 单标签分类 (CrossEntropy 损失)
        - "multi_label_classification": 多标签分类 (BCEWithLogits 损失)
    """

    def __init__(self, config, vit: CLIPVisionTransformer):
        super().__init__()
        self.vit = vit
        # 分类头: hidden_size → num_labels 的全连接层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config
        self.num_labels = config.num_labels

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ViT 编码器前向传播，提取视觉特征
        outputs = self.vit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取 pooler_output: outputs[1] 对应 CLIPVisionTransformer 输出的第二个元素
        # (即 pooled_output, shape [batch_size, hidden_size])
        pooler_output = outputs[1]

        # 分类头: 将 pooler 特征映射为各类别的 logits
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            # 自动检测任务类型 (problem_type):
            # - num_labels == 1 且不在字典中 → 回归任务
            # - labels 为整数类型 (long/int) → 单标签分类
            # - labels 为浮点类型 → 多标签分类 (每个标签有独立概率)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 回归损失: 均方误差 (MSE)，用于连续值预测
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            # 单标签分类损失: 交叉熵 (CrossEntropy)，每个样本属于一个类别
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            # 多标签分类损失: BCEWithLogitsLoss，每个样本可同时属于多个类别
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
