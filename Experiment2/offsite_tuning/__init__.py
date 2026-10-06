"""
Offsite-Tuning 包
=================
Offsite-Tuning: Transfer Learning without Full Model
论文: https://arxiv.org/abs/2302.04870

本包实现了一种隐私保护的迁移学习框架，允许数据拥有者在无法访问
完整大模型的情况下，仅使用轻量化的模拟器（emulator）和适配器
（adapter）来完成下游任务的微调。

核心模块：
  - utils: 教师-学生架构、KD loss、参数解析
  - run_clm: 语言模型训练脚本
  - run_image_classification: 视觉模型训练脚本
  - param_efficient: LoRA/Adapter/BitFit PEFT方法
  - data: 数据加载与预处理
  - tasks: 下游任务定义
  - eval_harness: 模型评估接口
"""
