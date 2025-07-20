# T5问答模型训练项目

## 项目概述
基于Mengzi-T5-base模型微调的问答系统，支持从JSON数据加载到模型训练、评估的全流程。主要功能包括：
- 数据预处理与自动分割（训练集70%/验证集20%/测试集10%）
- 混合精度训练（batch_size=1，学习率5e-5）
- 支持BLEU、精确匹配率（90%）等多维度评估
- 预测结果可视化与模型信息保存

## 环境依赖
```bash
Python 3.10+
PyTorch 2.6.0+ 
Transformers 4.30+
CUDA 11.8 (NVIDIA RTX 3090)
```

## 项目结构
<mcfile name="t5_base_train.ipynb" path="e:/Project/模型推理/t5_base_qa_model/t5_base_train.ipynb"></mcfile> 包含完整训练流程：
1. 数据加载与预处理（<mcsymbol name="preprocess_data" filename="t5_base_train.ipynb" path="e:/Project/模型推理/t5_base_qa_model/t5_base_train.ipynb" startline="249" type="function"></mcsymbol>）
2. Tokenization处理（<mcsymbol name="tokenize_data" filename="t5_base_train.ipynb" path="e:/Project/模型推理/t5_base_qa_model/t5_base_train.ipynb" startline="378" type="function"></mcsymbol>）
3. 训练参数配置（<mcsymbol name="setup_training_args" filename="t5_base_train.ipynb" path="e:/Project/模型推理/t5_base_qa_model/t5_base_train.ipynb" startline="462" type="function"></mcsymbol>）
4. 模型训练与评估（<mcsymbol name="evaluate_predictions" filename="t5_base_train.ipynb" path="e:/Project/模型推理/t5_base_qa_model/t5_base_train.ipynb" startline="1081" type="function"></mcsymbol>）

## 典型输出示例
```
📊 整体BLEU分数: 0.0000
🎯 精确匹配数量: 9/10 (90.00%)
💾 模型信息已保存到 model_info.json
```

## 使用指南
```python
# 加载训练好的模型
tokenizer, model, device = load_trained_model('./t5-qa-finetuned')

# 实时问答预测
answer = predict_answer(qa_model, 
                       context="微信支付每日转账限额...",
                       question="微信支付每日限额")
```

## 改进建议
1. 增加数据增强策略
2. 尝试t5-large等更大模型
3. 优化答案后处理规则