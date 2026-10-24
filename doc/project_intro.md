# Medical_Lora 项目介绍

## 一句话概述

基于 **Qwen2.5-7B-Instruct** 大语言模型，使用 **LoRA** 微调技术在 **cMedQA 中文医学问答数据集** 上进行领域适配，构建中文医学问答助手。

---

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 基座模型 | Qwen2.5-7B-Instruct（通义千问） |
| 微调方法 | LoRA（Low-Rank Adaptation） |
| 量化 | 4-bit NF4（BitsAndBytes） |
| 训练框架 | HuggingFace Transformers + TRL (SFTTrainer) |
| PEFT 库 | PEFT (Parameter-Efficient Fine-Tuning) |
| 数据集 | cMedQA（中文医学问答） |
| 硬件 | NVIDIA RTX 5080 |
| 模型下载 | ModelScope |

---

## 项目结构

```
Medical_Lora/
├── train_sft.py                      # 数据预处理：cMedQA → SFT 格式
├── finetune_qwen_lora.py             # LoRA 训练主脚本
├── finetune_qwen_lora_test.py        # LoRA 模型推理测试
├── finetune_qwen_lora_base_test.py   # 基座模型推理对比
├── test1.py                          # 模型加载冒烟测试
├── test2.py                          # GPU 环境检测
├── API_test1.py                      # Moonshot API 测试（无关）
├── cMedQA/                           # 原始数据集
│   ├── questions.csv/                # 问题库
│   ├── answers.csv/                  # 答案库
│   ├── train_candidates.txt/         # 训练集候选匹配
│   ├── dev_candidates.txt/           # 验证集候选匹配
│   └── test_candidates.txt/          # 测试集候选匹配
├── cmedqa_train_sft.jsonl            # SFT 格式训练集（5000条）
├── cmedqa_dev_sft.jsonl              # SFT 格式验证集（500条）
├── cmedqa_test_sft.jsonl             # SFT 格式测试集
├── qwen_medical_lora/                # 训练检查点
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── checkpoint-1250/
└── qwen_medical_lora_final/          # 最终 LoRA 模型
    ├── adapter_model.safetensors     # LoRA 权重（5 MB）
    ├── adapter_config.json           # LoRA 配置
    └── tokenizer.json                # 分词器
```

---

## 核心流程

### 1. 数据预处理（`train_sft.py`）

将 cMedQA 原始数据转换为 SFT（Supervised Fine-Tuning）格式：

```
原始格式:
  questions.csv:    question_id → question_text
  answers.csv:      answer_id   → answer_text
  candidates.txt:   question_id → answer_id (label=1 为正样本)

输出格式:
  {"instruction": "请根据患者描述进行医学分析...",
   "input":       "<患者问题>",
   "output":      "<医生回答>"}
```

处理步骤：加载 CSV → 列名清洗 → ID 统一 → 合并问题/答案 → 过滤短文本 → 去重 → 输出 JSONL。

### 2. LoRA 训练（`finetune_qwen_lora.py`）

- 加载 Qwen2.5-7B-Instruct 基座模型（4-bit NF4 量化）
- 应用 Qwen2.5 Chat Template 格式化数据
- 配置 LoRA：`r=8, alpha=16`，目标模块 `[q_proj, v_proj]`
- 训练参数：5000 条数据，1 epoch，batch_size=4（1×4 梯度累积）
- 每 500 步保存检查点，最终模型保存到 `qwen_medical_lora_final/`

### 3. 推理测试（`finetune_qwen_lora_test.py` / `finetune_qwen_lora_base_test.py`）

分别加载 LoRA 模型和基座模型，对同一医学问题生成回答，对比微调效果。

---

## 训练配置

| 参数 | 值 |
|------|-----|
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| 目标模块 | q_proj, v_proj |
| 学习率 | 2e-4 |
| 训练轮数 | 1 |
| 有效 batch size | 4 |
| 量化 | 4-bit NF4 + 双重量化 |
| 计算精度 | bfloat16 |
| 梯度检查点 | 开启 |
| 训练样本 | 5000 |
| 验证样本 | 500 |

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.x
- NVIDIA GPU（建议 ≥16GB 显存）

### 安装依赖

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

### 运行

```bash
# 1. 数据预处理
python train_sft.py

# 2. LoRA 训练
python finetune_qwen_lora.py

# 3. 推理测试
python finetune_qwen_lora_test.py
```

---

## LoRA 模型产物

最终模型仅 **5 MB**（仅 adapter 权重），不包含 7B 基座模型。使用时需同时加载：

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(base_model_path, ...)
lora_model = PeftModel.from_pretrained(base_model, "./qwen_medical_lora_final")
```

---

## 已知问题

详见 `doc/code_review.md`，关键问题包括：

- 学习率偏高（2e-4），建议降至 1e-4
- LoRA 目标模块仅 2 个（q_proj, v_proj），可扩展以获得更好效果
- 缺少 max_seq_length 设置
- 训练数据存在矛盾回答和低质量样本
