# 路由数据集 LoRA 微调运行计划

## 1. 环境

| 项目 | 值 |
|------|-----|
| GPU | RTX 5080 / 17GB VRAM / CUDA |
| 模型 | Qwen2.5-7B-Instruct（本地缓存） |
| 量化 | 4-bit QLoRA（NF4） |
| 框架 | transformers 5.6.2 / peft 0.19.1 / trl 1.2.0 |

## 2. 数据

来源：`data/synthetic/sft/`，由 `data/synthetic/raw/` 经流水线生成，再用 `training/clean_data.py` 修复 schema 不一致。

格式：ChatML messages，assistant.content 为字符串化 JSON。

| 集合 | 样本数 |
|------|--------|
| train | 1,068 |
| dev | 135 |
| test | 131 |
| **总计** | **1,334** |

已通过检查：JSON 合法率 100%、schema 统一、split 间无泄漏、无重复。

## 3. 目录结构

```text
training/
  clean_data.py            # 数据修复脚本（已执行）
  finetune_routing_lora.py # 训练脚本
  eval_routing_lora.py     # 评估脚本
```

## 4. 执行步骤

### 步骤 1：训练

```bash
cd training
python finetune_routing_lora.py
```

| 参数 | 值 |
|------|-----|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Target modules | q_proj, v_proj |
| Epochs | 3 |
| Batch size | 1 × grad_accum 4 = 有效 4 |
| Learning rate | 2e-4 |
| LR scheduler | cosine + 10% warmup |
| Max length | 1024 |
| 输出目录 | `training/qwen_routing_lora_final/` |

预计训练时间：10~15 分钟。

### 步骤 2：评估

```bash
cd training
python eval_routing_lora.py
```

指标：

| 指标 | 目标 |
|------|------|
| JSON 合法率 | ≥ 98% |
| Routing 准确率（每 flag 的 f1） | ≥ 90% |
| 多路由样本准确率 | ≥ 85% |
| 危险输出率 | 0% |

### 步骤 3：调参（如不达标）

- loss 不收敛 → 提高 lr 到 3e-4 或增加 epochs 到 5
- JSON 格式错误 → 增加 epochs 或增大 rank 到 16
- 过拟合 → 增大 lora_dropout 到 0.1

## 5. 产物

- `training/finetune_routing_lora.py` — 训练脚本
- `training/eval_routing_lora.py` — 评估脚本
- `training/qwen_routing_lora_final/` — 训练好的 LoRA adapter
