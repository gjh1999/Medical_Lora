# 路由数据集 LoRA 微调结果

## 训练

| 项目 | 值 |
|------|-----|
| 模型 | Qwen2.5-7B-Instruct |
| 方法 | 4-bit QLoRA (NF4) |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Target modules | q_proj, v_proj |
| 训练集 | 1,068 条 |
| 验证集 | 135 条 |
| Epochs | 3 |
| 有效 batch size | 4 |
| Learning rate | 2e-4, cosine + 10% warmup |
| 训练时间 | 20 分 36 秒（RTX 5080） |
| Train loss | 1.865 → 0.130 |
| Eval loss | 0.174 |
| Eval token accuracy | 95.8% |

## 评估

测试集 131 条，结果如下。

### 核心指标

| 指标 | 值 | 目标 | 达成 |
|------|-----|------|------|
| JSON 合法率 | 100.0% | ≥ 98% | ✓ |
| 多路由样本准确率 | 100.0% (76/76) | ≥ 85% | ✓ |
| 解析失败 | 0 条 | 0 条 | ✓ |

### 各 Flag 指标

| Flag | Precision | Recall | F1 |
|------|-----------|--------|-----|
| need_record | 1.000 | 1.000 | 1.000 |
| need_advice | 0.908 | 1.000 | 0.952 |
| need_plan | 1.000 | 1.000 | 1.000 |
| need_reminder | 1.000 | 1.000 | 1.000 |
| need_history | 1.000 | 1.000 | 1.000 |
| **Macro avg** | **0.982** | **1.000** | **0.990** |

### 分析

- 4 个 flag 满分，`need_advice` 偶尔多标（precision 0.908），但召回率 100%，即从不漏标真正的建议请求
- 多标影响很小：下游 advice 节点收到多余请求只会给出"建议"，不会产生危险操作
- 消除了解析失败问题，模型输出的 JSON 100% 可用

## 与原版模型对比

| 指标 | 原版 Qwen2.5-7B | LoRA 微调后 | 提升 |
|------|----------------|-------------|------|
| JSON 合法率 | 96.9% | **100.0%** | +3.1pp |
| need_record F1 | 0.000 | **1.000** | +1.000 |
| need_advice F1 | 0.000 | **0.952** | +0.952 |
| need_plan F1 | 0.000 | **1.000** | +1.000 |
| need_reminder F1 | 0.000 | **1.000** | +1.000 |
| need_history F1 | 0.000 | **1.000** | +1.000 |
| Macro avg F1 | 0.000 | **0.990** | +0.990 |
| 多路由样本准确率 | 0.0% | **100.0%** | +100pp |
| 解析失败 | 4 条 | **0 条** | — |

原版模型完全不理解路由 schema——不是答错了，而是根本不知道要输出什么格式的 JSON。例如：

- 用户说"饮食方案" → 原版输出了完整的饮食计划 JSON（不是路由）
- 用户说"设提醒" → 原版把记录和提醒混在一个输出里
- 原版甚至偶尔输出英文 JSON key，格式五花八门

LoRA 微调后模型完全掌握了 5-flag 路由 schema，输出格式稳定一致。

## 产物

```
training/
  clean_data.py              # 数据修复脚本
  finetune_routing_lora.py   # 训练脚本
  eval_routing_lora.py       # 评估脚本
  qwen_routing_lora_final/   # LoRA adapter（可直接加载推理）
```
