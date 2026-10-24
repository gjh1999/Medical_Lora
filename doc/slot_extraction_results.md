# 槽位抽取准确率对比实验结果

测试集: `data/synthetic/sft/test.jsonl`（131 条）

基础模型: Qwen2.5-7B-Instruct

## 总体对比

| 方法 | 描述 | JSON 合法率 | 槽位检测 Macro F1 | 槽位检测 Micro F1 | Key F1 | Value Acc | 路由 Exact | 路由 Macro F1 |
|------|------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| Base (short prompt) | 简短提示词 + 无微调 | 97.7% | 0.000 | 0.000 | — | — | 11.7% | 0.000 |
| Prompt (zero-shot) | 详细 Schema 提示词 + 无微调 | 100.0% | 0.698 | 0.775 | 92.1% | 21.1% | 52.7% | 0.842 |
| **LoRA fine-tuned** | 简短提示词 + LoRA adapter | **100.0%** | **0.975** | **0.978** | **97.0%** | **70.9%** | **95.4%** | **0.990** |

## 槽位检测详细指标

每个叶子槽位的 presence/absence 二分类 F1：

| 槽位 | Base (short prompt) | Prompt (zero-shot) | LoRA fine-tuned |
|------|:----:|:----:|:----:|
| record.blood_pressure | 0.000 | 0.857 | **0.968** |
| record.blood_glucose | 0.000 | 0.500 | **1.000** |
| record.weight | 0.000 | 0.800 | **1.000** |
| record.diet | 0.000 | 0.778 | **0.947** |
| record.exercise | 0.000 | 0.000 | **0.889** |
| record.medication | 0.000 | 0.714 | **1.000** |
| record.symptom | 0.000 | 0.583 | **0.970** |
| advice.advice | 0.000 | 0.642 | **0.952** |
| plan.plan | 0.000 | 0.957 | **1.000** |
| reminder.reminder | 0.000 | 1.000 | **1.000** |
| history.history | 0.000 | 0.847 | **1.000** |
| **Macro F1** | **0.000** | **0.698** | **0.975** |

## 槽位值匹配指标

对于检测正确的槽位（true 和 pred 都认为该填），评估填充质量：

| 指标 | Prompt (zero-shot) | LoRA fine-tuned | 提升 |
|------|:----:|:----:|:----:|
| Key Recall（子字段不漏填） | 89.0% | 97.4% | +8.4pp |
| Key Precision（子字段不幻觉） | 95.3% | 96.6% | +1.3pp |
| Key F1 | 92.1% | 97.0% | +4.9pp |
| Value Accuracy（值完全匹配） | 21.1% | 70.9% | +49.8pp |

## 路由指标（参考）

| Flag | Prompt (zero-shot) F1 | LoRA fine-tuned F1 |
|------|:----:|:----:|
| need_record | 0.686 | **1.000** |
| need_advice | 0.648 | **0.952** |
| need_plan | 0.957 | **1.000** |
| need_reminder | 1.000 | **1.000** |
| need_history | 0.921 | **1.000** |
| **Macro F1** | **0.842** | **0.990** |

路由完全匹配（5 个 flag 全部正确）：Prompt 69/131 = 52.7% → **LoRA 125/131 = 95.4%**

## LoRA 错误分析

LoRA 微调模型在 131 条样本中共 6 条路由错误，全部是 `need_advice` 多标（FP=6），零漏标（FN=0）。

槽位检测方面共 10 个 FP，分布在：
- `advice.advice`（6 个 FP）：模型倾向于对含健康描述的输入额外触发 advice 槽位
- `record.diet`（1 个 FP）、`record.exercise`（1 个 FP）、`record.symptom`（1 个 FP）：少量幻觉

`need_advice` 的多标影响可控 —— 下游 advice 节点收到多余请求只会给出通用建议，不会产生危险操作。

## 结论

1. **基础模型完全无法完成槽位抽取任务**：虽然 JSON 合法率有 97.7%，但所有槽位的 F1 均为 0，说明模型输出内容与预期 schema 完全不匹配。

2. **详细提示词显著改善了输出格式**：Prompt-only (zero-shot) 将 JSON 合法率拉满到 100%，槽位检测也有了一定水平（Macro F1=0.698）。但值准确率仅 21.1%，说明模型能猜对"该填什么槽位"，却填不准具体值。

3. **LoRA 微调在所有维度上最优**：
   - 槽位检测 Macro F1 从 0.698 → 0.975（+40%）
   - 值准确率从 21.1% → 70.9%（+236%）
   - 路由 Exact 从 52.7% → 95.4%（+81%）
   - `need_record` 和 `need_history` 的槽位检测从不及格提升到满分

4. **实际建议**：LoRA 微调是生产部署的推荐方案。Prompt-only 可作为快速冷启动 baseline 或 LoRA 训练数据的自动标注辅助。

## 输出文件

```
training/slot_eval_outputs/
  slot_summary_base_short_prompt.json
  slot_predictions_base_short_prompt.jsonl
  slot_summary_prompt_zero-shot.json
  slot_predictions_prompt_zero-shot.jsonl
  slot_summary_lora_fine-tuned.json
  slot_predictions_lora_fine-tuned.jsonl
```

## 复现命令

```powershell
# 单方法评估
python training\eval_slot_extraction.py --modes lora --print-examples 3

# 完整三方法对比
python training\eval_slot_extraction.py --modes all --print-examples 0
```
