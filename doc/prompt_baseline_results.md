# Prompt-only 与 LoRA 微调方案对比结果

## 实验设置

测试集：

```text
data/synthetic/sft/test.jsonl
```

样本数：

```text
131
```

基础模型：

```text
Qwen2.5-7B-Instruct
```

对比方案：

1. **Prompt zero-shot**：不微调，只给任务说明和 JSON schema。
2. **Prompt 5-shot**：不微调，在 prompt 中加入 5 条 dev 集示例。
3. **LoRA 微调方案**：加载 `training/qwen_routing_lora_final/` 中的 LoRA adapter。

## 总体对比

| 方案 | JSON 合法率 | Schema 合法率 | Routing Exact | Slot Exact | Multi-routing Exact | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Prompt zero-shot | 100.0% | 100.0% | 52.7% | 6.9% | 35.5% | 0.842 |
| Prompt 5-shot | 100.0% | 100.0% | 50.4% | 6.9% | 43.4% | 0.868 |
| LoRA 微调 | 100.0% | 未单独统计 | 约 100.0% | 未单独统计 | 100.0% | 0.990 |

说明：

- LoRA 的结果来自既有评估文档 `doc/finetune_results.md`。
- LoRA 文档中未单独统计 `Schema 合法率` 和 `Slot Exact`，但报告了 JSON 合法率、多路由准确率和每个 routing flag 的 F1。
- Prompt-only 新脚本额外统计了 `Schema 合法率` 和严格 `Slot Exact`。

## Prompt zero-shot 结果

运行命令：

```powershell
python training\eval_prompt_baseline.py --prompt-mode zero_shot --print-examples 0
```

结果：

| 指标 | 数值 |
|---|---:|
| JSON 合法率 | 131/131 = 100.0% |
| Schema 合法率 | 131/131 = 100.0% |
| Routing Exact | 69/131 = 52.7% |
| Slot Exact | 9/131 = 6.9% |
| Multi-routing Exact | 27/76 = 35.5% |
| Parse errors | 0 |

各 routing flag 指标：

| Flag | Precision | Recall | F1 |
|---|---:|---:|---:|
| need_record | 0.921 | 0.547 | 0.686 |
| need_advice | 0.739 | 0.576 | 0.648 |
| need_plan | 0.917 | 1.000 | 0.957 |
| need_reminder | 1.000 | 1.000 | 1.000 |
| need_history | 1.000 | 0.853 | 0.921 |
| Macro avg | 0.915 | 0.795 | 0.842 |

## Prompt 5-shot 结果

运行命令：

```powershell
python training\eval_prompt_baseline.py --prompt-mode few_shot --shots 5 --print-examples 0
```

结果：

| 指标 | 数值 |
|---|---:|
| JSON 合法率 | 131/131 = 100.0% |
| Schema 合法率 | 131/131 = 100.0% |
| Routing Exact | 66/131 = 50.4% |
| Slot Exact | 9/131 = 6.9% |
| Multi-routing Exact | 33/76 = 43.4% |
| Parse errors | 0 |

各 routing flag 指标：

| Flag | Precision | Recall | F1 |
|---|---:|---:|---:|
| need_record | 0.820 | 0.781 | 0.800 |
| need_advice | 0.631 | 0.695 | 0.661 |
| need_plan | 0.880 | 1.000 | 0.936 |
| need_reminder | 1.000 | 1.000 | 1.000 |
| need_history | 0.917 | 0.971 | 0.943 |
| Macro avg | 0.849 | 0.889 | 0.868 |

## LoRA 微调结果

LoRA 结果来自原有评估：

```text
doc/finetune_results.md
```

关键指标：

| 指标 | 数值 |
|---|---:|
| JSON 合法率 | 100.0% |
| 多路由样本准确率 | 100.0% (76/76) |
| Parse errors | 0 |
| Macro F1 | 0.990 |

各 routing flag 指标：

| Flag | Precision | Recall | F1 |
|---|---:|---:|---:|
| need_record | 1.000 | 1.000 | 1.000 |
| need_advice | 0.908 | 1.000 | 0.952 |
| need_plan | 1.000 | 1.000 | 1.000 |
| need_reminder | 1.000 | 1.000 | 1.000 |
| need_history | 1.000 | 1.000 | 1.000 |
| Macro avg | 0.982 | 1.000 | 0.990 |

## 结论

Prompt-only 方案可以很好地解决“输出格式”问题。本次 zero-shot 和 5-shot 的 JSON 合法率、Schema 合法率都达到 100%，说明基础模型在强 schema prompt 下能够稳定输出可解析 JSON。

但 prompt-only 的意图识别效果明显低于 LoRA 微调：

- Zero-shot 的 Routing Exact 只有 52.7%。
- 5-shot 的 Routing Exact 为 50.4%，没有超过 zero-shot。
- 5-shot 的 Macro F1 从 0.842 提升到 0.868，但仍低于 LoRA 的 0.990。
- 多意图样本是 prompt-only 的主要短板：zero-shot 为 35.5%，5-shot 为 43.4%，LoRA 为 100.0%。
- Slot Exact 只有 6.9%，说明即使 routing 大体正确，结构化字段和值也经常和标准答案不完全一致。

因此，本项目场景下可以得出当前结论：

```text
Prompt-only 适合作为快速 baseline 或冷启动方案；
LoRA 微调明显更适合作为稳定生产方案，尤其适合多意图路由和固定 schema 输出。
```

## 后续建议

后续可以继续补两个实验，让结论更完整：

1. 增加字段级 slot F1，而不是只看严格 `Slot Exact`。
2. 设计更强的 few-shot 示例选择策略，例如每种 routing 组合固定覆盖，而不是从 dev 集顺序选取。
