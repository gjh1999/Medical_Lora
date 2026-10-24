# Prompt-only 意图识别与结构化抽取对比实验流程

## 目标

验证不做 LoRA 微调、只通过 prompt 让基础模型完成慢病管理场景下的意图识别和结构化信息提取时，效果和已微调 LoRA 模型相比如何。

本实验使用同一份测试集：

```text
data/synthetic/sft/test.jsonl
```

测试集每条样本包含：

- `system`: 路由与信息抽取任务说明
- `user`: 用户自然语言输入
- `assistant`: 标准 JSON 输出，作为评估标签

## 实验脚本

新增脚本：

```text
training/eval_prompt_baseline.py
```

该脚本只加载基础模型，不加载任何 LoRA adapter：

```text
Qwen2.5-7B-Instruct
```

默认本地模型路径：

```text
C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct
```

## Prompt 方案

脚本内置一个强约束 system prompt，要求模型：

- 只输出合法 JSON
- 不输出 Markdown、解释、代码块或多余文本
- 判断 5 个 routing flag：
  - `need_record`
  - `need_advice`
  - `need_plan`
  - `need_reminder`
  - `need_history`
- 抽取 `slots`
- 输出 `missing_fields`
- 输出 `risk_flags`
- 保持和训练数据一致的顶层 JSON schema

## 实验模式

### Zero-shot

只给任务说明和 JSON schema，不给示例。

运行命令：

```powershell
python training\eval_prompt_baseline.py --prompt-mode zero_shot --print-examples 0
```

### Few-shot

从 dev 集中选取若干条不同 routing 模式的样本作为示例，测试集仍然只用于评估，避免测试标签泄漏。

本次使用 5-shot：

```powershell
python training\eval_prompt_baseline.py --prompt-mode few_shot --shots 5 --print-examples 0
```

## 评估指标

脚本统计以下指标：

| 指标 | 含义 |
|---|---|
| JSON valid rate | 模型输出是否能被 `json.loads()` 解析 |
| Schema valid rate | 输出是否包含约定的顶层字段、routing 和 confidence 字段 |
| Routing exact rate | 5 个 routing flag 是否全部预测正确 |
| Slot exact rate | `slots` 是否和标签完全一致 |
| Multi-routing exact rate | 多意图样本中 routing 是否全部预测正确 |
| Per-flag Precision/Recall/F1 | 分别统计 5 个 routing flag 的分类指标 |
| Macro F1 | 5 个 routing flag 的 F1 平均值 |

## 输出文件

实验结果保存在：

```text
training/prompt_baseline_outputs/
```

本次生成的文件：

```text
summary_zero_shot_0shot.json
predictions_zero_shot_0shot.jsonl
summary_few_shot_5shot.json
predictions_few_shot_5shot.jsonl
```

其中：

- `summary_*.json`: 汇总指标
- `predictions_*.jsonl`: 每条样本的输入、标准答案、模型预测和原始输出

## 复现实验顺序

建议按以下顺序运行：

```powershell
# 1. 语法检查
python -m py_compile training\eval_prompt_baseline.py

# 2. 小样本 smoke test
python training\eval_prompt_baseline.py --prompt-mode zero_shot --limit 3 --print-examples 3

# 3. 完整 zero-shot
python training\eval_prompt_baseline.py --prompt-mode zero_shot --print-examples 0

# 4. 完整 few-shot
python training\eval_prompt_baseline.py --prompt-mode few_shot --shots 5 --print-examples 0
```

## 注意事项

- 该实验评估的是基础模型加 prompt 的能力，不涉及训练。
- 本次实验使用确定性生成：`do_sample=False`。
- prompt-only 的输出 JSON 合法率很高，但 routing 和 slots 的准确性仍需用标签评估。
- `Slot exact rate` 是严格完全匹配指标，较难达到高分；后续可增加字段级 slot F1 作为更细粒度指标。
