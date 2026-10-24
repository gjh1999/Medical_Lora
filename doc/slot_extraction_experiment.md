# Slot 槽位抽取准确率对比实验

## 目标

比较三种方法在结构化信息提取中**槽位（Slot）抽取**的准确率：

1. **原版基础模型**（Base）：简短提示词 + Qwen2.5-7B-Instruct，无微调
2. **提示词工程**（Prompt）：详细 Schema 提示词（零样本）+ Qwen2.5-7B-Instruct
3. **LoRA 微调**（Fine-tuned）：简短提示词 + Qwen2.5-7B-Instruct + LoRA adapter

之前的实验只评估了路由（Routing）准确率，槽位抽取只有粗略的 `Slot Exact`（完全匹配），命中率仅 6.9%，不能反映真实的槽位抽取质量。本次实验新增**槽位粒度**的检测和填充指标。

## 测试数据

```text
data/synthetic/sft/test.jsonl  (131 条)
```

## 实验脚本

```text
training/eval_slot_extraction.py
```

脚本依次加载三种模型配置，在相同测试集上推理并对比。

## 评估指标

### 1. 槽位检测（Slot Presence Detection）

对 11 个叶子槽位逐一判断模型是否能正确区分「该填」和「不该填」：

| 槽位组 | 叶子槽位 |
|--------|----------|
| record | `blood_pressure`, `blood_glucose`, `weight`, `diet`, `exercise`, `medication`, `symptom` |
| advice | `advice`（topic/question 是否有值）|
| plan | `plan`（goal 是否有值）|
| reminder | `reminder`（task/action 是否有值）|
| history | `history`（metric 是否有值）|

每个叶子槽位计算 Precision / Recall / F1，汇总为 Micro F1 和 Macro F1。

### 2. 槽位值匹配（Slot Value Match）

对于检测正确的槽位（true 和 pred 都认为该填），评估填充质量：

- **Key Recall**：Ground Truth 子字段被模型包含的比例（不漏填）
- **Key Precision**：模型输出子字段是真实需要的比例（不幻觉）
- **Key F1**：上述两者的调和平均
- **Value Accuracy**：共有子字段值完全匹配的比例

### 3. 路由准确率（参考指标）

保留原有的 5 个 routing flag 的 Exact Match 和 Macro F1，作为参考对比。

## 输出文件

```text
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
# 1. 语法检查
python -m py_compile training\eval_slot_extraction.py

# 2. 小样本 smoke test (3 条)
python training\eval_slot_extraction.py --modes all --limit 3 --print-examples 3

# 3. 完整对比实验
python training\eval_slot_extraction.py --modes all --print-examples 0

# 4. 也可单独跑某一种方法
python training\eval_slot_extraction.py --modes lora --print-examples 0
```

## 注意事项

- 所有推理使用确定性生成（`do_sample=False`），保证可复现。
- 三种方法依次加载/卸载模型，避免 GPU 显存冲突。
- 基础模型可能输出不可解析的 JSON，这些样本会从槽位指标计算中排除（与之前实验一致）。
- `Slot Exact`（完全匹配）仍然极低是正常的——槽位中大量自由文本字段很难逐字命中，本次实验用更细粒度的检测 + 值匹配指标替代它。
- 槽位值匹配指标仅对「检测正确的槽位」计算，即 true 和 pred 都认为该槽位有值时才比较填充质量。
