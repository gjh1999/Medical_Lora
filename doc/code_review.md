# Medical_Lora 代码审查报告

## 项目概述

基于 Qwen2.5-7B-Instruct + LoRA 的两个子项目：
- **子项目 A**: cMedQA 中文医学问答助手
- **子项目 B**: 慢病健康管理系统意图路由与信息抽取器（核心）

核心链路：
- 医学问答: `train_sft.py` → `finetune_qwen_lora.py` → `finetune_qwen_lora_test.py`
- 路由抽取: `scripts/data_generation/generate_data.py` → `run_pipeline.py` → `training/finetune_routing_lora.py` → `training/eval_*.py`

实验设计严谨（base vs prompt-only vs LoRA 三项对照），评估指标完善（路由F1 + 槽位检测F1 + 值匹配率），结果可信度高。代码整体质量不错，但有以下问题需修复。

---

## P0 — 严重安全/正确性问题

### 1. API Key 硬编码并已提交 Git

**文件**: `API_test1.py:6`

```python
api_key=os.environ["MOONSHOT_API_KEY"],
```

API Key 以明文写在代码中，且文件状态为 `A`（已 git add）。即使后续从代码中删除，仍可通过 `git log` 回溯获取。

**修复建议**:
- 立即到 Moonshot 后台 revoke 该 key
- 改用环境变量：`api_key=os.environ["MOONSHOT_API_KEY"]`
- 将 `API_test1.py` 加入 `.gitignore`
- 清理 git 历史：`git filter-branch` 或 `BFG Repo-Cleaner`

### 2. `clean_data.py` 原地覆盖输入文件，无备份

**文件**: `training/clean_data.py:88-89`

```python
src_dir = Path("data/synthetic/sft")
out_dir = Path("data/synthetic/sft")  # same as src_dir!
```

脚本原地覆盖 train/dev/test.jsonl，如果 `clean_output()` 有 bug（如意外清空某个字段），原始数据直接丢失且无法恢复。此外，该脚本使用相对路径，必须在项目根目录运行才能正常工作。

**修复建议**: 输出到独立目录（如 `data/synthetic/sft_cleaned/`），或至少先备份原文件。

### 3. `.idea/` 和 `__MACOSX/` 文件已加入 Git

IDE 配置文件（`.idea/`）和 macOS 元数据文件夹（`cMedQA/*/__MACOSX/`）被 `git add` 到了暂存区。这些文件不应进入版本控制。

**修复建议**:
- `git rm --cached -r .idea/` 和所有 `__MACOSX/` 目录
- 添加 `.idea/` 和 `__MACOSX/` 到 `.gitignore`

---

## P1 — 高优先级

### 4. 缺少 max_seq_length，存在 OOM 风险

**文件**: `finetune_qwen_lora.py:144`, `training/finetune_routing_lora.py:84`

两个训练脚本均未设置 `max_seq_length`。SFTTrainer 默认使用模型最大长度（Qwen2.5-7B 为 131072），遇到长样本会 OOM。路由数据集的系统 prompt 按 3KB 计算，目前安全，但医学问答数据（cMedQA）的回答可能很长。

**修复建议**: 显式设置 `max_seq_length=2048`。

### 5. bf16=False 浪费训练效率

**文件**: `finetune_qwen_lora.py:126`, `training/finetune_routing_lora.py:77`

RTX 5080 支持 bf16，且 `bnb_4bit_compute_dtype` 已设为 `torch.bfloat16`。`bf16=False` 意味着 TrainingArguments 层面不用 bf16，浪费算力且与 compute dtype 不一致。

**修复建议**: 改为 `bf16=True`。

### 6. `deduplicate.py` 只对 `input` 去重，忽略 `output`

**文件**: `scripts/data_generation/deduplicate.py:42`

```python
inp = sample.get("input", "")
if inp in seen_inputs:
    exact_dup_count += 1
```

只检查 `input` 字段是否重复。如果两个样本 user input 相同但 output 不同（LLM 生成数据时常见），会导致训练标签冲突。此外，如果两个样本 input 不同但 output 完全相同，也会产生重复训练信号。

**修复建议**: 同时对 `input + json.dumps(output)` 做 exact dedup，或至少对 output 也做 fuzzy dedup。

### 7. `run_pipeline.py` Step 编号重复

**文件**: `scripts/data_generation/run_pipeline.py`

`step_normalize_slots()` 和 `step2_validate_jsonl()` 都打印 "STEP 2"。Normalize 步骤是后来插入的，但未更新后续步骤的编号。虽然不影响功能，但日志混乱。

**修复建议**: 重编号为 STEP 1-9，让每个 step 函数打印唯一的步骤号。

### 8. `validate_routing_slots.py` 依赖 raw 格式字段名

**文件**: `scripts/data_generation/validate_routing_slots.py:83-84`

```python
output = sample.get("output", {})
user_input = sample.get("input", "")
```

该脚本假设数据格式为 `{input, output}`，但最终 SFT 格式为 `{messages: [{role, content}, ...]}`。目前 pipeline 在 SFT 转换前运行验证，所以能正常工作，但这也意味着 **SFT 转换后的数据从未被验证过**。如果 `convert_to_sft.py` 或 `clean_data.py` 引入错误，不会被检测到。

**修复建议**: 增加对 SFT 格式数据的验证步骤，或至少验证转换前后的样本数量一致。

---

## P2 — 中优先级

### 9. 硬编码绝对路径，不可移植

涉及 5 个文件：

| 文件 | 行号 |
|------|------|
| `finetune_qwen_lora.py` | 15 |
| `finetune_qwen_lora_test.py` | 6 |
| `finetune_qwen_lora_base_test.py` | 4 |
| `test1.py` | 4 |
| `training/finetune_routing_lora.py` | 14 |
| `training/eval_routing_lora.py` | 10 |
| `training/eval_prompt_baseline.py` | 17 |
| `training/eval_slot_extraction.py` | 27 |

路径形式：`C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct`。换机器必炸。

**修复建议**: 统一从环境变量 `MODEL_PATH` 读取，或使用 project root 相对路径下的符号链接。

### 10. LoRA target_modules 过于保守

**文件**: `finetune_qwen_lora.py:107`, `training/finetune_routing_lora.py:58`

```python
target_modules=["q_proj", "v_proj"],
```

只训练 Q/V 投影。注释掉的配置包含了 `k_proj, o_proj, gate_proj, up_proj, down_proj` 共 7 个模块。路由任务 Macro F1 已达 0.990，瓶颈可能不在模型容量，但 `need_advice` Precision 0.908（多标）可能通过更多可训参数改善。

**修复建议**: 至少加上 `k_proj, o_proj`；若显存允许（RTX 5080 17GB），尝试全量 7 模块。

### 11. 学习率偏高，缺少显式 warmup

**文件**: `finetune_qwen_lora.py:118`, `training/finetune_routing_lora.py:69`

```python
learning_rate=2e-4,
```

LoRA 微调 7B 模型的典型学习率是 `5e-5` ~ `1e-4`。`2e-4` 配合 `r=8` 的小 rank 容易过拟合。代码中未设置 `warmup_steps` 或 `warmup_ratio`（结果文档提到有 "cosine + 10% warmup"，但代码中是 SFTTrainer 的默认行为，不是显式设置的）。

**修复建议**: 降至 `1e-4`，显式添加 `warmup_ratio=0.1`，并确认 lr_scheduler_type 为 cosine。

### 12. `eval_slot_extraction.py` 和 `compute_slot_metrics_offline.py` 大量代码重复

两个文件中的 `is_slot_present`、`get_leaf_value`、`compare_slot_values`、`compute_slot_detection_metrics`、`compute_slot_value_metrics`、`compute_routing_metrics` 函数几乎一字不差地重复。维护其中一个时容易忘记同步更新另一个。

**修复建议**: 抽取公共评估逻辑到 `training/metrics.py`。

### 13. `train_sft.py` 对 train 集静默依赖

**文件**: `train_sft.py:73-74`

```python
if "label" in candidates.columns:
    candidates = candidates[candidates["label"] == 1]
```

train 集没有 `label` 列，逻辑恰好跳过。但隐含假设"train 集全是正样本"，换数据格式时可能静默丢数据。

**修复建议**: 加 `assert "label" not in candidates.columns or ...` 显式处理。

### 14. 缺少 `requirements.txt` 或 `pyproject.toml`

项目没有任何依赖声明文件。当前依赖全部在 `.venv/` 中，但 `.venv/` 不应提交 git。其他人复现时需要反向推断依赖。

**修复建议**: 添加 `requirements.txt`（至少包含 torch、transformers、datasets、peft、trl、bitsandbytes、accelerate、pandas、openai、python-dotenv）。

---

## P3 — 低优先级 / 改进建议

### 15. instruction 字段冗余拼接浪费 token

**文件**: `finetune_qwen_lora.py:79`

```python
"content": example["instruction"] + "\n\n" + example["input"],
```

`instruction` 对所有样本完全相同，每条约 20 tokens × 5000 = 10 万 tokens 浪费。建议放入 system prompt，user 只放 `input`。

### 16. test1.py 非 raw string 路径

**文件**: `test1.py:4`

```python
model_path = "C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"
```

`\g`、`\m`、`\Q` 等不是合法转义序列，Python 按字面处理靠运气。应改为 raw string `r"..."`。

### 17. test1.py 全精度加载模型

不加任何量化的全精度加载，7B 模型约需 28GB 显存，RTX 5080（17GB）极易 OOM。

### 18. 大文件提交 Git 会持续膨胀仓库

`cmedqa_train_sft.json` (~58MB)、`medquad.csv` (~22.8MB)、`cmedqa_*_sft.jsonl` 等数据文件已 git add。即使后续 `.gitignore`，历史中也会永久保留。建议使用 Git LFS 或仅保留生成脚本，不提交生成后的数据文件。

### 19. 缺少单元测试

数据 pipeline（validate、normalize、dedup、split、convert）无任何单元测试。建议至少覆盖：
- `extract_jsonl_objects` 对各种 LLM 输出格式的鲁棒性
- `clean_output` 对各类 schema 不一致的修复正确性
- dedup 对边界条件的处理

---

## 数据质量问题

### cMedQA 数据（来自抽样）

| 问题 | 示例 |
|------|------|
| 非专业回答 | "我无法判断你是不是坐骨神经损伤" — 等于没回答 |
| 矛盾回答 | 同一问题 "流产多次又怀孕了怎么办" 出现两次，答案互相矛盾 |
| 格式不统一 | 有的以 "指导意见：" 开头，有的以 "你好" 开头 |
| 回答过短 | 部分回答不足 30 字，信息量低 |

### 合成路由数据

- 数据 pipeline 整体设计合理，9 步处理保证了质量
- `clean_data.py` 修复了 routing/confidence/slots 的 schema 不一致，统计：train 从 905 → 1068（修复了 163 条），dev 从 114 → 135（+21），test 从 112 → 131（+19）。样本数增加可能是因为 `clean_output` 为缺失字段补了默认值，从而使之前被过滤的样本重新可用
- SFT 转换后的数据未经验证（见 P1-8）

---

## 架构与实验结果评价

### 实验设计（优点）

1. **三项对照实验严谨**: base model → prompt-only (zero-shot + few-shot) → LoRA fine-tuned，层层递进，清楚展示了 prompt engineering 和 fine-tuning 各自的贡献
2. **评估指标全面**: 路由精确率/召回率/F1 + 槽位检测F1（11个叶子槽位）+ 槽位值匹配率（键精确率/召回率 + 值准确率）
3. **离线评估脚本** (`compute_slot_metrics_offline.py`) 方便无 GPU 环境复算指标
4. **危险内容检测** 在数据 pipeline 中集成了关键词过滤

### 实验结果（路由任务）

| 方法 | Macro F1 | JSON 合法率 |
|------|----------|------------|
| Base model | 0.000 | 0% |
| Prompt-only (zero-shot) | 0.842 | ~87% |
| Prompt-only (5-shot) | 0.868 | ~88% |
| **LoRA fine-tuned** | **0.990** | **100%** |

结论明确：对该自定义 schema 的结构化输出任务，prompt engineering 能解决 80%+，但要达到生产级可用（100% JSON 合法、F1 > 0.98），fine-tuning 是必需的。

### 论文级改进方向

- few-shot 的 dev examples 是通过 routing pattern 多样性挑选的（每种 pattern 选一条），但未控制 example 的 input 风格多样性。可考虑按 user input 长度、复杂度分层采样
- 目前实验用的 RTX 5080（单卡）。如有多卡，可尝试 DDP 或更大的 rank

---

## 总结

| 优先级 | 数量 | 关键问题 |
|--------|------|----------|
| P0 | 3 | API Key 泄露、clean_data 原地覆盖、IDE/系统文件入 Git |
| P1 | 5 | OOM 风险、bf16 未启用、数据验证盲区、dedup 只查 input、step 编号重复 |
| P2 | 6 | 绝对路径、LoRA 保守、学习率偏高、代码重复、隐式假设、无依赖文件 |
| P3 | 5 | token 浪费、raw string、全精度加载、大文件膨胀、无单元测试 |

**项目的核心亮点**：实验设计严谨、评估体系完整、数据 pipeline 考虑了多种边界情况（危险内容过滤、routing-slot 一致性校验、分层采样）。代码整体功能正确，主要问题集中在安全性（API Key）、可移植性（硬编码路径）和工程健壮性（原地覆盖、无备份、无测试）上。
