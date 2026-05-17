# 慢病管理意图路由数据集

## 1. 概述

本数据集用于 LoRA 微调 Qwen-7B，使其成为慢病健康管理系统的路由与信息抽取模块。

模型输入是用户的自然语言（如"我今天早上血压145/92，帮我记一下"），输出是结构化 JSON，包含路由开关、置信度、结构化槽位、缺失字段标记和风险标记。

数据集通过大模型（DeepSeek V4）生成合成数据，经自动校验、规范化、去重后产出。

## 2. 数据格式

采用标准 messages 格式，兼容 LLaMA-Factory、Unsloth、Firefly 等主流框架：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是慢病健康管理系统的路由与信息抽取器。你只能输出合法 JSON，不要输出解释。"
    },
    {
      "role": "user",
      "content": "我今天早上血压145/92，帮我记一下"
    },
    {
      "role": "assistant",
      "content": "{\"routing\":{\"need_record\":true,\"need_advice\":false,\"need_plan\":false,\"need_reminder\":false,\"need_history\":false},\"confidence\":{\"need_record\":0.97,\"need_advice\":0.08,\"need_plan\":0.03,\"need_reminder\":0.04,\"need_history\":0.02},\"slots\":{\"record\":{\"blood_pressure\":{\"systolic\":145,\"diastolic\":92,\"unit\":\"mmHg\",\"time\":\"今天早上\"},\"blood_glucose\":null,\"weight\":null,\"diet\":null,\"exercise\":null,\"medication\":null,\"symptom\":null},\"advice\":{\"topic\":null,\"question\":null,\"related_metrics\":[]},\"plan\":{\"goal\":null,\"duration\":null,\"preferences\":[],\"constraints\":[]},\"reminder\":{\"action\":null,\"task\":null,\"time\":null,\"repeat\":null},\"history\":{\"metric\":null,\"time_range\":null,\"aggregation\":null}},\"missing_fields\":[],\"risk_flags\":[],\"normalized\":{\"date\":null,\"time_of_day\":null,\"units\":{}}}"
    }
  ]
}
```

**关键点：**
- `assistant.content` 是字符串化的 JSON，不是 Markdown，不包含代码块
- 推理时需将模型输出做 `json.loads()` 解析后用程序消费

## 3. 输出结构说明

### 3.1 routing（路由开关）

5 个布尔值，决定激活哪些下游 LangGraph 节点：

| 字段 | 含义 | 触发示例 |
|------|------|----------|
| `need_record` | 记录健康数据 | "血压145/92"、"吃了二甲双胍" |
| `need_advice` | 请求健康建议 | "正常吗？"、"怎么回事？" |
| `need_plan` | 制定健康计划 | "帮我制定一周降压计划" |
| `need_reminder` | 设置/管理提醒 | "明早提醒我测血糖" |
| `need_history` | 查询历史数据 | "最近一周血压趋势怎么样" |

多个 routing 可以同时为 true（如"记录+询问是否异常"）。

### 3.2 confidence（置信度）

每个 routing 字段对应 0.0-1.0 的置信度。用于下游节点的阈值判断。

### 3.3 slots（结构化槽位）

| 槽位 | 子字段 | 说明 |
|------|--------|------|
| `record` | blood_pressure, blood_glucose, weight, diet, exercise, medication, symptom | 健康记录的数值/内容 |
| `advice` | topic, question, related_metrics | 用户咨询的主题和问题 |
| `plan` | goal, duration, preferences, constraints | 计划需求和偏好 |
| `reminder` | action, task, time, repeat | 提醒任务详情 |
| `history` | metric, time_range, aggregation | 查询的指标、时间范围、聚合方式 |

未触发的路由对应 slots 为 null/空值。

### 3.4 missing_fields（缺失字段）

用户表达了意图但缺少关键信息时，标记缺失字段路径（如 `"record.blood_glucose.value"`）。下游可用此字段决定是否需要追问。

### 3.5 risk_flags（风险标记）

标记初步风险信号，不输出诊断结论。常用值：`high_blood_pressure`、`possible_emergency`、`medication_risk`。

## 4. 数据集统计

### 4.1 划分

| 集合 | 样本数 | 占比 |
|------|--------|------|
| train | 905 | 80.0% |
| dev | 114 | 10.1% |
| test | 112 | 9.9% |

### 4.2 路由分布

| 路由模式 | 数量 | 占比 |
|----------|------|------|
| 仅记录 | 164 | 18.1% |
| 记录+提醒 | 158 | 17.5% |
| 记录+建议 | 143 | 15.8% |
| 历史+建议 | 120 | 13.3% |
| 无路由（闲聊/越界） | 111 | 12.3% |
| 建议+计划+历史 | 53 | 5.9% |
| 建议+计划 | 45 | 5.0% |
| 建议+计划+提醒+历史 | 32 | 3.5% |
| 仅建议 | 30 | 3.3% |
| 建议+计划+提醒 | 29 | 3.2% |
| 其他 | 20 | 2.2% |

### 4.3 Flag 频率

| Flag | 出现次数 | 占比 |
|------|----------|------|
| need_record | 465 | 51.4% |
| need_advice | 452 | 49.9% |
| need_reminder | 229 | 25.3% |
| need_history | 205 | 22.7% |
| need_plan | 170 | 18.8% |

### 4.4 其他统计

| 指标 | 值 |
|------|-----|
| 多路由样本（>1 flag=true） | ~48% |
| 含 missing_fields 样本 | 31 |
| 含 risk_flags 样本 | 90 |
| 输入平均长度 | 21 字 |
| 输出平均长度 | 839 字符 |

## 5. 目录结构

```text
data/synthetic/
  raw/                        # 原始生成数据（batch_*.jsonl）
  checked/                    # 校验中间产物
    combined_raw.jsonl        #   合并后的原始数据
    valid_json.jsonl          #   JSON 格式校验通过
    normalized.jsonl          #   字符串字段规范化后
    routing_consistent.jsonl  #   routing/slots 一致
    deduplicated.jsonl        #   去重后
    splits/                   #   8:1:1 划分
  sft/                        # 最终数据集
    train.jsonl               #   训练集 905条
    dev.jsonl                 #   验证集 114条
    test.jsonl                #   测试集 112条
```

## 6. 使用方法

### 6.1 LLaMA-Factory

在 `dataset_info.json` 中添加：

```json
{
  "routing_train": {
    "file_name": "data/synthetic/sft/train.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"}
  },
  "routing_dev": {
    "file_name": "data/synthetic/sft/dev.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"}
  }
}
```

训练命令：

```bash
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen-7B-Chat \
  --dataset routing_train \
  --eval_dataset routing_dev \
  --finetuning_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_target q_proj,v_proj,k_proj,o_proj \
  --output_dir ./output/routing-lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --max_length 1024 \
  --logging_steps 10 \
  --save_steps 100 \
  --eval_steps 100
```

### 6.2 Unsloth

```python
from unsloth import FastLanguageModel
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen-7B-Chat",
    max_seq_length=1024,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

dataset = load_dataset("json", data_files={
    "train": "data/synthetic/sft/train.jsonl",
    "dev": "data/synthetic/sft/dev.jsonl",
})
```

### 6.3 手动拼接 prompt

推理时构建 prompt：

```python
SYSTEM_PROMPT = "你是慢病健康管理系统的路由与信息抽取器。你只能输出合法 JSON，不要输出解释。"

def build_prompt(user_input: str) -> str:
    # ChatML 格式（Qwen 标准）
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n" \
           f"<|im_start|>user\n{user_input}<|im_end|>\n" \
           f"<|im_start|>assistant\n"
```

## 7. 推荐训练参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| LoRA rank | 16 | 结构化输出任务，r=8-16 足够 |
| LoRA alpha | 32 | alpha = 2× rank |
| Target modules | q_proj,v_proj,k_proj,o_proj | Qwen 全注意力的默认配置 |
| Epochs | 3-5 | 数据量 905，3 轮通常足够 |
| Learning rate | 2e-4 | LoRA 常用范围 1e-4 ~ 3e-4 |
| Batch size | 4 | 配合 gradient_accumulation=4 |
| Max length | 1024 | 输出最长 ~1021 字符 |
| LR scheduler | cosine | 带 warmup |

## 8. 评估建议

### 8.1 核心指标

| 指标 | 目标 | 说明 |
|------|------|------|
| Routing 准确率 | ≥ 90% | 每个 flag 的 precision/recall/f1 |
| JSON 合法率 | ≥ 98% | `json.loads()` 解析成功率 |
| 多路由样本准确率 | ≥ 85% | >1 flag=true 的样本 |
| 危险输出率 | 0% | 不应输出诊断/停药/开药建议 |

### 8.2 按场景评估

用 test 集（112条）分别统计每个路由模式的 f1：

```python
from sklearn.metrics import classification_report

# 对每个 routing flag 独立评估
for flag in ["need_record", "need_advice", "need_plan", "need_reminder", "need_history"]:
    y_true = [sample["routing"][flag] for sample in test_set]
    y_pred = [pred["routing"][flag] for pred in predictions]
    print(f"--- {flag} ---")
    print(classification_report(y_true, y_pred))
```

## 9. 数据再生

如需追加数据，修改 `--target` 参数重新生成：

```bash
cd scripts/data_generation

# 生成更多数据（每个场景 300 条）
python generate_data.py --target 300

# 运行完整流水线
python run_pipeline.py
```

流水线步骤：合并 → JSON 校验 → 槽位规范化 → Schema 校验 → Routing 一致性校验 → 去重 → 分布分析 → 划分 → SFT 转换。

## 10. 数据限制

- **合成数据**：由大模型生成，非真实用户数据。可能存在大模型的偏见和规律性
- **场景覆盖**：主要覆盖慢病管理常见场景（血压/血糖/体重/饮食/运动/用药/症状），罕见场景可能欠缺
- **中文**：所有数据为中文，不适用其他语言
- **数值范围**：已校验合理范围但未做真实世界分布对齐
