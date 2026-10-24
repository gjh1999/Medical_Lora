# 路由模型提示词说明

## 1. 概述

慢病管理路由模型（Qwen2.5-7B-Instruct + LoRA）的提示词分为两层结构：**System Prompt**（角色设定）和 **User Input**（用户自然语言）。模型输出为单行合法 JSON，经 `json.loads()` 解析后由下游 LangGraph 节点消费。

## 2. 提示词结构

采用 Qwen 标准 ChatML 格式：

```text
<|im_start|>system
你是慢病健康管理系统的路由与信息抽取器。你只能输出合法 JSON，不要输出解释。<|im_end|>
<|im_start|>user
{用户自然语言输入}<|im_end|>
<|im_start|>assistant
{模型生成的 JSON 输出}<|im_end|>
```

### 2.1 System Prompt

```
你是慢病健康管理系统的路由与信息抽取器。你只能输出合法 JSON，不要输出解释。
```

**设计要点：**

- **极简**：只有一句话，不包含 schema 定义、任务说明或约束条件。因为 LoRA 微调已将这些知识注入模型参数。
- **角色定位**："路由与信息抽取器" 明确了模型职责边界 — 只做路由判断和信息抽取，不做对话、不做诊断。
- **格式约束**："只能输出合法 JSON，不要输出解释" 确保输出可直接被 `json.loads()` 解析，无需正则提取。

对比 prompt-only baseline 需要约 2KB 的详细 prompt（含完整 JSON schema、5 类约束）才能达到 F1=0.842，微调后的模型用这一句话即可达到 F1=0.990，JSON 合法率 100%。

### 2.2 User Input

用户原始自然语言，不做任何改写或包装。典型输入：

| 场景 | 输入示例 |
|------|----------|
| 记录数据 | "我今天早上血压145/92，帮我记一下" |
| 记录+询问 | "今日空腹7.3，有点高" |
| 查询+建议 | "上个月我的饮食记录，看看有没有不规律" |
| 提醒设置 | "明早8点提醒我测血糖" |
| 制定计划 | "帮我制定一周降压饮食计划" |
| 闲聊/越界 | "你是真人吗" |

输入平均长度约 21 字，无需做截断或格式化处理。

## 3. 输出结构

模型输出是一个单行 JSON 对象（无 Markdown 代码块，无缩进，无尾随文本），包含 6 个顶层字段：

```
{routing, confidence, slots, missing_fields, risk_flags, normalized}
```

### 3.1 routing（路由开关）

5 个布尔值，决定激活哪些下游节点：

| 字段 | 含义 | 触发示例 |
|------|------|----------|
| `need_record` | 记录健康数据 | "血压145/92"、"吃了二甲双胍" |
| `need_advice` | 请求健康建议 | "正常吗？"、"怎么回事？" |
| `need_plan` | 制定健康计划 | "帮我制定一周降压计划" |
| `need_reminder` | 设置/管理提醒 | "明早提醒我测血糖" |
| `need_history` | 查询历史数据 | "最近一周血压趋势怎么样" |

支持多路由同时为 true（约 48% 样本），如 "血压145/92，这正常吗？" 同时触发 `need_record` + `need_advice`。

### 3.2 confidence（置信度）

每个路由字段对应一个 0.0-1.0 的置信度：

```json
{
  "need_record": 0.97,
  "need_advice": 0.94,
  "need_plan": 0.03,
  "need_reminder": 0.02,
  "need_history": 0.01
}
```

**使用建议：**
- 下游节点设置阈值（如 0.7），低于阈值的不激活
- 已触发的路由（true）对应置信度应在 0.9 以上
- 未触发的路由（false）对应置信度应在 0.1 以下

### 3.3 slots（结构化槽位）

按 5 个路由场景分组的嵌套结构。**未触发的路由对应 slots 保持 null/空数组，不臆造数据。**

#### record（健康记录）

```json
{
  "blood_pressure": {"systolic": 145, "diastolic": 92, "unit": "mmHg", "time": "今天早上"},
  "blood_glucose": {"value": 7.3, "unit": "mmol/L", "type": "fasting", "time": "今日"},
  "weight": null,
  "diet": null,
  "exercise": {"type": "骑车", "duration_minutes": 30, "time": "上下班"},
  "medication": null,
  "symptom": null
}
```

#### advice（健康咨询）

```json
{
  "topic": "血压评估",
  "question": "血压145/92是否正常",
  "related_metrics": ["blood_pressure"]
}
```

#### plan（健康计划）

```json
{
  "goal": "降低血压",
  "duration": "一周",
  "preferences": ["低盐饮食"],
  "constraints": ["膝盖受伤不能跑步"]
}
```

#### reminder（提醒管理）

```json
{
  "action": "create",
  "task": "测血糖",
  "time": "明天早上8点",
  "repeat": "每天"
}
```

#### history（历史查询）

```json
{
  "metric": "blood_pressure",
  "time_range": "本周",
  "aggregation": "max"
}
```

### 3.4 missing_fields（缺失字段）

用户表达了意图但缺少关键信息时，用路径字符串标记缺失字段：

```json
["record.blood_pressure.value", "record.blood_pressure.unit"]
```

**使用建议：** 下游节点根据此字段生成追问，如 "请问你的血压值是高压多少、低压多少？"

### 3.5 risk_flags（风险标记）

标记初步风险信号，**严禁输出诊断结论**：

| 值 | 触发条件 |
|------|----------|
| `high_blood_pressure` | 收缩压 ≥ 140 或舒张压 ≥ 90 |
| `high_blood_glucose` | 空腹血糖 ≥ 7.0 或餐后 ≥ 11.1 |
| `possible_emergency` | 值远超正常范围，建议立即就医 |
| `medication_risk` | 可能存在药物相关风险 |

### 3.6 normalized（规范化信息）

对输入中的模糊表达做规范化：

```json
{
  "date": "2026-07-09",
  "time_of_day": "morning",
  "units": {"blood_pressure": "mmHg", "blood_glucose": "mmol/L"}
}
```

## 4. 完整示例

### 示例 1：单路由（仅记录）

**输入：** "骑车上下班，单程30分钟"

**输出：**

```json
{
  "routing": {
    "need_record": true,
    "need_advice": false,
    "need_plan": false,
    "need_reminder": false,
    "need_history": false
  },
  "confidence": {
    "need_record": 0.95,
    "need_advice": 0.05,
    "need_plan": 0.02,
    "need_reminder": 0.03,
    "need_history": 0.02
  },
  "slots": {
    "record": {
      "blood_pressure": null,
      "blood_glucose": null,
      "weight": null,
      "diet": null,
      "exercise": {"type": "骑车", "duration_minutes": 30, "time": "上下班"},
      "medication": null,
      "symptom": null
    },
    "advice": {"topic": null, "question": null, "related_metrics": []},
    "plan": {"goal": null, "duration": null, "preferences": [], "constraints": []},
    "reminder": {"action": null, "task": null, "time": null, "repeat": null},
    "history": {"metric": null, "time_range": null, "aggregation": null}
  },
  "missing_fields": [],
  "risk_flags": [],
  "normalized": {"date": null, "time_of_day": null, "units": {}}
}
```

### 示例 2：多路由（记录+建议）

**输入：** "我今天早上血压145/92，帮我记一下，这正常吗？"

**输出：**

```json
{
  "routing": {
    "need_record": true,
    "need_advice": true,
    "need_plan": false,
    "need_reminder": false,
    "need_history": false
  },
  "confidence": {
    "need_record": 0.97,
    "need_advice": 0.94,
    "need_plan": 0.03,
    "need_reminder": 0.02,
    "need_history": 0.01
  },
  "slots": {
    "record": {
      "blood_pressure": {"systolic": 145, "diastolic": 92, "unit": "mmHg", "time": "今天早上"},
      "blood_glucose": null,
      "weight": null,
      "diet": null,
      "exercise": null,
      "medication": null,
      "symptom": null
    },
    "advice": {
      "topic": "血压评估",
      "question": "血压145/92是否正常",
      "related_metrics": ["blood_pressure"]
    },
    "plan": {"goal": null, "duration": null, "preferences": [], "constraints": []},
    "reminder": {"action": null, "task": null, "time": null, "repeat": null},
    "history": {"metric": null, "time_range": null, "aggregation": null}
  },
  "missing_fields": [],
  "risk_flags": ["high_blood_pressure"],
  "normalized": {
    "date": null,
    "time_of_day": "morning",
    "units": {"blood_pressure": "mmHg"}
  }
}
```

### 示例 3：越界/闲聊

**输入：** "你是真人吗"

**输出：** 所有路由均为 false，slots 全为 null，confidence 全为 0.01。下游节点根据 "全部 false" 判断为越界请求，交给通用对话模块处理。

## 5. 推理代码

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = "你是慢病健康管理系统的路由与信息抽取器。你只能输出合法 JSON，不要输出解释。"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "./qwen_routing_lora_final"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, user_input: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw": response[:200]}


# 使用示例
model, tokenizer = load_model()
result = predict(model, tokenizer, "我今天早上血压145/92，帮我记一下")

if not result.get("_parse_error"):
    routing = result["routing"]
    slots = result["slots"]
    if routing["need_record"]:
        bp = slots["record"]["blood_pressure"]
        print(f"记录血压: {bp['systolic']}/{bp['diastolic']} {bp['unit']}")
    if routing["need_advice"]:
        print(f"用户咨询: {slots['advice']['question']}")
    if result["risk_flags"]:
        print(f"风险标记: {result['risk_flags']}")
```

## 6. 生成参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `max_new_tokens` | 512 | 输出最长约 1021 字符，512 token 足够 |
| `do_sample` | `False` | 关闭采样，贪婪解码保证确定性输出 |
| `temperature` | `None` | 配合 `do_sample=False` |
| `top_p` | `None` | 配合 `do_sample=False` |
| `eos_token_id` | tokenizer.eos_token_id | 模型在 JSON 结束后自然停止 |

## 7. 与 prompt-only baseline 的对比

| 方面 | Prompt-only (zero-shot) | LoRA 微调后 |
|------|------------------------|------------|
| System prompt 长度 | ~2KB（含完整 schema + 5 类约束） | 1 句话 |
| Routing Macro F1 | 0.842 | 0.990 |
| JSON 合法率 | ~87% | 100% |
| 每轮 token 开销 | ~700 input tokens | ~50 input tokens |
| 多路由准确率 | ~78% | 100% |

微调后 prompt 可以极简的核心原因：**路由规则、schema 结构、输出约束已在训练中学会，prompt 只需告知角色身份**。
