# Medical Lora 外部调用与接口约定

## 1 当前结论

本仓库目前是模型训练与离线评测项目，**尚未实现可供外部项目调用的 HTTP API**。

现有能力包括：

- 慢病健康管理意图路由与槽位抽取，LoRA 产物位于 `training/qwen_routing_lora_final/`。
- 中文医学问答，LoRA 产物位于 `qwen_medical_lora_final/`。
- 本地推理示例，分别位于 `training/eval_routing_lora.py` 和 `finetune_qwen_lora_test.py`。

现有脚本会直接加载 Qwen2.5-7B-Instruct 基座模型和 LoRA adapter，适合本机验证，不适合作为外部系统接口，原因如下：

- 没有 FastAPI、Flask 或其他 Web 服务入口。
- 没有监听端口、鉴权、限流、超时和统一错误响应。
- 模型路径包含开发机绝对路径，换机器后需要修改。
- 部分脚本在导入时会立即加载模型或执行测试，不是稳定的 Python SDK。
- `API_test1.py` 是本项目调用 Moonshot API 的测试，不是本项目对外提供的接口。

因此，外部项目当前不能通过 HTTP 直接调用本项目。若调用方与模型部署在同一 Python 进程，可以参考第 2 节临时集成；跨项目、跨机器或生产环境应实现第 3 节定义的服务接口。

## 2 当前可用的本地集成方式

### 2.1 环境要求

- Python 3.10 或更高版本
- NVIDIA GPU 和兼容的 CUDA 环境
- 可访问 Qwen2.5-7B-Instruct 基座模型
- 安装 `torch`、`transformers`、`peft`、`bitsandbytes` 和 `accelerate`
- 同时保留基座模型与 LoRA adapter。仓库中的 adapter 不能单独推理

建议通过环境变量提供基座模型路径，不要继续使用脚本中的开发机绝对路径：

```powershell
$env:BASE_MODEL_PATH = "D:\models\Qwen2.5-7B-Instruct"
```

### 2.2 路由与槽位抽取

模型输入由固定 system prompt 和用户原始文本组成：

```text
system: 你是慢病健康管理系统的路由与信息抽取器。你只能输出合法 JSON，不要输出解释。
user:   用户原始输入
```

推理参数应保持确定性：

| 参数 | 值 |
|------|----|
| `max_new_tokens` | `512` |
| `do_sample` | `False` |
| `temperature` | `None` |
| `top_p` | `None` |

当前可参考 `training/eval_routing_lora.py` 中的 `load_model()` 和 `predict()`。调用方需要先修正其中的 `BASE_MODEL`、`LORA_PATH`，并确保进程启动时只加载一次模型。

示意代码：

```python
from training.eval_routing_lora import load_model, predict

# 模型应在进程启动阶段加载一次，不要每个请求重复加载。
model, tokenizer = load_model()

result = predict(
    model,
    tokenizer,
    "我今天早上血压145/92，帮我记一下，这正常吗？",
)

if result.get("_parse_error"):
    raise RuntimeError(result["_raw"])

print(result["routing"])
print(result["slots"])
```

注意：以上方式依赖现有评测脚本的路径约定，只适合临时验证。正式集成前应把模型加载和推理逻辑抽取到独立、无副作用的模块中。

### 2.3 中文医学问答

医学问答当前只有可执行测试脚本 `finetune_qwen_lora_test.py`，该脚本在导入时会立即加载模型并运行固定问题，不能作为稳定 SDK 直接导入。

其实际输入约定为：

```text
system: 你是一个专业、严谨的中文医学问答助手。
user:   用户问题
```

当前生成参数为：

| 参数 | 值 |
|------|----|
| `max_new_tokens` | `512` |
| `do_sample` | `True` |
| `temperature` | `0.7` |
| `top_p` | `0.9` |
| `repetition_penalty` | `1.1` |

该能力涉及医疗内容。对外提供时必须增加急症提示、免责声明、内容审核和人工兜底，不应把模型输出直接当作诊断或处方。

## 3 建议的 HTTP API 契约

> 本节是建议的接口规范，便于后续实现服务端和外部项目提前对接。仓库当前尚未实现这些 URL。

### 3.1 通用约定

建议基础地址：

```text
http://<host>:8020
```

通用请求头：

```http
Content-Type: application/json
Authorization: Bearer <api-key>
X-Request-ID: <可选的调用方请求 ID>
```

通用规则：

- 编码统一使用 UTF-8。
- 时间使用 ISO 8601；服务端返回 UTC 时间并以 `Z` 结尾。
- `X-Request-ID` 未提供时由服务端生成，并在响应中返回。
- API key 由服务端环境变量或密钥系统管理，不写入代码和仓库。
- 成功响应使用 HTTP 2xx；失败响应使用第 3.5 节的统一错误结构。
- 服务端启动时加载模型，单次请求不得重复加载模型。

### 3.2 健康检查

```http
GET /healthz
```

用途：检查服务进程是否存活，以及基座模型和 LoRA adapter 是否已加载。

成功响应：

```json
{
  "status": "ok",
  "model": "qwen2.5-7b-routing-lora"
}
```

模型未就绪时建议返回 `503 Service Unavailable`。

### 3.3 路由与槽位抽取

```http
POST /v1/intent/parse
```

#### 请求体

```json
{
  "text": "我今天早上血压145/92，帮我记一下，这正常吗？"
}
```

字段说明：

| 字段 | 类型 | 必填 | 约束 | 说明 |
|------|------|------|------|------|
| `text` | string | 是 | 去除首尾空白后 1 至 2000 个字符 | 用户原始自然语言输入 |

#### 成功响应

```json
{
  "request_id": "req_01JABCDEF1234567890",
  "model": "qwen2.5-7b-routing-lora",
  "data": {
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
        "blood_pressure": {
          "systolic": 145,
          "diastolic": 92,
          "unit": "mmHg",
          "time": "今天早上"
        },
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
      "plan": {
        "goal": null,
        "duration": null,
        "preferences": [],
        "constraints": []
      },
      "reminder": {
        "action": null,
        "task": null,
        "time": null,
        "repeat": null
      },
      "history": {
        "metric": null,
        "time_range": null,
        "aggregation": null
      }
    },
    "missing_fields": [],
    "risk_flags": ["high_blood_pressure"],
    "normalized": {
      "date": null,
      "time_of_day": "morning",
      "units": {
        "blood_pressure": "mmHg"
      }
    }
  },
  "usage": {
    "input_tokens": 32,
    "output_tokens": 196
  }
}
```

`routing` 的五个字段允许同时为 `true`。外部项目不应假设一次只能触发一个下游节点。

#### curl 示例

```bash
curl -X POST "http://127.0.0.1:8020/v1/intent/parse" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $MEDICAL_LORA_API_KEY" \
  -d '{"text":"明早8点提醒我测血糖"}'
```

#### Python 示例

```python
import requests

response = requests.post(
    "http://127.0.0.1:8020/v1/intent/parse",
    headers={"Authorization": "Bearer your-api-key"},
    json={"text": "明早8点提醒我测血糖"},
    timeout=30,
)
response.raise_for_status()

result = response.json()
if result["data"]["routing"]["need_reminder"]:
    reminder = result["data"]["slots"]["reminder"]
    print(reminder)
```

#### JavaScript 示例

```javascript
const response = await fetch("http://127.0.0.1:8020/v1/intent/parse", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-api-key"
  },
  body: JSON.stringify({ text: "明早8点提醒我测血糖" })
});

if (!response.ok) {
  throw new Error(`HTTP ${response.status}: ${await response.text()}`);
}

const result = await response.json();
console.log(result.data.routing, result.data.slots);
```

### 3.4 中文医学问答

```http
POST /v1/medical/answer
```

#### 请求体

```json
{
  "question": "我胸口剧痛，要不要去医院？",
  "generation": {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9
  }
}
```

字段说明：

| 字段 | 类型 | 必填 | 约束 | 说明 |
|------|------|------|------|------|
| `question` | string | 是 | 去除首尾空白后 1 至 4000 个字符 | 医学问题 |
| `generation` | object | 否 | 仅允许文档中列出的字段 | 生成参数 |
| `generation.max_new_tokens` | integer | 否 | 1 至 1024，默认 512 | 最大生成 token 数 |
| `generation.temperature` | number | 否 | 0.0 至 2.0，默认 0.7 | 采样温度 |
| `generation.top_p` | number | 否 | 大于 0 且不超过 1，默认 0.9 | nucleus sampling 参数 |

#### 成功响应

```json
{
  "request_id": "req_01JABCDEF1234567891",
  "model": "qwen2.5-7b-medical-lora",
  "answer": "胸口剧痛可能是急症信号，请立即拨打急救电话或尽快前往急诊，不要自行驾车。",
  "safety": {
    "emergency": true,
    "disclaimer": "本回答仅供健康信息参考，不能替代医生诊断和治疗。"
  },
  "usage": {
    "input_tokens": 18,
    "output_tokens": 48
  }
}
```

`safety` 是对外服务层建议增加的字段，不是当前 LoRA 模型原生输出。服务端应通过独立规则或审核模型产生该字段，不能仅依赖生成文本。

### 3.5 错误响应

所有失败响应建议采用统一结构：

```json
{
  "request_id": "req_01JABCDEF1234567892",
  "error": {
    "code": "INVALID_ARGUMENT",
    "message": "text 不能为空",
    "details": [
      {
        "field": "text",
        "reason": "required"
      }
    ]
  }
}
```

建议错误码：

| HTTP 状态码 | `error.code` | 场景 |
|-------------|--------------|------|
| 400 | `INVALID_ARGUMENT` | JSON 格式错误、字段缺失或参数越界 |
| 401 | `UNAUTHORIZED` | API key 缺失或无效 |
| 413 | `PAYLOAD_TOO_LARGE` | 输入超过长度限制 |
| 429 | `RATE_LIMITED` | 超过调用频率或并发限制 |
| 500 | `INFERENCE_ERROR` | 模型生成失败或输出无法解析 |
| 503 | `MODEL_NOT_READY` | 模型尚未加载完成或 GPU 不可用 |
| 504 | `INFERENCE_TIMEOUT` | 推理超时 |

路由模型输出无法被 `json.loads()` 解析时，服务端应返回 `500 INFERENCE_ERROR`，并把原始输出写入受控日志；不要在响应中返回完整内部提示词、文件路径或堆栈信息。

## 4 路由输出字段说明

### 4.1 routing

| 字段 | 类型 | 含义 |
|------|------|------|
| `need_record` | boolean | 记录血压、血糖、体重、饮食、运动、用药或症状 |
| `need_advice` | boolean | 请求健康建议，或询问是否正常、原因和处理方式 |
| `need_plan` | boolean | 制定饮食、运动、控糖、降压或减重计划 |
| `need_reminder` | boolean | 创建、修改、删除或查询提醒 |
| `need_history` | boolean | 查询历史记录、趋势、平均值或最近一次记录 |

### 4.2 confidence

与 `routing` 一一对应的 0.0 至 1.0 数值。是否执行下游动作应同时参考布尔路由、置信度和业务风险，不建议只依赖固定阈值。

### 4.3 slots

| 分组 | 主要字段 |
|------|----------|
| `record` | `blood_pressure`、`blood_glucose`、`weight`、`diet`、`exercise`、`medication`、`symptom` |
| `advice` | `topic`、`question`、`related_metrics` |
| `plan` | `goal`、`duration`、`preferences`、`constraints` |
| `reminder` | `action`、`task`、`time`、`repeat` |
| `history` | `metric`、`time_range`、`aggregation` |

未识别或未触发的值可能为 `null`、空数组或空对象，调用方必须做空值检查，不应直接读取深层字段。

### 4.4 missing_fields

字符串数组，表示完成下游任务仍缺少的信息，例如：

```json
["reminder.time"]
```

调用方可据此向用户追问，但不应自行编造缺失值。

### 4.5 risk_flags

初步风险标记数组，现有数据中常见值包括：

- `high_blood_pressure`
- `high_blood_glucose`
- `possible_emergency`
- `medication_risk`

风险标记只用于分流和提醒，不是诊断结论。调用方遇到 `possible_emergency` 时应进入急症处置流程，而不是继续普通对话。

### 4.6 normalized

| 字段 | 类型 | 含义 |
|------|------|------|
| `date` | string 或 null | 规范化日期；无法可靠解析时为 `null` |
| `time_of_day` | string 或 null | 如 `morning`、`fasting` |
| `units` | object | 指标到规范单位的映射 |

当前模型不知道请求发生时的真实系统日期。涉及“今天”“明天”“下周”等相对时间时，生产服务应显式传入当前时间并由服务层做二次规范化。

## 5 服务实现要求

正式实现对外 API 时至少应满足：

1. 从 `BASE_MODEL_PATH`、`ROUTING_LORA_PATH` 和 `MEDICAL_LORA_PATH` 环境变量读取路径。
2. 应用启动时完成模型加载，并通过 `/healthz` 暴露就绪状态。
3. 限制同一 GPU 的并发推理数，避免显存耗尽。
4. 对输入长度、生成参数和请求体大小进行校验。
5. 为模型推理设置超时、取消机制和结构化日志。
6. 不记录 API key；医疗文本日志应脱敏并设置保留期限。
7. 对路由输出执行 JSON 解析和 schema 校验，不能只判断是否为合法 JSON。
8. 对医学问答增加独立安全层和急症规则，模型输出不得直接替代医疗决策。
9. 不把 Transformers 或 CUDA 异常堆栈原样返回给调用方。
10. 提供 OpenAPI 文档并为请求、响应、鉴权和错误场景编写自动化测试。

## 6 推荐的后续落地顺序

1. 把两类模型的加载和推理逻辑抽取成 `app/inference/` 下的独立模块。
2. 先实现路由接口 `/v1/intent/parse`，因为该模型输出结构明确、评测结果完整。
3. 增加 Pydantic 请求和响应模型，对六个顶层字段做严格校验。
4. 增加健康检查、统一异常处理、请求 ID 和 API key 鉴权。
5. 完成单元测试和最小 GPU 冒烟测试后，再允许外部项目接入。
6. 医学问答接口应在安全规则、免责声明和人工升级流程完成后再开放。
