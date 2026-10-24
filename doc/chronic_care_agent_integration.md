# Medical Lora 与 Chronic Care Agent 集成方案

## 1 目标

将 `Medical_Lora` 已训练完成的慢病意图识别 LoRA 模型接入 `chronic_care_agent_2`，用于识别用户意图并抽取结构化槽位，再由 Chronic Care Agent 的 LangGraph 工作流执行记录、建议、计划、提醒和历史查询等业务节点。

涉及项目：

| 项目 | 路径 | 职责 |
|------|------|------|
| Medical Lora | `C:\Users\gjh\PycharmProjects\Medical_Lora` | 加载 Qwen2.5-7B 基座模型和路由 LoRA，对外提供意图解析服务 |
| Chronic Care Agent | `C:\Users\gjh\PycharmProjects\chronic_care_agent_2` | 接收用户请求、调用意图服务并执行 LangGraph 业务流程 |

本次集成只使用以下路由模型：

```text
Medical_Lora/training/qwen_routing_lora_final
```

`qwen_medical_lora_final` 是医学问答 adapter，不参与意图识别链路。

## 2 推荐架构

两个项目采用独立进程，通过本机 HTTP 通信：

```text
用户或前端
    |
    v
Chronic Care Agent
127.0.0.1:8000
    |
    | POST /v1/intent/parse
    v
Medical Lora Intent Service
127.0.0.1:8020
    |
    | Qwen2.5-7B-Instruct + routing LoRA
    v
routing + confidence + slots + risk flags
    |
    v
Chronic Care Agent LangGraph
    |
    +-- record node
    +-- advice node
    +-- plan node
    +-- reminder node
    +-- history node
```

不建议把 7B 模型直接加载到 Chronic Care Agent 进程，原因是：

- Agent Web 服务启停会导致模型重复加载。
- Uvicorn 多 worker 可能为每个进程复制模型，造成显存不足。
- 模型推理故障会直接影响业务 API 进程。
- 后续无法方便地把 GPU 推理迁移到独立服务器。
- Chronic Care Agent 已经实现了外部意图服务客户端，无需重新设计调用链。

## 3 当前项目状态

### 3.1 Chronic Care Agent 已完成的部分

目标项目已经具备下列代码：

```text
app/intent_client/client.py
app/intent_client/schemas.py
app/nodes/parser_node.py
app/config/settings.py
```

现有调用流程为：

1. `parser_node` 从 LangGraph state 读取 `user_input`。
2. 构造 `IntentRequest`。
3. `ExternalIntentClient` 请求 `POST /v1/intent/parse`。
4. 使用 Pydantic 校验意图服务响应。
5. 将解析结果写入 LangGraph state。

写入的状态字段包括：

```text
intent
intent_flags
extracted_data
confidence
missing_fields
risk_flags
normalized
```

默认配置已经指向：

```dotenv
INTENT_BASE_URL=http://127.0.0.1:8020
INTENT_API_KEY=
INTENT_TIMEOUT=60
```

### 3.2 Medical Lora 缺少的部分

Medical Lora 当前具有训练产物和离线推理脚本，但尚未提供 HTTP 服务。需要增加：

- FastAPI 服务入口。
- 模型启动加载逻辑。
- `GET /healthz` 健康检查。
- `POST /v1/intent/parse` 意图解析接口。
- 请求和响应 Schema 校验。
- 可选的 Bearer API key 鉴权。
- 推理异常和模型输出解析异常处理。

## 4 Medical Lora 服务设计

### 4.1 建议文件

在 Medical Lora 根目录新增：

```text
Medical_Lora/
|-- service_app.py
|-- service_requirements.txt
|-- training/
|   `-- qwen_routing_lora_final/
`-- doc/
    `-- chronic_care_agent_integration.md
```

后续代码规模扩大时，可以再拆分为：

```text
app/
|-- main.py
|-- schemas.py
|-- settings.py
`-- inference/
    `-- routing_model.py
```

第一版保持单一服务文件更容易验证，但模型加载与请求处理仍应分成独立类或函数。

### 4.2 运行环境变量

建议从环境变量读取配置：

```dotenv
BASE_MODEL_PATH=C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct
ROUTING_LORA_PATH=C:\Users\gjh\PycharmProjects\Medical_Lora\training\qwen_routing_lora_final
INTENT_SERVICE_HOST=127.0.0.1
INTENT_SERVICE_PORT=8020
INTENT_API_KEY=
MAX_NEW_TOKENS=512
```

注意事项：

- `BASE_MODEL_PATH` 是完整的 Qwen2.5-7B-Instruct 基座模型。
- `ROUTING_LORA_PATH` 是路由 LoRA adapter，不能单独运行。
- API key 为空时可以用于本机开发；跨机器部署时必须设置。
- 不应把真实 API key 写入仓库。

### 4.3 模型加载

服务启动时加载一次模型：

```python
tokenizer = AutoTokenizer.from_pretrained(
    ROUTING_LORA_PATH,
    trust_remote_code=True,
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, ROUTING_LORA_PATH)
model.eval()
```

模型必须在 FastAPI lifespan 或启动事件中加载，不能在每个请求内重新加载。

如果使用 Uvicorn，不要启动多个 worker：

```powershell
python -m uvicorn service_app:app --host 127.0.0.1 --port 8020 --workers 1
```

多个 worker 会分别加载一份 7B 模型，通常会耗尽显存。

### 4.4 推理提示词

必须继续使用训练时的固定 system prompt：

```text
你是慢病健康管理系统的路由与信息抽取器。你只能输出合法 JSON，不要输出解释。
```

消息结构：

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": request.text},
]
```

推荐生成参数：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,
    temperature=None,
    top_p=None,
    eos_token_id=tokenizer.eos_token_id,
)
```

意图识别属于结构化任务，应关闭随机采样，保证相同输入尽可能产生相同结果。

## 5 接口契约

### 5.1 健康检查

```http
GET /healthz
```

模型加载完成并可接受请求时返回 HTTP 200：

```json
{
  "status": "ok",
  "model": "qwen2.5-7b-routing-lora"
}
```

模型未完成加载或不可用时返回 HTTP 503。

### 5.2 意图解析

```http
POST /v1/intent/parse
Content-Type: application/json
```

如果配置了 API key，还需要：

```http
Authorization: Bearer <api-key>
```

请求体：

```json
{
  "text": "记录今天空腹血糖7.2，并给我一些建议",
  "user_id": "user-001",
  "conversation_id": "conversation-001"
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 是 | 用户原始输入，建议限制为 1 至 2000 个字符 |
| `user_id` | string 或 null | 否 | 透传的用户标识，不参与模型推理 |
| `conversation_id` | string 或 null | 否 | 透传的会话标识，不参与模型推理 |

成功响应：

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
    "need_record": 0.98,
    "need_advice": 0.95,
    "need_plan": 0.02,
    "need_reminder": 0.01,
    "need_history": 0.01
  },
  "slots": {
    "record": {
      "blood_pressure": null,
      "blood_glucose": {
        "value": 7.2,
        "unit": "mmol/L",
        "type": "fasting",
        "time": "今天"
      },
      "weight": null,
      "diet": null,
      "exercise": null,
      "medication": null,
      "symptom": null
    },
    "advice": {
      "topic": "blood_glucose",
      "question": "空腹血糖7.2是否正常",
      "related_metrics": ["blood_glucose"]
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
  "risk_flags": ["high_blood_glucose"],
  "normalized": {
    "date": null,
    "time_of_day": "fasting",
    "units": {
      "blood_glucose": "mmol/L"
    }
  }
}
```

Chronic Care Agent 同时支持统一包装格式：

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "routing": {},
    "confidence": {},
    "slots": {},
    "missing_fields": [],
    "risk_flags": [],
    "normalized": {}
  }
}
```

建议第一版直接返回未包装对象，减少一层协议处理。

## 6 响应校验

Medical Lora 服务不能把模型文本直接返回给 Agent。服务端至少需要完成两层校验：

1. 使用 `json.loads()` 验证输出是合法 JSON 对象。
2. 使用 Pydantic 验证输出满足接口 Schema。

必须验证：

- `routing` 包含五个布尔字段。
- `confidence` 的值在 0.0 至 1.0 之间。
- `slots` 是 JSON 对象。
- `missing_fields` 和 `risk_flags` 是字符串数组。
- `normalized` 是 JSON 对象。

解析失败时建议返回：

```http
HTTP/1.1 502 Bad Gateway
```

```json
{
  "detail": {
    "code": "INVALID_MODEL_OUTPUT",
    "message": "模型未返回符合 Schema 的 JSON"
  }
}
```

不要把内部提示词、绝对路径、完整模型原始输出或 Python 堆栈返回给调用方。

## 7 Chronic Care Agent 配置

在 `chronic_care_agent_2/.env` 中配置：

```dotenv
INTENT_BASE_URL=http://127.0.0.1:8020
INTENT_API_KEY=
INTENT_TIMEOUT=60
```

如果 Medical Lora 配置了 API key，两边的值必须相同：

```dotenv
INTENT_API_KEY=replace-with-a-random-secret
```

Chronic Care Agent 不需要安装 `torch`、`transformers`、`peft` 或 `bitsandbytes`，它只通过 `httpx` 调用意图服务。

## 8 本机启动流程

### 8.1 启动 Medical Lora

进入项目：

```powershell
cd C:\Users\gjh\PycharmProjects\Medical_Lora
```

设置模型路径：

```powershell
$env:BASE_MODEL_PATH = "C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"
$env:ROUTING_LORA_PATH = "C:\Users\gjh\PycharmProjects\Medical_Lora\training\qwen_routing_lora_final"
```

安装服务依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r service_requirements.txt
```

启动意图服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn service_app:app --host 127.0.0.1 --port 8020 --workers 1
```

首次启动时需要加载完整基座模型，健康检查只有在模型加载完成后才应返回成功。

### 8.2 验证 Medical Lora

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8020/healthz
```

接口测试：

```powershell
$body = @{
    text = "我今天早上血压145/92，帮我记录一下，这正常吗？"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8020/v1/intent/parse" `
    -ContentType "application/json" `
    -Body $body
```

预期结果：

- `need_record` 为 `true`。
- `need_advice` 为 `true`。
- `slots.record.blood_pressure` 包含收缩压和舒张压。
- 响应可以被 Chronic Care Agent 的 `IntentResponse` 校验通过。

### 8.3 启动 Chronic Care Agent

确认意图服务健康后再启动 Agent：

```powershell
cd C:\Users\gjh\PycharmProjects\chronic_care_agent_2

.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

检查外部依赖：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/dependencies
```

Medical Lora 意图服务应显示为可用状态。

## 9 Docker 部署注意事项

如果两个项目运行在不同容器中，`127.0.0.1` 指向各自容器，不能用于容器间通信。

Docker Compose 中应使用服务名：

```yaml
services:
  medical-lora:
    ports:
      - "8020:8020"

  chronic-care-agent:
    environment:
      INTENT_BASE_URL: http://medical-lora:8020
    depends_on:
      medical-lora:
        condition: service_healthy
```

Medical Lora 在容器内应监听：

```text
0.0.0.0:8020
```

本机直接运行时仍建议只监听 `127.0.0.1`，避免无意暴露模型服务。

## 10 稳定性和安全要求

### 10.1 GPU 并发

第一版建议使用单 worker，并通过锁或请求队列限制同一时间的推理数量。并发过高可能导致 CUDA 显存不足。

### 10.2 超时

Chronic Care Agent 当前默认等待 60 秒。服务端推理超时应返回明确错误，不应无限等待。

### 10.3 故障策略

意图服务是 LangGraph 的前置必需依赖。遇到以下情况时应终止本次工作流：

- 连接失败。
- 请求超时。
- 返回非 2xx 状态码。
- 返回非法 JSON。
- 响应 Schema 不匹配。

不要静默回退到普通聊天模型，因为错误路由可能导致错误写入健康数据或错误创建提醒。

### 10.4 日志和隐私

- 不记录 Authorization header。
- 用户医疗文本日志需要脱敏。
- 日志中记录 request ID、耗时、响应状态和错误类型。
- 不在生产响应中暴露模型路径、CUDA 错误堆栈和内部提示词。

## 11 验收清单

### Medical Lora

- [ ] 可以从环境变量读取基座模型和 adapter 路径。
- [ ] 服务启动时只加载一次模型。
- [ ] `GET /healthz` 在模型就绪后返回 200。
- [ ] `POST /v1/intent/parse` 接受 Agent 当前请求结构。
- [ ] 输出通过 JSON 和 Pydantic Schema 双重校验。
- [ ] 推理固定使用训练时的 system prompt。
- [ ] 推理关闭随机采样。
- [ ] API key 配置可选且两端一致。
- [ ] 单元测试不依赖真实 GPU，可注入假的推理 runtime。
- [ ] GPU 冒烟测试覆盖至少五种单意图和两个多意图样例。

### Chronic Care Agent

- [ ] `.env` 中的 `INTENT_BASE_URL` 指向正确服务地址。
- [ ] `ExternalIntentClient.health()` 返回 `true`。
- [ ] `/dependencies` 显示意图服务正常。
- [ ] `parser_node` 能把响应写入 LangGraph state。
- [ ] 记录意图可以进入 `record_node`。
- [ ] 建议意图可以进入 `advice_node`。
- [ ] 计划意图可以进入计划子图。
- [ ] 提醒意图可以进入提醒节点。
- [ ] 多意图输入能触发多个业务分支。

## 12 实施顺序

建议按照以下顺序落地：

1. 在 Medical Lora 中实现无模型依赖的 FastAPI 请求和响应 Schema。
2. 使用假的推理 runtime 完成接口单元测试。
3. 接入真实 Qwen2.5-7B 和 routing LoRA。
4. 在 Medical Lora 中完成单条 GPU 推理验证。
5. 启动 Chronic Care Agent，验证 `/dependencies`。
6. 完成单意图端到端测试。
7. 完成多意图、异常输出、超时和模型不可用测试。
8. 最后增加 API key、日志脱敏和部署配置。

完成以上步骤后，Medical Lora 负责稳定地产生结构化意图，Chronic Care Agent 只负责消费结果和执行业务流程，两边职责清晰，也便于后续独立扩容和部署。

## 13 实际代码修改和测试顺序

### 13.1 Medical Lora 新增文件

第一版实现增加以下文件：

```text
service_app.py                 FastAPI 服务、Schema 和模型 runtime
service_requirements.txt       服务及测试依赖
pytest.ini                     GPU 测试标记
tests/test_service_api.py      不加载模型的接口单元测试
tests/test_service_model.py    加载真实模型的 GPU 冒烟测试
```

`service_app.py` 提供 `create_app(runtime=...)` 工厂。接口单元测试注入假的 runtime，因此能够先验证 HTTP 契约、鉴权和异常映射，不需要加载模型。生产使用的模块级 `app` 则会在 FastAPI lifespan 中加载真实模型。

### 13.2 第一阶段 接口单元测试

安装依赖：

```powershell
cd C:\Users\gjh\PycharmProjects\Medical_Lora
.\.venv\Scripts\python.exe -m pip install -r service_requirements.txt
```

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_service_api.py -q
```

该阶段验证：

- 健康检查成功和未就绪状态。
- 请求与响应的六个顶层字段。
- 空文本校验。
- 可选 Bearer token 鉴权。
- 非法模型输出映射为 HTTP 502。

### 13.3 第二阶段 真实模型冒烟测试

```powershell
$env:BASE_MODEL_PATH = "C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"
$env:ROUTING_LORA_PATH = "C:\Users\gjh\PycharmProjects\Medical_Lora\training\qwen_routing_lora_final"
$env:RUN_MODEL_TESTS = "1"

.\.venv\Scripts\python.exe -m pytest tests/test_service_model.py -q -s
```

该测试只加载一次模型，并检查提醒意图和“记录加建议”多意图。默认不设置 `RUN_MODEL_TESTS=1` 时会跳过，避免普通单元测试意外占用 GPU。

### 13.4 第三阶段 服务接口测试

启动服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn service_app:app --host 127.0.0.1 --port 8020 --workers 1
```

另开终端执行：

```powershell
Invoke-RestMethod http://127.0.0.1:8020/healthz

$body = @{
    text = "我今天早上血压145/92，帮我记录一下，这正常吗？"
    user_id = "test-user"
    conversation_id = "test-conversation"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8020/v1/intent/parse" `
    -ContentType "application/json" `
    -Body $body
```

### 13.5 第四阶段 Agent 客户端联动测试

保持 8020 服务运行，然后执行：

```powershell
cd C:\Users\gjh\PycharmProjects\chronic_care_agent_2

.\.venv\Scripts\python.exe -m pytest tests/test_external_clients.py -q

.\.venv\Scripts\python.exe -c "from app.intent_client.client import ExternalIntentClient; from app.intent_client.schemas import IntentRequest; client=ExternalIntentClient('http://127.0.0.1:8020'); print(client.parse(IntentRequest(text='明早8点提醒我测血糖')).model_dump())"
```

第一个命令使用 Mock 验证客户端协议，第二个命令请求真实 Medical Lora 服务。只有第二个命令通过，才能证明两个项目实际连通。

### 13.6 第五阶段 Parser Node 和完整工作流测试

直接验证模型输出能写入 LangGraph state：

```powershell
.\.venv\Scripts\python.exe -c "from app.nodes.parser_node import parser_node; state={'user_input':'记录血压145/92，这正常吗？','user_id':'test-user','conversation_id':'test-conversation'}; print(parser_node(state))"
```

随后启动完整 Agent：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

检查依赖：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/dependencies
```

完整测试顺序为：

```text
接口假模型单测
  -> 真实模型 GPU 冒烟测试
  -> Medical Lora HTTP 测试
  -> Agent 客户端真实调用
  -> parser_node 联动
  -> Agent 完整工作流
```

## 14 首次实施测试记录

测试日期：2026-10-24。

| 阶段 | 结果 | 说明 |
|------|------|------|
| Medical Lora 接口单元测试 | 通过 | 健康检查、契约、输入校验、鉴权和异常映射全部通过 |
| Medical Lora 真实模型测试 | 通过 | RTX 5080 上成功加载 4-bit 基座模型和 routing LoRA |
| Medical Lora HTTP 测试 | 通过 | 8020 服务健康，记录加建议多意图和血压槽位正确 |
| Agent ExternalIntentClient Mock 测试 | 通过 | 4 个现有客户端测试通过 |
| Agent ExternalIntentClient 真实调用 | 通过 | Agent 客户端成功请求 8020 服务并校验响应 |
| Agent parser_node 真实调用 | 通过 | 意图、置信度和槽位成功写入 LangGraph state |
| Agent Web 服务依赖检查 | 通过 | `/dependencies` 返回 `intent_service=up` |
| Agent 完整提醒请求 | 通过 | 登录、创建会话、发送消息和提醒意图路由全部完成 |

首次真实联动暴露了两种历史训练 Schema 差异：

- 少量提醒样本输出 `reminder.recurrence`，Agent 使用 `reminder.repeat`。
- 模型可能在已经抽取 `reminder.time` 时仍将其列入 `missing_fields`。
- 部分样本会把 `plan.preferences`、`plan.constraints` 或 `advice.related_metrics` 输出为 `null`，而 Agent 节点更适合消费空数组。

服务边界因此增加以下确定性规范化：

1. 将 `reminder.recurrence` 转换为 `reminder.repeat`。
2. 将上述数组字段的 `null` 转换为空数组。
3. 当缺失字段路径已经有非空槽位值时，从 `missing_fields` 中移除该路径。

规范化只修复 Schema 表达，不修改路由判断、置信度或医学数值。
