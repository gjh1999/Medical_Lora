# Medical Lora 与 Chronic Care Agent 集成测试报告

## 1 报告信息

| 项目 | 内容 |
|------|------|
| 测试日期 | 2026-10-24 |
| 测试类型 | 接口单元测试、真实模型测试、HTTP 服务测试、跨项目联动测试 |
| 被测模型 | Qwen2.5-7B-Instruct 加 routing LoRA |
| LoRA 路径 | `training/qwen_routing_lora_final` |
| 模型服务 | Medical Lora，`127.0.0.1:8020` |
| 调用方 | Chronic Care Agent，`127.0.0.1:8000` |
| 总体结论 | 通过，意图识别服务已能被 Agent 调用 |

## 2 测试结论

Medical Lora 意图识别服务已完成实现，并成功接入 Chronic Care Agent 现有的 `ExternalIntentClient` 和 `parser_node`。

本次确认通过的核心链路为：

```text
用户输入
  -> Chronic Care Agent
  -> ExternalIntentClient
  -> POST http://127.0.0.1:8020/v1/intent/parse
  -> Qwen2.5-7B-Instruct + routing LoRA
  -> routing、confidence、slots、missing_fields、risk_flags、normalized
  -> parser_node 写入 LangGraph state
  -> LangGraph 根据 intent_flags 路由
```

自动化和人工联动验证结果：

| 测试集合 | 结果 |
|----------|------|
| Medical Lora 接口单元测试 | 7 passed |
| Medical Lora 默认测试集合 | 7 passed，2 skipped |
| Medical Lora 真实 GPU 模型测试 | 2 passed |
| Chronic Care Agent 外部客户端测试 | 4 passed |
| Medical Lora HTTP 健康检查 | 通过 |
| Medical Lora HTTP 意图解析 | 通过 |
| Agent 真实客户端调用 | 通过 |
| Agent parser_node 联动 | 通过 |
| Agent 应用依赖检查 | `intent_service=up` |
| Agent 完整提醒请求 | 通过 |

默认测试中的 2 个 skipped 是需要显式设置 `RUN_MODEL_TESTS=1` 的 GPU 测试，不是测试失败。这两个测试已经在真实 GPU 阶段单独运行并通过。

## 3 测试环境

### 3.1 硬件与系统

| 项目 | 版本或型号 |
|------|------------|
| 操作系统 | Windows 11 10.0.26200 |
| GPU | NVIDIA GeForce RTX 5080 |
| CUDA | 12.8 |
| Python | 3.14.0 64 bit |

### 3.2 主要依赖

| 依赖 | 版本 |
|------|------|
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.6.2 |
| PEFT | 0.19.1 |
| bitsandbytes | 0.49.2 |
| FastAPI | 0.162.0 |
| Uvicorn | 0.54.0 |
| pytest | 8.4.2 |

## 4 本次实现内容

### 4.1 Medical Lora 服务

新增 `service_app.py`，包含：

- FastAPI 应用工厂。
- 请求和响应 Pydantic Schema。
- Qwen2.5-7B 基座模型加载。
- routing LoRA adapter 加载。
- 4-bit NF4 量化配置。
- 固定 system prompt 和确定性生成参数。
- 单进程内推理锁。
- `GET /healthz`。
- `POST /v1/intent/parse`。
- 可选 Bearer token 鉴权。
- 模型输出 JSON 解析和顶层 Schema 校验。
- 异常到 HTTP 状态码的转换。

### 4.2 测试代码

新增：

```text
tests/test_service_api.py
tests/test_service_model.py
pytest.ini
service_requirements.txt
```

接口测试使用假的 runtime，不加载 GPU 模型。真实模型测试只有在设置 `RUN_MODEL_TESTS=1` 时执行。

### 4.3 服务边界规范化

联动过程中发现历史训练样本存在少量 Schema 差异，服务层增加了确定性规范化：

1. `reminder.recurrence` 转换为 `reminder.repeat`。
2. `plan.preferences=null` 转换为 `[]`。
3. `plan.constraints=null` 转换为 `[]`。
4. `advice.related_metrics=null` 转换为 `[]`。
5. 当 `missing_fields` 指向的槽位已经具有非空值时，移除该缺失字段。

规范化不修改路由结果、置信度、医学指标值或用户文本。

## 5 测试范围与结果

### 5.1 Medical Lora 接口单元测试

执行命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_service_api.py -q
```

结果：

```text
7 passed
```

覆盖用例：

| 编号 | 用例 | 预期 | 结果 |
|------|------|------|------|
| API-01 | 模型已就绪时请求 `/healthz` | HTTP 200 | 通过 |
| API-02 | 模型未就绪时请求 `/healthz` | HTTP 503 | 通过 |
| API-03 | 提交合法意图请求 | 返回六个顶层字段 | 通过 |
| API-04 | 提交空白文本 | HTTP 422 | 通过 |
| API-05 | 配置 API key 后不提供或提供错误 token | HTTP 401 | 通过 |
| API-06 | 模型输出不符合 Schema | HTTP 502 | 通过 |
| API-07 | 历史槽位格式规范化 | 转换为 Agent 可消费格式 | 通过 |

### 5.2 Medical Lora 默认测试集合

执行命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

结果：

```text
7 passed, 2 skipped
```

两个 GPU 用例默认跳过，以防普通测试意外加载 7B 模型。

### 5.3 真实 GPU 模型测试

执行命令：

```powershell
$env:BASE_MODEL_PATH = "C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"
$env:ROUTING_LORA_PATH = "C:\Users\gjh\PycharmProjects\Medical_Lora\training\qwen_routing_lora_final"
$env:RUN_MODEL_TESTS = "1"

.\.venv\Scripts\python.exe -m pytest tests\test_service_model.py -q -s
```

最终复测结果：

```text
2 passed in 60.37s
```

覆盖用例：

| 编号 | 输入 | 核心断言 | 结果 |
|------|------|----------|------|
| MODEL-01 | `明早8点提醒我测血糖` | `need_reminder=true`，提醒 Schema 已规范化 | 通过 |
| MODEL-02 | `我今天早上血压145/92，帮我记录一下，这正常吗？` | `need_record=true` 且 `need_advice=true` | 通过 |

模型成功完成：

- CUDA 检测。
- 4-bit 基座模型加载。
- LoRA adapter 加载。
- Chat template 构造。
- 确定性生成。
- JSON 解析。
- Pydantic 顶层 Schema 校验。

### 5.4 Medical Lora HTTP 测试

服务启动方式：

```powershell
.\.venv\Scripts\python.exe -m uvicorn service_app:app `
  --host 127.0.0.1 `
  --port 8020 `
  --workers 1
```

健康检查结果：

```json
{
  "status": "ok",
  "model": "qwen2.5-7b-routing-lora"
}
```

测试输入：

```text
我今天早上血压145/92，帮我记录一下，这正常吗？
```

关键响应：

```json
{
  "routing": {
    "need_record": true,
    "need_advice": true,
    "need_plan": false,
    "need_reminder": false,
    "need_history": false
  },
  "blood_pressure": {
    "systolic": 145,
    "diastolic": 92,
    "unit": "mmHg",
    "time": null
  }
}
```

结果：HTTP 接口、路由判断和血压槽位抽取均通过。

### 5.5 Chronic Care Agent 客户端测试

执行现有 Mock 测试：

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests\test_external_clients.py -q -p no:cacheprovider
```

结果：

```text
4 passed
```

随后使用真实 `ExternalIntentClient` 请求 Medical Lora：

```text
health=True
need_reminder=True
```

结果：Agent 客户端能够连接 8020 服务，并能够使用现有 `IntentResponse` 完成响应校验。

### 5.6 parser_node 联动测试

测试输入：

```text
记录血压145/92，这正常吗？
```

写入 LangGraph state 的关键结果：

```json
{
  "intent_flags": {
    "need_record": true,
    "need_advice": true,
    "need_plan": false,
    "need_reminder": false,
    "need_history": false
  },
  "blood_pressure": {
    "systolic": 145,
    "diastolic": 92,
    "unit": "mmHg",
    "time": null
  },
  "confidence": {
    "need_record": 0.96,
    "need_advice": 0.92,
    "need_plan": 0.02,
    "need_reminder": 0.03,
    "need_history": 0.04
  }
}
```

结果：真实模型结果已成功进入 Agent 的 LangGraph state。

### 5.7 Agent 应用级联动测试

Agent 使用隔离的临时 SQLite 数据库启动，未操作开发数据库。

系统接口结果：

| 接口 | 结果 |
|------|------|
| `GET /healthz` | HTTP 200，`status=ok` |
| `GET /readyz` | HTTP 200，`status=ready` |
| `GET /dependencies` | `intent_service=up`，`external_rag=down` |

完整请求流程：

1. 使用开发模式登录。
2. 创建测试会话。
3. 发送 `明早8点提醒我测血糖`。
4. Agent 调用 Medical Lora。
5. LangGraph 得到 `need_reminder=true`。
6. 工作流返回 `status=completed`。

结果：意图识别服务的应用级连接通过。

## 6 测试中发现的问题

### 6.1 已修复问题

| 问题 | 影响 | 处理 |
|------|------|------|
| `recurrence` 与 `repeat` 并存 | Agent 可能丢失重复提醒规则 | 服务层统一为 `repeat` |
| 数组槽位输出 `null` | Agent 使用 `join()` 时可能报错 | 服务层统一为空数组 |
| 已有提醒时间仍出现在 `missing_fields` | Agent 可能重复追问时间 | 服务层移除明显矛盾项 |

### 6.2 尚未阻塞但需要关注的问题

#### 风险标记不稳定

血压 `145/92` 的真实输出中 `risk_flags` 为空，未返回预期的 `high_blood_pressure`。这不影响意图路由和槽位抽取，但说明模型风险预标不能作为唯一安全判断。

建议继续保留 Chronic Care Agent 的本地血压、血糖风险规则，以规则引擎为安全判断依据。

#### 测试依赖警告

测试过程中出现以下非阻塞警告：

- Starlette 提示当前 `TestClient` 的 httpx 集成未来将迁移到 `httpx2`。
- PyTorch 提示 Python 3.14 下部分 `torch.jit` API 未来可能不兼容。
- bitsandbytes 使用了 PyTorch 将来会移除的内部检查 API。
- Transformers 提示模型 generation config 中的 `top_k` 在当前确定性生成下被忽略。
- 当前环境未安装 Triton，flop counting 不可用；不影响本次推理。

这些警告没有造成测试失败，但升级依赖时应重新执行完整测试。

## 7 未覆盖范围

以下内容未在本次报告中判定为通过：

- RAG 服务真实调用。测试时 `external_rag=down`。
- 提醒任务是否真正持久化到提醒表。
- reminder worker 是否按时间触发通知。
- 重复提醒、修改提醒和删除提醒的完整业务流程。
- 医学建议和计划生成使用的外部通用 LLM。
- 多用户并发和 GPU 压力测试。
- 长文本、恶意输入和极端 token 长度测试。
- API key 开启后的两个真实进程联动测试。鉴权逻辑仅完成接口单元测试。
- Docker 和跨机器部署测试。

因此，本报告的“通过”结论限定为：意图模型服务、HTTP 契约、Agent 客户端、parser_node 和基础 LangGraph 路由已经连通。

## 8 后续建议

按优先级建议：

1. 增加五类单意图和常见多意图的固定回归数据集。
2. 为 `slots` 增加更严格的嵌套 Pydantic Schema。
3. 对风险标记与本地规则引擎进行一致性测试。
4. 增加真实提醒持久化和 reminder worker 测试。
5. 启动 RAG 服务后补建议和计划链路端到端测试。
6. 增加 GPU 串行队列长度、超时和并发压力测试。
7. 生产环境启用 API key，并限制 Medical Lora 监听地址或网络访问范围。

## 9 复测命令

### 接口测试

```powershell
cd C:\Users\gjh\PycharmProjects\Medical_Lora
.\.venv\Scripts\python.exe -m pytest tests\test_service_api.py -q
```

### 真实模型测试

```powershell
$env:BASE_MODEL_PATH = "C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"
$env:ROUTING_LORA_PATH = "C:\Users\gjh\PycharmProjects\Medical_Lora\training\qwen_routing_lora_final"
$env:RUN_MODEL_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest tests\test_service_model.py -q -s
```

### 启动模型服务

```powershell
.\.venv\Scripts\python.exe -m uvicorn service_app:app --host 127.0.0.1 --port 8020 --workers 1
```

### Agent 客户端真实调用

```powershell
cd C:\Users\gjh\PycharmProjects\chronic_care_agent_2
.\.venv\Scripts\python.exe -c "from app.intent_client.client import ExternalIntentClient; from app.intent_client.schemas import IntentRequest; client=ExternalIntentClient('http://127.0.0.1:8020'); print(client.parse(IntentRequest(text='明早8点提醒我测血糖')).model_dump())"
```

## 10 最终判定

测试判定：**通过，可进入下一阶段功能联调。**

进入下一阶段前应明确：当前已经证明意图识别与 Agent 路由链路可以工作，但提醒持久化、RAG、外部 LLM 和生产并发能力仍需要分别验证。
