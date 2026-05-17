你是一个慢性病健康管理 App 的数据标注专家。

请生成 50 条"信息不完整，需要追问"的中文样本。用户表达了意图，但缺少关键信息。

要求：
- 至少一个 routing 为 true
- missing_fields 必须非空，包含缺失字段的路径（如 "record.blood_glucose.value"）
- slots 中允许相关字段为 null
- 覆盖以下不完整场景：
  1. 记录类缺信息：只说记录了但没说数值（如"帮我记一下血糖"）
  2. 提醒类缺信息：只说提醒但没说什么时间（如"提醒我吃药"）
  3. 计划类缺信息：只说做计划但没说周期（如"帮我做个降压计划"）
  4. 历史类缺信息：只说查询但没说什么时间范围（如"查一下我的血压"）
- 每种不完整场景至少 10 条
- 包含多种表达风格
- 不要重复句式
- 路由与slots一致性（非常重要）：
  * 信息不完整时，routing仍要标记用户意图（如要记录但没给数值，need_record仍为true）
  * routing=true但信息缺失的字段在slots中填null，同时在missing_fields中标记路径
  * 没有触发的routing，对应slots全部为null
- 只输出 JSONL，每行一个合法 JSON，不要用 ```json 代码块，不要输出解释

输出格式：
{"input": "用户自然语言", "output": {"routing": {"need_record": true/false, ...}, "confidence": {...}, "slots": {...}, "missing_fields": ["字段路径1", "字段路径2"], "risk_flags": [], "normalized": {"date": null, "time_of_day": null, "units": {}}}}

missing_fields 路径格式示例：
- record.blood_pressure.systolic
- record.blood_glucose.value
- record.blood_glucose.time
- reminder.time
- reminder.task
- plan.duration
- history.time_range
