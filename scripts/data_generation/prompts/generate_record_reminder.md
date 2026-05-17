你是一个慢性病健康管理 App 的数据标注专家。

请生成 100 条"记录健康数据 + 创建提醒"的中文样本。用户提供健康数据后，要求设置一个后续提醒（如复测、吃药等）。

要求：
- routing.need_record = true
- routing.need_advice = false
- routing.need_plan = false
- routing.need_reminder = true
- routing.need_history = false
- slots.record 必须包含健康数据
- slots.reminder 必须包含提醒任务，action 为 "create"，task 和 time 不能为空
- 覆盖的时间表达：明早、今晚、饭后两小时、睡前、每天、每周、具体时间点（如晚上八点）
- 覆盖的提醒任务：测血压、测血糖、吃药、运动、复测等
- 包含多种表达风格
- 不要重复句式
- 路由与slots一致性（非常重要）：
  * need_record=true => slots.record至少一个字段有数据
  * need_reminder=true => slots.reminder.task和time不能为null
  * need_advice/need_plan/need_history=false => 对应slots必须全部为null
- 只输出 JSONL，每行一个合法 JSON，不要用 ```json 代码块，不要输出解释

输出格式：
{"input": "用户自然语言", "output": {"routing": {"need_record": true, "need_advice": false, "need_plan": false, "need_reminder": true, "need_history": false}, "confidence": {"need_record": 0.95, "need_advice": 0.05, "need_plan": 0.03, "need_reminder": 0.94, "need_history": 0.02}, "slots": {"record": {"blood_pressure": null, "blood_glucose": null, "weight": null, "diet": null, "exercise": null, "medication": null, "symptom": null}, "advice": {"topic": null, "question": null, "related_metrics": []}, "plan": {"goal": null, "duration": null, "preferences": [], "constraints": []}, "reminder": {"action": "create", "task": "提醒任务", "time": "时间", "repeat": "重复规则或null"}, "history": {"metric": null, "time_range": null, "aggregation": null}}, "missing_fields": [], "risk_flags": [], "normalized": {"date": null, "time_of_day": null, "units": {}}}}
