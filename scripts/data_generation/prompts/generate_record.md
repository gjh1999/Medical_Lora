你是一个慢性病健康管理 App 的数据标注专家。

请生成 100 条"仅记录健康数据"的中文样本。用户只提供健康数据，不询问建议、不查询历史、不制定计划、不设置提醒。

要求：
- routing.need_record = true
- routing.need_advice = false
- routing.need_plan = false
- routing.need_reminder = false
- routing.need_history = false
- 覆盖以下所有类型：血压、血糖、体重、饮食、运动、用药、症状。每种至少 10 条
- 包含多种表达风格：标准表达（30条）、口语表达（25条）、省略表达（15条）、错别字/不规范表达（10条）、模糊时间表达（20条）
- slots.record 必须包含对应的健康数据，不能为空
- 不要生成医学诊断结论
- 不要重复句式
- 路由与slots一致性（非常重要）：
  * need_record=true => slots.record至少一个字段有数据
  * need_advice/need_plan/need_reminder/need_history=false => 对应slots必须全部为null
- 只输出 JSONL，每行一个合法 JSON，不要用 ```json 代码块，不要输出解释

输出格式：
{"input": "用户自然语言", "output": {"routing": {"need_record": true, "need_advice": false, "need_plan": false, "need_reminder": false, "need_history": false}, "confidence": {"need_record": 0.95, "need_advice": 0.05, "need_plan": 0.02, "need_reminder": 0.03, "need_history": 0.02}, "slots": {"record": {"blood_pressure": null, "blood_glucose": null, "weight": null, "diet": null, "exercise": null, "medication": null, "symptom": null}, "advice": {"topic": null, "question": null, "related_metrics": []}, "plan": {"goal": null, "duration": null, "preferences": [], "constraints": []}, "reminder": {"action": null, "task": null, "time": null, "repeat": null}, "history": {"metric": null, "time_range": null, "aggregation": null}}, "missing_fields": [], "risk_flags": [], "normalized": {"date": null, "time_of_day": null, "units": {}}}}

slots.record 字段说明：
- blood_pressure: {"systolic": 收缩压, "diastolic": 舒张压, "unit": "mmHg", "time": "时间"}
- blood_glucose: {"value": 数值, "unit": "mmol/L", "type": "fasting/postprandial/random/before_sleep/unknown", "time": "时间"}
- weight: {"value": 数值, "unit": "kg", "time": "时间"}
- diet: {"meal": "早餐/午餐/晚餐/零食", "content": "食物内容", "time": "时间"}
- exercise: {"type": "运动类型", "duration_minutes": 分钟数, "time": "时间"}
- medication: {"name": "药名", "dose": "剂量或null", "time": "时间"}
- symptom: {"name": "症状名", "duration": "持续时间或null", "severity": "mild/moderate/severe/unknown", "time": "时间"}
