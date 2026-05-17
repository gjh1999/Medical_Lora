你是一个慢性病健康管理 App 的数据标注专家。

请生成 100 条"记录健康数据 + 询问是否正常/是否有问题"的中文样本。用户既提供健康数据，又询问该数据是否正常、是否需要担心。

要求：
- routing.need_record = true
- routing.need_advice = true
- routing.need_plan = false
- routing.need_reminder = false
- routing.need_history = false
- slots.record 必须包含健康数据（血压/血糖/体重/症状等），至少一个字段非空
- slots.advice 必须包含用户想咨询的问题，topic 和 question 不能为空
- related_metrics 必须包含与 record 中对应的指标
- 覆盖血压、血糖、体重、症状等类型
- 包含多种表达风格：标准表达、口语表达、省略表达、模糊表达
- 不要输出诊断结论（如"你是高血压"）
- 问题可以是：高不高、正常吗、严重吗、需要去医院吗、怎么回事等
- 不要重复句式
- 路由与slots一致性（非常重要）：
  * need_record=true => slots.record至少一个字段有数据
  * need_advice=true => slots.advice.topic和question不能为null
  * need_plan/need_reminder/need_history=false => 对应slots必须全部为null
- 只输出 JSONL，每行一个合法 JSON，不要用 ```json 代码块，不要输出解释

输出格式：
{"input": "用户自然语言", "output": {"routing": {"need_record": true, "need_advice": true, "need_plan": false, "need_reminder": false, "need_history": false}, "confidence": {"need_record": 0.95, "need_advice": 0.92, "need_plan": 0.03, "need_reminder": 0.04, "need_history": 0.05}, "slots": {"record": {"blood_pressure": null, "blood_glucose": null, "weight": null, "diet": null, "exercise": null, "medication": null, "symptom": null}, "advice": {"topic": "topic_name", "question": "用户的具体问题", "related_metrics": ["对应指标"]}, "plan": {"goal": null, "duration": null, "preferences": [], "constraints": []}, "reminder": {"action": null, "task": null, "time": null, "repeat": null}, "history": {"metric": null, "time_range": null, "aggregation": null}}, "missing_fields": [], "risk_flags": [], "normalized": {"date": null, "time_of_day": null, "units": {}}}}

advice.topic 可选：blood_pressure, blood_glucose, weight, diet, exercise, medication, symptom, general_health
