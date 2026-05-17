你是一个慢性病健康管理 App 的数据标注专家。

请生成 100 条"制定健康计划"的中文样本。用户要求制定健康管理计划（饮食、运动、用药、监测等）。

要求：
- routing.need_plan = true
- routing.need_advice 通常为 true（用户在制定计划时往往需要建议）
- 如果用户要求基于历史数据制定个性化计划，则 routing.need_history = true（约40条）
- 如果用户同时要求设置提醒，则 routing.need_reminder = true（约20条）
- slots.plan 必须包含 goal、duration，preferences 和 constraints 按需填写
- 覆盖的目标类型：控制血压、控制血糖、减重、改善饮食、增加运动、规律用药、综合健康管理
- 覆盖的周期：一周、一个月、两周、三天、一个月
- 覆盖的偏好：饮食调整、轻度运动、低盐饮食、低糖饮食、有氧运动、力量训练等
- 包含多种表达风格：标准表达、口语表达、省略表达
- 不要生成具体危险医疗方案（如"每天只吃一顿"）
- 不要重复句式
- 路由与slots一致性（非常重要）：
  * need_plan=true => slots.plan.goal和duration不能为null
  * need_advice=true时 => slots.advice.topic/question不能为null
  * need_history=true时 => slots.history.metric/time_range不能为null
  * need_reminder=true时 => slots.reminder.task/time不能为null
  * routing为false的 => 对应slots必须全部为null
- 只输出 JSONL，每行一个合法 JSON，不要用 ```json 代码块，不要输出解释

输出格式：
{"input": "用户自然语言", "output": {"routing": {"need_record": false, "need_advice": true, "need_plan": true, "need_reminder": false, "need_history": false}, "confidence": {"need_record": 0.04, "need_advice": 0.88, "need_plan": 0.96, "need_reminder": 0.05, "need_history": 0.06}, "slots": {"record": {"blood_pressure": null, "blood_glucose": null, "weight": null, "diet": null, "exercise": null, "medication": null, "symptom": null}, "advice": {"topic": "对应topic", "question": "相关问题", "related_metrics": ["对应指标"]}, "plan": {"goal": "目标", "duration": "周期", "preferences": ["偏好1"], "constraints": []}, "reminder": {"action": null, "task": null, "time": null, "repeat": null}, "history": {"metric": null, "time_range": null, "aggregation": null}}, "missing_fields": [], "risk_flags": [], "normalized": {"date": null, "time_of_day": null, "units": {}}}}
