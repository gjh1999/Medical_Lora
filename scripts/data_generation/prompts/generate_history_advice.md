你是一个慢性病健康管理 App 的数据标注专家。

请生成 100 条"查询历史数据 + 请求解释建议"的中文样本。用户查询历史健康数据趋势，并要求解释或建议。

要求：
- routing.need_record = false
- routing.need_advice = true
- routing.need_plan = false
- routing.need_reminder = false
- routing.need_history = true
- slots.history 必须包含 metric、time_range、aggregation，都不能为空
- slots.advice 必须包含 question 和 related_metrics
- 覆盖的指标：blood_pressure、blood_glucose、weight、exercise、diet
- 覆盖的时间范围：最近一周、最近一个月、最近三天、这几天、上个月、本周、本月
- 覆盖的聚合类型：trend、average、max、min、latest、count、summary
- 问题类型：趋势怎么样、有没有变高/变低、稳定吗、有什么问题吗、需要调整吗
- 包含多种表达风格：标准表达、口语表达、省略表达
- 不要生成医学诊断结论
- 不要重复句式
- 路由与slots一致性（非常重要）：
  * need_advice=true => slots.advice.question和related_metrics不能为null
  * need_history=true => slots.history.metric、time_range、aggregation都不能为null
  * need_record/need_plan/need_reminder=false => 对应slots必须全部为null
- 只输出 JSONL，每行一个合法 JSON，不要用 ```json 代码块，不要输出解释

输出格式：
{"input": "用户自然语言", "output": {"routing": {"need_record": false, "need_advice": true, "need_plan": false, "need_reminder": false, "need_history": true}, "confidence": {"need_record": 0.03, "need_advice": 0.91, "need_plan": 0.05, "need_reminder": 0.02, "need_history": 0.95}, "slots": {"record": {"blood_pressure": null, "blood_glucose": null, "weight": null, "diet": null, "exercise": null, "medication": null, "symptom": null}, "advice": {"topic": "对应topic", "question": "用户的问题", "related_metrics": ["对应指标"]}, "plan": {"goal": null, "duration": null, "preferences": [], "constraints": []}, "reminder": {"action": null, "task": null, "time": null, "repeat": null}, "history": {"metric": "查询指标", "time_range": "时间范围", "aggregation": "聚合类型"}}, "missing_fields": [], "risk_flags": [], "normalized": {"date": null, "time_of_day": null, "units": {}}}}
