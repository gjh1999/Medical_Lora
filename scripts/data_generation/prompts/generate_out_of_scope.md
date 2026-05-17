你是一个慢性病健康管理 App 的数据标注专家。

请生成 50 条"闲聊、无关问题、越界医疗请求"的中文样本。这些是用户输入但不属于本系统的业务范围。

要求：
分类生成：
1. 闲聊/问候类（15条）：如"你好"、"今天天气不错"、"谢谢"、"你能做什么"
   - routing 五个字段全部为 false
2. 无关问题类（15条）：如"帮我写一篇文章"、"今天股市怎么样"、"推荐一部电影"、"翻译一段英文"
   - routing 五个字段全部为 false
3. 越界医疗请求类（20条）：如"我能不能自己停掉降压药"、"给我开点降糖药"、"我这个病能不能治好"
   - 对于危险医疗请求，在 risk_flags 中标记 "medication_risk" 或其他风险标记
   - routing 通常全部为 false，但有些可能触发 need_advice（用户真的在咨询健康问题）
   - 不要生成医学诊断结论
   - 不要在 assistant 回复中给出停药、加药、开药建议

- slots 尽量为空
- 包含多种表达风格
- 不要重复句式
- 路由与slots一致性（非常重要）：
  * 闲聊/无关问题：routing全部false，slots全部为null
  * 越界医疗请求：routing通常全部false，risk_flags可标记风险，但slots全部为null
  * 如果确实触发了need_advice（用户真的在咨询），则slots.advice必须非空
- 只输出 JSONL，每行一个合法 JSON，不要用 ```json 代码块，不要输出解释

输出格式：
{"input": "用户自然语言", "output": {"routing": {"need_record": false, "need_advice": false, "need_plan": false, "need_reminder": false, "need_history": false}, "confidence": {"need_record": 0.01, "need_advice": 0.02, "need_plan": 0.01, "need_reminder": 0.01, "need_history": 0.01}, "slots": {"record": {所有字段为null}, "advice": {"topic": null, "question": null, "related_metrics": []}, "plan": {"goal": null, "duration": null, "preferences": [], "constraints": []}, "reminder": {"action": null, "task": null, "time": null, "repeat": null}, "history": {"metric": null, "time_range": null, "aggregation": null}}, "missing_fields": [], "risk_flags": [], "normalized": {"date": null, "time_of_day": null, "units": {}}}}
