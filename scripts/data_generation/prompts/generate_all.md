你是一个慢性病健康管理 App 的数据标注专家。

请生成 50 条中文用户输入样本，并为每条样本标注结构化 JSON。

任务目标：
1. 判断用户输入需要触发哪些 LangGraph 节点
2. 输出 routing 五个布尔值：
   - need_record
   - need_advice
   - need_plan
   - need_reminder
   - need_history
3. 抽取健康数据、建议问题、计划需求、提醒任务、历史查询条件等 slots
4. 判断是否有缺失字段 missing_fields
5. 判断是否有风险信号 risk_flags
6. 输出 JSONL，每行一个 JSON，不要输出解释

routing 字段格式：
{
  "need_record": false,
  "need_advice": false,
  "need_plan": false,
  "need_reminder": false,
  "need_history": false
}

输出格式：
{
  "input": "用户自然语言",
  "output": {
    "routing": {
      "need_record": false,
      "need_advice": false,
      "need_plan": false,
      "need_reminder": false,
      "need_history": false
    },
    "confidence": {
      "need_record": 0.0,
      "need_advice": 0.0,
      "need_plan": 0.0,
      "need_reminder": 0.0,
      "need_history": 0.0
    },
    "slots": {
      "record": {
        "blood_pressure": null,
        "blood_glucose": null,
        "weight": null,
        "diet": null,
        "exercise": null,
        "medication": null,
        "symptom": null
      },
      "advice": {
        "topic": null,
        "question": null,
        "related_metrics": []
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
    "risk_flags": [],
    "normalized": {
      "date": null,
      "time_of_day": null,
      "units": {}
    }
  }
}

生成要求：
- 只输出 JSONL，不要输出 Markdown，不要输出解释
- 每行必须是一个合法 JSON
- 不要使用 ```json 代码块
- 覆盖血压、血糖、体重、饮食、运动、用药、症状、提醒、历史查询、计划制定
- 至少 10 条是口语表达
- 至少 5 条是省略表达
- 至少 5 条包含模糊时间
- 至少 3 条包含错别字或不规范表达
- 至少 8 条同时触发多个 routing=true
- 至少 5 条 missing_fields 非空
- 至少 3 条 routing 全部为 false
- 不要生成医学诊断结论
- 不要生成停药、加药、开药等危险建议
- 不要重复句式
- 路由与slots一致性（最重要）：
  * routing.X=true => slots.X对应字段非空；routing.X=false => slots.X全部为null
  * 每个样本生成后自检：routing和slots是否匹配？
