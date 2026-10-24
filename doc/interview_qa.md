# 意图识别与结构化信息提取 LoRA 微调 — 面试追问 Q&A

## 第一层：项目概述追问

### Q1: 你这个项目为什么要做微调，而不直接用 Prompt Engineering？

我做了 Prompt-only（Zero-shot / 5-shot）对比实验来验证。在 Qwen2.5-7B-Instruct 上，即使给了详细的 Schema 描述和 Few-shot 示例，路由完全匹配率只有 52.7%，多意图样本准确率只有 35.5%，Macro F1 0.842。而 LoRA 微调后 Macro F1 达 0.990，多路由准确率 100%。

核心原因：Prompt 是"提醒"模型怎么做，但基础模型不理解业务特有的路由 Schema 语义（比如什么算 need_record 什么算 need_advice），微调是将这种判断逻辑"内化"到模型权重中。

### Q2: 你的意图识别定义了哪几个意图？怎么划分的？

5 个路由开关：need_record（健康数据记录）、need_advice（健康建议咨询）、need_plan（健康计划制定）、need_reminder（提醒设置）、need_history（历史数据查询）。划分逻辑是按下游 LangGraph 节点的功能边界来定——每个 flag 对应一个下游处理节点。多个 flag 可以同时为 true，约 48% 的样本是多意图的。

### Q3: 为什么用合成数据而不是真实用户数据？

项目处于早期验证阶段，没有真实用户数据。我使用 DeepSeek V4 大模型生成合成数据，覆盖慢病管理常见场景（血压/血糖/体重/饮食/运动/用药/症状），经过 JSON 校验 → 槽位规范化 → Schema 校验 → Routing 一致性校验 → 去重 → 分布分析 → 划分的完整流水线。最终产出 1,334 条高质量数据（训练 1,068 / 验证 135 / 测试 131）。


## 第二层：技术细节追问

### Q4: LoRA 的原理是什么？为什么不直接全参数微调？

LoRA 假设预训练权重在领域适配时的变化量 ΔW 具有低内在秩，因此不需要更新整个权重矩阵，只需学习两个低秩矩阵 A（d×r）和 B（r×k），其中 r << min(d, k)。推理时直接合并：W' = W + BA，无额外延迟。

不用全参数微调的原因：7B 模型全量微调需要约 4×28GB = 112GB 显存，而 QLoRA 在 RTX 5080（17GB）上就可以跑，且只产生 5MB 的 adapter 权重，方便管理。

### Q5: LoRA 的 rank 和 alpha 你是怎么选的？为什么 target_modules 选了 q_proj 和 v_proj？

先解释几个基础概念。

**Q/K/V/O 是什么？** Transformer 的注意力层有 4 个线性投影矩阵，把输入向量映射到不同用途的空间：

```
输入向量 × W_q → Q（Query，查询向量）   "我要找什么？"
输入向量 × W_k → K（Key，键向量）       "我是什么？"
输入向量 × W_v → V（Value，值向量）      "我包含什么内容？"
注意力输出 × W_o → O（Output，输出投影） "整合后输出"
```

这 4 个矩阵就是 `q_proj`、`k_proj`、`v_proj`、`o_proj`。

**rank (r) 是什么？** LoRA 的核心思想是：微调时权重矩阵 W 的变化量 ΔW 不需要那么高的维度，可以用两个小矩阵的乘积来近似：

```
原始前向传播:  h = W · x            （W 是 4096×4096 的大矩阵，1600 万参数）
LoRA 前向传播: h = W · x + B·A·x   （W 冻结，只训练 A 和 B）

其中 A 的形状: r × 4096（比如 r=8，8000 参数）
     B 的形状: 4096 × r（8000 参数）
     B·A 的形状: 4096×4096（和 W 一样大，但由两个小矩阵拼出来）
```

这就是"低秩"——用一个瘦矩阵 A 和一个矮矩阵 B 来近似完整的大矩阵。r 控制 LoRA 能学到多少个"独立的变化方向"：r 越大，表达能力越强但参数越多。我的任务本质是格式学习（让模型学会输出固定 JSON），不是注入新知识，r=8 足够。

**alpha 是什么？** alpha 是缩放系数，控制 LoRA 对模型行为的影响力。实际计算时，LoRA 的贡献是 `(alpha / r) × B·A·x`。常见做法是 alpha = 2×r，我设 r=8, alpha=16，这样 alpha/r=2，LoRA 有较明显的影响力。

**为什么只选 q_proj 和 v_proj？** 直接原因是显存——RTX 5080 只有 17GB，加上 k_proj 和 o_proj 会 OOM。更深层的原因：q_proj 和 v_proj 直接影响注意力计算，是改变模型输出行为的最关键模块；k_proj 和 o_proj 是补充作用；FFN 层的 gate/up/down 更适合需要注入大量新知识的场景。从结果看，两个模块已足够（Macro F1 0.990）。

### Q6: 你用的学习率是 2e-4，为什么选这个值？

LoRA 微调的学习率可以比全参数微调高一个数量级，因为只训练少量参数且 B 矩阵初始化为零。典型范围 1e-4 ~ 3e-4，我选 2e-4 是因为数据量小（1000+ 条）、任务明确，配合 cosine 调度 + 10% warmup，在实际训练中 eval loss 从初始一直下降到 0.174，没有出现过拟合迹象。

### Q7: 微调后模型会"变傻"吗？你怎么防止灾难性遗忘？

我的任务场景对灾难性遗忘不敏感——这个模型在生产中是作为 LangGraph 的路由节点，只负责输出结构化 JSON，不承担通用对话任务。但如果需要保留通用能力，我会：

1. 在训练数据中混入 10-20% 通用数据
2. 使用 LoRA 本身就是一种防遗忘手段（只修改少量参数）
3. 用 C-Eval / MMLU 等基准监控通用能力变化


## 第三层：结构化输出追问

### Q8: 模型输出的 JSON 怎么保证格式合法？如果遇到 JSON 解析失败怎么办？

训练层面：训练数据中 assistant.content 是纯 JSON 字符串，不带 Markdown 代码块，模型学到了直接输出纯 JSON 的习惯。结果：测试集 JSON 合法率 100%（131/131）。

工程层面：推理时加 try-catch + 正则兜底（去除可能的代码块标记），如果仍解析失败则返回默认降级路由（全 false），下游走默认逻辑，不会让整个 pipeline 崩溃。

### Q9: 你的 Slot 槽位抽取准确率怎么样？为什么没有统计 Slot Exact？

我的 LoRA 评估脚本（eval_routing_lora.py）主要统计了 Routing 层面的指标——JSON 合法率、每个 flag 的 Precision/Recall/F1、多路由样本准确率，没有统计 Slot Exact。

原因：在这个系统架构中，**Routing 的准确性比 Slot 的精确匹配更重要**。路由决定了下游哪个节点被激活（控制流程走向），而 Slot 里的具体字段值可以交给下游节点做二次解析和纠错。比如模型把"今天早上"放到了 blood_pressure.time，即使时间格式和我们期望的不完全一致，下游节点仍然能解析出来。

作为参考，Prompt-only 方案我跑过 Slot Exact，结果只有 6.9%——说明严格完全匹配是个很苛刻的指标，连 Prompt 输出的 JSON 在结构完全合法的情况下，字段值和标准答案也很难逐字符一致。后续如果需要，可以在 LoRA 评估中也加上字段级 Slot F1（比 Slot Exact 更合理），以及人工抽检 Slot 填充质量。

### Q10: 你的模型输出了一个很复杂的 JSON（含 routing / confidence / slots / missing_fields / risk_flags），是怎么设计这个 Schema 的？Slot 槽位又是什么意思？

**先解释槽位（Slot）是什么。** 槽位就是从用户自然语言中提取出来的结构化字段，就像填表格一样，把散落在句子里的信息填到对应的格子里。比如用户说：

> "我今天早上血压 145/92，帮我记一下"

模型识别后填入 record 槽位：

```
record 槽位:
  ├─ blood_pressure: { systolic: 145, diastolic: 92, unit: "mmHg", time: "今天早上" }  ← 填了
  ├─ blood_glucose:  null   ← 没提到血糖
  ├─ weight:         null   ← 没提到体重
  ├─ diet:           null
  ├─ exercise:       null
  ├─ medication:     null
  └─ symptom:        null
```

每个路由 flag 下面各有一套槽位：need_record 下有 7 个健康指标槽位，need_advice 下有 topic/question 槽位，need_plan 下有 goal/duration 槽位，等等。

**Schema 设计原则**：先有下游需求，再定义输出结构。

- routing（5 个 bool）：对应 LangGraph 的 5 个下游节点，决定激活谁
- confidence（0~1 浮点）：给下游做阈值判断，比如 need_record > 0.5 才触发记录节点
- slots（嵌套对象）：按业务领域拆分为 5 个子槽位，存下游节点需要的结构化字段
- missing_fields（字符串数组）：标记缺失的关键信息，下游用它生成追问
- risk_flags（字符串数组）：标记风险信号（如 high_blood_pressure），不做诊断只做预警

关键原则：**每个字段都有明确的消费者**（下游代码或 UI），不输出没人用的字段。


## 第四层：方法论和思考追问

### Q11: 你做了 Prompt-only vs LoRA 的对比实验，这个实验设计有什么可以改进的地方？

三个改进方向：

1. 增加字段级 slot F1 指标，而不是只看严格的 Slot Exact（过严）
2. 设计更合理的 Few-shot 示例选择策略——我当前是从 dev 集顺序选 5 条，应该按路由组合模式分层采样，确保每种组合都有示例覆盖
3. 补充更多 baseline，比如 Prompt + JSON Mode（response_format）、不同基座模型（如 GPT-4o）的 prompt-only 表现

### Q12: 如果让你重新做这个项目，你会怎么做？

1. **数据层面**：先收集 200-300 条真实用户语料做种子，再用大模型做数据增强，比纯合成数据更贴近真实分布
2. **训练层面**：增加 k_proj 和 o_proj 作为 target modules，rank 提到 16 试试效果
3. **评估层面**：增加人工评估环节——随机抽 50 条让业务同学判断路由是否合理
4. **部署层面**：合并 LoRA 权重到基座模型后用 vLLM 部署，避免每次推理时加载 adapter 的开销

### Q13: 你项目的 Macro F1 是 0.990，need_advice 的 Precision 是 0.908，为什么这个指标偏低？你怎么看这个 trade-off？

先解释这几个指标的含义，以 need_advice 为例。测试集 131 条，每一条模型判断"是否需要给建议"：

```
                    标签=需要建议              标签=不需要建议

模型预测=需要建议      TP（猜对了）              FP（误报）
                                                   偶尔多标了

模型预测=不需要建议    FN（漏了）                TN（猜对了）
                       0 条
```

- **Precision（精确率 / 查准率）**：模型说"需要建议"的时候，有多少是真的需要。`TP/(TP+FP)`，need_advice 得 0.908——喊了 100 次"要建议"，有 9 次是多喊的
- **Recall（召回率 / 查全率）**：真正需要建议的样本，模型找出来多少。`TP/(TP+FN)`，need_advice 得 1.000——一条没漏
- **F1**：Precision 和 Recall 的调和平均，防止只看一个被蒙蔽。`2×P×R/(P+R)`

need_advice 的 Recall=1.0（该给建议的一条没漏），Precision=0.908（偶尔多给了建议），所以 F1 被拉低到 0.952。分析发现模型有时在用户纯报数据时也会标 need_advice——比如"血压 145/92"，模型觉得数值偏高，顺手把建议也开了。

从业务角度看，**这个 trade-off 是安全的**：多标 need_advice 只会让下游建议节点多回一条"您的血压偏高，建议注意饮食"，不会产生危险。但如果 need_record 漏标，健康数据就丢了。所以宁可多标不可漏标——Recall 比 Precision 更优先。


## 第五层：工程落地追问

### Q14: 1300 多条数据够吗？为什么不生成更多？

在这个任务的复杂度下够了。5 个二分类标签 + 固定 Schema 的 JSON 输出本质上是"格式学习"而非"知识学习"，1000+ 条数据已经覆盖了常见路由组合模式。eval loss 从 1.865 降到 0.174，测试集全部指标达标，说明数据量是充分的。如果后续增加场景（比如新增路由 flag 或复杂槽位），才需要追加数据。

### Q15: 模型训练只用了 20 分钟，这个训练效率你是怎么做到的？

QLoRA 的 4-bit 量化将 7B 模型压缩到约 4GB 显存，加上 LoRA 只训练约 0.1% 的参数（几百万参数），batch size=1 × 梯度累积 4，3 epochs × 1068 条 ≈ 800 步，RTX 5080 上约 20 分钟完成。这个速度意味着可以快速试错——改超参 → 重训 → 评估，一天可以跑十几轮迭代。


## 基础知识补充

### 训练框架详解：一个训练脚本里五个框架各干了什么

训练脚本 `finetune_routing_lora.py` 共用到 5 个框架，按执行顺序解释：

**1. Transformers（HuggingFace）— 模型加载 + 分词器**

```python
tokenizer = AutoTokenizer.from_pretrained(model_name)       # 加载分词器
model = AutoModelForCausalLM.from_pretrained(model_name, ...)  # 加载基座模型
TrainingArguments(...)  # 定义训练超参
```

- AutoTokenizer：把文本切成 token id 序列，Qwen 的 tokenizer 还自带 Chat Template（自动拼 `<|im_start|>system\n...<|im_end|>` 格式）
- AutoModelForCausalLM：加载 Qwen2.5-7B 的完整模型权重到内存
- TrainingArguments：定义 batch_size、学习率、保存策略等

**2. BitsAndBytes — 量化引擎，把 7B 模型从 ~28GB 压到 ~4GB**

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # 4-bit 量化
    bnb_4bit_compute_dtype=torch.bfloat16,      # 计算时仍用 BF16
    bnb_4bit_quant_type="nf4",                  # NF4（正态分布优化）量化
    bnb_4bit_use_double_quant=True,             # 双重量化
)
```

没有它，Qwen2.5-7B 需要约 28GB 显存（7B × 4 字节 FP32），RTX 5080（17GB）根本跑不了。BitsAndBytes 在加载模型时把每个参数从 32-bit 压到 4-bit（8 倍压缩），计算时仍用 BF16，精度损失很小。QLoRA 里的 "Q"（Quantized）就是这个库实现的。

**3. PEFT — 参数高效微调，在模型上"挂"LoRA 适配器**

```python
model = prepare_model_for_kbit_training(model)     # 预处理量化模型
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],           # 只在 Q 和 V 矩阵上挂 LoRA
)
```

PEFT 全称 Parameter-Efficient Fine-Tuning。两件事：
- `prepare_model_for_kbit_training`：让量化模型适配 LoRA 训练
- `LoraConfig`：定义适配器参数——在哪几层挂、rank 多大、alpha 多大

它只负责定义和注入适配器，不负责训练。

**4. TRL（SFTTrainer）— 训练调度核心**

```python
trainer = SFTTrainer(
    model=model, processing_class=tokenizer,
    train_dataset=..., eval_dataset=...,
    peft_config=lora_config,
    formatting_func=format_messages,    # 每条样本拼成 ChatML prompt
    args=training_args,
)
trainer.train()
```

TRL 全称 Transformer Reinforcement Learning，但 SFTTrainer 做的是监督微调。它封装了完整训练循环：遍历数据集 → 格式化 → Tokenize → 前向传播算 loss（只对 assistant 部分）→ 反向传播（只更新 LoRA 参数）→ 梯度累积 → 评估 → 保存 checkpoint → 学习率调度。

**TRL 是"总指挥"，它调用 Transformers 做推理、调用 PEFT 获取可训练参数、调用 PyTorch 做反向传播。**

**5. datasets（HuggingFace）— 数据加载**

```python
dataset = load_dataset("json", data_files={"train": train_file, "validation": eval_file})
```

把 JSONL 文件加载为标准 dataset 对象，支持流式读取。

**整体关系：**

```
datasets（加载数据）→ Transformers（加载模型 + tokenizer）→ BitsAndBytes（4-bit 量化压显存）
    → PEFT（在 Q/V 上挂 LoRA 适配器）→ TRL/SFTTrainer（调度整个训练循环）
```

**一句话总结**：Transformers 负责"有什么模型"，BitsAndBytes 负责"加载得动"，PEFT 负责"只训一小撮参数"，TRL 负责"把训练跑起来"。
