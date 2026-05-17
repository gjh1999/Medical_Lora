
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer


# ===== 模型 =====
model_name = r"C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"

# ===== 数据 =====
train_file = "cmedqa_train_sft.jsonl"
eval_file = "cmedqa_dev_sft.jsonl"

# ===== 加载 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# ===== 加载模型 =====

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype=torch.bfloat16,        # 别用 torch_dtype 了（新版本警告）
#     device_map={"": 0},          # 强制全上GPU
#     trust_remote_code=True
# )


from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)



# ===== 数据集 =====
dataset = load_dataset(
    "json",
    data_files={
        "train": train_file,
        "validation": eval_file
    }
)



# 取5000条数据训练
train_dataset = dataset["train"].shuffle(seed=42).select(range(5000))
eval_dataset = dataset["validation"].shuffle(seed=42).select(range(500))



# ===== 数据格式化 =====
def format_example(example):
    messages = [
        {"role": "system", "content": "你是一个专业、严谨的中文医学问答助手。"},
        {"role": "user", "content": example["instruction"] + "\n\n" + example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

# ===== LoRA配置 =====
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     lora_dropout=0.05,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj"
#     ],
#     bias="none",
#     task_type="CAUSAL_LM"
# )


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# ===== 训练参数（5080优化版）=====
training_args = TrainingArguments(
    output_dir="./qwen_medical_lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=3,
    bf16=False,
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False
)

# ===== Trainer =====
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["validation"],
#     peft_config=lora_config,
#     formatting_func=format_example,
#     args=training_args,
#     max_seq_length=1024
# )

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    # train_dataset=dataset["train"],
    # eval_dataset=dataset["validation"],
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    formatting_func=format_example,
    args=training_args,
)

# ===== 开始训练 =====
trainer.train()

# ===== 保存模型 =====
trainer.model.save_pretrained("./qwen_medical_lora_final")
tokenizer.save_pretrained("./qwen_medical_lora_final")

print("训练完成！")