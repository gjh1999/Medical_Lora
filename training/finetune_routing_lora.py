import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# ===== 模型 =====
model_name = r"C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"

# ===== 数据 =====
train_file = "../data/synthetic/sft/train.jsonl"
eval_file = "../data/synthetic/sft/dev.jsonl"

# ===== 加载 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ===== 4bit 量化加载模型 =====
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
    data_files={"train": train_file, "validation": eval_file},
)

# ===== 数据格式化 =====
def format_messages(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

# ===== LoRA 配置 =====
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# ===== 训练参数 =====
training_args = TrainingArguments(
    output_dir="./qwen_routing_lora",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=3,
    bf16=False,
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
)

# ===== Trainer =====
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
    formatting_func=format_messages,
    args=training_args,
)

# ===== 训练 =====
trainer.train()

# ===== 保存 =====
trainer.model.save_pretrained("./qwen_routing_lora_final")
tokenizer.save_pretrained("./qwen_routing_lora_final")

print("训练完成！")
