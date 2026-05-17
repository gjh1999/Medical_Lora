import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
# 测试微调版本的模型
# ===== 本地基础模型路径：和训练代码保持一致 =====
base_model_path = r"C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"

# ===== LoRA微调后的adapter路径 =====
lora_path = r"./qwen_medical_lora_final"

# ===== 加载 tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)

# ===== 4bit量化配置：和训练时一致 =====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# ===== 加载本地base模型 =====
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)

# ===== 加载LoRA adapter =====
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

# ===== 测试问题 =====
question = "我胸口剧痛，要不要去医院？？"

messages = [
    {"role": "system", "content": "你是一个专业、严谨的中文医学问答助手。"},
    {"role": "user", "content": question}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

answer = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print("\n===== 问题 =====")
print(question)

print("\n===== LoRA模型回答 =====")
print(answer)