from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"  # 比如 ./qwen_model

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("模型加载成功！")

# 简单推理测试
inputs = tokenizer("你好，介绍一下高血压", return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))