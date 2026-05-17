"""评估原版 Qwen2.5-7B-Instruct（未微调）在测试集上的表现。"""
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_MODEL = r"C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"
TEST_FILE = "../data/synthetic/sft/test.jsonl"
SYSTEM_PROMPT = "你是慢病健康管理系统的路由与信息抽取器。你只能输出合法 JSON，不要输出解释。"
FLAGS = ["need_record", "need_advice", "need_plan", "need_reminder", "need_history"]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, user_input: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"_parse_error": True, "_raw": response[:200]}


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    print("Loading base model...", flush=True)
    model, tokenizer = load_model()

    test_samples = load_jsonl(TEST_FILE)
    total = len(test_samples)
    print(f"Test samples: {total}", flush=True)

    parse_ok = 0
    per_flag = {f: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for f in FLAGS}
    multi_correct = 0
    multi_total = 0
    errors = []

    for i, s in enumerate(test_samples):
        user_input = s["messages"][1]["content"]
        true_obj = json.loads(s["messages"][-1]["content"])
        pred_obj = predict(model, tokenizer, user_input)

        if (i + 1) % 20 == 0:
            print(f"Progress: {i+1}/{total}", flush=True)

        if pred_obj.get("_parse_error"):
            errors.append((user_input, pred_obj.get("_raw", "")))
            continue
        parse_ok += 1

        true_routing = true_obj["routing"]
        pred_routing = pred_obj.get("routing", {})
        true_multi = sum(1 for f in FLAGS if true_routing.get(f)) > 1

        sample_correct = True
        for flag in FLAGS:
            t = bool(true_routing.get(flag, False))
            p = bool(pred_routing.get(flag, False))
            if t and p:
                per_flag[flag]["tp"] += 1
            elif not t and p:
                per_flag[flag]["fp"] += 1
                sample_correct = False
            elif t and not p:
                per_flag[flag]["fn"] += 1
                sample_correct = False
            else:
                per_flag[flag]["tn"] += 1

        if true_multi:
            multi_total += 1
            if sample_correct:
                multi_correct += 1

    # Report
    json_rate = parse_ok / total * 100
    print(f"\n{'='*50}")
    print(f"BASE MODEL (Qwen2.5-7B-Instruct, no LoRA)")
    print(f"JSON 合法率: {parse_ok}/{total} = {json_rate:.1f}%")
    print(f"\n{'Flag':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 50)
    macro_p = macro_r = macro_f = 0
    for flag in FLAGS:
        m = per_flag[flag]
        p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0
        r = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else 0
        f = 2 * p * r / (p + r) if (p + r) else 0
        macro_p += p
        macro_r += r
        macro_f += f
        print(f"{flag:<20} {p:>10.3f} {r:>10.3f} {f:>10.3f}")
    print("-" * 50)
    print(f"{'Macro avg':<20} {macro_p/5:>10.3f} {macro_r/5:>10.3f} {macro_f/5:>10.3f}")

    if multi_total:
        print(f"\n多路由样本准确率: {multi_correct}/{multi_total} = {multi_correct/multi_total*100:.1f}%")

    print(f"\n解析失败 ({total - parse_ok} 条):")
    for inp, raw in errors[:5]:
        print(f"  input: {inp[:50]}")
        print(f"  raw:   {raw}")
        print()


if __name__ == "__main__":
    main()
