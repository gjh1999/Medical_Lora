"""Evaluate prompt-only intent routing and slot extraction on the test set.

This baseline loads the base model only. It does not load any LoRA adapter.
Use it to compare prompt engineering against the fine-tuned routing LoRA.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = r"C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"
DEFAULT_TEST_FILE = PROJECT_ROOT / "data" / "synthetic" / "sft" / "test.jsonl"
DEFAULT_DEV_FILE = PROJECT_ROOT / "data" / "synthetic" / "sft" / "dev.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training" / "prompt_baseline_outputs"

FLAGS = ["need_record", "need_advice", "need_plan", "need_reminder", "need_history"]
TOP_LEVEL_KEYS = {"routing", "confidence", "slots", "missing_fields", "risk_flags", "normalized"}


ZERO_SHOT_SYSTEM_PROMPT = """你是慢病健康管理系统的意图识别与结构化信息提取器。

你必须只输出一个合法 JSON 对象，不要输出 Markdown、解释、代码块或多余文本。

任务：
1. 判断用户输入是否需要触发以下下游节点：
   - need_record: 记录健康数据，如血压、血糖、体重、饮食、运动、用药、症状
   - need_advice: 请求健康建议或询问是否正常、怎么办、原因
   - need_plan: 制定健康计划，如降压、控糖、减重、饮食、运动计划
   - need_reminder: 设置、修改、删除或查询提醒
   - need_history: 查询历史记录、趋势、平均值、最近一次记录
2. 抽取结构化 slots。
3. 如果信息不足，在 missing_fields 中写字段路径。
4. 只标记初步风险 risk_flags，不要给出诊断、处方、停药、换药等医疗决策。

输出 JSON 必须严格符合这个结构：
{
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

约束：
- 输出必须能被 json.loads 直接解析。
- routing 的 5 个 key 必须全部出现。
- 未触发的 slots 使用 null 或空数组，不要臆造。
- confidence 是 0.0 到 1.0 的数字。
"""


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_model(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(cleaned[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    return {"_parse_error": True, "_raw": text[:500]}


def get_true_object(sample: dict) -> dict:
    return json.loads(sample["messages"][-1]["content"])


def get_user_input(sample: dict) -> str:
    return sample["messages"][1]["content"]


def build_few_shot_messages(dev_samples: list[dict], shots: int) -> list[dict]:
    if shots <= 0:
        return []

    examples = []
    seen_patterns = set()
    for sample in dev_samples:
        true_obj = get_true_object(sample)
        pattern = tuple(flag for flag in FLAGS if true_obj.get("routing", {}).get(flag))
        if pattern in seen_patterns:
            continue
        seen_patterns.add(pattern)
        examples.append(
            {"role": "user", "content": get_user_input(sample)}
        )
        examples.append(
            {"role": "assistant", "content": json.dumps(true_obj, ensure_ascii=False)}
        )
        if len(examples) // 2 >= shots:
            break

    return examples


def predict(
    model,
    tokenizer,
    user_input: str,
    few_shot_messages: list[dict],
    max_new_tokens: int,
) -> tuple[dict, str]:
    messages = [{"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT}]
    messages.extend(few_shot_messages)
    messages.append({"role": "user", "content": user_input})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    return extract_json_object(raw), raw


def is_schema_valid(obj: dict) -> bool:
    if obj.get("_parse_error"):
        return False
    if not TOP_LEVEL_KEYS.issubset(obj.keys()):
        return False
    routing = obj.get("routing")
    confidence = obj.get("confidence")
    slots = obj.get("slots")
    if not isinstance(routing, dict) or not isinstance(confidence, dict) or not isinstance(slots, dict):
        return False
    return all(flag in routing for flag in FLAGS) and all(flag in confidence for flag in FLAGS)


def update_flag_metrics(per_flag: dict, true_routing: dict, pred_routing: dict) -> bool:
    sample_correct = True
    for flag in FLAGS:
        true_value = bool(true_routing.get(flag, False))
        pred_value = bool(pred_routing.get(flag, False))
        if true_value and pred_value:
            per_flag[flag]["tp"] += 1
        elif not true_value and pred_value:
            per_flag[flag]["fp"] += 1
            sample_correct = False
        elif true_value and not pred_value:
            per_flag[flag]["fn"] += 1
            sample_correct = False
        else:
            per_flag[flag]["tn"] += 1
    return sample_correct


def compute_flag_report(per_flag: dict) -> tuple[list[dict], dict]:
    rows = []
    macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for flag in FLAGS:
        m = per_flag[flag]
        precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
        recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"flag": flag, "precision": precision, "recall": recall, "f1": f1, **m})
        macro["precision"] += precision
        macro["recall"] += recall
        macro["f1"] += f1

    for key in macro:
        macro[key] /= len(FLAGS)
    return rows, macro


def evaluate(args):
    test_samples = load_jsonl(args.test_file)
    dev_samples = load_jsonl(args.dev_file) if args.prompt_mode == "few_shot" else []
    few_shot_messages = build_few_shot_messages(dev_samples, args.shots)

    print(f"Loading base model: {args.base_model}", flush=True)
    model, tokenizer = load_model(args.base_model)
    print(f"Test samples: {len(test_samples)}", flush=True)
    print(f"Prompt mode: {args.prompt_mode}, shots: {len(few_shot_messages) // 2}", flush=True)

    total = len(test_samples)
    parse_ok = 0
    schema_ok = 0
    routing_exact = 0
    multi_total = 0
    multi_correct = 0
    slot_exact = 0
    per_flag = {flag: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for flag in FLAGS}
    predictions = []
    parse_errors = []

    limit = args.limit if args.limit and args.limit > 0 else total
    samples_to_run = test_samples[:limit]

    for i, sample in enumerate(samples_to_run, 1):
        user_input = get_user_input(sample)
        true_obj = get_true_object(sample)
        pred_obj, raw = predict(model, tokenizer, user_input, few_shot_messages, args.max_new_tokens)

        row = {
            "index": i - 1,
            "input": user_input,
            "true": true_obj,
            "pred": pred_obj,
            "raw": raw,
        }
        predictions.append(row)

        if pred_obj.get("_parse_error"):
            parse_errors.append(row)
        else:
            parse_ok += 1
            if is_schema_valid(pred_obj):
                schema_ok += 1

            true_routing = true_obj.get("routing", {})
            pred_routing = pred_obj.get("routing", {})
            sample_correct = update_flag_metrics(per_flag, true_routing, pred_routing)

            if all(bool(true_routing.get(flag, False)) == bool(pred_routing.get(flag, False)) for flag in FLAGS):
                routing_exact += 1

            if true_obj.get("slots") == pred_obj.get("slots"):
                slot_exact += 1

            if sum(1 for flag in FLAGS if true_routing.get(flag)) > 1:
                multi_total += 1
                if sample_correct:
                    multi_correct += 1

        if i <= args.print_examples:
            print(f"\n--- Sample {i} ---", flush=True)
            print(f"Input: {user_input}", flush=True)
            print(f"Raw: {raw[:500]}", flush=True)
        if i % 20 == 0:
            print(f"Progress: {i}/{len(samples_to_run)}", flush=True)

    flag_rows, macro = compute_flag_report(per_flag)
    denominator = len(samples_to_run)
    summary = {
        "model": args.base_model,
        "prompt_mode": args.prompt_mode,
        "shots": len(few_shot_messages) // 2,
        "total": denominator,
        "parse_ok": parse_ok,
        "json_valid_rate": parse_ok / denominator if denominator else 0.0,
        "schema_ok": schema_ok,
        "schema_valid_rate": schema_ok / denominator if denominator else 0.0,
        "routing_exact": routing_exact,
        "routing_exact_rate": routing_exact / denominator if denominator else 0.0,
        "slot_exact": slot_exact,
        "slot_exact_rate": slot_exact / denominator if denominator else 0.0,
        "multi_total": multi_total,
        "multi_correct": multi_correct,
        "multi_exact_rate": multi_correct / multi_total if multi_total else 0.0,
        "macro": macro,
        "flags": flag_rows,
        "parse_errors": len(parse_errors),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.prompt_mode}_{summary['shots']}shot"
    summary_path = args.output_dir / f"summary_{suffix}.json"
    predictions_path = args.output_dir / f"predictions_{suffix}.jsonl"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print_report(summary)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved predictions: {predictions_path}")


def print_report(summary: dict):
    total = summary["total"]
    print("\n" + "=" * 60)
    print("PROMPT-ONLY BASELINE")
    print("=" * 60)
    print(f"Prompt mode: {summary['prompt_mode']} ({summary['shots']} shots)")
    print(f"JSON valid:     {summary['parse_ok']}/{total} = {summary['json_valid_rate'] * 100:.1f}%")
    print(f"Schema valid:   {summary['schema_ok']}/{total} = {summary['schema_valid_rate'] * 100:.1f}%")
    print(f"Routing exact:  {summary['routing_exact']}/{total} = {summary['routing_exact_rate'] * 100:.1f}%")
    print(f"Slot exact:     {summary['slot_exact']}/{total} = {summary['slot_exact_rate'] * 100:.1f}%")
    print(
        f"Multi-routing:  {summary['multi_correct']}/{summary['multi_total']} = "
        f"{summary['multi_exact_rate'] * 100:.1f}%"
    )
    print("\nFlag metrics:")
    print(f"{'Flag':<18} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    for row in summary["flags"]:
        print(f"{row['flag']:<18} {row['precision']:>10.3f} {row['recall']:>10.3f} {row['f1']:>10.3f}")
    print("-" * 52)
    macro = summary["macro"]
    print(f"{'Macro avg':<18} {macro['precision']:>10.3f} {macro['recall']:>10.3f} {macro['f1']:>10.3f}")
    print(f"\nParse errors: {summary['parse_errors']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate base-model prompt-only routing extraction.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--dev-file", type=Path, default=DEFAULT_DEV_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-mode", choices=["zero_shot", "few_shot"], default="zero_shot")
    parser.add_argument("--shots", type=int, default=5, help="Number of dev examples for few_shot mode.")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=0, help="Optional test subset size for smoke tests.")
    parser.add_argument("--print-examples", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
