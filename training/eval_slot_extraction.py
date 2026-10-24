"""Slot (槽位) 抽取准确率评估实验。

比较三种方法的槽位抽取准确率：
  1. 原版基础模型（无微调，简短提示词）
  2. 提示词工程（详细提示词，含完整 Schema）
  3. LoRA 微调模型

指标：
  - 槽位检测 F1：每个叶子槽位的 presence/absence 二分类检测
  - 槽位值部分匹配率：子字段键完整度 + 值准确率
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = r"C:\Users\gjh\.cache\modelscope\hub\models\Qwen\Qwen2___5-7B-Instruct"
LORA_PATH = PROJECT_ROOT / "training" / "qwen_routing_lora_final"
TEST_FILE = PROJECT_ROOT / "data" / "synthetic" / "sft" / "test.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "training" / "slot_eval_outputs"

SHORT_SYSTEM_PROMPT = "你是慢病健康管理系统的路由与信息抽取器。你只能输出合法 JSON，不要输出解释。"

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

FLAGS = ["need_record", "need_advice", "need_plan", "need_reminder", "need_history"]

# ---------------------------------------------------------------------------
# Slot leaf definitions
# ---------------------------------------------------------------------------
# Each leaf is (parent_key, leaf_key, is_present_fn)
# is_present_fn returns True if the slot is meaningfully filled (non-null / non-empty)
RECORD_LEAVES = [
    "blood_pressure", "blood_glucose", "weight",
    "diet", "exercise", "medication", "symptom",
]

SLOT_GROUPS = {
    "record": {
        "leaves": RECORD_LEAVES,
        "parent": "record",
        "is_present": lambda v: isinstance(v, dict) and v is not None,
    },
    "advice": {
        "leaves": ["advice"],
        "parent": "advice",
        "is_present": lambda v: (
            isinstance(v, dict)
            and (bool(v.get("topic")) or bool(v.get("question")))
        ),
    },
    "plan": {
        "leaves": ["plan"],
        "parent": "plan",
        "is_present": lambda v: isinstance(v, dict) and bool(v.get("goal")),
    },
    "reminder": {
        "leaves": ["reminder"],
        "parent": "reminder",
        "is_present": lambda v: (
            isinstance(v, dict)
            and (bool(v.get("task")) or bool(v.get("action")))
        ),
    },
    "history": {
        "leaves": ["history"],
        "parent": "history",
        "is_present": lambda v: isinstance(v, dict) and bool(v.get("metric")),
    },
}

ALL_LEAF_SLOTS = []
for group_name, group_info in SLOT_GROUPS.items():
    for leaf in group_info["leaves"]:
        ALL_LEAF_SLOTS.append((group_name, leaf))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_true_obj(sample: dict) -> dict:
    return json.loads(sample["messages"][-1]["content"])


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned]
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(cleaned[first: last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {"_parse_error": True, "_raw": text[:500]}


def safe_get(obj: dict, *keys, default=None):
    """Safely traverse nested dict keys."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def get_leaf_value(slots: dict, group: str, leaf: str):
    """Get the value of a leaf slot from the slots dict."""
    if group == "record":
        record = safe_get(slots, "record")
        if isinstance(record, dict):
            return record.get(leaf)
        return None
    else:
        # advice, plan, reminder, history are single units
        return safe_get(slots, group)


def is_slot_present(val) -> bool:
    """Check if a slot value is meaningfully present (non-null, non-empty)."""
    if val is None:
        return False
    if isinstance(val, dict):
        # dict with at least one non-null value is "present"
        return any(v is not None and v != [] and v != {} and v != "" for v in val.values())
    if isinstance(val, str):
        return len(val) > 0
    if isinstance(val, (list,)):
        return len(val) > 0
    return True


def compare_slot_values(true_val, pred_val) -> dict:
    """Compare true and predicted slot values, return match statistics.

    Returns dict with:
      - keys_jaccard: Jaccard similarity of keys (for dicts)
      - value_exact: fraction of shared keys with exact value match
      - shared_keys: number of shared keys
      - true_keys: number of keys in true
      - pred_keys: number of keys in pred
    """
    result = {"keys_jaccard": 1.0, "value_exact": 1.0, "shared_keys": 0, "true_keys": 0, "pred_keys": 0}

    # Both null/None -> perfect match
    if true_val is None and pred_val is None:
        return result
    if true_val is None or pred_val is None:
        result["keys_jaccard"] = 0.0
        result["value_exact"] = 0.0
        return result

    # Both strings
    if isinstance(true_val, str) and isinstance(pred_val, str):
        result["true_keys"] = 1
        result["pred_keys"] = 1
        result["shared_keys"] = 1
        result["value_exact"] = 1.0 if true_val == pred_val else 0.0
        result["keys_jaccard"] = 1.0
        return result

    # Both dicts
    if isinstance(true_val, dict) and isinstance(pred_val, dict):
        true_keys = {k for k, v in true_val.items() if v is not None and v != [] and v != {} and v != ""}
        pred_keys = {k for k, v in pred_val.items() if v is not None and v != [] and v != {} and v != ""}
        result["true_keys"] = len(true_keys)
        result["pred_keys"] = len(pred_keys)
        shared = true_keys & pred_keys
        result["shared_keys"] = len(shared)
        union = true_keys | pred_keys
        result["keys_jaccard"] = len(shared) / len(union) if union else 1.0
        if shared:
            matches = sum(1 for k in shared if true_val.get(k) == pred_val.get(k))
            result["value_exact"] = matches / len(shared)
        else:
            result["value_exact"] = 0.0 if (true_keys or pred_keys) else 1.0
        return result

    # Type mismatch (e.g., string vs dict)
    result["keys_jaccard"] = 0.0
    result["value_exact"] = 0.0
    return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_lora_model():
    model, tokenizer = load_base_model()
    model = PeftModel.from_pretrained(model, str(LORA_PATH))
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model, tokenizer, user_input: str, system_prompt: str, max_new_tokens: int = 768) -> tuple[dict, str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
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
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    ).strip()
    return extract_json_object(raw), raw


# ---------------------------------------------------------------------------
# Slot metrics computation
# ---------------------------------------------------------------------------

def compute_slot_detection_metrics(
    all_true_slots: list[dict],
    all_pred_slots: list[dict],
) -> dict:
    """Compute per-leaf slot presence detection metrics.

    Returns:
        per_leaf: dict mapping leaf_name -> {tp, fp, fn, tn, precision, recall, f1}
        macro_f1: average F1 across all leaves
        micro_f1: micro-averaged F1 across all leaves
    """
    per_leaf = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for true_slots, pred_slots in zip(all_true_slots, all_pred_slots):
        for group_name, leaf in ALL_LEAF_SLOTS:
            true_val = get_leaf_value(true_slots, group_name, leaf)
            pred_val = get_leaf_value(pred_slots, group_name, leaf)

            true_present = is_slot_present(true_val)
            pred_present = is_slot_present(pred_val)

            if true_present and pred_present:
                per_leaf[f"{group_name}.{leaf}"]["tp"] += 1
            elif not true_present and pred_present:
                per_leaf[f"{group_name}.{leaf}"]["fp"] += 1
            elif true_present and not pred_present:
                per_leaf[f"{group_name}.{leaf}"]["fn"] += 1
            else:
                per_leaf[f"{group_name}.{leaf}"]["tn"] += 1

    # Compute P/R/F1
    result = {}
    macro_f1 = 0.0
    micro_tp = micro_fp = micro_fn = 0
    n_leaves = len(ALL_LEAF_SLOTS)

    for leaf_name, counts in per_leaf.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result[leaf_name] = {**counts, "precision": precision, "recall": recall, "f1": f1}
        macro_f1 += f1
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    macro_f1 /= n_leaves
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "per_leaf": result,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }


def compute_slot_value_metrics(all_true_slots, all_pred_slots) -> dict:
    """Compute slot value partial match metrics for correctly detected slots."""
    total_shared_keys = 0
    total_true_keys = 0
    total_pred_keys = 0
    total_value_matches = 0
    total_comparisons = 0  # leaf slots where both true and pred are present

    for true_slots, pred_slots in zip(all_true_slots, all_pred_slots):
        for group_name, leaf in ALL_LEAF_SLOTS:
            true_val = get_leaf_value(true_slots, group_name, leaf)
            pred_val = get_leaf_value(pred_slots, group_name, leaf)

            if not (is_slot_present(true_val) and is_slot_present(pred_val)):
                continue

            total_comparisons += 1
            cmp = compare_slot_values(true_val, pred_val)
            total_shared_keys += cmp["shared_keys"]
            total_true_keys += cmp["true_keys"]
            total_pred_keys += cmp["pred_keys"]
            total_value_matches += cmp["value_exact"]

    if total_comparisons == 0:
        return {
            "present_slots_matched": 0,
            "key_recall": 1.0,
            "key_precision": 1.0,
            "key_f1": 1.0,
            "value_accuracy": 1.0,
        }

    key_recall = total_shared_keys / total_true_keys if total_true_keys > 0 else 1.0
    key_precision = total_shared_keys / total_pred_keys if total_pred_keys > 0 else 1.0
    key_f1 = (
        2 * key_recall * key_precision / (key_recall + key_precision)
        if (key_recall + key_precision) > 0
        else 0.0
    )
    value_accuracy = total_value_matches / total_comparisons

    return {
        "present_slots_matched": total_comparisons,
        "key_recall": key_recall,
        "key_precision": key_precision,
        "key_f1": key_f1,
        "value_accuracy": value_accuracy,
    }


def compute_routing_metrics(all_true_objs, all_pred_objs) -> dict:
    """Compute routing flag metrics (reused from existing eval scripts)."""
    per_flag = {f: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for f in FLAGS}
    routing_exact = 0

    for true_obj, pred_obj in zip(all_true_objs, all_pred_objs):
        true_routing = true_obj.get("routing", {})
        pred_routing = pred_obj.get("routing", {})

        if all(
            bool(true_routing.get(f, False)) == bool(pred_routing.get(f, False))
            for f in FLAGS
        ):
            routing_exact += 1

        for flag in FLAGS:
            t = bool(true_routing.get(flag, False))
            p = bool(pred_routing.get(flag, False))
            if t and p:
                per_flag[flag]["tp"] += 1
            elif not t and p:
                per_flag[flag]["fp"] += 1
            elif t and not p:
                per_flag[flag]["fn"] += 1
            else:
                per_flag[flag]["tn"] += 1

    flag_rows = []
    macro_f1 = 0.0
    for flag in FLAGS:
        m = per_flag[flag]
        precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
        recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        flag_rows.append({"flag": flag, "precision": precision, "recall": recall, "f1": f1, **m})
        macro_f1 += f1
    macro_f1 /= len(FLAGS)

    return {
        "routing_exact": routing_exact,
        "routing_exact_rate": routing_exact / len(all_true_objs) if all_true_objs else 0.0,
        "macro_f1": macro_f1,
        "flags": flag_rows,
    }


# ---------------------------------------------------------------------------
# Evaluate one approach
# ---------------------------------------------------------------------------

def evaluate_approach(
    model,
    tokenizer,
    test_samples: list[dict],
    system_prompt: str,
    label: str,
    max_new_tokens: int = 768,
    limit: int = 0,
    print_examples: int = 3,
) -> dict:
    """Run evaluation for a single approach, return metrics dict."""
    samples = test_samples if limit <= 0 else test_samples[:limit]
    total = len(samples)

    all_true_objs = []
    all_pred_objs = []
    all_true_slots = []
    all_pred_slots = []
    parse_ok = 0
    predictions = []

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Samples: {total}")
    print(f"{'='*60}")

    for i, sample in enumerate(samples, 1):
        user_input = sample["messages"][1]["content"]
        true_obj = get_true_obj(sample)
        pred_obj, raw = predict(model, tokenizer, user_input, system_prompt, max_new_tokens)

        predictions.append({
            "index": i - 1,
            "input": user_input,
            "true": true_obj,
            "pred": pred_obj,
            "raw": raw,
        })

        if pred_obj.get("_parse_error"):
            continue

        parse_ok += 1
        all_true_objs.append(true_obj)
        all_pred_objs.append(pred_obj)

        true_slots = true_obj.get("slots", {})
        pred_slots = pred_obj.get("slots", {})
        all_true_slots.append(true_slots)
        all_pred_slots.append(pred_slots)

        if i <= print_examples:
            print(f"\n--- Sample {i} ---")
            print(f"Input: {user_input}")
            print(f"Raw: {raw[:300]}")
        if i % 20 == 0:
            print(f"Progress: {i}/{total}")

    # Compute metrics on parse-OK samples only
    n = len(all_true_objs)
    if n == 0:
        return {
            "label": label,
            "total": total,
            "parse_ok": 0,
            "json_valid_rate": 0.0,
            "error": "No valid JSON predictions",
        }

    slot_detection = compute_slot_detection_metrics(all_true_slots, all_pred_slots)
    slot_value = compute_slot_value_metrics(all_true_slots, all_pred_slots)
    routing = compute_routing_metrics(all_true_objs, all_pred_objs)

    return {
        "label": label,
        "total": total,
        "parse_ok": parse_ok,
        "json_valid_rate": parse_ok / total,
        "slot_detection": slot_detection,
        "slot_value": slot_value,
        "routing": routing,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_slot_report(results: dict):
    print(f"\n{'='*70}")
    print(f"SLOT EXTRACTION — {results['label']}")
    print(f"{'='*70}")
    print(f"JSON valid: {results['parse_ok']}/{results['total']} = {results['json_valid_rate']*100:.1f}%")

    sd = results["slot_detection"]
    sv = results["slot_value"]

    # Slot detection table
    print(f"\n--- Slot Presence Detection (per leaf) ---")
    print(f"{'Slot':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 82)
    for leaf_name in sorted(sd["per_leaf"].keys()):
        m = sd["per_leaf"][leaf_name]
        print(
            f"{leaf_name:<30} {m['precision']:>10.3f} {m['recall']:>10.3f} "
            f"{m['f1']:>10.3f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6}"
        )
    print("-" * 82)
    print(f"{'Micro avg':<30} {sd['micro_precision']:>10.3f} {sd['micro_recall']:>10.3f} {sd['micro_f1']:>10.3f}")
    print(f"{'Macro avg':<30} {'':>10} {'':>10} {sd['macro_f1']:>10.3f}")

    # Slot value match (only for correctly detected present slots)
    print(f"\n--- Slot Value Match (for correctly detected present slots) ---")
    print(f"Present slots matched: {sv['present_slots_matched']}")
    print(f"Key Recall:    {sv['key_recall']*100:.1f}%  (sub-keys correctly included)")
    print(f"Key Precision: {sv['key_precision']*100:.1f}%  (sub-keys not hallucinated)")
    print(f"Key F1:        {sv['key_f1']*100:.1f}%")
    print(f"Value Accuracy:{sv['value_accuracy']*100:.1f}%  (exact match for shared keys)")

    # Routing
    r = results["routing"]
    print(f"\n--- Routing (reference) ---")
    print(f"Routing Exact: {r['routing_exact']}/{results['parse_ok']} = {r['routing_exact_rate']*100:.1f}%")
    print(f"Routing Macro F1: {r['macro_f1']:.3f}")


def print_comparison(all_results: list[dict]):
    """Print side-by-side comparison of all approaches."""
    print(f"\n{'='*90}")
    print("SLOT EXTRACTION ACCURACY COMPARISON")
    print(f"{'='*90}")

    # Table 1: Slot Detection F1 per leaf
    print(f"\n--- Slot Detection F1 Comparison ---")
    leaves = sorted(all_results[0]["slot_detection"]["per_leaf"].keys())
    header = f"{'Slot':<30}"
    for r in all_results:
        header += f" {r['label']:>18}"
    print(header)
    print("-" * (30 + 20 * len(all_results)))

    for leaf in leaves:
        row = f"{leaf:<30}"
        for r in all_results:
            f1 = r["slot_detection"]["per_leaf"].get(leaf, {}).get("f1", 0.0)
            row += f" {f1:>18.3f}"
        print(row)

    print("-" * (30 + 20 * len(all_results)))
    row = f"{'Macro F1':<30}"
    for r in all_results:
        row += f" {r['slot_detection']['macro_f1']:>18.3f}"
    print(row)
    row = f"{'Micro F1':<30}"
    for r in all_results:
        row += f" {r['slot_detection']['micro_f1']:>18.3f}"
    print(row)

    # Table 2: Slot Value Match
    print(f"\n--- Slot Value Match Comparison ---")
    print(f"{'Metric':<30}", end="")
    for r in all_results:
        print(f" {r['label']:>18}", end="")
    print()
    print("-" * (30 + 20 * len(all_results)))
    for metric_key, metric_label in [
        ("key_f1", "Key F1"),
        ("key_recall", "Key Recall"),
        ("key_precision", "Key Precision"),
        ("value_accuracy", "Value Accuracy"),
    ]:
        row = f"{metric_label:<30}"
        for r in all_results:
            val = r["slot_value"][metric_key]
            row += f" {val*100:>17.1f}%"
        print(row)

    # Table 3: Routing + JSON valid (for completeness)
    print(f"\n--- Routing + JSON ---")
    print(f"{'Metric':<30}", end="")
    for r in all_results:
        print(f" {r['label']:>18}", end="")
    print()
    print("-" * (30 + 20 * len(all_results)))
    for metric_key, metric_label in [
        ("json_valid_rate", "JSON Valid Rate"),
        ("routing_exact_rate", "Routing Exact"),
        ("macro_f1", "Routing Macro F1"),
    ]:
        row = f"{metric_label:<30}"
        for r in all_results:
            if metric_key == "macro_f1":
                val = r["routing"]["macro_f1"]
                row += f" {val:>18.3f}"
            elif metric_key == "routing_exact_rate":
                val = r["routing"]["routing_exact_rate"]
                row += f" {val*100:>17.1f}%"
            else:
                val = r[metric_key]
                row += f" {val*100:>17.1f}%"
        print(row)

    # Table 4: Summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    print(f"{'Approach':<30} {'Slot Macro F1':>15} {'Slot Micro F1':>15} {'Key F1':>15} {'Routing F1':>15}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['label']:<30} {r['slot_detection']['macro_f1']:>15.3f} "
            f"{r['slot_detection']['micro_f1']:>15.3f} "
            f"{r['slot_value']['key_f1']*100:>14.1f}% "
            f"{r['routing']['macro_f1']:>15.3f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Slot extraction accuracy experiment — compare base / prompt / LoRA"
    )
    parser.add_argument(
        "--modes", nargs="+", default=["all"],
        choices=["base", "prompt", "lora", "all"],
        help="Which approaches to evaluate (default: all)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit test samples (0=all)")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--print-examples", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    if "all" in args.modes:
        modes = ["base", "prompt", "lora"]
    else:
        modes = args.modes

    test_samples = load_jsonl(TEST_FILE)
    print(f"Test samples loaded: {len(test_samples)}")

    all_results = []
    base_model = None
    base_tokenizer = None

    # --- Base model ---
    if "base" in modes:
        print("\n>>> Loading BASE model (no LoRA, short prompt)...")
        base_model, base_tokenizer = load_base_model()
        result = evaluate_approach(
            base_model, base_tokenizer, test_samples,
            system_prompt=SHORT_SYSTEM_PROMPT,
            label="Base (short prompt)",
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            print_examples=args.print_examples,
        )
        all_results.append(result)
        print_slot_report(result)

    # --- Prompt-only (zero-shot with detailed system prompt) ---
    if "prompt" in modes:
        if base_model is None:
            print("\n>>> Loading model for PROMPT-ONLY evaluation (detailed schema prompt)...")
            base_model, base_tokenizer = load_base_model()
        else:
            print("\n>>> Reusing loaded model for PROMPT-ONLY evaluation...")
        result = evaluate_approach(
            base_model, base_tokenizer, test_samples,
            system_prompt=ZERO_SHOT_SYSTEM_PROMPT,
            label="Prompt (zero-shot)",
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            print_examples=args.print_examples,
        )
        all_results.append(result)
        print_slot_report(result)

    # Release base model before loading LoRA
    if base_model is not None:
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache()

    # --- LoRA fine-tuned ---
    if "lora" in modes:
        print("\n>>> Loading LORA fine-tuned model...")
        model, tokenizer = load_lora_model()
        result = evaluate_approach(
            model, tokenizer, test_samples,
            system_prompt=SHORT_SYSTEM_PROMPT,
            label="LoRA fine-tuned",
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            print_examples=args.print_examples,
        )
        all_results.append(result)
        print_slot_report(result)
        del model
        torch.cuda.empty_cache()

    # --- Comparison ---
    if len(all_results) >= 2:
        print_comparison(all_results)

    # --- Save ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for result in all_results:
        label_slug = result["label"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        summary = {k: v for k, v in result.items() if k != "predictions"}
        summary_path = args.output_dir / f"slot_summary_{label_slug}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        preds_path = args.output_dir / f"slot_predictions_{label_slug}.jsonl"
        with open(preds_path, "w", encoding="utf-8") as f:
            for row in result.get("predictions", []):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Saved: {summary_path}")
        print(f"Saved: {preds_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
