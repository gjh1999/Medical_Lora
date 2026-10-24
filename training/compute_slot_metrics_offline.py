"""离线计算槽位指标——从已保存的预测数据中计算，无需 GPU。

用法：
  python training/compute_slot_metrics_offline.py --predictions training/prompt_baseline_outputs/predictions_zero_shot_0shot.jsonl --label "Prompt (zero-shot)"
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

FLAGS = ["need_record", "need_advice", "need_plan", "need_reminder", "need_history"]
RECORD_LEAVES = ["blood_pressure", "blood_glucose", "weight", "diet", "exercise", "medication", "symptom"]

ALL_LEAF_SLOTS = []
for leaf in RECORD_LEAVES:
    ALL_LEAF_SLOTS.append(("record", leaf))
for group in ["advice", "plan", "reminder", "history"]:
    ALL_LEAF_SLOTS.append((group, group))


def safe_get(obj, *keys, default=None):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def get_leaf_value(slots, group, leaf):
    if group == "record":
        record = safe_get(slots, "record")
        if isinstance(record, dict):
            return record.get(leaf)
        return None
    else:
        return safe_get(slots, group)


def is_slot_present(val):
    if val is None:
        return False
    if isinstance(val, dict):
        return any(v is not None and v != [] and v != {} and v != "" for v in val.values())
    if isinstance(val, str):
        return len(val) > 0
    if isinstance(val, (list,)):
        return len(val) > 0
    return True


def compare_slot_values(true_val, pred_val):
    result = {"keys_jaccard": 1.0, "value_exact": 1.0, "shared_keys": 0, "true_keys": 0, "pred_keys": 0}
    if true_val is None and pred_val is None:
        return result
    if true_val is None or pred_val is None:
        result["keys_jaccard"] = 0.0
        result["value_exact"] = 0.0
        return result
    if isinstance(true_val, str) and isinstance(pred_val, str):
        result["true_keys"] = result["pred_keys"] = result["shared_keys"] = 1
        result["value_exact"] = 1.0 if true_val == pred_val else 0.0
        return result
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
    result["keys_jaccard"] = 0.0
    result["value_exact"] = 0.0
    return result


def compute_metrics(predictions: list[dict]) -> dict:
    valid_preds = [p for p in predictions if not p["pred"].get("_parse_error")]

    # Slot detection
    per_leaf = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    for p in valid_preds:
        true_slots = p["true"].get("slots", {})
        pred_slots = p["pred"].get("slots", {})
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

    macro_f1 = 0.0
    micro_tp = micro_fp = micro_fn = 0
    n_leaves = len(ALL_LEAF_SLOTS)
    per_leaf_result = {}

    for leaf_name, counts in per_leaf.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_leaf_result[leaf_name] = {**counts, "precision": precision, "recall": recall, "f1": f1}
        macro_f1 += f1
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    macro_f1 /= n_leaves
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    # Slot value
    total_shared_keys = total_true_keys = total_pred_keys = total_value_matches = total_comparisons = 0
    for p in valid_preds:
        true_slots = p["true"].get("slots", {})
        pred_slots = p["pred"].get("slots", {})
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

    if total_comparisons > 0:
        key_recall = total_shared_keys / total_true_keys if total_true_keys > 0 else 1.0
        key_precision = total_shared_keys / total_pred_keys if total_pred_keys > 0 else 1.0
        key_f1 = 2 * key_recall * key_precision / (key_recall + key_precision) if (key_recall + key_precision) > 0 else 0.0
        value_accuracy = total_value_matches / total_comparisons
    else:
        key_recall = key_precision = key_f1 = value_accuracy = 1.0

    # Routing
    per_flag = {f: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for f in FLAGS}
    routing_exact = 0
    for p in valid_preds:
        true_routing = p["true"].get("routing", {})
        pred_routing = p["pred"].get("routing", {})
        if all(bool(true_routing.get(f, False)) == bool(pred_routing.get(f, False)) for f in FLAGS):
            routing_exact += 1
        for flag in FLAGS:
            t = bool(true_routing.get(flag, False))
            pr = bool(pred_routing.get(flag, False))
            if t and pr:
                per_flag[flag]["tp"] += 1
            elif not t and pr:
                per_flag[flag]["fp"] += 1
            elif t and not pr:
                per_flag[flag]["fn"] += 1
            else:
                per_flag[flag]["tn"] += 1

    flag_rows = []
    routing_macro_f1 = 0.0
    for flag in FLAGS:
        m = per_flag[flag]
        precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
        recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        flag_rows.append({"flag": flag, "precision": precision, "recall": recall, "f1": f1, **m})
        routing_macro_f1 += f1
    routing_macro_f1 /= len(FLAGS)

    return {
        "total": len(predictions),
        "parse_ok": len(valid_preds),
        "json_valid_rate": len(valid_preds) / len(predictions),
        "slot_detection": {
            "per_leaf": per_leaf_result,
            "macro_f1": macro_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
        },
        "slot_value": {
            "present_slots_matched": total_comparisons,
            "key_recall": key_recall,
            "key_precision": key_precision,
            "key_f1": key_f1,
            "value_accuracy": value_accuracy,
        },
        "routing": {
            "routing_exact": routing_exact,
            "routing_exact_rate": routing_exact / len(valid_preds),
            "macro_f1": routing_macro_f1,
            "flags": flag_rows,
        },
    }


def print_report(results: dict, label: str):
    print(f"\n{'='*70}")
    print(f"SLOT EXTRACTION — {label}")
    print(f"{'='*70}")
    print(f"JSON valid: {results['parse_ok']}/{results['total']} = {results['json_valid_rate']*100:.1f}%")

    sd = results["slot_detection"]
    sv = results["slot_value"]

    print(f"\n--- Slot Presence Detection (per leaf) ---")
    print(f"{'Slot':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 82)
    for leaf_name in sorted(sd["per_leaf"].keys()):
        m = sd["per_leaf"][leaf_name]
        print(f"{leaf_name:<30} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6}")
    print("-" * 82)
    print(f"{'Micro avg':<30} {sd['micro_precision']:>10.3f} {sd['micro_recall']:>10.3f} {sd['micro_f1']:>10.3f}")
    print(f"{'Macro avg':<30} {'':>10} {'':>10} {sd['macro_f1']:>10.3f}")

    print(f"\n--- Slot Value Match (for correctly detected present slots) ---")
    print(f"Present slots matched: {sv['present_slots_matched']}")
    print(f"Key Recall:    {sv['key_recall']*100:.1f}%")
    print(f"Key Precision: {sv['key_precision']*100:.1f}%")
    print(f"Key F1:        {sv['key_f1']*100:.1f}%")
    print(f"Value Accuracy:{sv['value_accuracy']*100:.1f}%")

    r = results["routing"]
    print(f"\n--- Routing (reference) ---")
    print(f"Routing Exact: {r['routing_exact']}/{results['parse_ok']} = {r['routing_exact_rate']*100:.1f}%")
    print(f"Routing Macro F1: {r['macro_f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Offline slot metrics computation")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSONL file")
    parser.add_argument("--label", type=str, default="Model", help="Label for this evaluation")
    parser.add_argument("--output", type=Path, default=None, help="Output summary JSON path")
    args = parser.parse_args()

    predictions = []
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    print(f"Loaded {len(predictions)} predictions from {args.predictions}")

    results = compute_metrics(predictions)
    results["label"] = args.label
    print_report(results, args.label)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
