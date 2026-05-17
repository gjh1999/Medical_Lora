"""Analyze the distribution of routing flags, slots, and expression types in a dataset."""

import json
import sys
from collections import Counter
from pathlib import Path


def classify_routing(routing: dict) -> str:
    """Classify a sample by its routing pattern."""
    flags = []
    if routing.get("need_record"):
        flags.append("record")
    if routing.get("need_advice"):
        flags.append("advice")
    if routing.get("need_plan"):
        flags.append("plan")
    if routing.get("need_reminder"):
        flags.append("reminder")
    if routing.get("need_history"):
        flags.append("history")

    if not flags:
        return "no_routing"
    return "+".join(flags)


def analyze(filepath: str):
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(samples)
    print(f"Total samples: {total}\n")

    # Routing distribution
    routing_counter = Counter()
    for s in samples:
        routing = s.get("output", {}).get("routing", {})
        routing_counter[classify_routing(routing)] += 1

    print("=== Routing Distribution ===")
    for pattern, count in routing_counter.most_common():
        pct = count / total * 100
        print(f"  {pattern}: {count} ({pct:.1f}%)")

    # Individual flag frequency
    print("\n=== Flag Frequency ===")
    flag_counter = Counter()
    for s in samples:
        routing = s.get("output", {}).get("routing", {})
        for flag in ["need_record", "need_advice", "need_plan", "need_reminder", "need_history"]:
            if routing.get(flag):
                flag_counter[flag] += 1
    for flag, count in flag_counter.most_common():
        pct = count / total * 100
        print(f"  {flag}: {count} ({pct:.1f}%)")

    # Multi-routing
    print("\n=== Multi-Routing Samples ===")
    multi = sum(1 for s in samples
                if sum(1 for f in ["need_record", "need_advice", "need_plan", "need_reminder", "need_history"]
                       if s.get("output", {}).get("routing", {}).get(f)) > 1)
    print(f"  Samples with >1 routing=true: {multi} ({multi/total*100:.1f}%)")

    # Missing fields
    print("\n=== Missing Fields ===")
    has_missing = sum(1 for s in samples if len(s.get("output", {}).get("missing_fields", [])) > 0)
    print(f"  Samples with missing_fields: {has_missing} ({has_missing/total*100:.1f}%)")

    # Risk flags
    print("\n=== Risk Flags ===")
    has_risk = sum(1 for s in samples if len(s.get("output", {}).get("risk_flags", [])) > 0)
    print(f"  Samples with risk_flags: {has_risk} ({has_risk/total*100:.1f}%)")
    risk_counter = Counter()
    for s in samples:
        for flag in s.get("output", {}).get("risk_flags", []):
            risk_counter[flag] += 1
    for flag, count in risk_counter.most_common():
        print(f"    {flag}: {count}")

    # Slot metrics distribution
    print("\n=== Record Metric Distribution ===")
    metric_counter = Counter()
    record_fields = ["blood_pressure", "blood_glucose", "weight", "diet", "exercise", "medication", "symptom"]
    for s in samples:
        record = s.get("output", {}).get("slots", {}).get("record", {})
        if not isinstance(record, dict):
            continue
        for field in record_fields:
            val = record.get(field)
            if val is not None and isinstance(val, dict) and any(v is not None for v in val.values()):
                metric_counter[field] += 1
    for metric, count in metric_counter.most_common():
        print(f"  {metric}: {count}")

    # All-routing-false samples
    print("\n=== Out-of-Scope / No Routing ===")
    no_routing = sum(1 for s in samples
                     if all(not s.get("output", {}).get("routing", {}).get(f)
                            for f in ["need_record", "need_advice", "need_plan", "need_reminder", "need_history"]))
    print(f"  All routing=false: {no_routing} ({no_routing/total*100:.1f}%)")

    # Input length distribution
    print("\n=== Input Length ===")
    lengths = [len(s.get("input", "")) for s in samples]
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_distribution.py <file.jsonl>")
        sys.exit(1)
    analyze(sys.argv[1])


if __name__ == "__main__":
    main()
