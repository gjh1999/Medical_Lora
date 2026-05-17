"""Validate routing <-> slots consistency and check for dangerous content."""

import json
import sys
from pathlib import Path


def is_record_slots_nonempty(record) -> bool:
    """Check if at least one record slot field is non-null."""
    if not isinstance(record, dict):
        return False
    for field in ["blood_pressure", "blood_glucose", "weight", "diet", "exercise", "medication", "symptom"]:
        val = record.get(field)
        if val is not None and isinstance(val, dict) and len(val) > 0:
            # Check that at least one sub-field has a non-null value
            if any(v is not None for v in val.values()):
                return True
    return False


def is_advice_slots_nonempty(advice) -> bool:
    if not isinstance(advice, dict):
        return False
    return advice.get("topic") is not None or advice.get("question") is not None


def is_plan_slots_nonempty(plan) -> bool:
    if not isinstance(plan, dict):
        return False
    return plan.get("goal") is not None


def is_reminder_slots_nonempty(reminder) -> bool:
    if not isinstance(reminder, dict):
        return False
    return reminder.get("task") is not None or reminder.get("action") is not None


def is_history_slots_nonempty(history) -> bool:
    if not isinstance(history, dict):
        return False
    return history.get("metric") is not None or history.get("time_range") is not None


DANGEROUS_KEYWORDS = [
    "停药", "加药", "换药", "开药", "买药", "配药",
    "你是高血压", "你是糖尿病", "诊断", "确诊", "你得了",
    "高血压一级", "高血压二级", "高血压三级", "糖尿病前期",
]


def check_dangerous_content(text: str) -> list[str]:
    """Check for dangerous medical content in text."""
    found = []
    for kw in DANGEROUS_KEYWORDS:
        if kw in text:
            found.append(kw)
    return found


def validate_routing_slots(filepath: str) -> tuple[list[dict], list[dict]]:
    """Validate routing/slots consistency. Returns (valid, inconsistent)."""
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    valid = []
    inconsistent = []
    dangerous = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            output = sample.get("output", {})
            routing = output.get("routing", {})
            slots = output.get("slots", {})
            missing_fields = output.get("missing_fields", [])
            issues = []

            # Forward: routing=true => slots non-empty
            # Skip this check when missing_fields is non-empty: the model correctly
            # indicates intent but the user didn't provide the required values.
            if not missing_fields:
                if routing.get("need_record"):
                    if not is_record_slots_nonempty(slots.get("record")):
                        issues.append("need_record=true but slots.record is empty")

                if routing.get("need_advice"):
                    if not is_advice_slots_nonempty(slots.get("advice")):
                        issues.append("need_advice=true but slots.advice.topic/question is empty")

                if routing.get("need_plan"):
                    if not is_plan_slots_nonempty(slots.get("plan")):
                        issues.append("need_plan=true but slots.plan.goal is empty")

                if routing.get("need_reminder"):
                    if not is_reminder_slots_nonempty(slots.get("reminder")):
                        issues.append("need_reminder=true but slots.reminder.task/action is empty")

                if routing.get("need_history"):
                    if not is_history_slots_nonempty(slots.get("history")):
                        issues.append("need_history=true but slots.history.metric/time_range is empty")

            # Reverse: slots non-empty => routing=true
            if is_record_slots_nonempty(slots.get("record")):
                if not routing.get("need_record"):
                    issues.append("slots.record is non-empty but need_record=false")

            if is_advice_slots_nonempty(slots.get("advice")):
                if not routing.get("need_advice"):
                    issues.append("slots.advice is non-empty but need_advice=false")

            if is_plan_slots_nonempty(slots.get("plan")):
                if not routing.get("need_plan"):
                    issues.append("slots.plan is non-empty but need_plan=false")

            if is_reminder_slots_nonempty(slots.get("reminder")):
                if not routing.get("need_reminder"):
                    issues.append("slots.reminder is non-empty but need_reminder=false")

            if is_history_slots_nonempty(slots.get("history")):
                if not routing.get("need_history"):
                    issues.append("slots.history is non-empty but need_history=false")

            # Check dangerous content in input (what the model might learn)
            user_input = sample.get("input", "")
            dangerous_kws = check_dangerous_content(str(user_input))
            if dangerous_kws:
                dangerous.append({"line": i, "input": user_input, "keywords": dangerous_kws})

            if issues:
                inconsistent.append({"line": i, "input": sample.get("input", ""), "issues": issues})
            else:
                valid.append(sample)

    return valid, inconsistent, dangerous


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_routing_slots.py <file.jsonl>")
        sys.exit(1)

    filepath = sys.argv[1]
    valid, inconsistent, dangerous = validate_routing_slots(filepath)

    total = len(valid) + len(inconsistent)
    rate = len(valid) / total * 100 if total > 0 else 0

    print(f"Total checked: {total}")
    print(f"Consistent: {len(valid)} ({rate:.1f}%)")
    print(f"Inconsistent: {len(inconsistent)}")

    if inconsistent:
        print("\n--- Inconsistencies ---")
        for item in inconsistent[:20]:
            print(f"  Line {item['line']}: {item['input'][:60]}...")
            for issue in item["issues"]:
                print(f"    -> {issue}")
        if len(inconsistent) > 20:
            print(f"  ... and {len(inconsistent) - 20} more")

    if dangerous:
        print(f"\n--- Dangerous Content ({len(dangerous)}) ---")
        for item in dangerous:
            print(f"  Line {item['line']}: {item['input'][:80]}")
            print(f"    Keywords: {item['keywords']}")

    print(f"\nRouting-slots consistency rate: {rate:.1f}%")

    if inconsistent or dangerous:
        sys.exit(1)


if __name__ == "__main__":
    main()
