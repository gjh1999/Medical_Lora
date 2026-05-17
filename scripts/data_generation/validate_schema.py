"""Validate that slots, routing, and confidence follow the expected schema."""

import json
import sys
from pathlib import Path


SLOTS_RECORD_FIELDS = {
    "blood_pressure", "blood_glucose", "weight", "diet", "exercise", "medication", "symptom"
}
SLOTS_ADVICE_FIELDS = {"topic", "question", "related_metrics"}
SLOTS_PLAN_FIELDS = {"goal", "duration", "preferences", "constraints"}
SLOTS_REMINDER_FIELDS = {"action", "task", "time", "repeat"}
SLOTS_HISTORY_FIELDS = {"metric", "time_range", "aggregation"}

REMINDER_ACTIONS = {"create", "modify", "delete", "query", None}
AGGREGATION_TYPES = {"latest", "average", "trend", "max", "min", "count", "summary", None}
GLUCOSE_TYPES = {"fasting", "postprandial", "random", "before_sleep", "unknown", None}
TOPICS = {
    "blood_pressure", "blood_glucose", "weight", "diet", "exercise",
    "medication", "symptom", "general_health", None
}

# Numeric range checks
RANGES = {
    "blood_pressure.systolic": (60, 250),
    "blood_pressure.diastolic": (40, 150),
    "blood_glucose.value": (2.0, 30.0),
    "weight.value": (20, 250),
    "exercise.duration_minutes": (0, 300),
}


def check_slots_structure(slots: dict) -> list[str]:
    """Check slots structure, return list of warnings."""
    warnings = []

    # record
    record = slots.get("record", {})
    if not isinstance(record, dict):
        warnings.append("slots.record must be a dict")
    else:
        for field in SLOTS_RECORD_FIELDS:
            if field not in record:
                warnings.append(f"slots.record missing field: {field}")

    # advice
    advice = slots.get("advice", {})
    if not isinstance(advice, dict):
        warnings.append("slots.advice must be a dict")
    else:
        for field in SLOTS_ADVICE_FIELDS:
            if field not in advice:
                warnings.append(f"slots.advice missing field: {field}")
        if advice.get("topic") not in TOPICS:
            warnings.append(f"slots.advice.topic unknown value: {advice.get('topic')}")

    # plan
    plan = slots.get("plan", {})
    if not isinstance(plan, dict):
        warnings.append("slots.plan must be a dict")
    else:
        for field in SLOTS_PLAN_FIELDS:
            if field not in plan:
                warnings.append(f"slots.plan missing field: {field}")

    # reminder
    reminder = slots.get("reminder", {})
    if not isinstance(reminder, dict):
        warnings.append("slots.reminder must be a dict")
    else:
        for field in SLOTS_REMINDER_FIELDS:
            if field not in reminder:
                warnings.append(f"slots.reminder missing field: {field}")
        if reminder.get("action") not in REMINDER_ACTIONS:
            warnings.append(f"slots.reminder.action unknown value: {reminder.get('action')}")

    # history
    history = slots.get("history", {})
    if not isinstance(history, dict):
        warnings.append("slots.history must be a dict")
    else:
        for field in SLOTS_HISTORY_FIELDS:
            if field not in history:
                warnings.append(f"slots.history missing field: {field}")
        if history.get("aggregation") not in AGGREGATION_TYPES:
            warnings.append(f"slots.history.aggregation unknown value: {history.get('aggregation')}")

    return warnings


def check_numeric_ranges(record) -> list[str]:
    """Check numeric values are in reasonable ranges."""
    warnings = []
    if not isinstance(record, dict):
        return warnings

    bp = record.get("blood_pressure")
    if isinstance(bp, dict):
        systolic = bp.get("systolic")
        diastolic = bp.get("diastolic")
        if isinstance(systolic, (int, float)) and systolic is not None:
            lo, hi = RANGES["blood_pressure.systolic"]
            if systolic < lo or systolic > hi:
                warnings.append(f"blood_pressure.systolic out of range [{lo},{hi}]: {systolic}")
        if isinstance(diastolic, (int, float)) and diastolic is not None:
            lo, hi = RANGES["blood_pressure.diastolic"]
            if diastolic < lo or diastolic > hi:
                warnings.append(f"blood_pressure.diastolic out of range [{lo},{hi}]: {diastolic}")

    bg = record.get("blood_glucose")
    if isinstance(bg, dict):
        value = bg.get("value")
        if isinstance(value, (int, float)) and value is not None:
            lo, hi = RANGES["blood_glucose.value"]
            if value < lo or value > hi:
                warnings.append(f"blood_glucose.value out of range [{lo},{hi}]: {value}")
        if bg.get("type") not in GLUCOSE_TYPES:
            warnings.append(f"blood_glucose.type unknown: {bg.get('type')}")

    w = record.get("weight")
    if isinstance(w, dict):
        value = w.get("value")
        if isinstance(value, (int, float)) and value is not None:
            lo, hi = RANGES["weight.value"]
            if value < lo or value > hi:
                warnings.append(f"weight.value out of range [{lo},{hi}]: {value}")

    ex = record.get("exercise")
    if isinstance(ex, dict):
        dur = ex.get("duration_minutes")
        if isinstance(dur, (int, float)) and dur is not None:
            lo, hi = RANGES["exercise.duration_minutes"]
            if dur > hi:
                warnings.append(f"exercise.duration_minutes > {hi}: {dur}")

    return warnings


def validate_schema(filepath: str):
    """Validate schema of all samples in a JSONL file."""
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    total = 0
    schema_errors = 0
    range_warnings = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            slots = sample.get("output", {}).get("slots", {})
            record = slots.get("record") if isinstance(slots.get("record"), dict) else {}

            warnings = check_slots_structure(slots)
            range_warns = check_numeric_ranges(record)

            if warnings:
                schema_errors += 1
                print(f"Line {i} schema warnings:")
                for w in warnings:
                    print(f"  - {w}")

            if range_warns:
                range_warnings += 1
                print(f"Line {i} range warnings:")
                for w in range_warns:
                    print(f"  - {w}")

    schema_ok = total - schema_errors
    rate = schema_ok / total * 100 if total > 0 else 0
    print(f"\nTotal: {total}")
    print(f"Schema valid: {schema_ok} ({rate:.1f}%)")
    print(f"Schema warnings: {schema_errors}")
    print(f"Range warnings: {range_warnings}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_schema.py <file.jsonl>")
        sys.exit(1)
    validate_schema(sys.argv[1])


if __name__ == "__main__":
    main()
