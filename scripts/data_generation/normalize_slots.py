"""Normalize and repair synthetic data.

Three operations:
  1. repair_output_structure — fill missing routing/output/confidence keys
  2. normalize_record_schema — ensure all record dicts have 7 standard fields
  3. normalize_record_values — convert string-format values to proper dicts
"""

import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Standard schema definitions
# ---------------------------------------------------------------------------

ROUTING_KEYS = ["need_record", "need_advice", "need_plan", "need_reminder", "need_history"]
RECORD_FIELDS = ["blood_pressure", "blood_glucose", "weight", "diet", "exercise", "medication", "symptom"]
ADVICE_FIELDS = ["topic", "question", "related_metrics"]
PLAN_FIELDS = ["goal", "duration", "preferences", "constraints"]
REMINDER_FIELDS = ["action", "task", "time", "repeat"]
HISTORY_FIELDS = ["metric", "time_range", "aggregation"]


# ---------------------------------------------------------------------------
# 1. Repair output structure (fill missing keys)
# ---------------------------------------------------------------------------

def repair_output_structure(sample: dict) -> int:
    """Fill missing routing/output/confidence keys with defaults. Returns fix count."""
    fixes = 0

    if "output" not in sample:
        sample["output"] = {}
        fixes += 1
    output = sample["output"]

    # Fill missing top-level output keys
    if "routing" not in output or not isinstance(output["routing"], dict):
        output["routing"] = {k: False for k in ROUTING_KEYS}
        fixes += 1
    routing = output["routing"]

    for k in ROUTING_KEYS:
        if k not in routing:
            routing[k] = False
            fixes += 1
        elif not isinstance(routing[k], bool):
            routing[k] = False
            fixes += 1

    if "confidence" not in output or not isinstance(output["confidence"], dict):
        output["confidence"] = {k: 0.0 for k in ROUTING_KEYS}
        fixes += 1
    confidence = output["confidence"]

    for k in ROUTING_KEYS:
        if k not in confidence:
            confidence[k] = 0.0
            fixes += 1
        elif not isinstance(confidence[k], (int, float)) or not (0.0 <= confidence[k] <= 1.0):
            confidence[k] = 0.0
            fixes += 1

    if "slots" not in output or not isinstance(output["slots"], dict):
        output["slots"] = {}
        fixes += 1
    slots = output["slots"]

    # Ensure slot dicts exist with correct structure
    for slot_name, fields in [
        ("record", RECORD_FIELDS), ("advice", ADVICE_FIELDS),
        ("plan", PLAN_FIELDS), ("reminder", REMINDER_FIELDS), ("history", HISTORY_FIELDS)
    ]:
        if slot_name not in slots or not isinstance(slots[slot_name], dict):
            slots[slot_name] = {f: None for f in fields}
            fixes += 1

    if "missing_fields" not in output or not isinstance(output["missing_fields"], list):
        output["missing_fields"] = []
        fixes += 1
    if "risk_flags" not in output or not isinstance(output["risk_flags"], list):
        output["risk_flags"] = []
        fixes += 1
    if "normalized" not in output or not isinstance(output["normalized"], dict):
        output["normalized"] = {"date": None, "time_of_day": None, "units": {}}
        fixes += 1

    return fixes


# ---------------------------------------------------------------------------
# 2. Normalize record schema (ensure 7 standard field names)
# ---------------------------------------------------------------------------

# Mapping from LLM-invented record keys to standard fields (best-effort)
RECORD_KEY_ALIASES = {
    "blood_sugar": "blood_glucose",
    "heart_rate": None,  # not a standard field, skip
    "date": None,
    "timestamp": None,
    "notes": None,
    "note": None,
    "metric": None,
    "metric_type": None,
    "type": None,
    "value": None,
    "unit": None,
}


def normalize_record_schema(record: dict) -> tuple[dict, int]:
    """Ensure record dict has exactly the 7 standard fields.
    If record uses LLM-invented keys, keep the data under best-guess standard keys.
    Returns (normalized_record, fix_count).
    """
    if not isinstance(record, dict):
        return {f: None for f in RECORD_FIELDS}, 1

    # Check if already has standard schema
    current_keys = set(record.keys())
    if current_keys == set(RECORD_FIELDS):
        return record, 0

    # Non-standard schema: build a new dict with standard fields
    new_record = {}
    for field in RECORD_FIELDS:
        new_record[field] = record.get(field)  # may be None

    # Try to migrate data from non-standard keys
    for old_key, std_key in RECORD_KEY_ALIASES.items():
        if old_key in record and std_key is not None and new_record[std_key] is None:
            val = record[old_key]
            if val is not None:
                new_record[std_key] = val

    # Count non-standard keys as the fix indicator
    extra_keys = current_keys - set(RECORD_FIELDS)
    missing_keys = set(RECORD_FIELDS) - current_keys
    fix_count = 1 if (extra_keys or missing_keys) else 0

    return new_record, fix_count


# ---------------------------------------------------------------------------
# 3. Normalize record values (string -> dict)
# ---------------------------------------------------------------------------

def parse_blood_pressure(val: str) -> dict | None:
    m = re.match(r'(\d{2,3})\s*/\s*(\d{2,3})', str(val))
    if m:
        return {"systolic": int(m.group(1)), "diastolic": int(m.group(2)),
                "unit": "mmHg", "time": None}
    return None


def parse_blood_glucose(val: str) -> dict | None:
    try:
        v = float(str(val).strip())
        if 2.0 <= v <= 30.0:
            return {"value": v, "unit": "mmol/L", "type": "unknown", "time": None}
    except ValueError:
        pass
    return None


def parse_weight(val: str) -> dict | None:
    m = re.match(r'([\d.]+)\s*(kg|公斤|斤)?', str(val))
    if m:
        v = float(m.group(1))
        unit = m.group(2) if m.group(2) else None
        if unit == '斤':
            v = v / 2
            unit = 'kg'
        if unit is None:
            unit = 'kg'
        if 20 <= v <= 250:
            return {"value": v, "unit": unit, "time": None}
    return None


def parse_diet(val: str) -> dict | None:
    s = str(val).strip()
    if s:
        return {"meal": None, "content": s, "time": None}
    return None


def parse_exercise(val: str) -> dict | None:
    s = str(val).strip()
    if s:
        dur = None
        m = re.search(r'(\d+)\s*(分钟|min)', s)
        if m:
            dur = int(m.group(1))
        return {"type": s, "duration_minutes": dur, "time": None}
    return None


def parse_medication(val: str) -> dict | None:
    s = str(val).strip()
    if s:
        return {"name": s, "dose": None, "time": None}
    return None


def parse_symptom(val: str) -> dict | None:
    s = str(val).strip()
    if s:
        return {"name": s, "duration": None, "severity": "unknown", "time": None}
    return None


PARSERS = {
    "blood_pressure": parse_blood_pressure,
    "blood_glucose": parse_blood_glucose,
    "weight": parse_weight,
    "diet": parse_diet,
    "exercise": parse_exercise,
    "medication": parse_medication,
    "symptom": parse_symptom,
}


def normalize_record_values(record: dict) -> int:
    """Convert string-format record field values to proper dicts. Returns fix count."""
    if not isinstance(record, dict):
        return 0
    fixes = 0
    for field, parser in PARSERS.items():
        val = record.get(field)
        if val is None:
            continue
        if isinstance(val, dict):
            continue
        parsed = parser(val)
        if parsed is not None:
            record[field] = parsed
            fixes += 1
    return fixes


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def normalize_file(input_path: str, output_path: str) -> dict:
    """Normalize all samples. Returns stats dict."""
    stats = {"total": 0, "repaired": 0, "schema_fixed": 0, "values_fixed": 0}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1

            sample = json.loads(line)

            # Step A: Repair structure
            fixes = repair_output_structure(sample)
            if fixes > 0:
                stats["repaired"] += fixes

            # Step B: Normalize record schema
            slots = sample.get("output", {}).get("slots", {})
            record = slots.get("record")
            if record is not None:
                new_record, schema_fix = normalize_record_schema(record)
                slots["record"] = new_record
                if schema_fix > 0:
                    stats["schema_fixed"] += 1

            # Step C: Normalize record values
            val_fixes = normalize_record_values(slots["record"])
            if val_fixes > 0:
                stats["values_fixed"] += val_fixes

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Processed {stats['total']} samples:")
    print(f"  Structure repairs: {stats['repaired']} fixes")
    print(f"  Record schema normalized: {stats['schema_fixed']} samples")
    print(f"  String values converted: {stats['values_fixed']} fields")
    return stats


def main():
    if len(sys.argv) < 3:
        print("Usage: python normalize_slots.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    normalize_file(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
