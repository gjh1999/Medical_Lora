"""Validate that every line in a JSONL file is valid JSON and has required top-level keys."""

import json
import sys
from pathlib import Path


REQUIRED_TOP_KEYS = {"input", "output"}
REQUIRED_OUTPUT_KEYS = {
    "routing", "confidence", "slots", "missing_fields", "risk_flags", "normalized"
}
ROUTING_KEYS = {
    "need_record", "need_advice", "need_plan", "need_reminder", "need_history"
}


def validate_jsonl(filepath: str) -> tuple[list[dict], list[dict]]:
    """Validate a JSONL file. Returns (valid_samples, errors)."""
    valid = []
    errors = []
    path = Path(filepath)

    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Check 1: valid JSON
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append({"line": i, "error": f"Invalid JSON: {e}"})
                continue

            # Check 2: has "input"
            if "input" not in sample:
                errors.append({"line": i, "error": "Missing 'input' key"})
                continue

            # Check 3: has "output"
            if "output" not in sample:
                errors.append({"line": i, "error": "Missing 'output' key"})
                continue

            output = sample["output"]

            # Check 4: output has all required keys
            missing_keys = REQUIRED_OUTPUT_KEYS - set(output.keys())
            if missing_keys:
                errors.append({"line": i, "error": f"Output missing keys: {missing_keys}"})
                continue

            # Check 5: routing has 5 boolean fields
            routing = output.get("routing", {})
            if not isinstance(routing, dict):
                errors.append({"line": i, "error": "'routing' must be a dict"})
                continue
            missing_routing = ROUTING_KEYS - set(routing.keys())
            if missing_routing:
                errors.append({"line": i, "error": f"Routing missing keys: {missing_routing}"})
                continue
            for key in ROUTING_KEYS:
                if not isinstance(routing[key], bool):
                    errors.append({"line": i, "error": f"routing.{key} must be boolean"})
                    break

            # Check 6: confidence has 5 float fields in [0, 1]
            confidence = output.get("confidence", {})
            if not isinstance(confidence, dict):
                errors.append({"line": i, "error": "'confidence' must be a dict"})
                continue
            for key in ROUTING_KEYS:
                val = confidence.get(key)
                if not isinstance(val, (int, float)):
                    errors.append({"line": i, "error": f"confidence.{key} must be a number"})
                    break
                if val < 0.0 or val > 1.0:
                    errors.append({"line": i, "error": f"confidence.{key} out of range [0,1]: {val}"})
                    break

            valid.append(sample)

    return valid, errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_jsonl.py <file.jsonl>")
        sys.exit(1)

    filepath = sys.argv[1]
    valid, errors = validate_jsonl(filepath)

    print(f"Total lines processed: {len(valid) + len(errors)}")
    print(f"Valid: {len(valid)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\n--- Errors ---")
        for err in errors:
            print(f"  Line {err['line']}: {err['error']}")

    success_rate = len(valid) / (len(valid) + len(errors)) * 100 if (len(valid) + len(errors)) > 0 else 0
    print(f"\nJSON parse success rate: {success_rate:.1f}%")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
