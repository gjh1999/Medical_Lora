"""Master pipeline: merge, repair, validate, deduplicate, convert, and split synthetic data.

Steps:
  1. Merge all raw/*.jsonl files into one combined file
  2. Repair & normalize (fill missing keys, fix record schema, convert string values)
  3. Validate JSONL format (catches truly unparseable lines)
  4. Validate schema
  5. Validate routing/slots consistency
  6. Deduplicate
  7. Analyze distribution
  8. Split into train/dev/test
  9. Convert to SFT messages format
"""

import json
import sys
from pathlib import Path

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))

from validate_jsonl import validate_jsonl
from normalize_slots import normalize_file
from validate_schema import validate_schema
from validate_routing_slots import validate_routing_slots
from deduplicate import deduplicate
from analyze_distribution import analyze
from convert_to_sft import convert_file
from split_dataset import split_dataset


PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "synthetic" / "raw"
CHECKED_DIR = PROJECT_ROOT / "data" / "synthetic" / "checked"
SFT_DIR = PROJECT_ROOT / "data" / "synthetic" / "sft"


def step1_merge():
    """Merge all raw JSONL files into one combined file."""
    print("=" * 60)
    print("STEP 1: Merge raw files")
    print("=" * 60)

    CHECKED_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = CHECKED_DIR / "combined_raw.jsonl"

    all_samples = []
    all_jsonl = sorted(RAW_DIR.glob("*.jsonl"))

    if not all_jsonl:
        print("[ERROR] No JSONL files found in raw/ directory.")
        print("Run generate_data.py first, or place your JSONL files in data/synthetic/raw/")
        return None

    # When batch_* files exist, skip the combined files (e.g. record.jsonl)
    # to avoid double-counting. The combined files are just concatenations of batches.
    batch_files = [f for f in all_jsonl if f.name.startswith("batch_")]
    if batch_files:
        jsonl_files = batch_files
        print(f"Found {len(all_jsonl)} files, using {len(batch_files)} batch files "
              f"(skipping {len(all_jsonl) - len(batch_files)} combined files)")
    else:
        jsonl_files = all_jsonl
        print(f"Found {len(jsonl_files)} files")

    for f in jsonl_files:
        print(f"  {f.name}")

    for f in jsonl_files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_samples.append(line)

    with open(combined_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(sample + "\n")

    print(f"\nMerged {len(all_samples)} samples -> {combined_path}")
    return combined_path


def step2_validate_jsonl(filepath: str) -> bool:
    """Validate JSON format."""
    print("\n" + "=" * 60)
    print("STEP 2: Validate JSONL format")
    print("=" * 60)

    valid, errors = validate_jsonl(filepath)

    if errors:
        print(f"\n[WARNING] {len(errors)} lines have JSON errors.")
        # Filter out invalid lines
        valid_path = CHECKED_DIR / "valid_json.jsonl"
        with open(valid_path, "w", encoding="utf-8") as f:
            for sample in valid:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Filtered valid samples ({len(valid)}) -> {valid_path}")
        return True
    else:
        print("All samples have valid JSON.")
        return True


def step3_validate_schema(filepath: str):
    """Validate schema structure."""
    print("\n" + "=" * 60)
    print("STEP 3: Validate schema")
    print("=" * 60)
    validate_schema(filepath)


def step4_validate_routing(filepath: str) -> str:
    """Validate routing/slots consistency."""
    print("\n" + "=" * 60)
    print("STEP 4: Validate routing/slots consistency")
    print("=" * 60)

    valid, inconsistent, dangerous = validate_routing_slots(filepath)

    if dangerous:
        print(f"\n[WARNING] {len(dangerous)} samples contain dangerous medical content!")
        for item in dangerous:
            print(f"  Line {item['line']}: {item['input'][:80]}")

    if inconsistent:
        print(f"\n[WARNING] {len(inconsistent)} samples have routing/slots inconsistency.")
        # Save only consistent samples
        consistent_path = CHECKED_DIR / "routing_consistent.jsonl"
        with open(consistent_path, "w", encoding="utf-8") as f:
            for sample in valid:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Filtered consistent samples ({len(valid)}) -> {consistent_path}")
        return str(consistent_path)

    print("All samples have consistent routing/slots.")
    return filepath


def step5_deduplicate(filepath: str) -> str:
    """Deduplicate samples."""
    print("\n" + "=" * 60)
    print("STEP 5: Deduplicate")
    print("=" * 60)

    dedup_path = CHECKED_DIR / "deduplicated.jsonl"
    deduplicate(filepath, threshold=0.9, output=str(dedup_path))
    return str(dedup_path)


def step6_analyze(filepath: str):
    """Analyze distribution."""
    print("\n" + "=" * 60)
    print("STEP 6: Analyze distribution")
    print("=" * 60)
    analyze(filepath)


def step7_split(filepath: str) -> dict:
    """Split the original-format data into train/dev/test (before SFT conversion)."""
    print("\n" + "=" * 60)
    print("STEP 7: Split into train/dev/test (original format)")
    print("=" * 60)

    split_dir = CHECKED_DIR / "splits"
    split_dataset(filepath, str(split_dir), train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)

    result = {}
    for name in ["train", "dev", "test"]:
        p = split_dir / f"{name}.jsonl"
        if p.exists():
            result[name] = str(p)
    return result


def step8_convert_sft(split_files: dict):
    """Convert each split to SFT messages format."""
    print("\n" + "=" * 60)
    print("STEP 8: Convert splits to SFT format")
    print("=" * 60)

    SFT_DIR.mkdir(parents=True, exist_ok=True)

    for name, filepath in split_files.items():
        sft_path = SFT_DIR / f"{name}.jsonl"
        convert_file(filepath, str(sft_path))
        print(f"  {name}: {sft_path}")


def step_normalize_slots(filepath: str) -> str:
    """Repair structure + normalize record schema and values."""
    print("\n" + "=" * 60)
    print("STEP 2: Repair & normalize (structure, schema, values)")
    print("=" * 60)
    normalized_path = CHECKED_DIR / "normalized.jsonl"
    normalize_file(filepath, str(normalized_path))
    return str(normalized_path)


def main():
    print("Synthetic Data Pipeline")
    print("=" * 60)

    # Step 1: Merge
    combined = step1_merge()
    if combined is None:
        print("\nPipeline stopped: no data to process.")
        sys.exit(1)
    current_file = str(combined)

    # Step 2: Normalize & repair (before JSON validation to recover structural errors)
    current_file = step_normalize_slots(current_file)

    # Step 3: Validate JSONL (now only catches truly unparseable lines)
    step2_validate_jsonl(current_file)
    valid_json_path = CHECKED_DIR / "valid_json.jsonl"
    if valid_json_path.exists():
        current_file = str(valid_json_path)

    # Step 4: Validate schema
    step3_validate_schema(current_file)

    # Step 5: Validate routing/slots
    current_file = step4_validate_routing(current_file)

    # Step 6: Deduplicate
    current_file = step5_deduplicate(current_file)

    # Step 7: Analyze distribution
    step6_analyze(current_file)

    # Step 7: Split (original format)
    split_files = step7_split(current_file)

    # Step 8: Convert each split to SFT format
    step8_convert_sft(split_files)

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nFinal files:")
    print(f"  Checked (deduplicated): {current_file}")
    for name in ["train", "dev", "test"]:
        path = SFT_DIR / f"{name}.jsonl"
        if path.exists():
            count = sum(1 for _ in open(path, "r", encoding="utf-8"))
            print(f"  SFT {name}: {path} ({count} samples)")


if __name__ == "__main__":
    main()
