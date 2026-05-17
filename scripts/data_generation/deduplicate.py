"""Deduplicate JSONL samples: exact match and character-level similarity."""

import json
import sys
from difflib import SequenceMatcher
from pathlib import Path


def text_similarity(a: str, b: str) -> float:
    """Compute character-level similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def deduplicate(filepath: str, threshold: float = 0.9, output: str = None) -> list[dict]:
    """Deduplicate a JSONL file. Returns deduplicated samples."""
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

    original_count = len(samples)
    seen_inputs = set()
    deduped = []
    exact_dup_count = 0
    fuzzy_dup_count = 0

    for sample in samples:
        inp = sample.get("input", "")

        # Exact dedup
        if inp in seen_inputs:
            exact_dup_count += 1
            continue
        seen_inputs.add(inp)

        # Fuzzy dedup
        is_dup = False
        for existing in deduped:
            existing_inp = existing.get("input", "")
            if text_similarity(inp, existing_inp) > threshold:
                fuzzy_dup_count += 1
                is_dup = True
                break

        if not is_dup:
            deduped.append(sample)

    dup_rate = (original_count - len(deduped)) / original_count * 100 if original_count > 0 else 0

    print(f"Original samples: {original_count}")
    print(f"Exact duplicates removed: {exact_dup_count}")
    print(f"Fuzzy duplicates removed (threshold={threshold}): {fuzzy_dup_count}")
    print(f"Remaining: {len(deduped)}")
    print(f"Duplicate rate: {dup_rate:.1f}%")

    if output:
        out_path = Path(output)
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in deduped:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Deduplicated file saved to: {output}")

    return deduped


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deduplicate JSONL samples")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-o", "--output", help="Output JSONL file (optional)")
    parser.add_argument("-t", "--threshold", type=float, default=0.9,
                        help="Similarity threshold for fuzzy dedup (default: 0.9)")
    args = parser.parse_args()

    deduplicate(args.input, args.threshold, args.output)


if __name__ == "__main__":
    main()
