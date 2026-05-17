"""Split a JSONL dataset into train/dev/test sets with stratified sampling."""

import json
import random
import sys
from collections import defaultdict
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


def split_dataset(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Split a JSONL file into train/dev/test with stratified sampling by routing pattern."""
    in_path = Path(input_path)
    if not in_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    random.seed(seed)

    # Load all samples
    samples = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Group by routing pattern
    groups = defaultdict(list)
    for s in samples:
        routing = s.get("output", {}).get("routing", {})
        pattern = classify_routing(routing)
        groups[pattern].append(s)

    print(f"Total samples: {len(samples)}")
    print(f"Routing patterns: {len(groups)}")
    for pattern, items in sorted(groups.items()):
        print(f"  {pattern}: {len(items)}")

    # Stratified split
    train, dev, test = [], [], []

    for pattern, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, round(n * train_ratio))
        n_dev = max(1, round(n * dev_ratio))

        train.extend(items[:n_train])
        dev.extend(items[n_train:n_train + n_dev])
        test.extend(items[n_train + n_dev:])

    # Shuffle each split
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("dev", dev), ("test", test)]:
        out_path = out_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"\n{name}: {len(data)} samples -> {out_path}")

    total = len(train) + len(dev) + len(test)
    print(f"\nSplit ratios: train={len(train)/total:.1%}, dev={len(dev)/total:.1%}, test={len(test)/total:.1%}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python split_dataset.py <input.jsonl> <output_dir> [train_ratio] [dev_ratio] [test_ratio] [seed]")
        sys.exit(1)

    train_r = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    dev_r = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    test_r = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 42

    split_dataset(sys.argv[1], sys.argv[2], train_r, dev_r, test_r, seed)


if __name__ == "__main__":
    main()
