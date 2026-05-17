"""修复 SFT 数据中的 schema 不一致问题，输出清洗后的文件。"""
import json
import sys
from pathlib import Path
from collections import Counter

EXPECTED_ROUTING_KEYS = {"need_record", "need_advice", "need_plan", "need_reminder", "need_history"}
EXPECTED_SLOTS_KEYS = {"record", "advice", "plan", "reminder", "history"}


def clean_output(obj: dict) -> dict:
    """修复单条输出的 schema 问题，返回修复后的对象。"""
    changes = []

    # ---- routing: 只保留 5 个标准 key ----
    routing = obj.get("routing", {})
    clean_routing = {}
    for k in EXPECTED_ROUTING_KEYS:
        clean_routing[k] = bool(routing.get(k, False))
    if set(routing.keys()) != EXPECTED_ROUTING_KEYS:
        changes.append(f"routing keys: {sorted(routing.keys())} -> standard 5")
    obj["routing"] = clean_routing

    # ---- confidence: 只保留 5 个标准 key，同时对齐 routing ----
    confidence = obj.get("confidence", {})
    clean_confidence = {}
    for k in EXPECTED_ROUTING_KEYS:
        val = confidence.get(k, 0.0)
        if clean_routing[k] and val < 0.5:
            val = 0.95  # routing=true 时修到合理值
        elif not clean_routing[k] and val > 0.5:
            val = 0.02  # routing=false 时修到合理值
        clean_confidence[k] = val
    if set(confidence.keys()) != EXPECTED_ROUTING_KEYS:
        changes.append(f"confidence keys: {sorted(confidence.keys())} -> standard 5")
    obj["confidence"] = clean_confidence

    # ---- slots: 去掉嵌套在里面的 missing_fields/risk_flags/normalized ----
    slots = obj.get("slots", {})
    extra_in_slots = set(slots.keys()) - EXPECTED_SLOTS_KEYS
    if extra_in_slots:
        changes.append(f"slots extra keys: {sorted(extra_in_slots)} removed")
        for ek in extra_in_slots:
            del slots[ek]
    # 确保 5 个标准 slot 都存在
    for k in EXPECTED_SLOTS_KEYS:
        if k not in slots:
            slots[k] = {}
            changes.append(f"missing slot '{k}' added")
    obj["slots"] = slots

    # ---- 确保顶层有 missing_fields / risk_flags / normalized ----
    if "missing_fields" not in obj:
        obj["missing_fields"] = []
        changes.append("added top-level missing_fields")
    if "risk_flags" not in obj:
        obj["risk_flags"] = []
        changes.append("added top-level risk_flags")
    if "normalized" not in obj:
        obj["normalized"] = {"date": None, "time_of_day": None, "units": {}}
        changes.append("added top-level normalized")

    return obj, changes


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    src_dir = Path("data/synthetic/sft")
    out_dir = Path("data/synthetic/sft")
    stats = Counter()

    for name in ["train", "dev", "test"]:
        in_path = src_dir / f"{name}.jsonl"
        samples = load_jsonl(in_path)

        cleaned = []
        for s in samples:
            msg = s["messages"]
            assistant_content = msg[-1]["content"]
            obj = json.loads(assistant_content)
            fixed_obj, changes = clean_output(obj)

            if changes:
                stats[f"{name}_fixed"] += 1
                stats[tuple(changes)] += 1
            else:
                stats[f"{name}_clean"] += 1

            msg[-1]["content"] = json.dumps(fixed_obj, ensure_ascii=False)
            cleaned.append(s)

        out_path = out_dir / f"{name}.jsonl"
        save_jsonl(out_path, cleaned)
        print(f"{name}: {len(samples)} -> {len(cleaned)} (fixed {stats[name+'_fixed']}, clean {stats[name+'_clean']})")

    print(f"\nFix breakdown:")
    for change_tuple, cnt in stats.most_common():
        if isinstance(change_tuple, tuple):
            print(f"  [{cnt}] {change_tuple[0]}")
            for ch in change_tuple[1:]:
                print(f"       {ch}")


if __name__ == "__main__":
    main()
