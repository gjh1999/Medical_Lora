"""Convert {input, output} samples to Qwen messages format for SFT."""

import json
import sys
from pathlib import Path


SYSTEM_PROMPT = "你是慢病健康管理系统的路由与信息抽取器。你只能输出合法 JSON，不要输出解释。"


def convert_sample(sample: dict) -> dict:
    """Convert a single sample to messages format."""
    user_input = sample["input"]
    assistant_output = sample["output"]

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": json.dumps(assistant_output, ensure_ascii=False)}
        ]
    }


def convert_file(input_path: str, output_path: str):
    """Convert an entire JSONL file to SFT format."""
    in_path = Path(input_path)
    if not in_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    count = 0
    errors = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                sft_sample = convert_sample(sample)
                fout.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
                count += 1
            except json.JSONDecodeError as e:
                print(f"Line {i}: JSON decode error: {e}")
                errors += 1
            except KeyError as e:
                print(f"Line {i}: Missing key: {e}")
                errors += 1

    print(f"Converted: {count} samples")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_to_sft.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    convert_file(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
