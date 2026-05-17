"""Generate synthetic training data by calling an LLM API with prompt templates."""

import json
import os
import re
import sys
import time
from pathlib import Path

# Load .env file from project root
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

# Supports OpenAI-compatible APIs (OpenAI, vLLM, Ollama, etc.)
try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)


PROMPT_DIR = Path(__file__).parent / "prompts"
RAW_DIR = Path(__file__).parent.parent.parent / "data" / "synthetic" / "raw"

SCENARIOS = {
    "record": "generate_record.md",
    "record_advice": "generate_record_advice.md",
    "record_reminder": "generate_record_reminder.md",
    "history_advice": "generate_history_advice.md",
    "plan": "generate_plan.md",
    "incomplete": "generate_incomplete.md",
    "out_of_scope": "generate_out_of_scope.md",
    "all": "generate_all.md",
}


def load_prompt(scenario: str) -> str:
    """Load a prompt template for the given scenario."""
    filename = SCENARIOS.get(scenario)
    if not filename:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(SCENARIOS.keys())}")

    prompt_path = PROMPT_DIR / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8")


def extract_jsonl_objects(text: str) -> list[dict]:
    """Extract JSON objects from LLM output.

    Handles: clean JSONL, markdown code fences, multi-line JSON,
    and explanatory text mixed with JSON.
    """
    # Step 1: Extract content from markdown code fences if present
    fence_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)```', text, re.DOTALL)
    if fence_blocks:
        text = '\n'.join(fence_blocks)
    else:
        text = re.sub(r'```\w*\s*', '', text)

    json_objects = []
    buffer_lines = []

    for raw_line in text.strip().split('\n'):
        stripped = raw_line.strip()
        if not stripped:
            continue

        # If not currently building a multi-line object, skip non-JSON lines
        if not buffer_lines and not stripped.startswith('{'):
            continue

        # Fast path: try single-line JSON parse on lines that start a new object
        if not buffer_lines and stripped.startswith('{'):
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    json_objects.append(obj)
                continue
            except json.JSONDecodeError:
                pass

        # Multi-line path: accumulate until braces are balanced
        buffer_lines.append(stripped)
        combined = ''.join(buffer_lines)
        brace_count = combined.count('{') - combined.count('}')

        if brace_count == 0:
            try:
                obj = json.loads(combined)
                if isinstance(obj, dict):
                    json_objects.append(obj)
            except json.JSONDecodeError:
                pass
            buffer_lines = []  # clear regardless — balanced but invalid won't become valid
        elif brace_count < 0:
            buffer_lines = []  # malformed: more closing braces than opening
        elif len(buffer_lines) > 20:
            buffer_lines = []  # safety valve

    # Final flush for any leftover balanced content
    if buffer_lines:
        combined = ''.join(buffer_lines)
        try:
            obj = json.loads(combined)
            if isinstance(obj, dict):
                json_objects.append(obj)
        except json.JSONDecodeError:
            pass

    return json_objects


def generate_batch(
    scenario: str,
    client: OpenAI,
    model: str,
    extra_instruction: str = "",
) -> list[dict]:
    """Generate one batch of synthetic data for a given scenario."""
    system_prompt = load_prompt(scenario)

    if extra_instruction:
        system_prompt += f"\n\n额外要求：{extra_instruction}"

    print(f"Calling {model} for scenario '{scenario}'...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请直接输出 JSONL，每行一个 JSON，不要输出任何其他内容。"}
        ],
        temperature=0.8,
        max_tokens=16384,
    )

    content = response.choices[0].message.content
    print(f"  Response length: {len(content)} chars, {response.usage.total_tokens} tokens")

    samples = extract_jsonl_objects(content)
    print(f"  Extracted {len(samples)} JSON objects")

    return samples


def save_batch(samples: list[dict], scenario: str, batch_num: int = None):
    """Save a batch of samples to the raw directory."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if batch_num is not None:
        filename = f"batch_{batch_num:03d}_{scenario}.jsonl"
    else:
        filename = f"{scenario}.jsonl"

    filepath = RAW_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"  Saved {len(samples)} samples to {filepath}")
    return filepath


def generate_scenario(
    scenario: str,
    client: OpenAI,
    model: str,
    target_count: int = 100,
    samples_per_call: int = 50,
) -> str:
    """Generate data for one scenario, making multiple API calls if needed."""
    all_samples = []
    batch_num = 1

    while len(all_samples) < target_count:
        remaining = target_count - len(all_samples)
        extra = f"请生成恰好 {min(remaining, samples_per_call)} 条样本。"

        try:
            samples = generate_batch(scenario, client, model, extra_instruction=extra)
            if samples:
                all_samples.extend(samples)
                save_batch(samples, scenario, batch_num)
                batch_num += 1
            else:
                print(f"  Warning: No valid samples extracted, retrying...")
        except Exception as e:
            print(f"  Error: {e}, retrying in 5s...")
            time.sleep(5)

    # Save combined file for this scenario
    final_path = RAW_DIR / f"{scenario}.jsonl"
    with open(final_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"  Total: {len(all_samples)} samples saved to {final_path}")
    return str(final_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic routing data using an LLM")
    parser.add_argument("--scenario", "-s", choices=list(SCENARIOS.keys()), default="all",
                        help="Which scenario to generate (default: all)")
    parser.add_argument("--model", "-m", default=os.environ.get("LLM_MODEL", "gpt-4o"),
                        help="Model name (default: $LLM_MODEL or gpt-4o)")
    parser.add_argument("--api-key", "-k", default=os.environ.get("OPENAI_API_KEY", ""),
                        help="API key (default: $OPENAI_API_KEY)")
    parser.add_argument("--base-url", "-b", default=os.environ.get("OPENAI_BASE_URL", ""),
                        help="API base URL (default: $OPENAI_BASE_URL)")
    parser.add_argument("--target", "-t", type=int, default=100,
                        help="Target samples per scenario (default: 100)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Samples per API call (default: 50)")
    args = parser.parse_args()

    if not args.api_key:
        print("[ERROR] No API key provided. Set OPENAI_API_KEY or use --api-key.")
        sys.exit(1)

    client_kwargs = {"api_key": args.api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url

    client = OpenAI(**client_kwargs)

    scenarios_to_run = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    # Remove "all" from the list if present
    scenarios_to_run = [s for s in scenarios_to_run if s != "all"]

    print(f"Model: {args.model}")
    print(f"Scenarios: {scenarios_to_run}")
    print(f"Target per scenario: {args.target}")
    print(f"Batch size: {args.batch_size}")
    print(f"Raw output dir: {RAW_DIR}")
    print()

    for scenario in scenarios_to_run:
        print(f"\n{'='*60}")
        print(f"Generating: {scenario}")
        print(f"{'='*60}")
        generate_scenario(scenario, client, args.model, args.target, args.batch_size)

    print(f"\nDone! All generated files are in: {RAW_DIR}")


if __name__ == "__main__":
    main()
