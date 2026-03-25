from __future__ import annotations
import os
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional



# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# glossary_path = os.path.join(BASE_DIR, "dataset", "glossary_output_1.json")
  

# =========================================================
# Config
# =========================================================

USE_CLI_ARGS = False


@dataclass
class Config:
    input_path: str = "dataset/training_data_claude.jsonl"
    output_path: str = "dataset/training_data_claude_output_messages_type.jsonl"

    source_key: str = "영문"
    target_key: str = "한글"

    include_system: bool = True

    system_prompt: str = (
        "You are a professional technical translator for EMC and electrical "
        "engineering documents. Translate the user's English text into accurate "
        "Korean while preserving technical meaning and terminology."
    )

    instruction_template: str = (
        "Translate the following English technical text into Korean while "
        "preserving terminology and meaning.\n\n{source}"
    )

    strip_text: bool = True
    skip_empty: bool = True


CONFIG = Config()


# =========================================================
# Argument parsing
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert translation-pair JSONL to chat-style JSONL."
    )

    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file.")
    parser.add_argument("--source-key", type=str, default="영문", help="Source text key.")
    parser.add_argument("--target-key", type=str, default="한글", help="Target text key.")
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Include a system message in the output.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are a professional technical translator for EMC and electrical "
            "engineering documents. Translate the user's English text into accurate "
            "Korean while preserving technical meaning and terminology."
        ),
        help="System prompt text.",
    )
    parser.add_argument(
        "--instruction-template",
        type=str,
        default=(
            "Translate the following English technical text into Korean while "
            "preserving terminology and meaning.\n\n{source}"
        ),
        help="User instruction template. Use '{source}' placeholder.",
    )
    parser.add_argument("--strip", action="store_true", help="Strip whitespace.")
    parser.add_argument("--skip-empty", action="store_true", help="Skip empty samples.")

    return parser.parse_args()


def load_runtime_config() -> Config:
    """
    Load config from either CLI arguments or in-file CONFIG.
    """
    if not USE_CLI_ARGS:
        return CONFIG

    args = parse_args()
    return Config(
        input_path=args.input,
        output_path=args.output,
        source_key=args.source_key,
        target_key=args.target_key,
        include_system=args.include_system,
        system_prompt=args.system_prompt,
        instruction_template=args.instruction_template,
        strip_text=args.strip,
        skip_empty=args.skip_empty,
    )


# =========================================================
# Core helpers
# =========================================================

def safe_to_text(value: Any) -> str:
    """
    Convert input value to string safely.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def read_json_line(line: str, line_number: int) -> Optional[Dict[str, Any]]:
    """
    Parse one line from JSONL safely.
    """
    line = line.strip()
    if not line:
        return None

    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        print(f"[WARN] line {line_number}: invalid JSON skipped -> {exc}")
        return None

    if not isinstance(obj, dict):
        print(f"[WARN] line {line_number}: non-dict JSON skipped")
        return None

    return obj


def build_messages(
    source_text: str,
    target_text: str,
    include_system: bool,
    system_prompt: str,
    instruction_template: str,
) -> Dict[str, Any]:
    """
    Build chat-format sample.
    """
    user_content = instruction_template.format(source=source_text)

    messages = []

    if include_system:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": target_text,
        }
    )

    return {"messages": messages}


# =========================================================
# Conversion
# =========================================================

def convert_file(config: Config) -> None:
    """
    Convert translation-pair JSONL to chat-style JSONL.
    """
    input_path = Path(config.input_path)
    output_path = Path(config.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    valid_objects = 0
    converted_count = 0
    skipped_invalid = 0
    skipped_missing_key = 0
    skipped_empty = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_number, line in enumerate(fin, start=1):
            total_lines += 1

            obj = read_json_line(line, line_number)
            if obj is None:
                skipped_invalid += 1
                continue

            valid_objects += 1

            if config.source_key not in obj or config.target_key not in obj:
                skipped_missing_key += 1
                print(
                    f"[WARN] line {line_number}: missing key(s) "
                    f"'{config.source_key}' or '{config.target_key}'"
                )
                continue

            source_text = safe_to_text(obj.get(config.source_key))
            target_text = safe_to_text(obj.get(config.target_key))

            if config.strip_text:
                source_text = source_text.strip()
                target_text = target_text.strip()

            if config.skip_empty and (not source_text or not target_text):
                skipped_empty += 1
                print(f"[WARN] line {line_number}: empty source/target skipped")
                continue

            output_obj = build_messages(
                source_text=source_text,
                target_text=target_text,
                include_system=config.include_system,
                system_prompt=config.system_prompt,
                instruction_template=config.instruction_template,
            )

            fout.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
            converted_count += 1

    print("\n===== Conversion Summary =====")
    print(f"Input file        : {input_path}")
    print(f"Output file       : {output_path}")
    print(f"Source key        : {config.source_key}")
    print(f"Target key        : {config.target_key}")
    print(f"Include system    : {config.include_system}")
    print(f"Strip text        : {config.strip_text}")
    print(f"Skip empty        : {config.skip_empty}")
    print(f"Total lines       : {total_lines}")
    print(f"Valid JSON dicts  : {valid_objects}")
    print(f"Converted         : {converted_count}")
    print(f"Skipped invalid   : {skipped_invalid}")
    print(f"Skipped no key    : {skipped_missing_key}")
    print(f"Skipped empty     : {skipped_empty}")
    print("==============================\n")


def main() -> None:
    config = load_runtime_config()
    convert_file(config)


if __name__ == "__main__":
    main()