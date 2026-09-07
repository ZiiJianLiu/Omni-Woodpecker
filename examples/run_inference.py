#!/usr/bin/env python3
"""Run the complete OWP use case and read its final result records."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OWP integration example.")
    parser.add_argument("--model", choices=("videollama2", "qwen"), default="videollama2")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "sample_data" / "owp_input.jsonl",
        help="Request JSONL; defaults to the bundled representative requests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Result JSONL; defaults to results/owp_<model>_example.jsonl.",
    )
    parser.add_argument("--max-rows", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="auto", help="Torch device for the Qwen backend.")
    parser.add_argument("--model-path", default=None, help="Local checkpoint path or Hugging Face model id.")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else ROOT / "results" / f"owp_{args.model}_example.jsonl"
    )
    command = [
        sys.executable,
        str(ROOT / "src" / "owp_infer.py"),
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--model",
        args.model,
        "--max-rows",
        str(args.max_rows),
        "--dtype",
        args.dtype,
        "--device",
        str(args.device),
    ]
    if args.model_path:
        command.extend(["--model-path", str(args.model_path)])
    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            result = json.loads(line)
            print(result["sample_id"], result["answer"], result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
