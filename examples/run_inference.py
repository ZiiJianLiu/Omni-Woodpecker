#!/usr/bin/env python3
"""Run the complete OWP use case and read its final result records."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from owp import correct


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
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Rows to process; zero runs all three bundled CMM examples.",
    )
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
    results = correct(
        input_path,
        output,
        model=args.model,
        max_rows=args.max_rows,
        dtype=args.dtype,
        device=args.device,
        model_path=args.model_path,
    )
    for result in results:
        print(result["sample_id"], result["answer"], result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
