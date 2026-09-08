"""Stable integration API for the Omni-Woodpecker use case.

Integrators should call :func:`correct` instead of importing backend stages.
The function keeps the JSONL contract and hides model-specific probing,
intervention, and temporary runtime artifacts behind one call.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def correct(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    model: str = "videollama2",
    model_path: str | os.PathLike[str] | None = None,
    dtype: str = "bfloat16",
    device: str = "auto",
    max_rows: int = 0,
) -> list[dict[str, Any]]:
    """Run the cross-modal hallucination correction use case.

    Args:
        input_path: UTF-8 JSONL requests with ``sample_id``, ``question`` and
            at least one media path.
        output_path: Optional UTF-8 JSONL destination. If omitted, results are
            returned in memory without creating a caller-visible output file.
        model: ``videollama2`` or ``qwen`` backend.
        model_path: Optional local checkpoint directory or Hugging Face id.
        dtype: ``bfloat16`` or ``float16``.
        device: Torch device for the Qwen backend.
        max_rows: Process at most this many rows; zero means all rows.

    Returns:
        One final result dictionary per processed input, in input order.
    """
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {source}")
    temporary_destination: Path | None = None
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
    else:
        handle = tempfile.NamedTemporaryFile(prefix="owp_api_", suffix=".jsonl", delete=False)
        handle.close()
        temporary_destination = Path(handle.name)
        destination = temporary_destination
    command = [
        sys.executable,
        str(ROOT / "src" / "owp_infer.py"),
        "--input",
        str(source),
        "--output",
        str(destination),
        "--model",
        str(model),
        "--dtype",
        str(dtype),
        "--device",
        str(device),
        "--max-rows",
        str(max_rows),
    ]
    if model_path is not None:
        command.extend(["--model-path", str(model_path)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), environment.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    try:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        with destination.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    finally:
        if temporary_destination is not None:
            destination.unlink(missing_ok=True)


__all__ = ["correct"]
