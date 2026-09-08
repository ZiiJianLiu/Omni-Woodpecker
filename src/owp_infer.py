#!/usr/bin/env python3
"""Single business entry point for Omni-Woodpecker inference.

Input and output are JSONL.  The command keeps model-specific typing and
representation editing inside the selected backend and exposes only the final
answer record to callers.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_OUTPUT_FIELDS = ("sample_id", "answer", "status", "error")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from owp.assets import (
    build_dataset_manifest,
    cache_root,
    path_from_dataset,
    prepare_dataset,
)  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "owp_default.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def run_command(command: list[str], *, env: Mapping[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=dict(env), check=True)


def configured_cli_args(section: str) -> list[str]:
    """Convert one release-config section into deterministic CLI arguments."""
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    values = config.get(section)
    if not isinstance(values, Mapping):
        raise RuntimeError(f"Missing configuration section {section!r} in {DEFAULT_CONFIG}")
    arguments: list[str] = []
    for key, value in values.items():
        option = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(option)
            continue
        if isinstance(value, list):
            arguments.append(option)
            # Some argparse options use a comma-separated scalar (for example
            # the self-logit capture layers), whereas layer and coefficient
            # sweeps use nargs='+'. Keep the release config declarative while
            # emitting the syntax expected by each internal parser.
            if key == "evidence_validation_self_logit_layers":
                arguments.append(",".join(str(item) for item in value))
            else:
                arguments.extend(str(item) for item in value)
            continue
        if value is not None:
            arguments.extend([option, str(value)])
    return arguments


def materialize_input(path: Path, *, max_rows: int = 0) -> Path:
    """Normalize repository-relative paths and resolve supported cached datasets."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Input JSONL is empty: {path}")
    sample_ids: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("Every input row must contain a non-empty sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id in input: {sample_id!r}")
        sample_ids.add(sample_id)
        if not str(row.get("question") or "").strip():
            raise ValueError(f"Input row sample_id={sample_id!r} has no question")
        if not row.get("video_path") and not row.get("audio_path"):
            raise ValueError(
                f"Input row sample_id={row.get('sample_id')!r} must contain video_path or audio_path"
            )
    if max_rows < 0:
        raise ValueError("--max-rows must be zero or a positive integer")
    if max_rows > 0:
        rows = rows[:max_rows]
    missing = [
        (row.get("sample_id"), key, value)
        for row in rows
        for key in ("video_path", "audio_path")
        for value in (row.get(key),)
        if value
        and not (
            Path(str(value)).is_file()
            if Path(str(value)).is_absolute()
            else (ROOT / str(value)).is_file()
        )
    ]
    if missing:
        if any(Path(str(value)).is_absolute() for _, _, value in missing):
            sample_id, key, value = next(
                item for item in missing if Path(str(item[2])).is_absolute()
            )
            raise FileNotFoundError(
                f"Media file for sample_id={sample_id!r} is not available: {key}={value!r}"
            )
        benchmarks = {str(row.get("benchmark") or "").strip().lower() for row in rows}
        if benchmarks != {"cmm"}:
            sample_id, key, value = missing[0]
            raise FileNotFoundError(
                f"Media file for sample_id={sample_id!r} is not available: {key}={value!r}. "
                "Provide a local path or use benchmark=cmm for automatic dataset resolution."
            )
        layout = prepare_dataset("cmm")
        manifest_path = cache_root() / "manifests" / "cmm_runtime.jsonl"
        build_dataset_manifest(layout, manifest_path)
        lookup = {
            str(row.get("sample_id")): row
            for row in read_jsonl(manifest_path)
            if row.get("sample_id")
        }
        resolved: list[dict[str, Any]] = []
        for row in rows:
            cached = lookup.get(str(row.get("sample_id")))
            merged = dict(row)
            for key in ("video_path", "audio_path"):
                value = row.get(key)
                if value:
                    candidates = [str(value)]
                    prefix = "data/CMM/"
                    if str(value).startswith(prefix):
                        candidates.insert(0, str(value)[len(prefix) :])
                    resolved_value = next(
                        (
                            path_from_dataset(layout.root, candidate)
                            for candidate in candidates
                            if Path(path_from_dataset(layout.root, candidate)).is_file()
                        ),
                        "",
                    )
                    if resolved_value:
                        merged[key] = resolved_value
                if not Path(str(merged.get(key) or "")).is_file() and cached and cached.get(key):
                    merged[key] = cached[key]
            if not any(
                Path(str(merged.get(key) or "")).is_file()
                for key in ("video_path", "audio_path")
            ):
                raise FileNotFoundError(
                    f"Could not resolve CMM media for sample_id={row.get('sample_id')!r}"
                )
            resolved.append(merged)
        rows = resolved
    else:
        for row in rows:
            for key in ("video_path", "audio_path"):
                value = row.get(key)
                if value and not Path(str(value)).is_absolute():
                    row[key] = str((ROOT / str(value)).resolve())

    temporary = tempfile.NamedTemporaryFile(
        prefix="owp_input_", suffix=".jsonl", delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    write_jsonl(temporary_path, rows)
    return temporary_path


def build_qwen_manifest(input_path: Path, typing_path: Path, output_path: Path) -> None:
    """Join public requests with runtime typing without exposing typing as input."""
    source = {
        str(row.get("sample_id")): dict(row)
        for row in read_jsonl(input_path)
        if row.get("sample_id")
    }
    merged: list[dict[str, Any]] = []
    for typed in read_jsonl(typing_path):
        sample_id = str(typed.get("sample_id") or "")
        if not sample_id:
            continue
        row = dict(source.get(sample_id) or {})
        for field in (
            "sample_id",
            "question",
            "video_path",
            "audio_path",
            "benchmark",
            "target_modality",
            "branches",
            "probe_full_answer",
            "current_answer_y0",
            "candidate_answer_y",
            "evidence_certificate",
            "current_best_accepts",
        ):
            if field in typed and typed.get(field) not in (None, ""):
                row[field] = typed[field]
        merged.append(row)
    if not merged:
        raise RuntimeError(f"Qwen typing produced no usable records: {typing_path}")
    write_jsonl(output_path, merged)


def normalize_qwen_rows(edit_path: Path, typing_path: Path) -> list[dict[str, Any]]:
    records = read_jsonl(edit_path)
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        if not sample_id:
            continue
        # The internal executor can evaluate a coefficient grid. The public
        # wrapper requests one configuration, but keep the first row per input
        # as a guard against accidental duplicate internal records.
        selected.setdefault(
            sample_id,
            {
                "sample_id": record.get("sample_id"),
                "answer": record.get("edited_prediction"),
                "baseline_answer": record.get("baseline_prediction"),
                "target_modality": record.get("target_modality"),
                "intervention_applied": bool(record.get("edit_applied")),
                "status": "error"
                if record.get("error") or record.get("skipped")
                else "ok",
                "error": str(record.get("error") or record.get("skipped_reason") or "") or None,
            },
        )
    rows: list[dict[str, Any]] = []
    for typed in read_jsonl(typing_path):
        sample_id = str(typed.get("sample_id") or "")
        edited = selected.get(sample_id)
        if edited is not None:
            rows.append(edited)
            continue
        branches = typed.get("branches") if isinstance(typed.get("branches"), Mapping) else {}
        full = branches.get("full") if isinstance(branches.get("full"), Mapping) else {}
        baseline = typed.get("probe_full_answer") or full.get("answer")
        failed = bool(typed.get("error") or typed.get("skipped") or not baseline)
        rows.append(
            {
                "sample_id": typed.get("sample_id"),
                "answer": baseline,
                "baseline_answer": baseline,
                "target_modality": typed.get("target_modality") or "unknown",
                "intervention_applied": False,
                "status": "error" if failed else "ok",
                "error": (
                    str(typed.get("error") or typed.get("skipped_reason") or "") or None
                    if failed
                    else None
                ),
            }
        )
    return rows


def normalize_videollama_rows(path: Path) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "sample_id": record.get("sample_id"),
                "answer": record.get("intervened_prediction"),
                "baseline_answer": record.get("baseline_prediction"),
                "target_modality": record.get("target_modality"),
                "intervention_applied": bool(record.get("intervention_applied")),
                "status": "error"
                if record.get("error") or record.get("skipped")
                else "ok",
                "error": str(record.get("error") or record.get("skipped_reason") or "") or None,
            }
        )
    return rows


def order_and_validate_outputs(
    input_path: Path,
    output_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enforce the public one-input/one-output contract and input ordering."""
    inputs = read_jsonl(input_path)
    expected_ids = [str(row["sample_id"]) for row in inputs]
    by_id: dict[str, dict[str, Any]] = {}
    for row in output_rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise RuntimeError("Backend returned an output without sample_id")
        if sample_id in by_id:
            raise RuntimeError(f"Backend returned duplicate output for sample_id={sample_id!r}")
        by_id[sample_id] = row
    missing = [sample_id for sample_id in expected_ids if sample_id not in by_id]
    extra = sorted(set(by_id) - set(expected_ids))
    if missing or extra:
        raise RuntimeError(
            "Backend output does not match the processed input IDs: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    ordered: list[dict[str, Any]] = []
    for sample_id in expected_ids:
        row = by_id[sample_id]
        if row.get("status") not in {"ok", "error"}:
            raise RuntimeError(f"Invalid status for sample_id={sample_id!r}: {row.get('status')!r}")
        if row.get("status") == "ok" and row.get("answer") not in {"Yes", "No"}:
            raise RuntimeError(
                f"Backend returned a non-binary answer for sample_id={sample_id!r}: "
                f"{row.get('answer')!r}"
            )
        if row.get("status") == "ok":
            row["error"] = None
        elif not str(row.get("error") or "").strip():
            row["error"] = "Backend failed without an error message"
        ordered.append({field: row.get(field) for field in PUBLIC_OUTPUT_FIELDS})
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete Omni-Woodpecker hallucination-correction use case."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with question and media paths.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL with corrected answers.")
    parser.add_argument(
        "--model",
        choices=("videollama2", "qwen"),
        default="videollama2",
        help="Inference backend. Both backends implement the same use-case contract.",
    )
    parser.add_argument("--model-path", default=None, help="Local checkpoint path or Hugging Face model id.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="auto", help="Torch device for the Qwen backend.")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {args.input}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    runtime_input = materialize_input(args.input, max_rows=args.max_rows)
    atexit.register(runtime_input.unlink, missing_ok=True)

    if args.model == "videollama2":
        with tempfile.NamedTemporaryFile(prefix="owp_raw_", suffix=".jsonl", delete=False) as handle:
            raw_output = Path(handle.name)
        try:
            command = [
                sys.executable,
                str(ROOT / "src" / "run_owp.py"),
                "--input",
                str(runtime_input),
                "--output",
                str(raw_output),
                "--repeat",
                "1",
                "--dtype",
                args.dtype,
            ]
            if args.model_path:
                command.extend(["--model-path", str(args.model_path)])
            run_command(command, env=env)
            output_rows = normalize_videollama_rows(raw_output)
        finally:
            raw_output.unlink(missing_ok=True)
    else:
        with tempfile.TemporaryDirectory(prefix="owp_qwen_") as temporary:
            workspace = Path(temporary)
            typing_dir = workspace / "typing"
            intervention_dir = workspace / "intervention"
            typing_command = [
                sys.executable,
                str(ROOT / "src" / "probe_qwen.py"),
                "--records",
                str(runtime_input),
                "--output-dir",
                str(typing_dir),
                "--device",
                str(args.device),
                "--dtype",
                args.dtype,
                "--skip-on-error",
                "--no-progress",
            ]
            if args.model_path:
                typing_command.extend(["--model-path", str(args.model_path)])
            run_command(typing_command, env=env)
            records_path = typing_dir / "records.jsonl"
            manifest_path = workspace / "manifest.jsonl"
            build_qwen_manifest(runtime_input, records_path, manifest_path)
            intervention_command = [
                sys.executable,
                str(ROOT / "src" / "intervention.py"),
                "--manifest",
                str(manifest_path),
                "--runtime-rows",
                str(records_path),
                "--output-dir",
                str(intervention_dir),
                "--device",
                str(args.device),
                "--dtype",
                args.dtype,
                "--no-progress",
            ]
            intervention_command.extend(configured_cli_args("qwen_hard"))
            if args.model_path:
                intervention_command.extend(["--model-path", str(args.model_path)])
            intervention_command.extend(["--max-rows", "0"])
            run_command(intervention_command, env=env)
            output_rows = normalize_qwen_rows(
                intervention_dir / "edit_records.jsonl",
                records_path,
            )

    output_rows = order_and_validate_outputs(runtime_input, output_rows)
    runtime_input.unlink(missing_ok=True)
    write_jsonl(args.output, output_rows)


if __name__ == "__main__":
    main()
