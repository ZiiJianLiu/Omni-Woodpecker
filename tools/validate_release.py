#!/usr/bin/env python3
"""Validate the source-only OWP release or an optional local asset bundle."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


REQUIRED_FILES = (
    ".gitattributes",
    "README.md",
    "requirements.txt",
    "run_smoke.sh",
    "configs/owp_default.json",
    "data/AVHBench/QA.json",
    "data/CMM/all_data_final_reorg.json",
    "owp/assets.py",
    "tools/prepare_assets.py",
    "src/run_owp.py",
    "src/intervention.py",
    "src/probe_qwen.py",
    "src/probe_videollama2.py",
    "src/intervention_videollama2.py",
    "owp/models/qwen_omni.py",
    "owp/modules/question_conditioned_evidence.py",
    "third_party/VideoLLaMA2/videollama2/__init__.py",
    "third_party/HulluEdit/hulluedit/steer.py",
)

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"[FAIL] {message}")


def is_unresolved_lfs_pointer(path: Path) -> bool:
    """Return true when checkout contains a Git LFS pointer, not its payload."""
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(LFS_POINTER_PREFIX)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check OWP source files and optional local assets.")
    parser.add_argument("--skip-imports", action="store_true", help="Do not import lightweight Python modules.")
    parser.add_argument(
        "--require-bundled-assets",
        action="store_true",
        help="Require the development checkout's local model shards and benchmark media.",
    )
    args = parser.parse_args()
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            fail(f"missing required file: {relative}", errors)

    for subtree in (root / "data", root / "models"):
        if subtree.exists():
            links = [str(path.relative_to(root)) for path in subtree.rglob("*") if path.is_symlink()]
            if links:
                fail(f"symbolic links are not allowed in {subtree.name}: {links[:3]}", errors)

    try:
        avh = load_json(root / "data/AVHBench/QA.json")
        cmm = load_json(root / "data/CMM/all_data_final_reorg.json")
        if not isinstance(avh, list) or len(avh) != 6408:
            fail(f"AVHBench/QA.json expected 6408 rows, found {len(avh)}", errors)
        if not isinstance(cmm, list) or len(cmm) != 2400:
            fail(f"CMM/all_data_final_reorg.json expected 2400 rows, found {len(cmm)}", errors)
        if args.require_bundled_assets:
            missing = []
            for row in avh:
                video = root / "data" / "AVHBench" / "videos" / f"{row['video_id']}.mp4"
                audio = root / "data" / "AVHBench" / "audios" / f"{row['video_id']}.wav"
                if (
                    not video.is_file()
                    or not audio.is_file()
                    or is_unresolved_lfs_pointer(video)
                    or is_unresolved_lfs_pointer(audio)
                ):
                    missing.append(str(row.get("video_id")))
            if missing:
                fail(f"AVHBench media missing for {len(missing)} rows", errors)
            missing = []
            cmm_root = root / "data" / "CMM"
            for row in cmm:
                video_value = row.get("video_path")
                if video_value:
                    video = cmm_root / str(video_value).lstrip("./")
                    if not video.is_file() or is_unresolved_lfs_pointer(video):
                        missing.append(str(video_value))
                audio_value = row.get("audio_path")
                if audio_value:
                    audio = cmm_root / str(audio_value).lstrip("./")
                    if not audio.is_file() or is_unresolved_lfs_pointer(audio):
                        missing.append(str(audio_value))
            if missing:
                fail(f"CMM media missing for {len(missing)} paths", errors)
        else:
            print("[INFO] Local media/model payloads are optional; runtime HF resolution is enabled.")
    except Exception as exc:
        fail(f"dataset validation failed: {type(exc).__name__}: {exc}", errors)

    model_indexes = (
        "models/Qwen2.5-Omni-7B/model.safetensors.index.json",
        "models/VideoLLaMA2.1-7B-AV/model.safetensors.index.json",
    )
    for relative in model_indexes:
        if not (root / relative).is_file():
            if args.require_bundled_assets:
                fail(f"missing required local model index: {relative}", errors)
            continue
        try:
            index = load_json(root / relative)
            parent = (root / relative).parent
            shards = sorted(set(index["weight_map"].values()))
            missing = [shard for shard in shards if not (parent / shard).is_file()]
            if missing:
                fail(f"missing model shards for {relative}: {missing}", errors)
            unresolved = [
                shard for shard in shards if is_unresolved_lfs_pointer(parent / shard)
            ]
            if unresolved:
                fail(
                    f"unresolved Git LFS model shards for {relative}: {unresolved}; "
                    "run `git lfs pull`",
                    errors,
                )
        except Exception as exc:
            fail(f"model index validation failed for {relative}: {exc}", errors)

    siglip_weights = root / "models/siglip-so400m-patch14-384/model.safetensors"
    if args.require_bundled_assets and not siglip_weights.is_file():
        fail("missing local SigLIP weights", errors)
    if is_unresolved_lfs_pointer(siglip_weights):
        fail("unresolved local SigLIP weights", errors)

    if not args.skip_imports:
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "src"))
        for module_name in (
            "owp.assets",
            "owp.evaluation.answer_policy",
            "owp.evaluation.datasets",
            "owp.modules.question_conditioned_evidence",
        ):
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                fail(f"import failed for {module_name}: {type(exc).__name__}: {exc}", errors)

    if errors:
        print(f"Release validation failed with {len(errors)} error(s).")
        return 1
    mode = "bundled assets" if args.require_bundled_assets else "source-only with runtime HF assets"
    print(f"Release validation passed: {mode}, annotations, and imports are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
