#!/usr/bin/env python3
"""Fetch OWP models/datasets on first use and build a runtime manifest.

Large binary assets are stored in the Hugging Face cache, not in the GitHub
checkout.  The command is idempotent: a second invocation reuses completed
snapshots without downloading them again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow ``python tools/prepare_assets.py`` from the checkout root without an
# editable package installation.
RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from OMhallucination.asset_manager import (
    build_dataset_manifest,
    cache_root,
    model_repo,
    prepare_dataset,
    resolve_model_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare OWP assets from local files or Hugging Face.")
    parser.add_argument("--model", choices=("qwen", "videollama2", "siglip", "all"), default="all")
    parser.add_argument("--dataset", choices=("cmm", "avhbench", "all", "none"), default="none")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to the asset cache for a selected dataset.",
    )
    args = parser.parse_args()

    models = ("qwen", "videollama2", "siglip") if args.model == "all" else (args.model,)
    resolved_models: dict[str, str] = {}
    for kind in models:
        path = resolve_model_path(model_repo(kind), kind=kind)
        resolved_models[kind] = path
        print(f"{kind}: {path}")

    datasets = () if args.dataset == "none" else (("cmm", "avhbench") if args.dataset == "all" else (args.dataset,))
    manifests: dict[str, str] = {}
    for kind in datasets:
        layout = prepare_dataset(kind)
        output = args.manifest if args.manifest and len(datasets) == 1 else (
            cache_root() / "manifests" / f"{kind}.jsonl"
        )
        manifest = build_dataset_manifest(layout, output, max_rows=args.max_rows)
        manifests[kind] = str(manifest)
        print(f"{kind}: {layout.root} ({layout.annotation.name})")
        print(f"manifest: {manifest}")

    summary = {"models": resolved_models, "datasets": manifests, "cache": str(cache_root())}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
