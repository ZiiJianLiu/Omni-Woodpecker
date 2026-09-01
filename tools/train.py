#!/usr/bin/env python3
"""Record the OWP training-free protocol.

OWP does not update model parameters.  This command intentionally validates
the release configuration and writes a machine-readable status record so that
training-oriented pipelines can call a stable entry point without silently
fine-tuning a backbone.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate the training-free OWP setup.")
    parser.add_argument("--config", type=Path, default=root / "configs" / "owp_default.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "training_status.json",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("method") != "Omni-Woodpecker":
        raise ValueError("The supplied config is not an OWP configuration.")
    if config.get("training_required") is not False:
        raise ValueError("OWP is training-free; training_required must be false.")
    payload = {
        "method": "Omni-Woodpecker",
        "training_required": False,
        "parameters_updated": False,
        "config": str(args.config.relative_to(root) if args.config.is_relative_to(root) else args.config),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "message": "No optimization was run. OWP is applied at inference time to a frozen backbone.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
