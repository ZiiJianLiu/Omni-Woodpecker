#!/usr/bin/env python3
"""Create runtime JSONL manifests from the bundled AVHBench or CMM data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build an OWP evaluation manifest.")
    parser.add_argument("--benchmark", choices=("avhbench", "cmm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    rows = []
    if args.benchmark == "avhbench":
        source = json.loads((root / "data/AVHBench/QA.json").read_text(encoding="utf-8"))
        allowed = {
            "Audio-driven Video Hallucination",
            "Video-driven Audio Hallucination",
            "AV Matching",
        }
        for index, item in enumerate(source):
            if item.get("task") not in allowed:
                continue
            video_id = str(item["video_id"])
            rows.append(
                {
                    "sample_id": f"avhbench:{index:05d}:{video_id}",
                    "benchmark": "avhbench",
                    "mad_protocol_task": item["task"],
                    "question": item["text"],
                    "reference_answer": item["label"],
                    "video_id": video_id,
                    "video_path": f"data/AVHBench/videos/{video_id}.mp4",
                    "audio_path": f"data/AVHBench/audios/{video_id}.wav",
                    "_efficiency_order": len(rows),
                }
            )
    else:
        source = json.loads((root / "data/CMM/all_data_final_reorg.json").read_text(encoding="utf-8"))
        for index, item in enumerate(source):
            video_value = item.get("video_path")
            video = str(video_value).lstrip("./") if video_value else ""
            audio_value = item.get("audio_path")
            audio = f"data/CMM/{str(audio_value).lstrip('./')}" if audio_value else ""
            rows.append(
                {
                    "sample_id": f"cmm:{index:05d}",
                    "benchmark": "cmm",
                    "question": item["question"],
                    "reference_answer": item.get("answer"),
                    "video_path": f"data/CMM/{video}" if video else "",
                    "audio_path": audio,
                    "category": item.get("category", ""),
                    "sub_category": item.get("sub_category", ""),
                    "modality": item.get("modality", ""),
                    "granularity": item.get("granularity", ""),
                    "correlation_type": item.get("correlation_type", ""),
                    "_efficiency_order": len(rows),
                }
            )

    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
