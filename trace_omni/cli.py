"""
CLI entrypoint for TRACe-Omni.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..config import PipelineConfig
from .config import TraceOmniConfig
from .pipeline import TraceOmniPipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run TRACe-Omni on a single video-question pair.')
    parser.add_argument('--video_path', default=None, help='Path to the input video.')
    parser.add_argument('--audio_path', default=None, help='Optional external audio path.')
    parser.add_argument('--question', required=True, help='Question to answer.')
    parser.add_argument('--max_new_tokens', type=int, default=None, help='Override generation max_new_tokens.')
    parser.add_argument('--device', default=None, help='Override device, e.g. cuda:0 or auto.')
    parser.add_argument(
        '--probe_mode',
        choices=['heuristic', 'branch_contrastive'],
        default='heuristic',
        help='Dependency probe mode.',
    )
    parser.add_argument('--output', default=None, help='Optional JSON output path.')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON output.')
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not args.video_path and not args.audio_path:
        raise SystemExit('At least one of --video_path or --audio_path must be provided.')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    )

    base_config = PipelineConfig()
    config = TraceOmniConfig.from_pipeline_config(base_config)
    if args.device:
        config.device = args.device
    config.probe.mode = args.probe_mode
    pipeline = TraceOmniPipeline(config=config)
    result = pipeline.run_media(
        args.question,
        video_path=args.video_path,
        audio_path=args.audio_path,
        max_new_tokens=args.max_new_tokens,
    )
    payload = result.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        Path(args.output).write_text(text + '\n', encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
