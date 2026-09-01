"""
Aggregate POPE metrics across splits (random, popular, adversarial).

Reads the JSON outputs produced by hulluedit.eval.pope_eval and computes
micro-averaged Accuracy/Precision/Recall/F1 over all samples combined.
Optionally prints per-split metrics and writes a summary JSON.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def _safe_read(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _compute_metrics_from_results(results: List[Dict[str, str]]) -> Dict[str, float]:
    tp = sum(1 for r in results if r.get("prediction") == "yes" and r.get("label") == "yes")
    fp = sum(1 for r in results if r.get("prediction") == "yes" and r.get("label") == "no")
    tn = sum(1 for r in results if r.get("prediction") == "no" and r.get("label") == "no")
    fn = sum(1 for r in results if r.get("prediction") == "no" and r.get("label") == "yes")

    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate POPE metrics across splits")
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "outputs/pope_random.json",
            "outputs/pope_popular.json",
            "outputs/pope_adversarial.json",
        ],
        help="List of POPE result JSON files to aggregate",
    )
    parser.add_argument("--output", type=str, default="outputs/pope_all_metrics.json", help="Summary JSON output path")
    args = parser.parse_args()

    # Collect results from each file
    combined_results: List[Dict[str, str]] = []
    per_split_metrics = {}

    for file_path in args.files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Result file not found: {file_path}")
        data = _safe_read(file_path)
        results = data.get("results", [])
        combined_results.extend(results)
        per_split_metrics[os.path.basename(file_path)] = _compute_metrics_from_results(results)

    overall_metrics = _compute_metrics_from_results(combined_results)

    summary = {
        "files": args.files,
        "num_samples": len(combined_results),
        "overall": overall_metrics,
        "per_split": per_split_metrics,
    }

    # Write summary JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    # Print concise report
    print("\n[POPE Aggregated Results]")
    print(f"  Samples: {summary['num_samples']}")
    print(
        "  Overall  Acc={accuracy:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}".format(
            **overall_metrics
        )
    )
    for name, m in per_split_metrics.items():
        print(
            f"  {name:<24} Acc={m['accuracy']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}"
        )


if __name__ == "__main__":
    main()


