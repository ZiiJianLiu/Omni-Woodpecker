import json
import argparse
import json
from collections import defaultdict
from pathlib import Path

from score import extract_answer, normalize_answer


def compute_accuracy(entries):
    """Return (correct, total) using score.py normalization."""
    correct = 0
    total = len(entries)

    for item in entries:
        pred = normalize_answer(extract_answer(item["pred"]))
        gt = normalize_answer(item["answer"])
        if pred == gt:
            correct += 1

    return correct, total


def load_results(path: Path):
    """Load newline-delimited JSON results."""
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Compute overall and per-subcategory accuracy for CMM results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/evaluation_cmm_v.json"),
        help="Path to newline-delimited JSON results.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/scores_cmm.json"),
        help="Path to save aggregated accuracy JSON.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Result file not found: {args.input}")

    entries = load_results(args.input)
    total_correct, total_samples = compute_accuracy(entries)

    by_subcategory = defaultdict(list)
    for item in entries:
        subcat = item.get("sub_category", "UNKNOWN")
        by_subcategory[subcat].append(item)

    print(f"Input file: {args.input}")
    print(f"Total samples: {total_samples}")
    print(f"Total correct: {total_correct}")
    overall_acc = total_correct / total_samples if total_samples else 0.0
    print(f"Overall accuracy: {overall_acc * 100:.2f}%\n")

    print("Per-subcategory accuracy:")
    per_subcat_stats = {}
    for subcat, items in sorted(by_subcategory.items()):
        correct, total = compute_accuracy(items)
        acc = correct / total if total else 0.0
        per_subcat_stats[subcat] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }
        print(f"  {subcat}: {acc * 100:.2f}% ({correct}/{total})")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_file": str(args.input),
        "overall": {
            "accuracy": overall_acc,
            "correct": total_correct,
            "total": total_samples,
        },
        "per_subcategory": per_subcat_stats,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved scores to {output_path}")


if __name__ == "__main__":
    main()
