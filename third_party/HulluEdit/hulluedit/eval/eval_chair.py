#!/usr/bin/env python3
"""
CHAIR evaluation script
Uses DeCo's chair.py to compute CHAIR metrics (CHAIRs, CHAIRi), and outputs Recall and Len.

Dependencies:
- A local copy of DeCo's ``chair.py`` (set ``DECO_ROOT`` to its directory)
- NLTK (tokenization, POS tagging, WordNet lemmatization); set ``NLTK_DATA`` when needed
"""
import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

DECO_ROOT = os.environ.get(
    "DECO_ROOT",
    str(Path(__file__).resolve().parents[3] / "third_party" / "DeCo"),
)
if DECO_ROOT not in sys.path:
    sys.path.insert(0, DECO_ROOT)

try:
    from chair import CHAIR  # type: ignore
except Exception as e:
    print(f"[ERROR] Failed to import DeCo's chair.py: {e}")
    sys.exit(1)


def read_jsonl(file_path: str):
    """Read JSONL file"""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def parse_coco_int_id(image_id) -> int:
    """
    Extract COCO image_id (integer) from string
    
    Compatible formats:
    - "COCO_val2014_000000395849"
    - "000000395849"
    - "395849"
    - 395849 (int)
    """
    if isinstance(image_id, int):
        return image_id
    
    s = str(image_id)
    m = re.search(r"(\d{6,12})$", s)
    if m:
        return int(m.group(1))
    
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    
    raise ValueError(f"Failed to parse COCO image_id: {image_id}")


def compute_chair_metrics(jsonl_file: str, coco_annotations_dir: str, cache_file: str = "") -> Dict[str, Any]:
    """
    Compute metrics using DeCo's CHAIR implementation.

    Args:
        jsonl_file: Prediction JSONL file (each line contains image_id, caption)
        coco_annotations_dir: COCO annotations directory (must contain captions_*2014.json and instances_*2014.json)
        cache_file: Optional, CHAIR evaluator cache (pickle) path

    Returns:
        dict: Output consistent with DeCo/chair.py (sentences and overall_metrics)
    """
    import pickle

    if cache_file and os.path.exists(cache_file):
        try:
            evaluator = pickle.load(open(cache_file, 'rb'))
            print(f"loaded evaluator from cache: {cache_file}")
        except Exception as e:
            print(f"Cache loading failed (will rebuild): {e}")
            evaluator = CHAIR(coco_annotations_dir)
            if cache_file:
                pickle.dump(evaluator, open(cache_file, 'wb'))
                print(f"cached evaluator to: {cache_file}")
    else:
        print("cache not set or not exist yet, building from scratch...")
        evaluator = CHAIR(coco_annotations_dir)
        if cache_file:
            pickle.dump(evaluator, open(cache_file, 'wb'))
            print(f"cached evaluator to: {cache_file}")

    cap_dict = evaluator.compute_chair(jsonl_file, "image_id", "caption")
    return cap_dict




def parse_args():
    parser = argparse.ArgumentParser(description="CHAIR Evaluation")
    parser.add_argument("--input", type=str, required=True, 
                       help="Generated caption JSONL file")
    parser.add_argument("--coco-annotations", type=str, 
                       default=None,
                       help="COCO annotations directory (contains captions/instances *2014.json)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file (auto-generated if not specified)")
    parser.add_argument("--cache", type=str, default=None,
                       help="CHAIR evaluator cache (pickle) path; leave empty to disable caching")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file does not exist: {args.input}")
        sys.exit(1)
    
    print("=" * 80)
    print("CHAIR Evaluation")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"COCO Annotations: {args.coco_annotations}")
    print("=" * 80)
    
    chair_result = compute_chair_metrics(
        jsonl_file=args.input,
        coco_annotations_dir=args.coco_annotations,
        cache_file=args.cache if args.cache else "",
    )
    
    halc_caption_result = chair_result.get("sentences", [])
    halc_result: Dict[int, Dict[str, Any]] = {}
    for item in halc_caption_result:
        img_id = item["image_id"]
        metrics = item.get("metrics", {})
        halc_result[img_id] = {
            "caption": item.get("caption", ""),
            "chairs": metrics.get("CHAIRs", 0),
            "chairi": metrics.get("CHAIRi", 0.0),
            "recall": metrics.get("Recall", 0.0),
            "objects_num": len(item.get("mscoco_generated_words", [])),
            "words_num": len(item.get("words", [])),
            "hallucinate_num": len(item.get("hallucination_idxs", [])),
        }
    
    overall = chair_result.get("overall_metrics", {})
    chairs_avg = float(overall.get("CHAIRs", 0.0))
    chairi_avg = float(overall.get("CHAIRi", 0.0))
    recall_avg = float(overall.get("Recall", 0.0))
    len_avg = float(overall.get("Len", 0.0))
    n_samples = len(halc_result)
    
    print("\n" + "=" * 80)
    print("CHAIR Evaluation Results")
    print("=" * 80)
    print(f"Samples:          {n_samples}")
    print(f"CHAIRs (avg):     {chairs_avg:.4f}")
    print(f"CHAIRi (avg):     {chairi_avg:.4f}")
    print(f"Recall (avg):     {recall_avg:.4f}")
    print(f"Len (avg):        {len_avg:.4f}")
    print("=" * 80)
    
    if args.verbose:
        print("\nPer-image detailed results:")
        for img_id, res in sorted(halc_result.items()):
            print(f"\nImage {img_id}:")
            print(f"  Caption: {res['caption']}")
            print(f"  CHAIRs: {res['chairs']:.4f}, CHAIRi: {res['chairi']:.4f}, Recall: {res['recall']:.4f}")
            print(f"  Objects: {res['objects_num']}, Hallucinations: {res['hallucinate_num']}")
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        output_file = args.input.replace(".jsonl", "_chair_result.json")
    
    result = {
        "input_file": args.input,
        "num_samples": n_samples,
        "overall_metrics": overall,
        "per_image_results": halc_result,
        "full_chair_output": chair_result
    }
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
