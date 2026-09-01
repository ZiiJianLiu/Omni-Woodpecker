"""
COCO Caption Generation Evaluation (for CHAIR computation)
"""
import argparse
import json
import os
import sys
import random
from pathlib import Path
from glob import glob
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hulluedit.engines.llava7b import LLaVAHullueditEngine, EngineConfig
from hulluedit.steer import HullueditConfig


def main():
    parser = argparse.ArgumentParser(description="COCO Caption Generation (Hulluedit)")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples (for quick testing)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Initialize engine
    hulluedit_cfg = HullueditConfig(
        rank_evidence=cfg.get("rank_evidence", 6),
        rank_prior=cfg.get("rank_prior", 4),
        kappa=cfg.get("kappa", 0.6),
        lambda_prior=cfg.get("lambda_prior", 0.3),
        eps=cfg.get("eps", 1e-6)
    )
    
    eng_cfg = EngineConfig(
        model_name=cfg["model_name"],
        anchor_layer=cfg.get("anchor_layer", 26),
        max_new_tokens=cfg.get("max_new_tokens", 128),
        top_p=cfg.get("top_p", 0.9),
        temperature=cfg.get("temperature", 0.2),
        precision=cfg.get("precision", "bf16")
    )
    
    print("[Caption] Initializing Hulluedit engine...")
    engine = LLaVAHullueditEngine(eng_cfg, hulluedit_cfg)
    
    # Load image list
    coco_img_dir = Path(cfg["coco_images"])
    image_files = sorted(glob(str(coco_img_dir / "*.jpg")))
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {coco_img_dir}")
    
    # Random sampling
    random.shuffle(image_files)
    if args.max_samples:
        image_files = image_files[:args.max_samples]
    
    print(f"[Caption] Number of images: {len(image_files)}")
    
    # Generate caption
    results = []
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
    
    for img_path in tqdm(image_files, desc="Generating Captions"):
        image_id = Path(img_path).stem  # COCO_val2014_000000123456
        
        try:
            output = engine.generate(prompt, img_path)
            
            results.append({
                "image_id": image_id,
                "image_path": str(img_path),
                "caption": output["text"],
                "certs": output["certs"]
            })
            
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
            continue
    
    # Save results
    output_data = {
        "config": args.config,
        "num_samples": len(results),
        "results": results
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[Caption] Results saved: {args.output}")
    print(f"  Samples: {len(results)}")


if __name__ == "__main__":
    main()
