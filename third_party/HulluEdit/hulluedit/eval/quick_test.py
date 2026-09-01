"""
Quick test script (few-shot validation)
Used to verify if Hulluedit system is working correctly
"""
import argparse
import json
import sys
from pathlib import Path
from glob import glob
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hulluedit.engines.llava7b import LLaVAHullueditEngine, EngineConfig
from hulluedit.steer import HullueditConfig


def main():
    parser = argparse.ArgumentParser(description="Hulluedit Quick Test")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--num-images", type=int, default=10, help="Number of test images")
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Initialize engine
    print("[Quick Test] Initializing Hulluedit engine...")
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
        max_new_tokens=50,  # Shorter length for quick test
        top_p=cfg.get("top_p", 0.9),
        temperature=cfg.get("temperature", 0.2),
        precision=cfg.get("precision", "bf16")
    )
    
    engine = LLaVAHullueditEngine(eng_cfg, hulluedit_cfg)
    
    # Load test images
    coco_img_dir = Path(cfg["coco_images"])
    image_files = sorted(glob(str(coco_img_dir / "*.jpg")))[:args.num_images]
    
    if not image_files:
        print(f"[Error] No images found: {coco_img_dir}")
        return
    
    print(f"[Quick Test] Number of test images: {len(image_files)}")
    
    # Test generation
    test_prompts = [
        "USER: <image>\nWhat objects do you see in this image?\nASSISTANT:",
        "USER: <image>\nDescribe this image.\nASSISTANT:",
        "USER: <image>\nIs there a person in this image?\nASSISTANT:",
    ]
    
    for i, img_path in enumerate(image_files, 1):
        prompt = test_prompts[i % len(test_prompts)]
        
        try:
            output = engine.generate(prompt, img_path, max_new_tokens=30)
            
        except Exception as e:
            print(f"[Error] {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("[Quick Test] Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
