#!/usr/bin/env python3
"""
Generate captions for CHAIR evaluation (using Hulluedit engine)
Reference Nullu's CHAIR evaluation process for consistency

Output format: JSONL file, one JSON per line
{"image_id": 123456, "caption": "A person riding a bike..."}
"""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import yaml
import random
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hulluedit.engines.llava7b import LLaVAHullueditEngine, EngineConfig
from hulluedit.steer import HullueditConfig
from hulluedit.datasets.chair_dataset import build_chair_dataset


def setup_seeds(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CHAIR Caption (Hulluedit)")
    
    parser.add_argument("--config", type=str, required=True, help="YAML config file path")
    
    parser.add_argument("--split", type=str, default=None, choices=["val", "train"])
    parser.add_argument("--sampling", type=str, default=None, choices=["first", "random"])
    parser.add_argument("--num-samples", type=int, default=None)
    
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="Print generation results for each image")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    setup_seeds(seed)
    
    split = args.split if args.split else cfg.get("split", "val")
    sampling = args.sampling if args.sampling else cfg.get("sampling", "random")
    num_samples = args.num_samples if args.num_samples else cfg.get("num_samples", 500)
    data_root = cfg.get("coco_root", "data/coco")
    
    output_dir = args.output_dir if args.output_dir else cfg.get("output_dir", "results/hulluedit_chair")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Hulluedit CHAIR Caption Generation")
    print("=" * 80)
    print(f"Dataset settings:")
    print(f"  Split:       {split}")
    print(f"  Sampling:    {sampling}")
    print(f"  Num Samples: {num_samples}")
    print(f"  Seed:        {seed}")
    print("=" * 80)
    print(f"Hulluedit parameters:")
    print(f"  Rank (r,q):  ({cfg.get('rank_evidence', 6)}, {cfg.get('rank_prior', 4)})")
    print(f"  Kappa:       {cfg.get('kappa', 0.6)}")
    print(f"  Lambda:      {cfg.get('lambda_prior', 0.3)}")
    print(f"  Anchor:      {cfg.get('anchor_layer', 26)}")
    print("=" * 80)
    print(f"Generation parameters:")
    print(f"  Temperature: {cfg.get('temperature', 0.2)}")
    print(f"  Top-P:       {cfg.get('top_p', 0.9)}")
    print(f"  Max Tokens:  {cfg.get('max_new_tokens', 128)}")
    print("=" * 80)
    
    print(f"Building CHAIR dataset...")
    data = build_chair_dataset(
        split=split,
        data_root=data_root,
        sampling=sampling,
        num_samples=num_samples,
        seed=seed
    )
    print(f"Loaded {len(data)} images")
    
    print("Initializing Hulluedit parameters...")
    hulluedit_cfg = HullueditConfig(
        rank_evidence=cfg.get("rank_evidence", 6),
        rank_prior=cfg.get("rank_prior", 4),
        kappa=cfg.get("kappa", 0.6),
        lambda_prior=cfg.get("lambda_prior", 0.3),
        eps=cfg.get("eps", 1e-6),
        lambda_n_max=cfg.get("lambda_n_max", 4.0),
        lambda_p_max=cfg.get("lambda_p_max", 4.0),
        vcr_floor=cfg.get("vcr_floor", 0.05),
        pcr_ceiling=cfg.get("pcr_ceiling", 0.95),
        pcr_threshold=cfg.get("pcr_threshold", 0.02),
        blend_tau=cfg.get("blend_tau", 0.7),
        norm_preserve=cfg.get("norm_preserve", True),
        norm_beta=cfg.get("norm_beta", 0.5),
        weight_temp=cfg.get("weight_temp", 1.5),
        uniform_svd=cfg.get("uniform_svd", False),
        no_complement=cfg.get("no_complement", False),
        no_gating=cfg.get("no_gating", False),
        use_fixed_strengths=cfg.get("use_fixed_strengths", False),
        fixed_lambda_n=cfg.get("fixed_lambda_n", 0.0),
        fixed_lambda_p=cfg.get("fixed_lambda_p", 0.0),
        only_residual=cfg.get("only_residual", False),
        only_anti_prior=cfg.get("only_anti_prior", False),
    )
    
    engine_name = str(cfg.get("engine", "llava")).lower()
    print(f"Initializing inference engine: {engine_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and cfg.get("gpu_id") is not None:
        device = f"cuda:{cfg.get('gpu_id')}"
    
    if engine_name == "llava":
        estimate_layer = cfg.get("estimate_layer", None)
        edit_layer = cfg.get("edit_layer", None)
        if estimate_layer == -1:
            estimate_layer = -1
        if edit_layer == -1:
            edit_layer = -1
        
        eng_cfg = EngineConfig(
            model_name=cfg["model_name"],
            anchor_layer=cfg.get("anchor_layer", 26),
            estimate_layer=estimate_layer,
            edit_layer=edit_layer,
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.2),
            precision=cfg.get("precision", "bf16")
        )
        engine = LLaVAHullueditEngine(eng_cfg, hulluedit_cfg)
    elif engine_name == "minigpt4":
        from hulluedit.engines.minigpt4 import MiniGPT4HullueditEngine, MiniGPT4EngineConfig
        gpu_id = 0
        if device.startswith("cuda:"):
            try:
                gpu_id = int(device.split(":")[1])
            except (ValueError, IndexError):
                gpu_id = 0
        mg_cfg = MiniGPT4EngineConfig(
            cfg_path=cfg["minigpt4_cfg_path"],
            anchor_layer=cfg.get("anchor_layer", 26),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.2),
            gpu_id=cfg.get("gpu_id", gpu_id),
        )
        engine = MiniGPT4HullueditEngine(mg_cfg, hulluedit_cfg, device=device)
    elif engine_name in ["mplug", "mplug_owl2", "mplug-owl2"]:
        from hulluedit.engines.mplug_owl2 import MplugOwl2Engine, MplugOwl2EngineConfig
        if "model_path" not in cfg:
            raise ValueError("mPLUG-Owl2 requires model_path configuration (HF or local path)")
        mplug_cfg = MplugOwl2EngineConfig(
            model_path=cfg["model_path"],
            model_name=cfg.get("model_name", "mplug_owl2"),
            anchor_layer=cfg.get("anchor_layer", 26),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.2),
            precision=cfg.get("precision", "fp16"),
        )
        engine = MplugOwl2Engine(mplug_cfg, hulluedit_cfg)
    else:
        raise ValueError(f"Unsupported engine type: {engine_name}")
    print("[Hulluedit] Engine initialization complete")
    
    if args.output_file:
        output_file = args.output_file
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir,
            f"chair_captions_hulluedit_{split}_{timestamp}.jsonl"
        )
    
    print(f"Output file: {output_file}")
    print("=" * 80)
    
    if engine_name == "llava":
        prompt_template = "USER: <image>\n{question}\nASSISTANT:"
    elif engine_name in ["mplug", "mplug_owl2", "mplug-owl2"]:
        prompt_template = "USER: <|image|>\n{question}\nASSISTANT:"
    else:
        prompt_template = "{question}"
    written = 0
    total = len(data)
    
    with open(output_file, "w") as f:
        for idx, item in enumerate(tqdm(data, desc="Generating Caption"), start=1):
            image_id = int(item["image_id"])
            image_path = item["image_path"]
            question = item.get("question", "Please describe this image in detail.")
            
            if not os.path.exists(image_path):
                print(f"[WARN] Image does not exist: {image_path}")
                continue
            
            try:
                prompt = prompt_template.format(question=question)
                output = engine.generate(prompt, image_path)
                caption = output["text"]
                
                if args.verbose:
                    pass
                
                record = {
                    "image_id": image_id,
                    "caption": caption
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                written += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to process image {image_id}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                continue
    
    print(f"\n{'='*80}")
    print(f"Complete! Generated {written} captions")
    print(f"Saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
