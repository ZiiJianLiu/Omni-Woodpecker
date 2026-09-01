"""
POPE (Polling-based Object Probing Evaluation) Evaluation
Evaluates object hallucination issues in multimodal models
Supports LLaVA-1.5, MiniGPT-4 and mPLUG-Owl2
"""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import yaml
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hulluedit.engines.llava7b import LLaVAHullueditEngine, EngineConfig
# from hulluedit.engines.minigpt4 import MiniGPT4HullueditEngine, MiniGPT4EngineConfig
# mPlug-Owl2 lazy import
from hulluedit.steer import HullueditConfig


def load_pope_data(pope_root: str, split: str = "adversarial"):
    """
    Load POPE dataset
    POPE format: each line is a JSON containing image, text, label
    """
    pope_file = Path(pope_root) / f"coco_pope_{split}.json"
    if not pope_file.exists():
        raise FileNotFoundError(f"POPE file not found: {pope_file}")
    
    data = []
    with open(pope_file) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    return data


def evaluate_pope(predictions, labels):
    """Compute POPE metrics: Accuracy, Precision, Recall, F1"""
    assert len(predictions) == len(labels)
    
    tp = sum(1 for p, l in zip(predictions, labels) if p == "yes" and l == "yes")
    fp = sum(1 for p, l in zip(predictions, labels) if p == "yes" and l == "no")
    tn = sum(1 for p, l in zip(predictions, labels) if p == "no" and l == "no")
    fn = sum(1 for p, l in zip(predictions, labels) if p == "no" and l == "yes")
    
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


def extract_yes_no(text: str) -> str:
    """Extract yes/no answer from model output"""
    if not text:
        return "no"
    
    text = text.lower().strip()
    
    # Check first 30 characters for yes/no
    prefix = text[:30]
    
    # Check yes first
    if "yes" in prefix:
        return "yes"
    elif "no" in prefix:
        return "no"
    else:
        # Default to no (conservative strategy)
        return "no"


def main():
    parser = argparse.ArgumentParser(description="POPE Evaluation (Hulluedit, supports LLaVA-1.5, MiniGPT-4 and mPLUG-Owl2)")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--split", type=str, default="adversarial", 
                       choices=["random", "popular", "adversarial"],
                       help="POPE dataset split")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples (for quick testing)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Model name (LLaVA, MiniGPT-4 or mPLUG-Owl2), inferred from config if not specified")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Model path (for mPLUG-Owl2, overrides config if provided)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # Override model_path from command line if provided
    if args.model_path:
        cfg["model_path"] = args.model_path
    
    # Determine model type
    model_name = args.model_name or cfg.get("model_name", "")
    model_name_lower = model_name.lower()
    
    # Initialize Hulluedit config
    hulluedit_cfg = HullueditConfig(
        rank_evidence=cfg.get("rank_evidence", 8),
        rank_prior=cfg.get("rank_prior", 5),
        kappa=cfg.get("kappa", 0.50),
        lambda_prior=cfg.get("lambda_prior", 0.22),
        eps=cfg.get("eps", 1e-6),
        lambda_n_max=cfg.get("lambda_n_max", 3.2),
        lambda_p_max=cfg.get("lambda_p_max", 3.8),
        vcr_floor=cfg.get("vcr_floor", 0.045),
        pcr_ceiling=cfg.get("pcr_ceiling", 0.92),
        pcr_threshold=cfg.get("pcr_threshold", 0.015),
        blend_tau=cfg.get("blend_tau", 0.76),
        norm_preserve=cfg.get("norm_preserve", True),
        norm_beta=cfg.get("norm_beta", 0.74),
        weight_temp=cfg.get("weight_temp", 1.15),
    )
    
    # Initialize engine
    print(f"[POPE] Initializing Hulluedit engine: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if "llava" in model_name_lower:
        eng_cfg = EngineConfig(
            model_name=cfg["model_name"],
            anchor_layer=cfg.get("anchor_layer", 28),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.12),
            precision=cfg.get("precision", "bf16")
        )
        engine = LLaVAHullueditEngine(eng_cfg, hulluedit_cfg, device=device)
        is_llava = True
        is_mplug = False
    elif "minigpt" in model_name_lower:
        # Extract gpu_id from device string (e.g., "cuda:0" -> 0)
        gpu_id = 0
        if device.startswith("cuda:"):
            try:
                gpu_id = int(device.split(":")[1])
            except (ValueError, IndexError):
                gpu_id = 0
        
        eng_cfg = MiniGPT4EngineConfig(
            cfg_path=cfg.get("minigpt4_cfg_path"),
            anchor_layer=cfg.get("anchor_layer", 26),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.12),
            gpu_id=cfg.get("gpu_id", gpu_id),
        )
        engine = MiniGPT4HullueditEngine(eng_cfg, hulluedit_cfg, device=device)
        is_llava = False
        is_mplug = False
    elif "mplug" in model_name_lower:
        # Lazy import mPlug-Owl2
        try:
            from hulluedit.engines.mplug_owl2 import MplugOwl2Engine, MplugOwl2EngineConfig
        except ImportError as e:
            raise ImportError(
                f"Cannot import mPlug-Owl2 module. Please ensure mplug_owl2 dependencies are installed."
                f"Error details: {e}"
            )
        
        model_path = cfg.get("model_path")
        if not model_path:
            raise ValueError("mPLUG-Owl2 requires model_path (HF or local path)")
        eng_cfg = MplugOwl2EngineConfig(
            model_path=model_path,
            anchor_layer=cfg.get("anchor_layer", 26),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.12),
            precision=cfg.get("precision", "fp16")
        )
        engine = MplugOwl2Engine(eng_cfg, hulluedit_cfg, device=device)
        is_llava = False
        is_mplug = True
    else:
        raise ValueError(f"Unsupported model: {model_name}, please use LLaVA, MiniGPT-4 or mPLUG-Owl2")
    
    # Load POPE data
    print(f"[POPE] Loading dataset: {args.split}")
    pope_data = load_pope_data(cfg["pope_root"], args.split)
    
    if args.max_samples:
        pope_data = pope_data[:args.max_samples]
    
    print(f"[POPE] Number of samples: {len(pope_data)}")
    
    # Inference
    results = []
    predictions = []
    labels = []
    
    coco_img_dir = Path(cfg["coco_images"])
    
    # Print frequency: print detailed info every N samples, or every sample if few
    print_every = max(1, len(pope_data) // 100) if len(pope_data) > 100 else 1
    
    for idx, item in enumerate(tqdm(pope_data, desc="POPE Evaluation")):
        # item["image"] may be relative or absolute path
        if "/" in item["image"] and Path(item["image"]).exists():
            image_path = Path(item["image"])
        else:
            image_path = coco_img_dir / item["image"]
        
        if not image_path.exists():
            tqdm.write(f"[WARNING] Image not found: {image_path}")
            continue
        
        question = item["text"]
        label = item["label"]  # "yes" or "no"
        
        # Build different prompts based on model type
        if is_llava:
            # LLaVA format prompt
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
        elif is_mplug:
            # mPLUG-Owl2 format prompt
            prompt = f"USER: <|image|>\n{question}\nASSISTANT:"
        else:
            # MiniGPT-4 format prompt (use question directly)
            prompt = question
        
        # Generate
        try:
            if is_llava:
                # LLaVA accepts PIL Image or path
                image = Image.open(image_path).convert("RGB")
                output_dict = engine.generate(prompt, image, max_new_tokens=cfg.get("max_new_tokens", 128))
                pred_text = output_dict.get("text", output_dict.get("output", ""))
            elif is_mplug:
                # mPLUG-Owl2 accepts path
                output_dict = engine.generate(prompt, str(image_path), max_new_tokens=cfg.get("max_new_tokens", 128))
                pred_text = output_dict.get("text", output_dict.get("output", ""))
            else:
                # MiniGPT-4 accepts path
                output_dict = engine.generate(prompt, str(image_path), max_new_tokens=cfg.get("max_new_tokens", 128))
                pred_text = output_dict.get("text", output_dict.get("output", ""))
            
            pred_label = extract_yes_no(pred_text)
            
            # Compute average Hulluedit metrics (if available)
            certs = output_dict.get("certs", [])
            avg_vcr = sum(c.get("vcr", 0.0) for c in certs) / len(certs) if certs else 0.0
            avg_pcr = sum(c.get("pcr", 0.0) for c in certs) / len(certs) if certs else 0.0
            avg_gate = sum(c.get("gate", 0.0) for c in certs) / len(certs) if certs else 0.0
            
            # Check correctness
            is_correct = "✓" if pred_label == label else "✗"
            
            # Print output (use tqdm.write to avoid interfering with progress bar)
            if idx % print_every == 0 or idx < 10:  # Always print first 10 samples
                pass
            
            results.append({
                "image": item["image"],
                "question": question,
                "label": label,
                "prediction": pred_label,
                "raw_output": pred_text,
                "certs": certs
            })
            
            predictions.append(pred_label)
            labels.append(label)
            
            # Periodically print intermediate statistics (every 100 samples)
            if len(predictions) > 0 and len(predictions) % 100 == 0:
                pass
        
        except Exception as e:
            tqdm.write(f"[ERROR] Sample {idx+1} ({image_path}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute metrics
    if len(predictions) == 0:
        print("[ERROR] No predictions generated successfully")
        return
    
    metrics = evaluate_pope(predictions, labels)
    
    print("\n[POPE Results]")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  TP/FP/TN/FN: {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")
    
    # Save results
    output_data = {
        "config": args.config,
        "model_name": model_name,
        "split": args.split,
        "num_samples": len(results),
        "metrics": metrics,
        "results": results
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[POPE] Results saved: {args.output}")


if __name__ == "__main__":
    main()
