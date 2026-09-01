"""
MME (Multi-Modal Evaluation) Evaluation
Based on Hulluedit method, evaluates LLaVA and MiniGPT-4 on MME dataset

Reference: DeCo/mme_llava.py implementation
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Any

import yaml
import torch
from PIL import Image
from tqdm import tqdm

from hulluedit.steer import HullueditConfig


def _load_mme_questions(question_file: str, image_folder: str) -> List[Dict[str, Any]]:
    """
    Read MME test file, return question list
    
    Args:
        question_file: MME test file path (JSONL format)
        image_folder: MME image root directory
    
    Returns:
        Question list, each element contains: question_id, image, prompt, text
    """
    image_folder_path = Path(image_folder).expanduser()
    if not image_folder_path.exists():
        raise FileNotFoundError(f"MME image directory does not exist: {image_folder_path}")
    
    questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Build full image path
                image_path = item.get("image", "")
                if image_path.startswith("/"):
                    # Absolute path
                    full_image_path = image_path
                else:
                    # Relative path
                    full_image_path = str(image_folder_path / image_path)
                
                questions.append({
                    "question_id": item.get("question_id"),
                    "image": image_path,
                    "image_path": full_image_path,
                    "prompt": item.get("prompt", ""),
                    "text": item.get("text", ""),  # May contain reference answer
                })
            except Exception as e:
                print(f"Failed to parse line: {line}, error: {e}")
                continue
    
    if not questions:
        raise RuntimeError("MME test set parsing is empty, please check question_file content")
    
    return questions


def recorder(output: str) -> str:
    """
    Convert model output to Yes/No answer
    Reference: DeCo/mme_llava.py implementation
    """
    NEG_WORDS = ["No", "not", "no", "NO"]
    
    output = output.replace('.', '')
    output = output.replace(',', '')
    words = output.split(' ')
    
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"


def main():
    parser = argparse.ArgumentParser(description="MME Evaluation (based on Hulluedit)")
    parser.add_argument("--model_name", type=str, default="LLaVA-7B",
                        help="Model name, used for output file naming")
    parser.add_argument("--question_file", type=str, required=True,
                        help="MME test file path (JSONL format)")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="MME image root directory")
    parser.add_argument("--config", type=str, required=True,
                        help="Hulluedit config file path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Max samples (None means all)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    print(f"[INFO] Loading config file: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    # Load questions
    print(f"[INFO] Loading MME test set: {args.question_file}")
    questions = _load_mme_questions(args.question_file, args.image_folder)
    
    if args.num_samples and args.num_samples < len(questions):
        print(f"[INFO] Sampling {args.num_samples} questions (total: {len(questions)})")
        questions = random.sample(questions, args.num_samples)
    
    print(f"[INFO] Total {len(questions)} questions to evaluate")
    
    # Initialize Hulluedit config
    print(f"[INFO] Initializing Hulluedit config")
    hulluedit_cfg = HullueditConfig(
        rank_evidence=config_dict.get("rank_evidence", 8),
        rank_prior=config_dict.get("rank_prior", 5),
        kappa=config_dict.get("kappa", 0.50),
        lambda_prior=config_dict.get("lambda_prior", 0.22),
        eps=config_dict.get("eps", 1e-6),
        lambda_n_max=config_dict.get("lambda_n_max", 3.2),
        lambda_p_max=config_dict.get("lambda_p_max", 3.8),
        vcr_floor=config_dict.get("vcr_floor", 0.045),
        pcr_ceiling=config_dict.get("pcr_ceiling", 0.92),
        pcr_threshold=config_dict.get("pcr_threshold", 0.015),
        blend_tau=config_dict.get("blend_tau", 0.76),
        norm_preserve=config_dict.get("norm_preserve", True),
        norm_beta=config_dict.get("norm_beta", 0.74),
        weight_temp=config_dict.get("weight_temp", 1.15),
    )
    
    # Load model
    model_name = config_dict.get("model_name", args.model_name)
    print(f"[INFO] Loading model: {model_name}")
    from hulluedit.engines.llava7b import LLaVAHullueditEngine, EngineConfig
    from hulluedit.engines.minigpt4 import MiniGPT4HullueditEngine, MiniGPT4EngineConfig
    
    model_name_lower = args.model_name.lower()
    if "llava" in model_name_lower:
        eng_cfg = EngineConfig(
            model_name=model_name,
            anchor_layer=config_dict.get("anchor_layer", 28),
            max_new_tokens=config_dict.get("max_new_tokens", 128),
            top_p=config_dict.get("top_p", 0.90),
            temperature=config_dict.get("temperature", 0.12),
            precision=config_dict.get("precision", "bf16"),
        )
        engine = LLaVAHullueditEngine(
            eng_cfg=eng_cfg,
            hulluedit_cfg=hulluedit_cfg,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    elif "minigpt" in model_name_lower:
        eng_cfg = MiniGPT4EngineConfig(
            cfg_path=config_dict.get("minigpt4_cfg_path"),
            anchor_layer=config_dict.get("anchor_layer", 28),
            max_new_tokens=config_dict.get("max_new_tokens", 128),
            top_p=config_dict.get("top_p", 0.90),
            temperature=config_dict.get("temperature", 0.12),
        )
        engine = MiniGPT4HullueditEngine(
            eng_cfg=eng_cfg,
            hulluedit_cfg=hulluedit_cfg,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")
    
    # Generate answers
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.model_name.lower()}_mme_answers_{timestamp}.jsonl"
    
    print(f"[INFO] Start generating answers, output to: {output_file}")
    
    results = []
    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in tqdm(questions, desc="MME Evaluation"):
            question_id = item["question_id"]
            image_path = item["image_path"]
            prompt = item["prompt"]
            
            # Check if image exists
            if not Path(image_path).exists():
                print(f"[WARNING] Image not found: {image_path}")
                continue
            
            try:
                # Generate answer (use different interface based on engine type)
                if "llava" in model_name_lower:
                    # LLaVA accepts PIL Image or path
                    image = Image.open(image_path).convert("RGB")
                    result_dict = engine.generate(prompt, image)
                    output = result_dict.get("output", "")
                elif "minigpt" in model_name_lower:
                    # MiniGPT-4 accepts path
                    result_dict = engine.generate(prompt, image_path)
                    output = result_dict.get("output", "")
                else:
                    raise ValueError(f"Unsupported model: {args.model_name}")
                
                # Convert to Yes/No
                answer = recorder(output)
                
                # Save result
                result = {
                    "question_id": question_id,
                    "prompt": prompt,
                    "text": answer,
                    "model_id": args.model_name,
                    "image": item["image"],
                    "metadata": {
                        "raw_output": output
                    }
                }
                
                results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                
            except Exception as e:
                print(f"[ERROR] Failed to process question {question_id}: {e}")
                continue
    
    print(f"[INFO] Evaluation complete, generated {len(results)} answers")
    print(f"[INFO] Results saved in: {output_file}")
    
    # Output statistics
    yes_count = sum(1 for r in results if r["text"] == "Yes")
    no_count = sum(1 for r in results if r["text"] == "No")
    print(f"[INFO] Answer statistics: Yes={yes_count}, No={no_count}")
    
    return output_file


if __name__ == "__main__":
    main()
