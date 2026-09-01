import sys
import json
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator, PartialState
from tqdm import tqdm
import time

sys.path.append('./')
from videollama2 import model_init, mm_infer

class AVHBenchDataset(Dataset):
    """Dataset class for AVHBench evaluation"""

    def __init__(self, qa_data, video_dir, modal_type):
        self.qa_data = qa_data
        self.video_dir = video_dir

        # Filter out samples with missing videos
        self.valid_samples = []
        for qa_item in qa_data:
            if 'AV' in qa_item['task'] :
                continue
            if modal_type == "a":
                video_path = os.path.join(video_dir, f"{qa_item['video_id']}.wav")
            else:
                video_path = os.path.join(video_dir, f"{qa_item['video_id']}.mp4")
            
            if os.path.exists(video_path):
                self.valid_samples.append({
                    **qa_item,
                    'video_path': video_path
                })
            else:
                if PartialState().is_main_process:
                    print(f"Warning: Video {video_path} not found, skipping...")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        return self.valid_samples[idx]

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return batch

def load_dataset(dataset_path):
    """Load QA annotations from JSON file"""
    with open(os.path.join(dataset_path, 'QA.json'), 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    return qa_data

def run_inference(model, processor, tokenizer, sample, modal_type, accelerator, max_new_tokens):
    """Run inference on a single sample"""
    video_path = sample['video_path']
    question = sample['text']
    ground_truth = sample['label']
    task = sample['task']
    video_id = sample['video_id']

    try:
        start_time = time.time()

        preprocess = processor['audio' if modal_type == "a" else "video"]

        if modal_type == "a":
            audio_video_tensor = preprocess(video_path)
        else:
            audio_video_tensor = preprocess(video_path, va=True if modal_type == "av" else False)

        if task != "AV Captioning" :
            question = question + " Answer only 'Yes' or 'No'."
            add_prompt = ""
        else:
            add_prompt = ""
        
        
        output = mm_infer(
            audio_video_tensor,
            question,
            model=model,
            tokenizer=tokenizer,
            modal='audio' if modal_type == "a" else "video",
            add_prompt=add_prompt,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

        inference_time = time.time() - start_time

        result = {
            'video_id': video_id,
            'task': task,
            'question': question,
            'ground_truth': ground_truth,
            'prediction': output,
            'inference_time': inference_time,
            'device': str(accelerator.device)
        }

        return result

    except Exception as e:
        if accelerator.is_main_process:
            print(f"Error processing {video_id} on {accelerator.device}: {str(e)}")

        result = {
            'video_id': video_id,
            'task': task,
            'question': question,
            'ground_truth': ground_truth,
            'prediction': f"ERROR: {str(e)}",
            'inference_time': 0,
            'device': str(accelerator.device)
        }
        return result

def evaluate_dataset_accelerate(args):
    """Main evaluation function using accelerate"""

    # Initialize accelerator
    accelerator = Accelerator()

    # Check available GPUs
    if accelerator.is_main_process:
        print(f"Using accelerate with {accelerator.num_processes} processes")
        if torch.cuda.is_available():
            print(f"CUDA available with {torch.cuda.device_count()} GPUs")
        print(f"Current process device: {accelerator.device}")

    # Load model with device_map for multi-GPU distribution
    if accelerator.is_main_process:
        print(f"Loading model from {args.model_path}")

    # Configure device_map for multi-GPU
    if args.device_map == "auto":
        device_map = "auto"
    elif args.device_map == "balanced":
        device_map = "balanced"
    elif args.device_map == "sequential":
        device_map = "sequential"
    else:
        device_map = {
            "": accelerator.device
        }

    # Initialize model with device_map
    model, processor, tokenizer = model_init(
        model_path=args.model_path,
        device_map=accelerator.device,
        use_flash_attn=True,
        torch_dtype=torch.bfloat16,
    )
    
    # model, processor, tokenizer = accelerator.prepare(model, processor, tokenizer)

    # Configure model based on modal type
    if args.modal_type == "a":
        model.model.vision_tower = None
    elif args.modal_type == "v":
        model.model.audio_tower = None
    elif args.modal_type == "av":
        pass
    else:
        raise NotImplementedError(f"Modal type {args.modal_type} not supported")

    # Load dataset
    if accelerator.is_main_process:
        print(f"Loading dataset from {args.dataset_path}")

    qa_data = load_dataset(args.dataset_path)
    video_dir = os.path.join(args.dataset_path, 'videos' if args.modal_type != "a" else 'audios')

    dataset = AVHBenchDataset(qa_data, video_dir, args.modal_type)

    if accelerator.is_main_process:
        print(f"Total valid samples: {len(dataset)}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Prepare dataloader with accelerate
    dataloader = accelerator.prepare(dataloader)

    # Evaluation loop
    all_results = []
    start_time = time.time()

    if accelerator.is_main_process:
        print(f"Starting evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader,
                                             desc=f"GPU {accelerator.process_index}",
                                             disable=not accelerator.is_local_main_process)):

            batch_results = []

            for sample in batch:
                result = run_inference(model, processor, tokenizer, sample, args.modal_type, accelerator, args.max_new_tokens)
                batch_results.append(result)

                if args.verbose and accelerator.is_main_process:
                    print(f"\nProcessed: {sample['video_id']}")
                    print(f"Task: {sample['task']}")
                    print(f"Question: {sample['question']}")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Time: {result['inference_time']:.2f}s")

            all_results.extend(batch_results)

            # Periodic saving and memory cleanup
            if (batch_idx + 1) % args.save_interval == 0:
                if accelerator.is_main_process:
                    print(f"Processed {(batch_idx + 1) * args.batch_size * accelerator.num_processes} samples")

                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Gather results from all processes
    all_results = accelerator.gather_for_metrics(all_results)

    total_time = time.time() - start_time

    # Save results (only on main process)
    if accelerator.is_main_process:
        # Sort results by video_id for consistency
        all_results.sort(key=lambda x: x['video_id'])

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path.replace('.json', f'_{args.modal_type}.json'), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\nEvaluation completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Results saved to: {args.output_path}")
        print(f"Total samples processed: {len(all_results)}")

        # Print basic statistics
        successful_samples = [r for r in all_results if not r['prediction'].startswith('ERROR:')]
        print(f"Successful inferences: {len(successful_samples)}")
        print(f"Failed inferences: {len(all_results) - len(successful_samples)}")

        if successful_samples:
            avg_time = sum(r['inference_time'] for r in successful_samples) / len(successful_samples)
            print(f"Average inference time per sample: {avg_time:.2f}s")
            print(f"Throughput: {len(successful_samples) / total_time:.2f} samples/s")

        # Print per-device statistics
        device_stats = {}
        for result in all_results:
            device = result.get('device', 'unknown')
            if device not in device_stats:
                device_stats[device] = {'total': 0, 'successful': 0, 'time': 0}

            device_stats[device]['total'] += 1
            if not result['prediction'].startswith('ERROR:'):
                device_stats[device]['successful'] += 1
                device_stats[device]['time'] += result['inference_time']

        print(f"\nPer-Device Statistics:")
        for device, stats in device_stats.items():
            if stats['successful'] > 0:
                avg_time = stats['time'] / stats['successful']
                print(f"  {device}: {stats['successful']}/{stats['total']} successful, avg time: {avg_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VideoLLaMA2 on AVHBench dataset with accelerate")

    parser.add_argument('--model-path', type=str, required=False,
                       default=str(Path(__file__).resolve().parents[2] / 'models' / 'VideoLLaMA2.1-7B-AV'),
                       help='Path to the model')
    parser.add_argument('--modal-type', choices=["a", "v", "av"], required=True,
                       help='Modal type: a=audio, v=video, av=audio-video')
    parser.add_argument('--dataset-path', type=str, required=False,
                       default=str(Path(__file__).resolve().parents[2] / 'data' / 'AVHBench'),
                       help='Path to AVHBench dataset')
    parser.add_argument('--output-path', type=str, required=False,
                       default='./results/evaluation_results_accelerate.json',
                       help='Path to save evaluation results')
    parser.add_argument('--device-map', type=str, choices=["auto", "balanced", "sequential", "single"],
                       default="auto",
                       help='Device mapping strategy for multi-GPU')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of dataloader workers')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save intermediate results every N batches')
    parser.add_argument('--load-8bit', action='store_true',
                       help='Load model in 8-bit precision')
    parser.add_argument('--load-4bit', action='store_true',
                       help='Load model in 4-bit precision')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results for each sample')
    parser.add_argument('--max-new-tokens', type=int, default=2048,
                       help='Maximum new tokens; yes/no benchmark generation stops early after a valid answer')

    args = parser.parse_args()

    evaluate_dataset_accelerate(args)
