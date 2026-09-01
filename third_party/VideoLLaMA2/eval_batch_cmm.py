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

class CMMDataset(Dataset):
    """Dataset class for CMM benchmark evaluation"""

    def __init__(self, qa_data, video_base_dir):
        self.qa_data = qa_data
        self.video_base_dir = video_base_dir

        # Filter out samples with missing videos
        self.valid_samples = []
        for qa_item in qa_data:
            # Convert relative path to absolute path
            video_path = qa_item.get('video_path', '')
            if video_path:
                # Remove leading "./" if present
                if video_path.startswith('./'):
                    video_path = video_path[2:]
                full_video_path = os.path.join(video_base_dir, video_path)

                if os.path.exists(full_video_path):
                    self.valid_samples.append({
                        **qa_item,
                        'full_video_path': full_video_path
                    })
                else:
                    if PartialState().is_main_process:
                        print(f"Warning: Video {full_video_path} not found, skipping...")
            else:
                if PartialState().is_main_process:
                    print(f"Warning: No video path in sample, skipping...")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        return self.valid_samples[idx]

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return batch

def load_dataset(dataset_path, category=None):
    """Load QA annotations from JSON file"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # Filter by category if specified
    if category and category != 'all':
        qa_data = [item for item in qa_data if item.get('category') == category]

    return qa_data

def build_cmm_full_av_tensor(processor, video_path, audio_path=None):
    """CMM stores audio as sidecar wav; many mp4 files do not contain an audio stream."""
    if not audio_path:
        audio_path = os.path.splitext(video_path)[0] + ".wav"
    elif not os.path.isabs(audio_path):
        candidate = os.path.join(os.path.dirname(video_path), os.path.basename(audio_path))
        audio_path = candidate if os.path.exists(candidate) else os.path.splitext(video_path)[0] + ".wav"
    video = processor['video'](video_path, va=False)
    audio = processor['audio'](audio_path)
    return {'video': video, 'audio': audio}

def run_inference(model, processor, tokenizer, sample, modal_type, accelerator, max_new_tokens):
    """Run inference on a single sample"""
    video_path = sample['full_video_path']
    question = sample['question']
    ground_truth = sample['answer']

    # Get metadata
    category = sample.get('category', '')
    sub_category = sample.get('sub_category', '')
    modality = sample.get('modality', '')
    granularity = sample.get('granularity', '')
    correlation_type = sample.get('correlation_type', '')
    original_video_path = sample.get('video_path', '')
    audio_path = sample.get('audio_path', None)

    try:
        start_time = time.time()

        # Preprocess video based on modality
        # if modality == 'audio':
        #     # For audio modality, we might need to extract audio or use audio file
        #     # For now, assume we use video file with audio processing
        #     preprocess = processor['video']
        #     video_tensor = preprocess(video_path, va=False)  # video only for now
        # elif modality == 'visual':
        #     preprocess = processor['video']
        #     video_tensor = preprocess(video_path, va=False)
        # elif modality == 'audio-visual' or modality == 'visual-audio-language':
        #     preprocess = processor['video']
        #     video_tensor = preprocess(video_path, va=True)
        # else:
        #     # Default to visual
        #     preprocess = processor['video']
        #     video_tensor = preprocess(video_path, va=False)
        
        if modal_type == "a":
            audio_video_tensor = processor['audio'](audio_path if audio_path else os.path.splitext(video_path)[0] + ".wav")
            infer_modal = 'audio'
        elif modal_type == "av":
            audio_video_tensor = build_cmm_full_av_tensor(processor, video_path, audio_path)
            infer_modal = 'video'
        else:
            audio_video_tensor = processor['video'](video_path, va=False)
            infer_modal = 'video'

        question_with_prompt = question + " Only yes/no. No extra text." # " Answer yes or no. Do not include any explanation."

        infer_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "modal": infer_modal,
            "add_prompt": "",
            "do_sample": False,
        }
        if max_new_tokens is not None:
            infer_kwargs["max_new_tokens"] = max_new_tokens

        output = mm_infer(
            audio_video_tensor,
            question_with_prompt,
            **infer_kwargs,
        )

        inference_time = time.time() - start_time

        result = {
            'category': category,
            'sub_category': sub_category,
            'modality': modality,
            'granularity': granularity,
            'correlation_type': correlation_type,
            'video_path': original_video_path,
            'audio_path': audio_path,
            'question': question,
            'answer': ground_truth,
            'pred': output,
            'inference_time': inference_time,
            'device': str(accelerator.device)
        }

        return result

    except Exception as e:
        if accelerator.is_main_process:
            print(f"Error processing {original_video_path} on {accelerator.device}: {str(e)}")

        result = {
            'category': category,
            'sub_category': sub_category,
            'modality': modality,
            'granularity': granularity,
            'correlation_type': correlation_type,
            'video_path': original_video_path,
            'audio_path': audio_path,
            'question': question,
            'answer': ground_truth,
            'pred': f"ERROR: {str(e)}",
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

    # Load model
    if accelerator.is_main_process:
        print(f"Loading model from {args.model_path}")

    # Initialize model with device_map
    model, processor, tokenizer = model_init(
        model_path=args.model_path,
        device_map=accelerator.device,
        use_flash_attn=True,
        torch_dtype=torch.bfloat16,
    )

    # Load dataset
    if accelerator.is_main_process:
        print(f"Loading dataset from {args.dataset_path}")
        if args.category != 'all':
            print(f"Filtering category: {args.category}")

    qa_data = load_dataset(args.dataset_path, args.category)

    if accelerator.is_main_process:
        print(f"Total samples in dataset: {len(qa_data)}")

    dataset = CMMDataset(qa_data, args.video_base_dir)

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
                    print(f"\nProcessed: {sample.get('video_path', 'unknown')}")
                    print(f"Category: {sample.get('category', 'unknown')}")
                    print(f"Question: {sample.get('question', 'unknown')}")
                    print(f"Answer: {sample.get('answer', 'unknown')}")
                    print(f"Prediction: {result['pred']}")
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
        # Create output directory if needed
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)

        # Modify output path to include category if not 'all'
        output_path = args.output_path
        if args.category != 'all' and args.output_path == './output.json':
            # Extract filename and extension
            base_name, ext = os.path.splitext(args.output_path)
            output_path = f"{base_name}_{args.category}{ext}"

        # Save in CMM format (JSON lines format)
        with open(output_path.replace('.json', f'_{args.modal_type}.json'), 'w', encoding='utf-8') as f:
            for result in all_results:
                # Remove the extra fields used for tracking
                result_clean = {k: v for k, v in result.items() if k not in ['inference_time', 'device']}
                f.write(json.dumps(result_clean, ensure_ascii=False) + '\n')

        print(f"\nEvaluation completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Results saved to: {output_path}")
        print(f"Total samples processed: {len(all_results)}")

        # Print basic statistics
        successful_samples = [r for r in all_results if not r['pred'].startswith('ERROR:')]
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
            if not result['pred'].startswith('ERROR:'):
                device_stats[device]['successful'] += 1
                device_stats[device]['time'] += result['inference_time']

        print(f"\nPer-Device Statistics:")
        for device, stats in device_stats.items():
            if stats['successful'] > 0:
                avg_time = stats['time'] / stats['successful']
                print(f"  {device}: {stats['successful']}/{stats['total']} successful, avg time: {avg_time:.2f}s")

        # Print per-category statistics
        category_stats = {}
        for result in all_results:
            category = result.get('category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'successful': 0}

            category_stats[category]['total'] += 1
            if not result['pred'].startswith('ERROR:'):
                category_stats[category]['successful'] += 1

        print(f"\nPer-Category Statistics:")
        for category, stats in category_stats.items():
            print(f"  {category}: {stats['successful']}/{stats['total']} successful")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VideoLLaMA2 on CMM benchmark with accelerate")

    parser.add_argument('--model-path', type=str, required=False,
                       default=str(Path(__file__).resolve().parents[2] / 'models' / 'VideoLLaMA2.1-7B-AV'),
                       help='Path to the model')
    parser.add_argument('--dataset-path', type=str, required=False,
                       default=str(Path(__file__).resolve().parents[2] / 'data' / 'CMM' / 'all_data_final_reorg.json'),
                       help='Path to CMM dataset JSON file')
    parser.add_argument('--video-base-dir', type=str, required=False,
                       default=str(Path(__file__).resolve().parents[2] / 'data' / 'CMM'),
                       help='Base directory containing CMM videos')
    parser.add_argument('--modal-type', choices=["a", "v", "av"], required=True,
                    help='Modal type: a=audio, v=video, av=audio-video')
    parser.add_argument('--category', type=str, required=False,
                       default='over-reliance_unimodal_priors',
                       choices=['all', 'inter-modality_spurious_correlation', 'over-reliance_unimodal_priors'],
                       help='Category to evaluate (default: over-reliance_unimodal_priors)')
    parser.add_argument('--output-path', type=str, required=False,
                       default='./output.json',
                       help='Path to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size per GPU')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of dataloader workers')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save intermediate results every N batches')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed results for each sample')
    parser.add_argument('--max-new-tokens', type=int, default=None,
                       help='Optional maximum new tokens override; omit to keep the original mm_infer default')

    args = parser.parse_args()

    evaluate_dataset_accelerate(args)
