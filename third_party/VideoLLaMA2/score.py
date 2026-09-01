import json
import argparse
import re
from collections import defaultdict
from typing import Dict, List, Tuple

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    # Convert to lowercase and strip whitespace
    answer = answer.lower().strip()

    # Remove punctuation at the end
    answer = re.sub(r'[^\w\s]$', '', answer)

    # Handle common variations
    if answer in ['yes', 'y', 'true', '1', 'correct']:
        return 'yes'
    elif answer in ['no', 'n', 'false', '0', 'incorrect']:
        return 'no'
    
    if 'yes' in answer:
        return 'yes'
    elif 'no' in answer:
        return 'no'

    return answer

def extract_answer(prediction: str) -> str:
    """Extract answer from model prediction"""
    prediction = prediction.strip()

    # Look for yes/no patterns
    yes_patterns = [
        r'\byes\b',
        r'\btrue\b',
        r'\bcorrect\b',
        r'\baffirmative\b'
    ]

    no_patterns = [
        r'\bno\b',
        r'\bfalse\b',
        r'\bincorrect\b',
        r'\bnegative\b'
    ]

    prediction_lower = prediction.lower()

    # Check for explicit patterns
    for pattern in yes_patterns:
        if re.search(pattern, prediction_lower):
            return 'yes'

    for pattern in no_patterns:
        if re.search(pattern, prediction_lower):
            return 'no'

    # If no clear pattern, return the first word
    words = prediction.split()
    if words:
        return normalize_answer(words[0])

    return prediction.lower().strip()

def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Calculate accuracy between predictions and ground truths"""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")

    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        pred_normalized = normalize_answer(extract_answer(pred))
        gt_normalized = normalize_answer(gt)

        if pred_normalized == gt_normalized:
            correct += 1

    return correct / total if total > 0 else 0.0

def evaluate_by_task(results: List[Dict]) -> Dict[str, Dict]:
    """Evaluate results grouped by task"""
    task_results = defaultdict(list)

    # Group results by task
    for result in results:
        if not result['prediction'].startswith('ERROR:'):
            task_results[result['task']].append(result)

    task_scores = {}

    for task, task_data in task_results.items():
        predictions = [r['prediction'] for r in task_data]
        ground_truths = [r['ground_truth'] for r in task_data]

        accuracy = calculate_accuracy(predictions, ground_truths)

        task_scores[task] = {
            'accuracy': accuracy,
            'total_samples': len(task_data),
            'correct_predictions': int(accuracy * len(task_data)),
            'samples': task_data
        }

    return task_scores

def print_detailed_results(task_scores: Dict[str, Dict], show_examples: bool = False):
    """Print detailed evaluation results"""
    print("\n" + "="*80)
    print("DETAILED EVALUATION RESULTS")
    print("="*80)

    total_samples = 0
    total_correct = 0

    for task, scores in task_scores.items():
        print(f"\nTask: {task}")
        print("-" * 50)
        print(f"Accuracy: {scores['accuracy']:.4f} ({scores['correct_predictions']}/{scores['total_samples']})")

        
        if task != "AV Captioning" and task != "AV Matching":
            total_samples += scores['total_samples']
            total_correct += scores['correct_predictions']

        if show_examples:
            print("\nSample predictions:")
            for i, sample in enumerate(scores['samples'][:3]):  # Show first 3 examples
                pred_normalized = normalize_answer(extract_answer(sample['prediction']))
                gt_normalized = normalize_answer(sample['ground_truth'])
                status = "✓" if pred_normalized == gt_normalized else "✗"

                print(f"  {status} Video {sample['video_id']}:")
                print(f"    Q: {sample['question']}")
                print(f"    GT: {sample['ground_truth']} | Pred: {sample['prediction']}")
                print(f"    Normalized - GT: {gt_normalized} | Pred: {pred_normalized}")

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"\n" + "="*80)
    print(f"OVERALL ACCURACY: {overall_accuracy:.4f} ({total_correct}/{total_samples})")
    print("="*80)

def analyze_errors(task_scores: Dict[str, Dict]):
    """Analyze common error patterns"""
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)

    for task, scores in task_scores.items():
        print(f"\nTask: {task}")
        print("-" * 50)

        errors = []
        for sample in scores['samples']:
            pred_normalized = normalize_answer(extract_answer(sample['prediction']))
            gt_normalized = normalize_answer(sample['ground_truth'])

            if pred_normalized != gt_normalized:
                errors.append({
                    'video_id': sample['video_id'],
                    'question': sample['question'],
                    'ground_truth': gt_normalized,
                    'prediction': pred_normalized,
                    'raw_prediction': sample['prediction']
                })

        if errors:
            print(f"Error rate: {len(errors)}/{scores['total_samples']} ({len(errors)/scores['total_samples']:.2%})")

            # Analyze error patterns
            error_patterns = defaultdict(int)
            for error in errors:
                pattern = f"{error['ground_truth']} -> {error['prediction']}"
                error_patterns[pattern] += 1

            print("Most common error patterns:")
            for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {pattern}: {count} times")
        else:
            print("No errors found!")

def save_detailed_scores(task_scores: Dict[str, Dict], output_path: str):
    """Save detailed scores to JSON file"""
    # Prepare data for JSON serialization
    output_data = {
        'task_scores': {}
    }

    total_samples = 0
    total_correct = 0

    for task, scores in task_scores.items():
        total_samples += scores['total_samples']
        total_correct += scores['correct_predictions']

        output_data['task_scores'][task] = {
            'accuracy': scores['accuracy'],
            'total_samples': scores['total_samples'],
            'correct_predictions': scores['correct_predictions']
        }

    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    output_data['overall'] = {
        'accuracy': overall_accuracy,
        'total_samples': total_samples,
        'correct_predictions': total_correct
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed scores saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Score VideoLLaMA2 evaluation results")

    parser.add_argument('--f', type=str, required=True,
                       help='Path to evaluation results JSON file')
    parser.add_argument('--output-path', type=str, required=False,
                       default='./results/scores.json',
                       help='Path to save detailed scores')
    parser.add_argument('--show-examples', action='store_true',
                       help='Show example predictions for each task')
    parser.add_argument('--analyze-errors', action='store_true',
                       help='Analyze error patterns')

    args = parser.parse_args()

    # Load evaluation results
    print(f"Loading results from: {args.f}")
    with open(args.f, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"Loaded {len(results)} evaluation results")

    # Remove duplicates based on (video_id, question)
    seen = set()
    unique_results = []
    for r in results:
        key = (r['video_id'], r['question'])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    dup_count = len(results) - len(unique_results)
    if dup_count > 0:
        print(f"Removed {dup_count} duplicate entries (same video_id + question)")
    results = unique_results

    # Filter out error cases
    valid_results = [r for r in results if not r['prediction'].startswith('ERROR:')]
    error_count = len(results) - len(valid_results)

    if error_count > 0:
        print(f"Warning: {error_count} results had errors and will be excluded from scoring")

    # Calculate scores by task
    task_scores = evaluate_by_task(valid_results)

    # Print results
    print_detailed_results(task_scores, args.show_examples)

    # Analyze errors if requested
    if args.analyze_errors:
        analyze_errors(task_scores)

    # Save detailed scores
    save_detailed_scores(task_scores, args.output_path)

if __name__ == "__main__":
    main()