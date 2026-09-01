"""
Convert MME answers to MME official evaluation format
Reference: DeCo/eval_tool/convert_answer_to_mme.py implementation
"""
import os
import json
import argparse
from collections import defaultdict
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Convert MME answers to official evaluation format")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Model output answer file path (JSONL format)")
    parser.add_argument('--seed', default=0, type=int,
                        help="Random seed")
    parser.add_argument('--log_path', type=str, required=True,
                        help="Output directory for converted results")
    parser.add_argument('--data_path', type=str,
                        default=None,
                        help="MME dataset root directory (for getting ground truth)")
    args = parser.parse_args()
    return args


def get_gt(data_path):
    """
    Read ground truth from MME dataset
    Reference: DeCo/eval_tool/convert_answer_to_mme.py
    """
    GT = {}
    if not data_path:
        print("[WARNING] No MME dataset path supplied; ground truth lookup is skipped.")
        return GT
    data_path = Path(data_path).expanduser()
    
    if not data_path.exists():
        print(f"[WARNING] MME dataset path does not exist: {data_path}")
        return GT
    
    for category in os.listdir(data_path):
        category_dir = data_path / category
        if not category_dir.is_dir():
            continue
        
        # Check directory structure
        if (category_dir / 'images').exists():
            image_path = category_dir / 'images'
            qa_path = category_dir / 'questions_answers_YN'
        else:
            image_path = qa_path = category_dir
        
        if not image_path.is_dir():
            continue
        if not qa_path.is_dir():
            continue
        
        # Read questions and answers
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            
            qa_file = qa_path / file
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                question = parts[0]
                                answer = parts[1]
                                GT[(category, file, question)] = answer
                        except Exception as e:
                            print(f"[WARNING] Failed to parse line: {line}, error: {e}")
            except Exception as e:
                print(f"[WARNING] Failed to read file: {qa_file}, error: {e}")
    
    return GT


def main():
    args = get_args()
    
    # Read ground truth
    print(f"[INFO] Reading ground truth: {args.data_path}")
    GT = get_gt(args.data_path)
    print(f"[INFO] Loaded {len(GT)} ground truth entries")
    
    # Create output directory
    result_dir = Path(args.log_path).expanduser()
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {result_dir}")
    
    # Read model answers
    print(f"[INFO] Reading model answers: {args.output_path}")
    answers = []
    with open(args.output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                answers.append(json.loads(line))
            except Exception as e:
                print(f"[WARNING] Failed to parse answer: {line}, error: {e}")
    
    print(f"[INFO] Loaded {len(answers)} model answers")
    
    # Group by category
    results = defaultdict(list)
    for answer in answers:
        question_id = answer.get('question_id', '')
        if not question_id:
            continue
        
        # Parse question_id: "category/image_name.png"
        parts = question_id.split('/')
        if len(parts) < 2:
            print(f"[WARNING] Invalid question_id: {question_id}")
            continue
        
        category = parts[0]
        image_name = parts[-1]
        file = image_name.rsplit('.', 1)[0] + '.txt'
        
        prompt = answer.get('prompt', '')
        text = answer.get('text', '')
        
        results[category].append((file, prompt, text))
    
    print(f"[INFO] Total {len(results)} categories")
    
    # Write converted results
    total_matched = 0
    total_unmatched = 0
    
    for category, cate_tups in results.items():
        output_file = result_dir / f'{category}.txt'
        matched = 0
        unmatched = 0
        
        with open(output_file, 'w', encoding='utf-8') as fp:
            for file, prompt, answer in cate_tups:
                # Clean prompt
                if 'Answer the question using a single word or phrase.' in prompt:
                    prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
                
                # Try to match ground truth
                gt_ans = None
                
                # Try different prompt formats
                prompt_variants = [
                    prompt,
                    prompt + ' Please answer yes or no.',
                    prompt + '  Please answer yes or no.',  # Two spaces
                ]
                
                for p in prompt_variants:
                    if (category, file, p) in GT:
                        gt_ans = GT[(category, file, p)]
                        matched += 1
                        break
                
                if gt_ans is None:
                    # Ground truth not found
                    gt_ans = "UNKNOWN"
                    unmatched += 1
                
                # Write result: file \t prompt \t gt_ans \t answer
                tup = (file, prompt, gt_ans, answer)
                fp.write('\t'.join(tup) + '\n')
        
        total_matched += matched
        total_unmatched += unmatched
        print(f"[INFO] {category}: {matched} matched, {unmatched} unmatched")
    
    print(f"\n[INFO] Conversion complete!")
    print(f"[INFO] Total: {total_matched} matched, {total_unmatched} unmatched")
    print(f"[INFO] Results saved in: {result_dir}")
    
    # Statistics
    print(f"\n[INFO] Category files:")
    for category_file in sorted(result_dir.glob('*.txt')):
        print(f"  - {category_file.name}")


if __name__ == "__main__":
    main()
