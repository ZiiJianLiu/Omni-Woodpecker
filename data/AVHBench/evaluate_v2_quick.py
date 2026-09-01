#!/usr/bin/env python
"""
快速评估V2版本：对抗No偏向策略

只运行少量样本（10-20个）来快速验证改进效果
"""
import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from OMhallucination.config import PipelineConfig
from OMhallucination.qwen_omni_adapter import QwenOmniAdapter
from OMhallucination.modules.modality_extractor import ModalityExtractor
from OMhallucination.modules.token_level_suppressor_v2 import ImprovedTokenLevelSuppressor

# 只显示关键日志
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("v2_quick_eval")
logger.setLevel(logging.INFO)

# 读取之前的结果来对比
PREV_RESULTS = ROOT / "results" / "enhanced_comparison" / "raw_results.jsonl"
BENCHMARK_JSON = ROOT / "data" / "AVHBench" / "QA.json"
VIDEO_DIR = ROOT / "data" / "AVHBench" / "videos"

def main():
    print("="*70)
    print("V2版本快速评估：对抗No偏向策略")
    print("="*70)

    # 加载之前的结果
    prev_results = {}
    if PREV_RESULTS.exists():
        with open(PREV_RESULTS) as f:
            for line in f:
                data = json.loads(line)
                prev_results[data['video_id']] = data

    # 初始化模型
    logger.info("初始化模型...")
    config = PipelineConfig()
    adapter = QwenOmniAdapter(
        model_path=config.model_path,
        device=config.device,
    )
    extractor = ModalityExtractor(device='cpu')

    # 初始化V2抑制器
    suppressor = ImprovedTokenLevelSuppressor(
        emotion_conflict_threshold=0.25,
        content_conflict_threshold=0.20,
        yes_boost_strength=2.5,  # 更激进的增强
        no_penalty_strength=1.5,  # 更激进的惩罚
    )

    # 加载benchmark数据
    with open(BENCHMARK_JSON) as f:
        benchmark_data = json.load(f)

    # 只测试前20个样本
    samples = benchmark_data[:20]

    results = {
        'baseline': {'correct': 0, 'yes_count': 0, 'no_count': 0},
        'v2': {'correct': 0, 'yes_count': 0, 'no_count': 0, 'suppressed': 0},
    }

    logger.info(f"开始评估 {len(samples)} 个样本...")

    for i, item in enumerate(samples, 1):
        video_id = item['video_id']
        video_path = str(VIDEO_DIR / f"{video_id}.mp4")
        question = item['question']
        label = item['label']

        # 获取之前的baseline结果
        if video_id in prev_results:
            baseline_ans = prev_results[video_id]['baseline']['text_answer'].strip().lower()
        else:
            # 如果没有之前的结果，跳过
            logger.warning(f"[{i}/{len(samples)}] {video_id}: 没有baseline结果，跳过")
            continue

        # 提取特征
        try:
            features = extractor.extract(video_path)
        except Exception as e:
            logger.warning(f"[{i}/{len(samples)}] {video_id}: 特征提取失败 - {e}")
            continue

        # 检测冲突
        conflict_signal = suppressor.detect_conflict(
            visual_emotion=features.visual_emotion or 'neutral',
            audio_emotion=features.audio_emotion or 'neutral',
            visual_emotion_conf=features.visual_emotion_conf or 0.0,
            audio_emotion_conf=features.audio_emotion_conf or 0.0,
            visual_objects=features.visual_objects or [],
            asr_text=features.asr_text or '',
        )

        # V2生成
        try:
            if conflict_signal.should_suppress:
                # 创建logits处理器
                logits_processor = suppressor.create_logits_processor(
                    tokenizer=adapter._processor.tokenizer,
                    conflict_signal=conflict_signal,
                )

                # 使用底层生成方法
                messages = adapter._build_messages(video_path, question, emotion_constraint=None)
                audio_array = adapter._load_audio(video_path) if Path(video_path).exists() else None
                v2_ans = adapter._generate(
                    messages=messages,
                    audio_array=audio_array,
                    max_new_tokens=10,
                    logits_processor=logits_processor,
                ).strip().lower()

                results['v2']['suppressed'] += 1
            else:
                # 无冲突，标准生成
                v2_ans = adapter.answer(video_path, question, max_new_tokens=10).strip().lower()
        except Exception as e:
            logger.warning(f"[{i}/{len(samples)}] {video_id}: V2推理失败 - {e}")
            v2_ans = baseline_ans  # 失败时使用baseline结果

        # 统计结果
        baseline_correct = ('yes' in baseline_ans and label == 'Yes') or ('no' in baseline_ans and label == 'No')
        v2_correct = ('yes' in v2_ans and label == 'Yes') or ('no' in v2_ans and label == 'No')

        if baseline_correct:
            results['baseline']['correct'] += 1
        if v2_correct:
            results['v2']['correct'] += 1

        if 'yes' in baseline_ans:
            results['baseline']['yes_count'] += 1
        else:
            results['baseline']['no_count'] += 1

        if 'yes' in v2_ans:
            results['v2']['yes_count'] += 1
        else:
            results['v2']['no_count'] += 1

        # 打印进度
        status = "✓" if v2_correct else "✗"
        supp_mark = "[SUPP]" if conflict_signal.should_suppress else ""
        logger.info(
            f"[{i}/{len(samples)}] {video_id} {status} {supp_mark} "
            f"label={label}, baseline={baseline_ans[:3]}, v2={v2_ans[:3]}, "
            f"conflict={conflict_signal.overall_conflict_score:.2f}"
        )

    # 打印结果
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)

    total = len(samples)
    baseline_acc = results['baseline']['correct'] / total * 100
    v2_acc = results['v2']['correct'] / total * 100
    improvement = v2_acc - baseline_acc

    print(f"\nBaseline:")
    print(f"  准确率: {results['baseline']['correct']}/{total} = {baseline_acc:.1f}%")
    print(f"  回答分布: Yes={results['baseline']['yes_count']}, No={results['baseline']['no_count']}")

    print(f"\nV2 (对抗No偏向):")
    print(f"  准确率: {results['v2']['correct']}/{total} = {v2_acc:.1f}%")
    print(f"  回答分布: Yes={results['v2']['yes_count']}, No={results['v2']['no_count']}")
    print(f"  触发抑制: {results['v2']['suppressed']}/{total}")

    print(f"\n改进:")
    print(f"  准确率提升: {improvement:+.1f}%")

    if improvement > 0:
        print(f"\n✓ V2版本有效！准确率提升了 {improvement:.1f}%")
        print("  建议运行完整的100样本评估")
    else:
        print(f"\n✗ V2版本效果不佳，准确率下降了 {abs(improvement):.1f}%")
        print("  需要进一步调整参数")

    print("="*70)

if __name__ == '__main__':
    main()
