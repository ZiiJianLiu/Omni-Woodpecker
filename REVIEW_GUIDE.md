# Implementation audit guide

The release contains two complete OWP inference paths:

1. `src/run_owp.py` runs the VideoLLaMA2 five-pass benchmark workflow.
2. `src/probe_qwen.py` produces Qwen2.5-Omni query-conditioned typing records.
3. `src/intervention.py` performs the Qwen2.5-Omni evidence/prior decomposition
   and token-head plus residual intervention.
4. `src/probe_videollama2.py` and `src/intervention_videollama2.py` expose the
   corresponding VideoLLaMA2 typing and representation-editing stages.

The shared implementation is organized as:

- `owp/models/qwen_omni.py`: Qwen2.5-Omni model and media adapter.
- `owp/evaluation/`: benchmark loaders and answer-space normalization.
- `owp/modules/`: query-conditioned evidence scoring and attention utilities.
- `src/analyze_attention.py`, `src/causal_consistency.py`, `src/avcd.py`, and
  `src/mad.py`: diagnostics used by the intervention executors.

Run `python tools/validate_release.py` before inference. It checks the source
layout and bundled annotations without requiring model downloads.
