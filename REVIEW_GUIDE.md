# Implementation Audit Guide

The delivery has one public use-case entry point:

```bash
python src/owp_infer.py --input <input.jsonl> --output <output.jsonl> --model videollama2
```

The `--model qwen` option selects the Qwen2.5-Omni backend. The backend keeps
typing, four-view diagnosis, carrier construction and representation editing
inside a temporary run workspace; callers receive only the final output JSONL.

The implementation is organized as:

- `owp/models/qwen_omni.py`: Qwen2.5-Omni model and media adapter.
- `owp/evaluation/`: benchmark loaders and answer-space normalization.
- `owp/modules/`: query-conditioned evidence scoring and attention utilities.
- `src/run_owp.py`, `src/probe_qwen.py`, `src/intervention.py`,
  `src/probe_videollama2.py`, and `src/intervention_videollama2.py`: internal
  backend stages called by `src/owp_infer.py`.
- `src/analyze_attention.py`, `src/causal_consistency.py`, `src/avcd.py`, and
  `src/mad.py`: internal diagnostics used by those stages.

Run `python tools/validate_release.py` before inference. It checks the source
layout and bundled annotations without requiring model downloads.
