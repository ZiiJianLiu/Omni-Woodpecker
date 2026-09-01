# Implementation audit guide

1. Start with `src/run_aaai27_videollama2_owp_5pass_efficiency_20260731.py`.
   `run_row` executes full, audio-only, visual-only, and text-only views and
   then performs one edited full-view pass.
2. Inspect `choose_target` and `carriers_from_cache` in the same file. They use
   the question, branch predictions, attention, and hidden states; reference
   labels are not read by the runtime path.
3. Inspect `build_base_conflict_budget` and `build_frozen_carriers` in
   `src/run_videollama2_strict_online_prior_typing_probe.py` for the target
   evidence carrier, orthogonal competing carrier, and conflict budget.
4. Inspect `frozen_structured_edit_yesno` in
   `src/run_videollama2_strict_prior_carrier_executor.py` for token-head path
   scaling and same-layer residual write-back.
5. Run `python tools/validate_release.py` to verify that all code, shards,
   annotations, and media are present before loading a checkpoint.

OWP is inference-only. The supplied `tools/train.py` entry point is a
deliberate no-op that records this fact rather than silently changing model
weights.
