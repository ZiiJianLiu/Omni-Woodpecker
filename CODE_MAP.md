# Code structure

`owp.api.correct` is the Python entry point and `src/owp_infer.py` is the
equivalent command-line entry point. `owp/` contains the API and shared
components, `src/` contains the OWP pipeline, and `third_party/VideoLLaMA2/`
contains the runtime needed by the VideoLLaMA2 backbone.

| Paper component | Files | Main symbols |
| --- | --- | --- |
| Entry points | `owp.api.correct`, `src/owp_infer.py` | JSONL input and output |
| Four-view conflict diagnosis | `src/intervention.py`, `src/run_owp.py`, `src/probe_videollama2.py` | `diagnose_view`, `choose_target`, `run_row` |
| Query-conditioned target typing | `src/probe_qwen.py`, `src/probe_videollama2.py` | `online_target_modality_from_question`, `build_base_conflict_budget`, `build_frozen_carriers` |
| Evidence-prior decomposition | `src/intervention.py`, `src/intervention_videollama2.py` | `orthonormal_basis`, `subspace_projection`, `soft_minimal_suppression_delta`, `up_r_projection_delta` |
| Token-head and residual rebalancing | `src/intervention.py`, `src/intervention_videollama2.py` | `frozen_structured_edit_yesno`, path scaling and residual write-back helpers |
| Qwen2.5-Omni backbone | `owp/models/qwen_omni.py` | `QwenOmniAdapter` |
| Model and media preparation | `tools/prepare_assets.py` | Hugging Face download and local cache manifest |

A standard OWP run performs four diagnostic forward passes and one edited
full-view forward pass. All intervention coefficients are stored in
`configs/owp_default.json`; per-sample directions and strengths are computed
from the views of the frozen model.

`sample_data/` contains example requests and the output format, and
`examples/run_inference.py` shows a minimal end-to-end run.
