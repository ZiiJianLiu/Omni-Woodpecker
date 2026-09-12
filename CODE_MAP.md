# OWP code map

The release keeps the public use-case wrapper and backend implementation
separate. `owp.api.correct` is the importable integration interface and
`src/owp_infer.py` is its CLI equivalent;
`owp/` contains the integration API and shared components; the remaining `src/`
files are internal backend stages and are not separate use cases; and
`third_party/VideoLLaMA2/` is the bundled runtime
needed by the VideoLLaMA2 backend.

| Paper operation | Release entry point | Main symbols |
| --- | --- | --- |
| Public use-case wrapper | `owp.api.correct`, `src/owp_infer.py` | JSONL input/output contract |
| Four-view conflict diagnosis | `src/intervention.py`, `src/run_owp.py`, and `src/probe_videollama2.py` | `diagnose_view`, `choose_target`, `run_row` |
| Query-conditioned target typing | `src/probe_qwen.py` and `src/probe_videollama2.py` | `online_target_modality_from_question`, `build_base_conflict_budget`, `build_frozen_carriers` |
| Evidence/prior decomposition | `src/intervention.py` and `src/intervention_videollama2.py` | `orthonormal_basis`, `subspace_projection`, `soft_minimal_suppression_delta`, `up_r_projection_delta` |
| Token-head and residual intervention | `src/intervention.py` and `src/intervention_videollama2.py` | `frozen_structured_edit_yesno`, path scaling and residual write-back helpers |
| Qwen2.5-Omni adapter | `owp/models/qwen_omni.py` | `QwenOmniAdapter` |
| Runtime asset preparation | `tools/prepare_assets.py` | Hugging Face model/media download and cache-local manifest generation |

The standard OWP run performs four diagnostic forwards and one edited full-view
forward. All intervention coefficients are stored in
`configs/owp_default.json`; per-sample directions and strengths are computed
from the current frozen model views.

`sample_data/` contains the representative request/output fixtures, while
`examples/run_inference.py` demonstrates how an integrator invokes the single
use-case entry point and consumes only final records.
