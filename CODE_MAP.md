# OWP code map

The release keeps the research implementation and executable wrappers in
separate locations. `owp/` is the importable package; `src/` contains the
benchmark and intervention executors;
`third_party/VideoLLaMA2/` is the bundled runtime used for the transfer
experiment.

| Paper operation | Release entry point | Main symbols |
| --- | --- | --- |
| Four-view conflict diagnosis | `src/intervention.py`, `src/run_owp.py`, and `src/probe_videollama2.py` | `diagnose_view`, `choose_target`, `run_row` |
| Query-conditioned target typing | `src/probe_qwen.py` and `src/probe_videollama2.py` | `online_target_modality_from_question`, `build_base_conflict_budget`, `build_frozen_carriers` |
| Evidence/prior decomposition | `src/intervention.py` and `src/intervention_videollama2.py` | `orthonormal_basis`, `subspace_projection`, `soft_minimal_suppression_delta`, `up_r_projection_delta` |
| Token-head and residual intervention | `src/intervention.py` and `src/intervention_videollama2.py` | `frozen_structured_edit_yesno`, path scaling and residual write-back helpers |
| Qwen2.5-Omni adapter | `owp/models/qwen_omni.py` | `QwenOmniAdapter` |
| Portable data manifests | `tools/build_manifests.py` | AVHBench/CMM manifest generation |

The standard OWP run performs four diagnostic forwards and one edited full-view
forward. All intervention coefficients are stored in
`configs/owp_default.json`; per-sample directions and strengths are computed
from the current frozen model views.
