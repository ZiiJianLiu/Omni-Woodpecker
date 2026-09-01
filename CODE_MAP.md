# OWP code map

The release keeps the research implementation and the executable wrappers in
separate locations. `OMhallucination/` is the importable Qwen package;
`src/` contains the paper's benchmark and intervention executors;
`third_party/VideoLLaMA2/` is the bundled runtime used for the transfer
experiment.

| Paper operation | Release entry point | Main symbols |
| --- | --- | --- |
| Four-view conflict diagnosis | `src/run_aaai27_videollama2_owp_5pass_efficiency_20260731.py` | `diagnose_view`, `choose_target`, `carriers_from_cache`, `run_row` |
| Query-conditioned target typing | `src/run_videollama2_strict_online_prior_typing_probe.py` | `online_target_modality_from_question`, `build_base_conflict_budget`, `build_frozen_carriers` |
| Evidence/prior decomposition | `src/run_prior_carrier_suppression_executor.py` | `orthonormal_basis`, `subspace_projection`, `soft_minimal_suppression_delta`, `up_r_projection_delta` |
| Token-head intervention | `src/run_prior_carrier_suppression_executor.py` | `_allpath_head_scaled_output`, `_allpath_token_scaled_output`, head selection helpers |
| Residual write-back | `src/run_videollama2_strict_prior_carrier_executor.py` | `frozen_structured_edit_yesno` |
| Qwen2.5-Omni media/model adapter | `OMhallucination/qwen_omni_adapter.py` | `QwenOmniAdapter` |
| Portable data manifests | `tools/build_manifests.py` | AVHBench/CMM manifest generation |

The standard OWP run performs four diagnostic forwards and one edited full-view
forward. All intervention coefficients are stored in
`configs/owp_default.json`; per-sample directions and strengths are computed
from the current frozen model views.
