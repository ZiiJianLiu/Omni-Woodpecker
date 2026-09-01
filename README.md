# Omni-Woodpecker (OWP)

This directory is the source release of Omni-Woodpecker (OWP), a training-free
representation intervention for query-dependent cross-modal hallucinations in
omni-modal language models. GitHub stores the method code, configurations,
annotations, and small manifests. Large checkpoints and benchmark media are
resolved automatically from Hugging Face on first use and kept in a user cache;
they are not part of the Git history.

## Quick Start

Run these commands from the `omni-woodpecker` checkout root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/validate_release.py
bash run_smoke.sh
```

The validator is CPU-safe. The smoke launcher requires one CUDA GPU and
automatically resolves the VideoLLaMA2-AV checkpoint and its media sample.

## Automatic assets

The default model sources are `Qwen/Qwen2.5-Omni-7B`,
`DAMO-NLP-SG/VideoLLaMA2.1-7B-AV`, and
`google/siglip-so400m-patch14-384`. A missing repository-relative model path is
interpreted as the corresponding Hugging Face model id, so existing local
checkouts remain valid and source-only checkouts download on demand.

```bash
# Optional: choose a cache location outside the repository.
export OWP_CACHE_DIR="$HOME/.cache/omni-woodpecker"

# Optional explicit preparation. The normal model-loading entry points perform
# the same resolution lazily, so this step is not required before inference.
python tools/prepare_assets.py --model all
```

The official CMM dataset is available as `DAMO-NLP-SG/CMM` and can be prepared
with:

```bash
python tools/prepare_assets.py --dataset cmm
```

The official AVHBench README currently distributes its media through a Google
Drive subset rather than a verified Hugging Face mirror. To reproduce the
paper's 6,408-row annotation/media release without a manual download, publish
that exact release in a Hugging Face dataset repository and set
`OWP_AVHBENCH_DATASET_ID` to its repository id. OWP deliberately does not
substitute a different dataset with a similar name.

## Method

For each query, OWP evaluates four frozen views: full audiovisual input,
audio-only, video-only, and text-only. The view agreement and the query
condition determine which perceptual stream should be protected. Hidden states
from the evidence layers (12, 16, 20) provide a target-evidence direction;
late states (24, 26, 27) provide an orthogonal competing-prior direction. OWP
then allocates a non-negative correction budget between token-head paths and
the residual stream. The language head and all backbone parameters remain
frozen. The standard evaluation path uses five forward passes per example
(four diagnostic views plus one edited full view).

The implementation is split into three auditable pieces:

| Component | Entry point |
| --- | --- |
| Four-view diagnosis and cached five-pass execution | `src/run_aaai27_videollama2_owp_5pass_efficiency_20260731.py` |
| Query-conditioned target typing and carrier construction | `src/run_videollama2_strict_online_prior_typing_probe.py` |
| Orthogonal decomposition, path intervention, and residual write-back | `src/run_prior_carrier_suppression_executor.py` and `src/run_videollama2_strict_prior_carrier_executor.py` |
| Qwen2.5-Omni adapter | `OMhallucination/qwen_omni_adapter.py` |
| VideoLLaMA2 runtime | `third_party/VideoLLaMA2/` |

## Included assets

All paths below are relative to this directory.

| Asset | Contents |
| --- | --- |
| `models/` | Optional local development cache; ignored by the source-only Git release |
| `data/AVHBench/QA.json` | 6,408-row annotation index retained for protocol inspection |
| `data/CMM/all_data_final_reorg.json` | 2,400-row annotation index retained for protocol inspection |
| `data/samples/` | Example manifests and the one-row smoke schema (annotations only; media are resolved at runtime) |
| `data/manifests/` | Small materialized manifests used by diagnostic executors |
| `third_party/HulluEdit/` | Lightweight baseline steering implementation used by the comparison executor |

Runtime media are stored under `OWP_CACHE_DIR` (or the default user cache), not
under the repository. The release validator accepts both this source-only
layout and an optional local development bundle.

## Release scope

This bundle covers the OWP main evaluation path: Qwen2.5-Omni-7B, the
VideoLLaMA2-AV transfer path, CMM, and AVHBench. It also includes the local OWP
executors and the lightweight HulluEdit steering code used by the comparison
scripts. Research-only assets outside this path (for example, auxiliary
backbones or unrelated external benchmark pools) are not required by the
documented commands and are intentionally not represented as bundled data.

## Environment

The tested environment is Python 3.10/3.12 with CUDA, PyTorch 2.7.1,
Transformers 4.55.0, Accelerate 1.7.0, and the packages listed in
`requirements.txt`. A version snapshot for the validated core environment is
provided in `requirements.lock`. Install the dependencies inside a fresh environment:

```bash
# Run from the release checkout root.
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# For the validated CUDA 12.6 environment, use:
# python -m pip install -r requirements.lock
```

Inference is CUDA-only in the supplied executors. Use BF16 on a GPU with enough
memory for the selected checkpoint; the VideoLLaMA2 path also loads the SigLIP
tower from Hugging Face when it is not available locally. The first model/data
preparation can require tens of gigabytes of cache space, but the GitHub
checkout itself remains small.

Validate the copied release before running a model:

```bash
python tools/validate_release.py
```

Use `--skip-imports` when checking files on a machine where the Python
dependencies are not installed yet. To validate a development checkout that
still contains all local media and weights, add `--require-bundled-assets`.

The shortest end-to-end check is available as a repository-relative launcher;
it can be called from any current working directory after the dependencies are
installed:

```bash
bash run_smoke.sh
```

The Python commands below are intended to be run after `cd` into this release
directory. The launcher is the only command that intentionally changes into its
own directory, so it remains convenient to call from elsewhere.

## Training and testing

OWP is training-free: it does not optimize weights or require preference data.
The training entry point records this protocol and validates the configuration;
it intentionally performs no parameter update.

```bash
python tools/train.py --config configs/owp_default.json
```

The command writes `results/training_status.json` with
`training_required=false` and `parameters_updated=false`.

Run the one-row VideoLLaMA2 smoke test (requires one CUDA GPU). Missing model
files are downloaded automatically:

```bash
CUDA_VISIBLE_DEVICES=0 python src/run_aaai27_videollama2_owp_5pass_efficiency_20260731.py \
  --input data/samples/owp_smoke.jsonl \
  --output results/owp_smoke.jsonl \
  --repeat 1 --max-rows 1
```

Run the complete AVHBench directional and AV Matching protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python src/run_aaai27_videollama2_owp_5pass_efficiency_20260731.py \
  --input data/samples/avhbench_owp.jsonl \
  --output results/avhbench_owp.jsonl \
  --repeat 1
```

The same executor accepts `data/samples/cmm_owp.jsonl` for the 2,400 CMM
questions; if media are absent, it automatically prepares the CMM snapshot in
the user cache. `--max-rows N` is useful for a bounded run; `--dtype float16` can be
used when BF16 is unavailable. The Qwen2.5-Omni diagnostic and audit entry
points use repository-relative manifests and the same local-or-Hugging-Face
model resolution, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python src/run_strict_online_prior_typing_probe.py \
  --records data/manifests/unary_rebalance_manifest.jsonl \
  --max-samples 16 \
  --output-dir results/qwen_typing_smoke
```

All documented paths are resolved from the release directory or from the
launcher location. The code dynamically discovers that checkout root at runtime;
no workstation-specific absolute path is embedded in the package, and the
runtime does not rewrite manifests into machine-specific paths.

## Results reported in the paper

The table below reproduces the paper's method comparison. Values are accuracy
unless marked PA, HR, or F1. Bold values are the best method within each
backbone block.

| Backbone / method | CMM overall | CMM PA | CMM HR | AVHBench overall | AVHBench F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Omni backbone | 73.2 | 89.5 | 57.0 | 75.1 | 76.3 |
| Qwen2.5-Omni + MAD | 81.2 | 89.7 | 72.8 | 76.0 | 75.0 |
| Qwen2.5-Omni + HulluEdit | 79.2 | 86.2 | 72.3 | 71.6 | 66.4 |
| **Qwen2.5-Omni + OWP** | **85.7** | 85.8 | **85.5** | **83.7** | **83.3** |
| VideoLLaMA2-AV backbone | 76.2 | **87.2** | 65.3 | 70.2 | 72.9 |
| VideoLLaMA2-AV + MAD | 79.2 | **88.0** | 70.3 | 70.0 | 71.7 |
| VideoLLaMA2-AV + HulluEdit | 77.2 | 82.2 | 72.2 | 71.7 | 70.1 |
| **VideoLLaMA2-AV + OWP** | **79.9** | 82.2 | **77.7** | **72.1** | **73.9** |

Relative to the frozen backbone, OWP improves CMM by 12.5 points and AVHBench
by 8.6 points on Qwen2.5-Omni, and by 3.7 and 1.9 points on VideoLLaMA2-AV.
The paper also reports 74.5 to 81.1 on AV Matching under the general
omni-modal transfer configuration, together with an inference cost of five
forward calls for OWP.

## Reproducibility notes

- `configs/owp_default.json` records the frozen layers and coefficients used by
  the main protocol.
- `tools/build_manifests.py` regenerates JSONL manifests from local annotation
  copies; `tools/prepare_assets.py` performs the equivalent operation after a
  Hugging Face dataset snapshot has been cached.
- `tools/validate_release.py` checks source files, annotation counts, optional
  local media/model payloads, and lightweight imports.
- `docs/ASSET_NOTICES.md` records the provenance and redistribution notes for
  the remote assets and their licenses.

## GitHub packaging

The recommended GitHub artifact is this source-only checkout. Model and media
directories are ignored, and the runtime obtains them from Hugging Face. This
avoids GitHub's per-file and repository storage limits while preserving a
single-command first run for readers.

To initialize and publish this directory as a new repository, first create an
empty GitHub repository (do not pre-create its README), then run the following
from this directory:

```bash
git init -b main
git add .gitattributes .gitignore README.md
git add -A
git commit -m "Initial Omni-Woodpecker release"
git remote add origin https://github.com/<USER>/<REPOSITORY>.git
git push -u origin main
```

No model or dataset binary is staged by these commands. Never put a Hugging
Face or GitHub access token in a command or commit; use `HF_TOKEN` in the
environment only when a gated or private repository requires it.
