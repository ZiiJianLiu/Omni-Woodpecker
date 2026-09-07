# Omni-Woodpecker

Training-free cross-modal hallucination correction for frozen omni-modal
language models. The delivery exposes one business use case: give OWP a
binary question and its video/audio context, and receive a corrected `Yes` or
`No` answer record.

## Quick Start

```bash
git clone https://github.com/ZiiJianLiu/Omni-Woodpecker.git
cd Omni-Woodpecker
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python tools/validate_release.py
python -m unittest discover -s tests
```

The package is Level 2: it uses standard PyTorch GPU wheels and requires no
custom CUDA compilation. One CUDA GPU with at least 24 GB memory is recommended
for the 7B BF16 defaults. Use `--dtype float16` when BF16 is unavailable.
Docker is not required; use an isolated Python environment with a
CUDA-compatible PyTorch wheel.

## Input and Output

Input is UTF-8 JSONL, one request per line. Required fields are `sample_id`,
`question`, and at least one of `video_path` or `audio_path`. Paths may be
relative to the repository root. Questions must use a binary `Yes`/`No`
answer space. Three representative CMM requests are included in
`sample_data/owp_input.jsonl`; their media paths are resolved from the CMM
Hugging Face dataset on first use, so benchmark files are not required in the
checkout.

Output is UTF-8 JSONL, one result per input, with `sample_id`, `answer`,
`baseline_answer`, `target_modality`, `intervention_applied`, and `status`.
The record also contains `error` (null for successful requests and a readable
message when `status` is `error`).
`intervention_applied` records whether OWP wrote a hidden/path update; it can
be `true` even when the constrained answer remains unchanged.
Typing, modality diagnosis, hidden-state decomposition, and editing are
internal stages and are not separate use cases.

`sample_data/owp_expected_output.jsonl` is a contract fixture showing the output
schema and representative values. It is not a performance claim: generated
answers are determined by the selected checkpoint and resolved media.

## Run

The default VideoLLaMA2 backend:

```bash
CUDA_VISIBLE_DEVICES=0 python src/owp_infer.py \
  --input sample_data/owp_input.jsonl \
  --output results/owp_output.jsonl \
  --model videollama2 \
  --max-rows 1
```

The Qwen2.5-Omni backend uses the same input/output contract:

```bash
CUDA_VISIBLE_DEVICES=0 python src/owp_infer.py \
  --input sample_data/owp_input.jsonl \
  --output results/owp_output_qwen.jsonl \
  --model qwen \
  --max-rows 1
```

The equivalent complete Python example is `examples/run_inference.py`. It
defaults to the bundled sample and accepts the same input/output paths for an
integrator's own JSONL:

```bash
python examples/run_inference.py --model videollama2 --max-rows 1
# python examples/run_inference.py --input requests.jsonl --output results.jsonl
```

Models are fetched automatically from Hugging Face on first use and cached in
`~/.cache/omni-woodpecker`. Set `OWP_CACHE_DIR` to change the cache location.
CMM media are fetched automatically. AVHBench media require local files or an
equivalent Hugging Face dataset configured with `OWP_AVHBENCH_DATASET_ID`.

## Validation

```bash
python tools/validate_release.py
```

The command validates the source layout, bundled annotations and lightweight
imports. Complete GPU inference requires the selected model and media.
