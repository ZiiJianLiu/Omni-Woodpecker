# Omni-Woodpecker

Training-free cross-modal hallucination correction for frozen omni-modal
language models. The delivery exposes one business use case: give OWP a
binary question and its video/audio context, and receive a corrected `Yes` or
`No` answer record.

The repository provides one stable integration interface, `owp.correct`, and
one equivalent CLI, `src/owp_infer.py`. Backend-specific probing and
representation editing are internal implementation details.

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
`sample_data/owp_input.jsonl`, and all four media files they reference are
tracked in `sample_data/media/`. A fresh clone can therefore validate the
input/output contract without downloading the full CMM dataset. The model
checkpoint is still downloaded or supplied locally at runtime.

Output is UTF-8 JSONL, one final result per input, with only `sample_id`,
`answer`, `status`, and `error`. `error` is null for successful requests and a
readable message when `status` is `error`. Typing, modality diagnosis,
baseline answers, hidden-state decomposition, and editing remain internal
stages and are not separate use cases or public output fields.

`sample_data/owp_expected_output.jsonl` is a contract fixture showing the output
schema and representative values. It is not a performance claim: generated
answers are determined by the selected checkpoint and resolved media.

## Integration Interface

Call the packaged function when integrating from Python:

```python
from owp import correct

results = correct(
    "sample_data/owp_input.jsonl",
    "results/owp_output.jsonl",
    model="videollama2",
)
for result in results:
    print(result["sample_id"], result["answer"], result["status"])
```

`correct` accepts the same JSONL request contract described below and returns
one final result dictionary per input, in input order. The output file is
optional; omit it to receive results in memory.

## Run

The default VideoLLaMA2 backend:

```bash
CUDA_VISIBLE_DEVICES=0 python src/owp_infer.py \
  --input sample_data/owp_input.jsonl \
  --output results/owp_output.jsonl \
  --model videollama2 \
  --max-rows 0
```

The Qwen2.5-Omni backend uses the same input/output contract:

```bash
CUDA_VISIBLE_DEVICES=0 python src/owp_infer.py \
  --input sample_data/owp_input.jsonl \
  --output results/owp_output_qwen.jsonl \
  --model qwen \
  --max-rows 0
```

The equivalent complete Python example is `examples/run_inference.py`. It
defaults to the bundled sample and accepts the same input/output paths for an
integrator's own JSONL:

```bash
python examples/run_inference.py --model videollama2
# python examples/run_inference.py --input requests.jsonl --output results.jsonl
```

Models are fetched automatically from Hugging Face on first use and cached in
`~/.cache/omni-woodpecker`. Set `OWP_CACHE_DIR` to change the cache location.
The bundled three-row example does not fetch CMM media. For custom CMM rows
that point to the full benchmark, media are fetched automatically. AVHBench
media require local files or an equivalent Hugging Face dataset configured with
`OWP_AVHBENCH_DATASET_ID`.

## Validation

```bash
python tools/validate_release.py
```

The command validates the source layout, bundled annotations and lightweight
imports. Complete GPU inference requires the selected model and media.
