# Omni-Woodpecker

This repository contains the inference code for Omni-Woodpecker (OWP). OWP is
training-free: the documented workflow only downloads checkpoints, prepares
benchmark media, and runs inference.

## 1. Install

```bash
git clone https://github.com/ZiiJianLiu/Omni-Woodpecker.git
cd Omni-Woodpecker

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use a CUDA-compatible PyTorch installation for your GPU before installing the
remaining requirements when the default PyPI wheel is not suitable. OWP's
inference executors require one CUDA GPU.

Check the checkout and its lightweight imports:

```bash
python tools/validate_release.py
```

Use `--skip-imports` if dependencies have not been installed yet.

## 2. Models and data

Model checkpoints are downloaded automatically on first use and stored outside
the repository at `~/.cache/omni-woodpecker/hub`. The defaults are:

| Component | Hugging Face repository |
| --- | --- |
| Qwen2.5-Omni | `Qwen/Qwen2.5-Omni-7B` |
| VideoLLaMA2-AV | `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV` |
| SigLIP tower | `google/siglip-so400m-patch14-384` |

Set `OWP_CACHE_DIR` to change the cache location. Set `HF_TOKEN` only when a
model or dataset requires authentication.

The annotation files are included in `data/`. CMM media are downloaded
automatically from `DAMO-NLP-SG/CMM` when a CMM run starts. The AVHBench media
are not stored in this Git repository; for an AVHBench run, either place the
media under `data/AVHBench/videos/` and `data/AVHBench/audios/`, or set
`OWP_AVHBENCH_DATASET_ID` to a Hugging Face dataset repository containing the
paper's `QA.json`, `videos/`, and `audios/` files:

```bash
export OWP_AVHBENCH_DATASET_ID=<your-hf-user>/<your-avhbench-repository>
```

## 3. Minimal check

Run one example end to end. The script prepares one CMM sample when local
media are absent and downloads the VideoLLaMA2 checkpoint as needed:

```bash
bash run_smoke.sh
```

The prediction is written to `results/owp_smoke.jsonl`.

## 4. Run evaluations

### CMM

Run all rows in the included CMM manifest:

```bash
CUDA_VISIBLE_DEVICES=0 python src/run_aaai27_videollama2_owp_5pass_efficiency_20260731.py \
  --input data/samples/cmm_owp.jsonl \
  --output results/cmm_owp.jsonl \
  --repeat 1
```

For a bounded check, append `--max-rows 10`.

### AVHBench

After making the AVHBench media available as described above, run:

```bash
CUDA_VISIBLE_DEVICES=0 python src/run_aaai27_videollama2_owp_5pass_efficiency_20260731.py \
  --input data/samples/avhbench_owp.jsonl \
  --output results/avhbench_owp.jsonl \
  --repeat 1
```

The runner accepts `--max-rows N` for a smaller run and `--dtype float16` when
BF16 is unavailable.

### Qwen2.5-Omni probe

This probe uses the Qwen2.5-Omni checkpoint and the repository-relative
manifests. It requires the referenced media to be available locally or through
the AVHBench setup above:

```bash
CUDA_VISIBLE_DEVICES=0 python src/run_strict_online_prior_typing_probe.py \
  --records data/manifests/unary_rebalance_manifest.jsonl \
  --output-dir results/qwen_typing \
  --max-samples 16 \
  --device cuda
```

## 5. Outputs and options

Each probe run creates its output directory and writes `records.jsonl` and
`summary.json`. The five-pass runner writes the JSONL path supplied with
`--output`.

To prefetch the default models and CMM media before inference:

```bash
python tools/prepare_assets.py --model all --dataset cmm
```

Configuration defaults are in `configs/owp_default.json`.
