# Algorithm: Omni-Woodpecker

## Algorithm Overview

Omni-Woodpecker (OWP) corrects cross-modal hallucinations in a frozen
audio-visual language model. Given a binary question and its video/audio context, it
diagnoses disagreement between complementary modality views, repairs the
evidence-prior mixture in the hidden representation, and returns the corrected
answer without training or changing model weights.

## Use Cases

| No. | Use case | Description |
|---:|---|---|
| 1 | Cross-modal hallucination correction | Read question/media records and return one corrected answer record per input. |

The use case has one public entry point: `src/owp_infer.py`. Typing, view
diagnosis, carrier construction and hidden-state editing are internal stages of
this use case and are not separate delivery interfaces.

---

# Use Case 1: Cross-Modal Hallucination Correction

## 1. Business Problem

An omni-modal language model can follow a language prior or an irrelevant
modality when answering a question about audio-visual content. This use case
preserves the frozen model and repairs the answer before decoding.

## 2. Input

The input is UTF-8 JSONL. Each line is one independent request.

| Item | Description | Format | Example |
|---|---|---|---|
| `sample_id` | Caller-defined stable identifier | string | `demo-0001` |
| `question` | Binary question about the supplied context | string | `Is the car making sound in the audio?` |
| `video_path` | Video file path; use an empty string when unavailable | repository-relative or absolute string | `data/AVHBench/videos/02060.mp4` |
| `audio_path` | Audio file path; use an empty string when unavailable | repository-relative or absolute string | `data/AVHBench/audios/02060.wav` |
| `benchmark` | Optional dataset name used for media resolution | `avhbench` or `cmm` | `avhbench` |
| `reference_answer` | Optional offline evaluation label; never used for intervention | string | `Yes` |

The input must contain at least one valid media stream. Paths may be relative
to the repository root, and the answer space must be `Yes`/`No`. For benchmark
manifests, missing CMM media are fetched
to the user cache on first use. AVHBench media require local files or a dataset
repository configured with `OWP_AVHBENCH_DATASET_ID`.

## 3. Output

The output is UTF-8 JSONL with one line per processed input. Intermediate views,
hidden states, carriers and model logits remain inside the run workspace and
are not part of the public output.

| Item | Description | Format | Example |
|---|---|---|---|
| `sample_id` | Input identifier | string | `demo-0001` |
| `answer` | OWP-corrected answer | `Yes` or `No` | `Yes` |
| `baseline_answer` | Frozen model answer before editing | string | `Yes` |
| `target_modality` | Query-relevant modality inferred by OWP | `audio`/`visual`/`unknown` | `audio` |
| `intervention_applied` | Whether OWP applied a representation edit | boolean | `true` |
| `status` | Processing status | `ok` or `error` | `ok` |
| `error` | Error message when `status=error`; otherwise null | string or null | `null` |

## 4. Complete Example

Use the representative requests in `sample_data/owp_input.jsonl` and run one backend:

```bash
CUDA_VISIBLE_DEVICES=0 python src/owp_infer.py \
  --input sample_data/owp_input.jsonl \
  --output results/owp_output.jsonl \
  --model videollama2 \
  --max-rows 1
```

The same use case is available through Qwen2.5-Omni:

```bash
CUDA_VISIBLE_DEVICES=0 python src/owp_infer.py \
  --input sample_data/owp_input.jsonl \
  --output results/owp_output_qwen.jsonl \
  --model qwen \
  --max-rows 1
```

Read the final result without depending on internal files:

```python
import json

with open("results/owp_output.jsonl", encoding="utf-8") as handle:
    for line in handle:
        result = json.loads(line)
        print(result["sample_id"], result["answer"], result["status"])
```

The same call is packaged as `examples/run_inference.py` for integration
testing. It defaults to the bundled requests and accepts custom JSONL paths:

```bash
python examples/run_inference.py --model videollama2 --max-rows 1
# python examples/run_inference.py --input requests.jsonl --output results.jsonl
```

## 5. Runtime Environment

- **Level**: Level 2 (standard Python/PyTorch GPU inference).
- **Python**: 3.10 or newer.
- **Core dependencies**: PyTorch, torchaudio, torchvision, Transformers,
  Accelerate, Hugging Face Hub, Qwen Omni utilities, decord, librosa,
  OpenCV, and soundfile. Install the pinned lower bounds from
  `requirements.txt`.
- **System**: Linux is recommended; no custom CUDA extension or source build
  is required.
- **Network**: Required on first run to fetch model checkpoints and CMM media;
  cached runs can operate offline when all requested files are present.
- **GPU**: One CUDA GPU with at least 24 GB memory for the 7B BF16 default
  configuration. Use `--dtype float16` when BF16 is unavailable.
- **Models**: `Qwen/Qwen2.5-Omni-7B` or
  `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV`, downloaded automatically into the user
  cache. Override with `--model-path` when a local checkpoint is available.
- **Container**: Docker is not required for this Level 2 package; the release
  uses standard PyTorch wheels and has no custom CUDA extension.

## 6. Expected Output Example

For the representative requests in `sample_data/owp_input.jsonl`,
`sample_data/owp_expected_output.jsonl` records a valid output shape:

```json
{"sample_id":"cmm:00000","answer":"Yes","baseline_answer":"Yes","target_modality":"visual","intervention_applied":false,"status":"ok","error":null}
```

The answer values depend on the selected checkpoint and media. The fixture is
therefore a schema example rather than a fixed model-regression target.
Integration validation should check that every input identifier appears
exactly once, all required output fields are present, and `status` is `ok`.
