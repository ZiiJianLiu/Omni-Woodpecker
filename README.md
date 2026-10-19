# Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs

<!-- LOGO: <p align="center"><img src="./assets/omni_woodpecker.png" width="75%" height="75%"></p> -->


<font size=7><div align='center' > :grapes: \[[Read our arXiv Paper](https://arxiv.org/abs/XXXX.XXXXX)\] </div></font>

-----------------

<p align="center">
    <img src="./assets/pipeline.jpg" width="96%" height="96%">
</p>

> Recent Omni LLMs (OLLMs) have demonstrated remarkable capabilities in joint audio-visual reasoning, yet cross-modal hallucinations remain a major challenge. Existing mitigation methods mainly operate at the output level, overlooking the entanglement of grounding evidence and learned priors within fused representations. To address this issue, we propose Omni-Woodpecker (OWP), a training-free framework that shifts hallucination mitigation from output-level manipulation to conflict-conditioned representation rebalancing. Specifically, OWP first identifies cross-modal conflicts from complementary modality-specific views. It then employs orthogonal decomposition to separate grounding evidence from competing priors within the hidden states. Finally, OWP adaptively rebalances fused representations by suppressing misleading priors and restoring ignored evidence before decoding. Extensive experiments demonstrate the effectiveness and strong cross-architecture generalization of OWP. Using Qwen2.5-Omni-7B as the backbone, OWP improves performance by 12.9 and 8.6 points on CMM and AVHBench, respectively, while consistently achieving gains of 0.67–2.25 points across four general OLLM reasoning benchmarks. Furthermore, OWP generalizes effectively to VideoLLaMA2-AV, improving performance by 4.6 and 3.9 points on the same benchmarks.

If you have any question, please feel free to email bradyfu24@gmail.com.


## 🌋 Evaluation
We perform experiments based on two baseline models:
- [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), [VideoLLaMA2-AV](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2.1-7B-AV)

The experimental results are shown below. For more details, please check out [our paper](https://arxiv.org/abs/XXXX.XXXXX).

<p align="center">
    <img src="./assets/qualitative.jpg" width="96%" height="96%">
</p>

### 📍 CMM and AVHBench Result

This part focuses on cross-modal hallucinations. PA and HR are CMM perception accuracy and hallucination resistance.

| Backbone | Method | CMM Acc. | CMM PA | CMM HR | AVHBench Acc. | AVHBench F1 |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-Omni-7B | Backbone | 73.2 | 89.5 | 57.0 | 75.1 | 76.3 |
| | AVCD | 74.8 | 83.5 | 66.0 | 74.8 | 72.0 |
| | MAD | 81.2 | **89.7** | 72.8 | 76.0 | 75.0 |
| | HulluEdit | 79.2 | 86.2 | 72.3 | 71.6 | 66.4 |
| | **OWP** | **86.1 (+12.9)** | 86.7 | **85.5** | **83.7 (+8.6)** | **83.3** |
| VideoLLaMA2-AV | Backbone | 76.2 | 87.2 | 65.3 | 70.2 | 72.9 |
| | AVCD | 77.9 | 85.0 | 70.8 | 72.1 | **73.9** |
| | MAD | 79.2 | **88.0** | 70.3 | 70.0 | 71.7 |
| | HulluEdit | 77.2 | 82.2 | 72.2 | 71.7 | 70.1 |
| | **OWP** | **80.8 (+4.6)** | 83.8 | **77.7** | **74.1 (+3.9)** | **73.9** |

### 📍 General Omni-Modal Reasoning Result

This part focuses on general omni-modal reasoning with Qwen2.5-Omni-7B.

| Model | OmniBench | WorldSense | MUSIC-AVQA | AVQA |
| --- | :---: | :---: | :---: | :---: |
| Qwen2.5-Omni-7B | 45.36 | 45.21 | 85.08 | 91.83 |
| **+ OWP** | **47.29 (+1.93)** | **47.46 (+2.25)** | **86.08 (+1.00)** | **92.50 (+0.67)** |


## 🛠️ Preliminary

1. Create conda environment

```bash
conda create -n owp python=3.10
conda activate owp
pip install -r requirements.txt
pip install -e .
```

2. Install required models

- OWP supports two backbones. The checkpoints are downloaded from Hugging Face on first use and cached in `~/.cache/omni-woodpecker` (set `OWP_CACHE_DIR` to change it, or pass a local checkpoint with `--model-path`).

| Backbone | `--model` | Checkpoint |
| --- | --- | --- |
| VideoLLaMA2-AV | `videollama2` | [DAMO-NLP-SG/VideoLLaMA2.1-7B-AV](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2.1-7B-AV) |
| Qwen2.5-Omni-7B | `qwen` | [Qwen/Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) |

- A GPU with at least 24 GB of memory is required for the 7B backbones in BF16 (use `--dtype float16` when BF16 is unavailable).

## ⭐ Usage

**1. Inference**

To correct hallucinations for a set of audio-visual questions, run the inference code as follows:

```Shell
CUDA_VISIBLE_DEVICES=0 python src/owp_infer.py \
        --input sample_data/owp_input.jsonl \
        --output results/owp_output.jsonl \
        --model videollama2
```
Each input line contains `sample_id`, a Yes/No `question`, and at least one of `video_path` and `audio_path`. The corrected answers are saved to the output JSONL file. Use `--model qwen` to run OWP on Qwen2.5-Omni-7B.

***

**2. Python API**

OWP can also be called from Python:

```python
from owp import correct

results = correct("sample_data/owp_input.jsonl", "results/owp_output.jsonl", model="videollama2")
```
A complete example is provided in `examples/run_inference.py`. All intervention coefficients are stored in `configs/owp_default.json`, and [CODE_MAP.md](CODE_MAP.md) shows where each component of the paper is implemented.



## 🌻 Acknowledgement
This repository benefits from [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni), [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2), and [Woodpecker](https://github.com/BradyFU/Woodpecker). Thanks for their awesome works.



## 📑 Citation
If you find our project helpful to your research, please consider citing:
```
@article{liu2026omniwoodpecker,
  title={Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs},
  author={Liu, Zijian and Fu, Chaoyou and Yang, Yan and Shan, Caifeng and He, Ran},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
