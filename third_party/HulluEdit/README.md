# 🚀 HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models

<p align="center">
  <a href="https://arxiv.org/pdf/2602.22727">📄arXiv</a> •
  <a href="https://opensource.org/licenses/MIT">📜License</a> •
  <a href="https://www.python.org/">🐍Python</a> •
  <a href="https://pytorch.org/">🔥PyTorch</a>
</p>

<p align="center"><strong>CVPR 2026</strong></p>

> [**HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models**](https://arxiv.org/pdf/2602.22727) 
>
> Yangguang Lin<sup>1</sup>, Quan Fang<sup>1</sup>, Yufei Li<sup>1</sup>, Jiachen Sun<sup>1</sup>, Junyu Gao<sup>2</sup>, Jitao Sang<sup>3</sup> 
>
> <sup>1</sup>Beijing University of Posts and Telecommunications, <sup>2</sup>Institute of Automation, Chinese Academy of Sciences, <sup>3</sup>Beijing Jiaotong University

---

## 🎯 Overview

We introduce **HulluEdit**, a novel single-pass evidence-consistent subspace editing method designed to effectively mitigate object hallucinations in Large Vision-Language Models (LVLMs). HulluEdit mitigates hallucinations by decomposing model hidden states into orthogonal subspaces—visual evidence, conflicting priors, and residual uncertainty—enabling selective suppression of hallucinatory patterns without interfering with visual grounding.

The key components of our approach are:

- 🔬 **Evidence Subspace Extraction**: Constructs a visual evidence subspace from image-conditioned token representations by analyzing hidden states through SVD, identifying the main directions of visual grounding
- 🛡️ **Anti-Prior Subspace Construction**: Builds an anti-prior subspace to suppress language-only biases that lead to hallucinations
- ✏️ **Closed-Form Weight Editing**: Applies closed-form editing to MLP weights in the LLM backbone, projecting to the evidence subspace while suppressing anti-prior directions

![Model Diagram](docs/model.png)

---

## ✨ Key Features

- 🎨 **Single-Pass Editing** — Edit model weights in a single forward pass without requiring additional reference models
- 🔒 **Evidence-Based** — Grounded in visual evidence extraction with theoretical guarantees
- 🚀 **Closed-Form Solution** — No iterative optimization required, ensuring computational efficiency
- 📈 **Effective** — Significantly reduces object hallucinations while preserving general capabilities

---

## 🛠️ Getting Started

### 📦 Environment Installation

```bash
# Clone the repository
git clone https://github.com/VioAgnes/HulluEdit.git
cd HulluEdit

# Create and activate environment
conda create -n hullu python=3.10
conda activate hullu

# Install dependencies
pip install -r requirements.txt
```

### 🤖 Model Setup

Our method generalizes well and supports the following models. We recommend starting with **LLaVA-1.5-7B**:

| Model                             | Download Link                                                |
| :-------------------------------- | :----------------------------------------------------------- |
| 🟦 **LLaVA-1.5 7B**                | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) |
| 🟦 **LLaVA-1.5 13B**               | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) |
| 🟨 **MiniGPT-4 (LLaMA-2 Chat 7B)** | [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) |
| 🟨 **MiniGPT-4 LLM**               | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| 🟪 **mPLUG-Owl2**                  | [MAGAer13/mplug-owl2-llama2-7b](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b) |
| 🟧 **QwenVL**                      | [Qwen/Qwen-VL](https://huggingface.co/Qwen/Qwen-VL)          |
| 🟧 **QwenVL 2.5**                  | [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |

### 📊 Dataset Preparation

#### MSCOCO 2014

Download from [COCO Dataset](https://cocodataset.org/#download) and organize as:

```
DATA/
├── annotations/
│   ├── instances_val2014.json
│   └── captions_val2014.json
└── val2014/
    ├── COCO_val2014_000000000042.jpg
    └── ...
```

#### POPE Dataset

Download from [POPE](https://github.com/shikiw/OPERA/blob/main/pope_coco) and place under:

```
DATA/POPE/
├── pope_random.json
├── pope_popular.json
└── pope_adversarial.json
```

---

## 📈 Evaluation

### 🎯 Benchmarks

HulluEdit is designed to be evaluated on multiple hallucination benchmarks: **POPE, AMBER, Hallu-Bench, MME, CHAIR, LlaVA-Bench, and MMVet**.

### 🔬 POPE Evaluation

> 💡 **You can even use an RTX 4090 (GPU memory ≥ 20GB) for evaluation.**

```bash
# Navigate to project directory
cd /Path/to/HulluEdit
conda activate hullu

# Run POPE evaluation for LLaVA
bash scripts/run_pope_llava.sh
```

### 📂 Project Structure

> 💡 **Core Implementation:** The main implementation of HulluEdit is located in the `hulluedit/` directory.

```
HulluEdit/
├── hulluedit/              # 🔥 Core implementation
│   ├── steer.py           # Main weight editing method
│   ├── steer_alternatives.py  # Alternative steering implementations
│   ├── engines/           # Model engines (LLaVA, MiniGPT-4, mPLUG-Owl2)
│   ├── datasets/           # Dataset implementations
│   ├── eval/              # Evaluation scripts
│   └── analysis/          # Analysis tools
├── configs/                # Configuration files
│   ├── pope_llava_run.yaml
│   └── pope_mplug.yaml
├── scripts/                # Evaluation scripts
├── docs/                   # Documentation & figures
├── DATA/                   # Dataset directory
└── README.md
```

---

## 📝 Citation

```bibtex
@inproceedings{hulluedit2026,
  title={HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models},
  author={Lin, Yangguang and Fang, Quan and Li, Yufei and Sun, Jiachen and Gao, Junyu and Sang, Jitao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 🙏 Acknowledgments

This work is built upon the excellent foundations of [DeCo](https://github.com/zjunlp/Deco) and [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit). We sincerely thank the authors for their valuable contributions to the research community.

---

⭐ **Star us on GitHub** if this project helps you!
