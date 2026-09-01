# CMM annotation index

This directory contains the CMM annotation index used by Omni-Woodpecker. The
file `all_data_final_reorg.json` has 2,400 questions over 1,200 audio/video
samples. The media are not tracked in the source release. At runtime,
`python tools/prepare_assets.py --dataset cmm` downloads the official
`DAMO-NLP-SG/CMM` snapshot to the user cache, unpacks any archive, and writes a
manifest with the same row schema as `../samples/cmm_owp.jsonl`.

Preserve the dataset citation and usage terms when using or redistributing the
downloaded media.

## Citation

```bibtex
@article{leng2024curse,
  title={The Curse of Multi-Modalities: Evaluating Hallucinations of Large Multimodal Models across Language, Visual, and Audio},
  author={Sicong Leng and Yun Xing and Zesen Cheng and Yang Zhou and Hang Zhang and Xin Li and Deli Zhao and Shijian Lu and Chunyan Miao and Lidong Bing},
  journal={arXiv},
  year={2024}
}
```
