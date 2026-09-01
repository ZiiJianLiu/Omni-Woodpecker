# Asset notices

The source release keeps large model checkpoints and benchmark media out of Git.
The runtime resolves them from the Hugging Face repositories recorded in
`owp/assets.py` and stores them in `OWP_CACHE_DIR` (or the
default user cache). The first run therefore needs network access and enough
local cache space; later runs reuse the same snapshots.

- `Qwen/Qwen2.5-Omni-7B`, `DAMO-NLP-SG/VideoLLaMA2.1-7B-AV`, and
  `google/siglip-so400m-patch14-384` are the default model sources. Review each
  upstream model card and license before redistribution or commercial use.
- `DAMO-NLP-SG/CMM` is the default CMM source. Its dataset card and original
  usage terms apply to downloaded annotations and media.
- The repository retains the OWP annotation indexes for protocol inspection.
  AVHBench's official README currently points to a Google Drive media subset;
  the package therefore requires an explicit `OWP_AVHBENCH_DATASET_ID` for a
  verified Hugging Face mirror of the exact release used in the paper.
- `third_party/VideoLLaMA2/` contains the runtime source required to execute
  the remote VideoLLaMA2 checkpoint. Its upstream copyright headers are kept
  in the source files.
- `third_party/HulluEdit/` contains the lightweight steering module needed by
  the optional HulluEdit comparison path; its upstream README is retained.

Before public redistribution outside the paper artifact, review the original
dataset and checkpoint licenses for the intended use and jurisdiction.
