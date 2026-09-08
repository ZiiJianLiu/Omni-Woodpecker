# Sample requests

`owp_input.jsonl` contains three representative CMM requests covering
visual-only, audio-only, and audio-video input. The four small media files in
`media/` are tracked in this repository, so these requests run after a plain
clone without downloading the CMM benchmark. `owp_expected_output.jsonl` shows
the public output schema. The expected file is a contract fixture, not a fixed
model-regression target.

For integration, call the packaged interface shown in
`../examples/run_inference.py`:

```python
from owp import correct

results = correct("sample_data/owp_input.jsonl", "results/owp_output.jsonl")
print(results[0]["sample_id"], results[0]["answer"])
```
