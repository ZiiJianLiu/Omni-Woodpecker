import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "owp_infer", ROOT / "src" / "owp_infer.py"
)
OWP_INFER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWP_INFER)

from owp import api as OWP_API


class OwpInferContractTest(unittest.TestCase):
    def write_jsonl(self, path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_materialize_validates_before_limiting_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            media = temp / "sample.mp4"
            media.write_bytes(b"test fixture")
            source = temp / "input.jsonl"
            self.write_jsonl(
                source,
                [
                    {
                        "sample_id": "sample-1",
                        "video_path": str(media),
                        "question": "Is speech audible?",
                    },
                    {
                        "sample_id": "sample-2",
                        "video_path": str(media),
                        "question": "",
                    },
                ],
            )

            with self.assertRaisesRegex(ValueError, "no question"):
                OWP_INFER.materialize_input(source, max_rows=1)

    def test_materialize_rejects_duplicate_sample_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            media = temp / "sample.wav"
            media.write_bytes(b"test fixture")
            source = temp / "input.jsonl"
            self.write_jsonl(
                source,
                [
                    {
                        "sample_id": "duplicate",
                        "audio_path": str(media),
                        "question": "Is there music?",
                    },
                    {
                        "sample_id": "duplicate",
                        "audio_path": str(media),
                        "question": "Is there speech?",
                    },
                ],
            )

            with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
                OWP_INFER.materialize_input(source)

    def test_output_is_reordered_and_has_stable_fields(self):
        inputs = [{"sample_id": "first"}, {"sample_id": "second"}]
        outputs = [
            {
                "sample_id": "second",
                "answer": None,
                "status": "error",
                "error": "backend failed",
            },
            {
                "sample_id": "first",
                "answer": "Yes",
                "status": "ok",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            self.write_jsonl(input_path, inputs)
            normalized = OWP_INFER.order_and_validate_outputs(input_path, outputs)

        self.assertEqual(
            [row["sample_id"] for row in normalized], ["first", "second"]
        )
        self.assertEqual(set(normalized[0]), {"sample_id", "answer", "status", "error"})
        self.assertIsNone(normalized[0]["error"])
        self.assertEqual(normalized[1]["error"], "backend failed")

    def test_output_rejects_missing_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.jsonl"
            self.write_jsonl(input_path, [{"sample_id": "first"}, {"sample_id": "second"}])
            with self.assertRaisesRegex(RuntimeError, r"missing=\['second'\]"):
                OWP_INFER.order_and_validate_outputs(
                    input_path,
                    [{"sample_id": "first", "status": "ok", "answer": "No"}],
                )

    def test_public_api_returns_records_and_cleans_implicit_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            media = temp / "sample.mp4"
            media.write_bytes(b"test fixture")
            source = temp / "input.jsonl"
            self.write_jsonl(
                source,
                [{"sample_id": "sample-1", "video_path": str(media), "question": "Is it visible?"}],
            )

            def fake_run(command, cwd, env, check):
                output = Path(command[command.index("--output") + 1])
                output.write_text(
                    json.dumps(
                        {
                            "sample_id": "sample-1",
                            "answer": "Yes",
                            "baseline_answer": "No",
                            "target_modality": "visual",
                            "intervention_applied": True,
                            "status": "ok",
                            "error": None,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            with patch.object(OWP_API.subprocess, "run", side_effect=fake_run):
                records = OWP_API.correct(source)

            self.assertEqual(records[0]["answer"], "Yes")
            self.assertEqual(list(temp.glob(".input.owp_output.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
