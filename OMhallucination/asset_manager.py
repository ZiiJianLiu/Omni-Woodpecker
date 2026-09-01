"""Resolve OWP models and benchmark assets from local paths or Hugging Face.

The source release deliberately keeps large checkpoints and media outside the
Git repository.  A local checkout is still accepted when it contains the
bundled assets (useful for development); otherwise the same argument may be a
Hugging Face model or dataset id and the asset is fetched into a user cache.
"""
from __future__ import annotations

import json
import os
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MODEL_REPOS = {
    "qwen": "Qwen/Qwen2.5-Omni-7B",
    "videollama2": "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
    "siglip": "google/siglip-so400m-patch14-384",
}

DATASET_REPOS = {
    "cmm": "DAMO-NLP-SG/CMM",
    # The official AVHBench release currently distributes its media through
    # Google Drive.  Set OWP_AVHBENCH_DATASET_ID after publishing an equivalent
    # dataset repository to Hugging Face; do not substitute another benchmark.
    "avhbench": "",
}


def cache_root() -> Path:
    configured = os.environ.get("OWP_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "omni-woodpecker").resolve()


def model_repo(kind: str) -> str:
    key = str(kind).strip().lower()
    env_name = {
        "qwen": "OWP_QWEN_MODEL_ID",
        "videollama2": "OWP_VIDEOLLAMA2_MODEL_ID",
        "siglip": "OWP_SIGLIP_MODEL_ID",
    }.get(key)
    if env_name:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return MODEL_REPOS[key]


def dataset_repo(kind: str) -> str:
    key = str(kind).strip().lower()
    env_name = {
        "cmm": "OWP_CMM_DATASET_ID",
        "avhbench": "OWP_AVHBENCH_DATASET_ID",
    }[key]
    value = os.environ.get(env_name, "").strip()
    return value or DATASET_REPOS[key]


def _known_local_kind(value: str) -> str | None:
    name = Path(value).name.lower()
    if "qwen2.5-omni" in name or name == "qwen2.5-omni-7b":
        return "qwen"
    if "videollama2" in name:
        return "videollama2"
    if "siglip" in name:
        return "siglip"
    return None


def _bundled_model_path(kind: str) -> Path:
    root = Path(__file__).resolve().parents[1] / "models"
    names = {
        "qwen": "Qwen2.5-Omni-7B",
        "videollama2": "VideoLLaMA2.1-7B-AV",
        "siglip": "siglip-so400m-patch14-384",
    }
    return root / names[str(kind).lower()]


def _hf_snapshot(repo_id: str, *, repo_type: str, destination: Path | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "Hugging Face assets are not present locally. Install the release "
            "dependencies (huggingface_hub) or provide a local asset path."
        ) from exc

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "cache_dir": str(cache_root() / "hub"),
        "token": os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
    }
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        # local_dir makes the result independent of the internal HF cache
        # layout, which is important for dataset manifests.
        kwargs["local_dir"] = str(destination)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        return Path(snapshot_download(**kwargs)).expanduser().resolve()
    except Exception as exc:
        token_hint = " (the repository may require an HF token)" if "gated" in str(exc).lower() else ""
        raise RuntimeError(f"Could not download Hugging Face {repo_type} '{repo_id}'{token_hint}: {exc}") from exc


def resolve_model_path(value: str | os.PathLike[str] | None, *, kind: str | None = None) -> str:
    """Return a usable local model directory, downloading an HF snapshot if needed."""
    raw = str(value).strip() if value is not None else ""
    inferred_kind = kind or _known_local_kind(raw)
    if not raw:
        if inferred_kind is None:
            raise ValueError("A model path or model kind is required")
        raw = model_repo(inferred_kind)

    # Development bundles are preferred even when the caller supplied the
    # canonical repository id.  This keeps validation offline and avoids a
    # needless network request on machines that already have the payload.
    if inferred_kind is not None and raw == model_repo(inferred_kind):
        bundled = _bundled_model_path(inferred_kind)
        if bundled.is_dir() and (bundled / "config.json").is_file():
            return str(bundled.resolve())

    candidate = Path(raw).expanduser()
    if candidate.is_dir():
        return str(candidate.resolve())

    # Repository-relative defaults (models/...) are mapped to their canonical
    # HF id when the source-only checkout has no bundled models directory.
    if inferred_kind is None:
        inferred_kind = _known_local_kind(raw)
    if inferred_kind is not None and (
        raw.startswith("models/")
        or raw.endswith(".safetensors")
        or candidate.name.lower() in {
            "qwen2.5-omni-7b",
            "videollama2.1-7b-av",
            "siglip-so400m-patch14-384",
        }
    ):
        raw = model_repo(inferred_kind)
    elif candidate.is_absolute():
        raise FileNotFoundError(f"Model path does not exist: {candidate}")

    return str(_hf_snapshot(raw, repo_type="model"))


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Refusing archive path outside destination: {member.filename}")
        handle.extractall(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Refusing archive path outside destination: {member.name}")
        if "filter" in tarfile.TarFile.extractall.__code__.co_varnames:
            handle.extractall(destination, filter="data")
        else:  # Python 3.10/3.11
            handle.extractall(destination)


def extract_archives(root: Path) -> list[Path]:
    """Extract archives found below *root* and return the extracted archives."""
    extracted: list[Path] = []
    for archive in sorted(root.rglob("*")):
        if not archive.is_file() or archive.name.startswith("."):
            continue
        suffix = archive.name.lower()
        if suffix.endswith(".zip"):
            _safe_extract_zip(archive, archive.parent)
            extracted.append(archive)
        elif suffix.endswith((".tar", ".tar.gz", ".tgz")):
            _safe_extract_tar(archive, archive.parent)
            extracted.append(archive)
    return extracted


def _first_file(root: Path, names: Iterable[str]) -> Path | None:
    wanted = {name.lower() for name in names}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name.lower() in wanted:
            return path
    return None


@dataclass(frozen=True)
class DatasetLayout:
    name: str
    root: Path
    annotation: Path


def prepare_dataset(kind: str, *, destination: str | os.PathLike[str] | None = None) -> DatasetLayout:
    """Download and unpack a benchmark into the OWP cache.

    CMM is available from the official Hugging Face dataset repository.  For
    AVHBench, ``OWP_AVHBENCH_DATASET_ID`` must point to a repository containing
    the same QA.json and media release used by the paper.
    """
    key = str(kind).strip().lower()
    if key not in DATASET_REPOS:
        raise ValueError(f"Unknown dataset: {kind}")
    repo_id = dataset_repo(key)
    if not repo_id:
        raise RuntimeError(
            "AVHBench has no verified official Hugging Face mirror for the media "
            "release used by OWP. Publish an equivalent dataset repository and "
            "set OWP_AVHBENCH_DATASET_ID to its id; another AVHBench-like dataset "
            "would not reproduce the paper protocol."
        )
    bundled_root = Path(__file__).resolve().parents[1] / "data" / {
        "cmm": "CMM",
        "avhbench": "AVHBench",
    }[key]
    bundled_annotation = _first_file(bundled_root, ("QA.json", "all_data_final_reorg.json"))
    if destination is None and bundled_annotation is not None:
        if key == "avhbench":
            has_media = any(bundled_root.glob("videos/*")) and any(bundled_root.glob("audios/*"))
        else:
            media_root = bundled_root / "reorg_raw_files"
            has_media = media_root.is_dir() and any(
                path.is_file() and path.suffix.lower() in {".mp4", ".wav", ".avi", ".mov"}
                for path in media_root.rglob("*")
            )
        if has_media:
            return DatasetLayout(name=key, root=bundled_root.resolve(), annotation=bundled_annotation.resolve())
    root = Path(destination).expanduser().resolve() if destination else cache_root() / "datasets" / key
    root.mkdir(parents=True, exist_ok=True)
    _hf_snapshot(repo_id, repo_type="dataset", destination=root)
    extract_archives(root)
    annotation = _first_file(root, ("QA.json", "all_data_final_reorg.json"))
    if annotation is None:
        raise RuntimeError(
            f"Downloaded dataset '{repo_id}' has no QA.json or all_data_final_reorg.json under {root}"
        )
    return DatasetLayout(name=key, root=root, annotation=annotation)


def path_from_dataset(root: Path, value: Any) -> str:
    """Resolve a dataset-relative media path without embedding workstation paths."""
    text = str(value or "").strip()
    if not text:
        return ""
    candidate = Path(text).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return str(candidate.resolve())
    clean = text.lstrip("./")
    direct = (root / clean).resolve()
    if direct.is_file():
        return str(direct)
    matches = list(root.rglob(Path(clean).name))
    if len(matches) == 1 and matches[0].is_file():
        return str(matches[0].resolve())
    return str(direct)


def build_dataset_manifest(layout: DatasetLayout, output: str | os.PathLike[str], *, max_rows: int = 0) -> Path:
    """Create the same portable row schema consumed by the OWP evaluators."""
    source = json.loads(layout.annotation.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise RuntimeError(f"Expected a JSON list in {layout.annotation}")
    rows: list[dict[str, Any]] = []
    if layout.name == "avhbench":
        allowed = {"Audio-driven Video Hallucination", "Video-driven Audio Hallucination", "AV Matching"}
        for index, item in enumerate(source):
            if item.get("task") not in allowed:
                continue
            video_id = str(item["video_id"])
            video = next(iter(layout.root.rglob(f"{video_id}.mp4")), layout.root / "videos" / f"{video_id}.mp4")
            audio = next(iter(layout.root.rglob(f"{video_id}.wav")), layout.root / "audios" / f"{video_id}.wav")
            rows.append(
                {
                    "sample_id": f"avhbench:{index:05d}:{video_id}",
                    "benchmark": "avhbench",
                    "mad_protocol_task": item["task"],
                    "question": item["text"],
                    "reference_answer": item["label"],
                    "video_id": video_id,
                    "video_path": str(video.resolve()),
                    "audio_path": str(audio.resolve()),
                    "_efficiency_order": len(rows),
                }
            )
    else:
        for index, item in enumerate(source):
            video = path_from_dataset(layout.root, item.get("video_path"))
            audio = path_from_dataset(layout.root, item.get("audio_path"))
            rows.append(
                {
                    "sample_id": f"cmm:{index:05d}",
                    "benchmark": "cmm",
                    "question": item["question"],
                    "reference_answer": item.get("answer"),
                    "video_path": video,
                    "audio_path": audio,
                    "category": item.get("category", ""),
                    "sub_category": item.get("sub_category", ""),
                    "modality": item.get("modality", ""),
                    "granularity": item.get("granularity", ""),
                    "correlation_type": item.get("correlation_type", ""),
                    "_efficiency_order": len(rows),
                }
            )
    if max_rows > 0:
        rows = rows[: int(max_rows)]
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


__all__ = [
    "DATASET_REPOS",
    "MODEL_REPOS",
    "DatasetLayout",
    "build_dataset_manifest",
    "cache_root",
    "dataset_repo",
    "model_repo",
    "prepare_dataset",
    "resolve_model_path",
]
