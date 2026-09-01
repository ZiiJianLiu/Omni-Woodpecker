from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf


DEFAULT_SHIFT_SECONDS = 1.5


@dataclass
class ResolvedMedia:
    dataset: str
    sample_id: str
    media_id: str
    video_path: Optional[Path]
    audio_path: Optional[Path]
    source_kind: str


class MediaResolver:
    def __init__(
        self,
        data_root: str | Path,
        cache_dir: str | Path = "/tmp/omni_media_cache",
    ) -> None:
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._worldsense_index: Optional[Dict[str, Path]] = None
        self._daily_tar_path = self.data_root / "daily_omni" / "Videos.tar"
        self._audio_cache: Dict[str, Optional[np.ndarray]] = {}

    def resolve_pair_media(self, pair: Dict[str, Any]) -> ResolvedMedia:
        return self.resolve_media_record(
            dataset=str(pair.get("dataset") or ""),
            sample_id=str(pair.get("sample_id") or ""),
            media=pair.get("input", {}).get("media", {}) or {},
        )

    def resolve_media_record(
        self,
        *,
        dataset: str,
        sample_id: str,
        media: Dict[str, Any],
    ) -> ResolvedMedia:
        if dataset == "omnimmi":
            return self._resolve_omnimmi(sample_id=sample_id, media=media)
        if dataset == "omnivideobench":
            return self._resolve_omnivideobench(sample_id=sample_id, media=media)
        if dataset == "daily_omni":
            return self._resolve_daily_omni(sample_id=sample_id, media=media)
        if dataset == "worldsense":
            return self._resolve_worldsense(sample_id=sample_id, media=media)
        raise ValueError(f"Unsupported dataset: {dataset}")

    def prepare_pair_inputs(self, pair: Dict[str, Any]) -> Dict[str, Any]:
        resolved = self.resolve_pair_media(pair)
        intervention = pair.get("input", {}).get("intervention")
        return self.apply_intervention(
            resolved=resolved,
            intervention=intervention,
        )

    def apply_intervention(
        self,
        *,
        resolved: ResolvedMedia,
        intervention: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        intervention = intervention or {}
        transform = intervention.get("transform") or {}
        transform_type = str(transform.get("type") or "")

        base_video = str(resolved.video_path) if resolved.video_path else None
        base_audio = self._load_audio_array_cached(resolved.audio_path or resolved.video_path)

        if not transform_type:
            return {
                "video_path": base_video,
                "audio_array": base_audio,
                "applied_transform": "none",
            }
        if transform_type == "drop_audio":
            return {
                "video_path": base_video,
                "audio_array": None,
                "applied_transform": transform_type,
            }
        if transform_type == "drop_video":
            return {
                "video_path": None,
                "audio_array": base_audio,
                "applied_transform": transform_type,
            }
        if transform_type == "shift_audio":
            seconds = float(transform.get("seconds") or DEFAULT_SHIFT_SECONDS)
            return {
                "video_path": base_video,
                "audio_array": self.shift_audio(base_audio, seconds=seconds),
                "applied_transform": transform_type,
                "seconds": seconds,
            }
        if transform_type in {"swap_audio", "swap_video"}:
            swap_source = intervention.get("swap_source") or {}
            if not swap_source.get("dataset") or not swap_source.get("sample_id") or not (swap_source.get("media") or {}):
                raise ValueError(
                    "Swap intervention is missing swap_source metadata. "
                    "Rebuild the CIRPO/OmniDPO data artifacts so each swap branch has a concrete donor sample."
                )
            swap_media = self.resolve_media_record(
                dataset=str(swap_source.get("dataset") or ""),
                sample_id=str(swap_source.get("sample_id") or ""),
                media=swap_source.get("media", {}) or {},
            )
            swap_video = str(swap_media.video_path) if swap_media.video_path else None
            swap_audio = self._load_audio_array_cached(swap_media.audio_path or swap_media.video_path)
            if transform_type == "swap_audio":
                return {
                    "video_path": base_video,
                    "audio_array": swap_audio,
                    "applied_transform": transform_type,
                    "swap_source_sample_id": swap_source.get("sample_id"),
                }
            return {
                "video_path": swap_video,
                "audio_array": base_audio,
                "applied_transform": transform_type,
                "swap_source_sample_id": swap_source.get("sample_id"),
            }

        raise ValueError(f"Unsupported transform type: {transform_type}")

    def _load_audio_array_cached(self, path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None:
            return None
        cache_key = str(path.resolve())
        if cache_key not in self._audio_cache:
            self._audio_cache[cache_key] = self.load_audio_array(path)
        return self._audio_cache[cache_key]

    @staticmethod
    def shift_audio(audio_array: Optional[np.ndarray], seconds: float) -> Optional[np.ndarray]:
        if audio_array is None:
            return None
        sample_rate = 16000
        offset = int(round(seconds * sample_rate))
        if offset == 0:
            return audio_array
        if offset > 0:
            padding = np.zeros(offset, dtype=audio_array.dtype)
            shifted = np.concatenate([padding, audio_array], axis=0)[: audio_array.shape[0]]
        else:
            shift = min(audio_array.shape[0], abs(offset))
            padding = np.zeros(shift, dtype=audio_array.dtype)
            shifted = np.concatenate([audio_array[shift:], padding], axis=0)
        return shifted

    @staticmethod
    def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
        audio, sample_rate = sf.read(str(path), dtype="float32")
        if isinstance(audio, np.ndarray) and audio.ndim == 2:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)

    @classmethod
    def load_audio_array(cls, path: Optional[Path]) -> Optional[np.ndarray]:
        if path is None or not path.exists():
            return None
        try:
            if path.suffix.lower() == ".wav":
                audio, sample_rate = cls._read_wav_mono(path)
                if sample_rate == 16000:
                    return audio
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(path),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        tmp_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                return None
            audio, _ = cls._read_wav_mono(Path(tmp_path))
            return audio
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _resolve_omnimmi(self, *, sample_id: str, media: Dict[str, Any]) -> ResolvedMedia:
        root = self.data_root / "omnimmi"
        candidates = []
        for key in ("video_resolved", "video_ref"):
            value = media.get(key)
            if value:
                candidates.append(str(value))
        for value in media.get("video_candidates") or []:
            if value:
                candidates.append(str(value))

        for name in candidates:
            for subdir in ("clips", "videos", ""):
                path = root / subdir / name if subdir else root / name
                if path.exists():
                    return ResolvedMedia(
                        dataset="omnimmi",
                        sample_id=sample_id,
                        media_id=Path(name).stem,
                        video_path=path,
                        audio_path=None,
                        source_kind=f"omnimmi:{subdir or 'root'}",
                    )
        raise FileNotFoundError(f"Could not resolve OmniMMI media for {sample_id}: {candidates}")

    def _resolve_omnivideobench(self, *, sample_id: str, media: Dict[str, Any]) -> ResolvedMedia:
        root = self.data_root / "omnivideobench"
        candidates: List[str] = []
        for key in ("video_resolved", "video_ref"):
            value = media.get(key)
            if value:
                candidates.append(str(value))
        for value in media.get("video_candidates") or []:
            if value:
                candidates.append(str(value))

        for name in candidates:
            stem = Path(name).stem
            path_candidates = [
                root / "videos" / name,
                root / "clips" / name,
                root / name,
            ]
            if Path(name).suffix:
                path_candidates.extend(
                    [
                        root / "videos" / stem,
                        root / "clips" / stem,
                        root / stem,
                    ]
                )
            else:
                for ext in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
                    path_candidates.extend(
                        [
                            root / "videos" / f"{stem}{ext}",
                            root / "clips" / f"{stem}{ext}",
                            root / f"{stem}{ext}",
                        ]
                    )
            for path in path_candidates:
                if path.exists():
                    return ResolvedMedia(
                        dataset="omnivideobench",
                        sample_id=sample_id,
                        media_id=Path(path).stem,
                        video_path=path,
                        audio_path=None,
                        source_kind=f"omnivideobench:{path.parent.name or 'root'}",
                    )
        raise FileNotFoundError(f"Could not resolve OmniVideoBench media for {sample_id}: {candidates}")

    def _resolve_daily_omni(self, *, sample_id: str, media: Dict[str, Any]) -> ResolvedMedia:
        video_ref = str(media.get("video_ref") or "")
        media_id = Path(video_ref).stem
        extracted_dir = self.cache_dir / "daily_omni" / media_id
        video_path = extracted_dir / f"{media_id}_video.mp4"
        audio_path = extracted_dir / f"{media_id}_audio.wav"
        if not video_path.exists():
            self._extract_daily_member(media_id, f"Videos/{media_id}/{media_id}_video.mp4", video_path)
        if not audio_path.exists():
            try:
                self._extract_daily_member(media_id, f"Videos/{media_id}/{media_id}_audio.wav", audio_path)
            except FileNotFoundError:
                audio_path = None
        return ResolvedMedia(
            dataset="daily_omni",
            sample_id=sample_id,
            media_id=media_id,
            video_path=video_path if video_path.exists() else None,
            audio_path=audio_path if audio_path and audio_path.exists() else None,
            source_kind="daily_omni:tar",
        )

    def _extract_daily_member(self, media_id: str, member_name: str, target_path: Path) -> None:
        if not self._daily_tar_path.exists():
            raise FileNotFoundError(f"Missing Daily-Omni archive: {self._daily_tar_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(self._daily_tar_path, "r") as tf:
            member = tf.getmember(member_name)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(member_name)
            with target_path.open("wb") as out_f:
                shutil.copyfileobj(extracted, out_f)

    def _resolve_worldsense(self, *, sample_id: str, media: Dict[str, Any]) -> ResolvedMedia:
        video_ref = str(media.get("video_ref") or "")
        if not video_ref:
            raise FileNotFoundError(f"Missing WorldSense video_ref for {sample_id}")
        media_id = Path(video_ref).name
        extracted_path = self.cache_dir / "worldsense" / media_id
        if not extracted_path.exists():
            zip_path = self._worldsense_entry_index().get(media_id)
            if zip_path is None:
                raise FileNotFoundError(f"Could not locate WorldSense video {media_id}")
            extracted_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                with zf.open(media_id) as src, extracted_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        return ResolvedMedia(
            dataset="worldsense",
            sample_id=sample_id,
            media_id=Path(media_id).stem,
            video_path=extracted_path,
            audio_path=None,
            source_kind="worldsense:zip",
        )

    def _worldsense_entry_index(self) -> Dict[str, Path]:
        if self._worldsense_index is not None:
            return self._worldsense_index
        root = self.data_root / "worldsense"
        index: Dict[str, Path] = {}
        for zip_path in sorted(root.glob("worldsense_videos_*.zip")):
            with zipfile.ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    if name.endswith(".mp4"):
                        index[Path(name).name] = zip_path
        self._worldsense_index = index
        return index
