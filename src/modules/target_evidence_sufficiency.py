from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


YES_NO_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)


STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "be",
    "can",
    "do",
    "does",
    "did",
    "in",
    "is",
    "it",
    "no",
    "of",
    "only",
    "or",
    "the",
    "this",
    "that",
    "to",
    "video",
    "visible",
    "with",
    "yes",
}


def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def yn(value: Any) -> str:
    match = YES_NO_RE.search(safe_text(value))
    if not match:
        return ""
    return "Yes" if match.group(1).lower() == "yes" else "No"


def strip_yes_no_instruction(question: str) -> str:
    text = safe_text(question)
    text = re.sub(r"\bAnswer\s+with\s+only\s+Yes\s+or\s+No\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAnswer\s+Yes\s+or\s+No\.?", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def content_phrases_from_evidence(question: str, evidence: Optional[Mapping[str, Any]]) -> List[str]:
    phrases: List[str] = []
    if evidence:
        debug = evidence.get("question_content_token_debug")
        if isinstance(debug, Sequence):
            for item in debug:
                if not isinstance(item, Mapping):
                    continue
                token = safe_text(item.get("normalized") or item.get("token")).lower()
                token = re.sub(r"[^a-z0-9']+", " ", token).strip()
                if token and token not in STOPWORDS and token not in phrases:
                    phrases.append(token)
    if len(phrases) < 2:
        question_clean = strip_yes_no_instruction(question).lower()
        for token in re.findall(r"[a-z0-9']+", question_clean):
            if len(token) < 3 or token in STOPWORDS or token in phrases:
                continue
            phrases.append(token)
            if len(phrases) >= 8:
                break
    return phrases[:8]


def token_item_score(item: Mapping[str, Any]) -> float:
    for field in (
        "causal_token_score",
        "decision_score",
        "target_support_score",
        "attention_score",
        "read_score_max",
        "read_score_mean",
        "saliency",
        "grad_norm",
        "tube_score",
    ):
        value = item.get(field)
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(out):
            return out
    return 1.0


@dataclass(frozen=True)
class VisualCarrierConfig:
    top_k_evidence_tokens: int = 24
    top_k_decision_tokens: int = 8
    top_k_tube_points: int = 4
    frame_window: int = 1
    spatial_cell_expansion: int = 1
    point_radius_ratio: float = 0.10
    carrier_background_alpha: float = 0.18
    masked_region_alpha: float = 0.08
    blur_kernel_fraction: float = 0.055
    tube_bbox_expansion_ratio: float = 0.06
    output_fps: float = 2.0


def _as_int_triplet(value: Any) -> Optional[Tuple[int, int, int]]:
    if not isinstance(value, Sequence) or len(value) != 3:
        return None
    try:
        t, h, w = int(value[0]), int(value[1]), int(value[2])
    except (TypeError, ValueError):
        return None
    if t <= 0 or h <= 0 or w <= 0:
        return None
    return t, h, w


def _token_grid(evidence: Mapping[str, Any]) -> Optional[Tuple[int, int, int]]:
    return _as_int_triplet(evidence.get("token_grid_thw")) or _as_int_triplet(evidence.get("video_grid_thw"))


def _frame_count_loaded(evidence: Mapping[str, Any], fallback_t: int) -> int:
    try:
        value = int(evidence.get("frame_count_loaded") or fallback_t)
    except (TypeError, ValueError):
        value = fallback_t
    return max(1, value)


def visual_ordinal_to_cell(evidence: Mapping[str, Any], visual_ordinal: int) -> Optional[Tuple[int, int, int]]:
    grid = _token_grid(evidence)
    if grid is None:
        return None
    t_grid, h_grid, w_grid = grid
    try:
        ordinal = int(visual_ordinal)
    except (TypeError, ValueError):
        return None
    if ordinal < 0:
        return None
    cells_per_t = h_grid * w_grid
    if cells_per_t <= 0:
        return None
    t = ordinal // cells_per_t
    rem = ordinal % cells_per_t
    h = rem // w_grid
    w = rem % w_grid
    if not (0 <= t < t_grid and 0 <= h < h_grid and 0 <= w < w_grid):
        return None
    return int(t), int(h), int(w)


def token_index_to_cell(evidence: Mapping[str, Any], token_index: int) -> Optional[Tuple[int, int, int]]:
    try:
        start = int(evidence.get("video_token_start") or 0)
        ordinal = int(token_index) - start
    except (TypeError, ValueError):
        return None
    return visual_ordinal_to_cell(evidence, ordinal)


def temporal_cell_to_loaded_frames(
    temporal_cell: int,
    *,
    temporal_grid: int,
    frame_count_loaded: int,
    frame_window: int = 0,
) -> List[int]:
    if temporal_grid <= 0 or frame_count_loaded <= 0:
        return []
    start = int(math.floor(float(temporal_cell) * float(frame_count_loaded) / float(temporal_grid)))
    end = int(math.ceil(float(temporal_cell + 1) * float(frame_count_loaded) / float(temporal_grid))) - 1
    start = max(0, min(frame_count_loaded - 1, start))
    end = max(0, min(frame_count_loaded - 1, end))
    frames: List[int] = []
    for idx in range(start - max(0, int(frame_window)), end + max(0, int(frame_window)) + 1):
        idx = max(0, min(frame_count_loaded - 1, idx))
        if idx not in frames:
            frames.append(idx)
    return frames


def _add_cell(
    cells: Dict[Tuple[int, int, int], float],
    cell: Optional[Tuple[int, int, int]],
    score: float,
) -> None:
    if cell is None:
        return
    cells[cell] = max(float(cells.get(cell, -1.0)), float(score))


def collect_visual_evidence_cells(
    evidence: Optional[Mapping[str, Any]],
    decision: Optional[Mapping[str, Any]],
    *,
    config: VisualCarrierConfig,
) -> Tuple[Dict[Tuple[int, int, int], float], List[Dict[str, Any]]]:
    """Collect query-conditioned visual cells without using labels or benchmark metadata."""
    cells: Dict[Tuple[int, int, int], float] = {}
    sources: List[Dict[str, Any]] = []
    if not evidence:
        return cells, sources

    for field in ("selected_topk_query_attended_object_tokens", "selected_topk_query_attended_tokens"):
        values = evidence.get(field)
        if not isinstance(values, Sequence):
            continue
        for item in values[: max(0, int(config.top_k_evidence_tokens))]:
            if not isinstance(item, Mapping):
                continue
            try:
                token_index = int(item.get("token_index"))
            except (TypeError, ValueError):
                continue
            cell = token_index_to_cell(evidence, token_index)
            score = token_item_score(item)
            before = dict(cells)
            _add_cell(cells, cell, score)
            if cell is not None and cells != before:
                sources.append(
                    {
                        "source": field,
                        "token_index": token_index,
                        "cell": list(cell),
                        "score": float(score),
                    }
                )

    if decision:
        values = decision.get("decision_tokens")
        if isinstance(values, Sequence):
            for item in values[: max(0, int(config.top_k_decision_tokens))]:
                if not isinstance(item, Mapping):
                    continue
                cell: Optional[Tuple[int, int, int]] = None
                if item.get("visual_ordinal") is not None:
                    try:
                        cell = visual_ordinal_to_cell(evidence, int(item.get("visual_ordinal")))
                    except (TypeError, ValueError):
                        cell = None
                if cell is None and item.get("token_index") is not None:
                    try:
                        cell = token_index_to_cell(evidence, int(item.get("token_index")))
                    except (TypeError, ValueError):
                        cell = None
                score = token_item_score(item)
                before = dict(cells)
                _add_cell(cells, cell, score)
                if cell is not None and cells != before:
                    sources.append(
                        {
                            "source": "decision_tokens",
                            "token_index": item.get("token_index"),
                            "visual_ordinal": item.get("visual_ordinal"),
                            "cell": list(cell),
                            "score": float(score),
                        }
                    )
    return cells, sources


def collect_tube_prompt_points(
    evidence: Optional[Mapping[str, Any]],
    *,
    config: VisualCarrierConfig,
) -> List[Dict[str, Any]]:
    if not evidence:
        return []
    tubes = evidence.get("tube_records")
    if not isinstance(tubes, Sequence):
        return []
    out: List[Dict[str, Any]] = []
    for tube in sorted(
        [item for item in tubes if isinstance(item, Mapping)],
        key=lambda item: token_item_score(item),
        reverse=True,
    )[: max(0, int(config.top_k_tube_points))]:
        point = tube.get("prompt_point")
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        try:
            frame = int(tube.get("prompt_frame_index"))
            x = float(point[0])
            y = float(point[1])
            score = token_item_score(tube)
        except (TypeError, ValueError):
            continue
        out.append({"loaded_frame": frame, "point": [x, y], "score": float(score), "source": "tube_prompt_point"})
    return out


def collect_tube_frame_regions(
    evidence: Optional[Mapping[str, Any]],
    *,
    config: VisualCarrierConfig,
) -> List[Dict[str, Any]]:
    if not evidence:
        return []
    tubes = evidence.get("tube_records")
    if not isinstance(tubes, Sequence):
        return []
    out: List[Dict[str, Any]] = []
    ranked_tubes = sorted(
        [item for item in tubes if isinstance(item, Mapping)],
        key=lambda item: token_item_score(item),
        reverse=True,
    )[: max(0, int(config.top_k_tube_points))]
    for tube_rank, tube in enumerate(ranked_tubes):
        score = token_item_score(tube)
        frame_records = tube.get("frame_records")
        if not isinstance(frame_records, Sequence):
            continue
        for frame_record in frame_records:
            if not isinstance(frame_record, Mapping):
                continue
            bbox = frame_record.get("bbox_xyxy")
            if not isinstance(bbox, Sequence) or len(bbox) < 4:
                continue
            try:
                frame_index = int(frame_record.get("frame_index"))
                x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
            except (TypeError, ValueError):
                continue
            patch_indices: List[List[int]] = []
            raw_patch_indices = frame_record.get("patch_indices")
            if isinstance(raw_patch_indices, Sequence):
                for patch in raw_patch_indices:
                    if not isinstance(patch, Sequence) or len(patch) < 2:
                        continue
                    try:
                        patch_indices.append([int(patch[0]), int(patch[1])])
                    except (TypeError, ValueError):
                        continue
            out.append(
                {
                    "source": "sam2_object_tube_bbox",
                    "tube_rank": int(tube_rank),
                    "object_id": tube.get("object_id"),
                    "loaded_frame": int(frame_index),
                    "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
                    "patch_indices": patch_indices,
                    "patch_index_count": len(patch_indices),
                    "score": float(score),
                    "area_ratio": frame_record.get("area_ratio"),
                    "temporal_cell": frame_record.get("temporal_cell"),
                }
            )
    return out


def original_frame_indices_for_loaded_frames(
    video_path: str,
    loaded_frames: Sequence[int],
    frame_count_loaded: int,
) -> List[int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        return []
    loaded_total = max(1, int(frame_count_loaded) if frame_count_loaded else max(loaded_frames or [0]) + 1)
    linspace = np.linspace(0, total - 1, loaded_total)
    out: List[int] = []
    for frame in loaded_frames:
        idx = int(round(float(linspace[max(0, min(loaded_total - 1, int(frame)))])))
        idx = max(0, min(total - 1, idx))
        if idx not in out:
            out.append(idx)
    return out


def load_loaded_video_frames(
    video_path: str,
    *,
    frame_count_loaded: int,
) -> Tuple[List[np.ndarray], List[int], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        cap.release()
        return [], [], source_fps
    loaded_frames = list(range(max(1, int(frame_count_loaded))))
    original_indices = original_frame_indices_for_loaded_frames(video_path, loaded_frames, frame_count_loaded)
    frames: List[np.ndarray] = []
    for idx in original_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames, original_indices, source_fps


def load_video_frames_by_original_indices(
    video_path: str,
    original_indices: Sequence[Any],
) -> Tuple[List[np.ndarray], List[int], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        cap.release()
        return [], [], source_fps
    frames: List[np.ndarray] = []
    used_indices: List[int] = []
    for raw_idx in original_indices:
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        idx = max(0, min(total - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
            used_indices.append(idx)
    cap.release()
    return frames, used_indices, source_fps


def _odd_kernel(value: int) -> int:
    value = max(3, int(value))
    if value % 2 == 0:
        value += 1
    return value


def build_visual_masks(
    evidence: Mapping[str, Any],
    decision: Optional[Mapping[str, Any]],
    *,
    frame_shape: Tuple[int, int],
    config: VisualCarrierConfig,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    grid = _token_grid(evidence)
    if grid is None:
        return [], {"mask_ready": False, "mask_reason": "missing_token_grid"}
    t_grid, h_grid, w_grid = grid
    frame_count = _frame_count_loaded(evidence, fallback_t=t_grid)
    height, width = int(frame_shape[0]), int(frame_shape[1])
    masks = [np.zeros((height, width), dtype=np.float32) for _ in range(frame_count)]
    cells, cell_sources = collect_visual_evidence_cells(evidence, decision, config=config)

    expansion = max(0, int(config.spatial_cell_expansion))
    for (t_cell, h_cell, w_cell), score in cells.items():
        loaded_indices = temporal_cell_to_loaded_frames(
            t_cell,
            temporal_grid=t_grid,
            frame_count_loaded=frame_count,
            frame_window=int(config.frame_window),
        )
        h0 = max(0, h_cell - expansion)
        h1 = min(h_grid, h_cell + expansion + 1)
        w0 = max(0, w_cell - expansion)
        w1 = min(w_grid, w_cell + expansion + 1)
        y0 = int(math.floor(float(h0) * height / float(h_grid)))
        y1 = int(math.ceil(float(h1) * height / float(h_grid)))
        x0 = int(math.floor(float(w0) * width / float(w_grid)))
        x1 = int(math.ceil(float(w1) * width / float(w_grid)))
        y0, y1 = max(0, y0), min(height, max(y0 + 1, y1))
        x0, x1 = max(0, x0), min(width, max(x0 + 1, x1))
        for idx in loaded_indices:
            masks[idx][y0:y1, x0:x1] = np.maximum(masks[idx][y0:y1, x0:x1], float(score))

    tube_points = collect_tube_prompt_points(evidence, config=config)
    radius = max(4, int(min(height, width) * float(config.point_radius_ratio)))
    for point in tube_points:
        loaded = int(point["loaded_frame"])
        for idx in range(loaded - max(0, int(config.frame_window)), loaded + max(0, int(config.frame_window)) + 1):
            if idx < 0 or idx >= frame_count:
                continue
            x, y = point["point"]
            cv2.circle(masks[idx], (int(round(x)), int(round(y))), radius, float(point["score"]), thickness=-1)

    tube_regions = collect_tube_frame_regions(evidence, config=config)
    expansion_ratio = max(0.0, float(config.tube_bbox_expansion_ratio))
    patch_grid_h = 16
    patch_grid_w = 16
    for region in tube_regions:
        for ph, pw in region.get("patch_indices") or []:
            patch_grid_h = max(patch_grid_h, int(ph) + 1)
            patch_grid_w = max(patch_grid_w, int(pw) + 1)
    for region in tube_regions:
        loaded = int(region["loaded_frame"])
        if loaded < 0 or loaded >= frame_count:
            continue
        patch_indices = region.get("patch_indices") or []
        for idx in range(loaded - max(0, int(config.frame_window)), loaded + max(0, int(config.frame_window)) + 1):
            if idx < 0 or idx >= frame_count:
                continue
            if patch_indices:
                for ph, pw in patch_indices:
                    ph_i, pw_i = int(ph), int(pw)
                    if ph_i < 0 or pw_i < 0:
                        continue
                    y0_i = int(math.floor(float(ph_i) * height / float(patch_grid_h)))
                    y1_i = int(math.ceil(float(ph_i + 1) * height / float(patch_grid_h)))
                    x0_i = int(math.floor(float(pw_i) * width / float(patch_grid_w)))
                    x1_i = int(math.ceil(float(pw_i + 1) * width / float(patch_grid_w)))
                    y0_i, y1_i = max(0, y0_i), min(height, max(y0_i + 1, y1_i))
                    x0_i, x1_i = max(0, x0_i), min(width, max(x0_i + 1, x1_i))
                    masks[idx][y0_i:y1_i, x0_i:x1_i] = np.maximum(
                        masks[idx][y0_i:y1_i, x0_i:x1_i],
                        float(region["score"]),
                    )
            else:
                x0, y0, x1, y1 = [float(v) for v in region["bbox_xyxy"]]
                bw = max(1.0, x1 - x0)
                bh = max(1.0, y1 - y0)
                x0 -= bw * expansion_ratio
                x1 += bw * expansion_ratio
                y0 -= bh * expansion_ratio
                y1 += bh * expansion_ratio
                x0_i = max(0, min(width - 1, int(math.floor(x0))))
                y0_i = max(0, min(height - 1, int(math.floor(y0))))
                x1_i = max(x0_i + 1, min(width, int(math.ceil(x1))))
                y1_i = max(y0_i + 1, min(height, int(math.ceil(y1))))
                masks[idx][y0_i:y1_i, x0_i:x1_i] = np.maximum(
                    masks[idx][y0_i:y1_i, x0_i:x1_i],
                    float(region["score"]),
                )

    max_value = max(float(mask.max()) for mask in masks) if masks else 0.0
    if max_value <= 0.0:
        return masks, {
            "mask_ready": False,
            "mask_reason": "no_visual_evidence_cells",
            "cell_count": len(cells),
            "tube_point_count": len(tube_points),
            "tube_region_count": len(tube_regions),
        }
    normalized = []
    for mask in masks:
        m = np.clip(mask / max_value, 0.0, 1.0)
        k = _odd_kernel(int(min(height, width) * 0.035))
        if k > 1:
            m = cv2.GaussianBlur(m, (k, k), 0)
        if float(m.max()) > 0:
            m = np.clip(m / float(m.max()), 0.0, 1.0)
        normalized.append(m.astype(np.float32))

    active_frames = [idx for idx, mask in enumerate(normalized) if float(mask.max()) > 0.0]
    coverage = float(sum(float((mask > 0.05).mean()) for mask in normalized) / max(1, len(normalized)))
    return normalized, {
        "mask_ready": True,
        "mask_reason": "ok",
        "frame_count_loaded": frame_count,
        "token_grid_thw": [t_grid, h_grid, w_grid],
        "active_loaded_frames": active_frames,
        "active_frame_count": len(active_frames),
        "cell_count": len(cells),
        "tube_point_count": len(tube_points),
        "tube_region_count": len(tube_regions),
        "mask_mean_coverage": coverage,
        "cell_sources": cell_sources[:64],
        "tube_point_sources": tube_points,
        "tube_region_sources": [
            {key: value for key, value in region.items() if key != "patch_indices"}
            for region in tube_regions[:64]
        ],
    }


def _write_video(path: Path, frames: Sequence[np.ndarray], *, fps: float) -> bool:
    if not frames:
        return False
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
    return path.is_file() and path.stat().st_size > 0


def build_visual_tes_counterfactual_videos(
    *,
    sample_id: str,
    source_video: str,
    evidence: Optional[Mapping[str, Any]],
    decision: Optional[Mapping[str, Any]],
    output_dir: Path,
    config: Optional[VisualCarrierConfig] = None,
) -> Dict[str, Any]:
    """Build carrier-only and carrier-masked videos for visual TES.

    carrier-only keeps target evidence cells and suppresses context.
    carrier-masked keeps the original frames but removes the same target evidence cells.
    """
    config = config or VisualCarrierConfig()
    if not evidence:
        return {"ready": False, "reason": "missing_visual_evidence_record"}
    if not source_video or not Path(source_video).is_file():
        return {"ready": False, "reason": "missing_source_video", "source_video": source_video}

    grid = _token_grid(evidence)
    fallback_t = grid[0] if grid else 8
    frame_count = _frame_count_loaded(evidence, fallback_t=fallback_t)
    frames, original_indices, source_fps = load_loaded_video_frames(source_video, frame_count_loaded=frame_count)
    if not frames:
        return {"ready": False, "reason": "source_frame_read_failed", "source_video": source_video}
    height, width = frames[0].shape[:2]
    masks, mask_info = build_visual_masks(
        evidence,
        decision,
        frame_shape=(height, width),
        config=config,
    )
    if not mask_info.get("mask_ready"):
        return {
            "ready": False,
            "reason": safe_text(mask_info.get("mask_reason")) or "mask_unavailable",
            "source_video": source_video,
            **mask_info,
        }

    if len(masks) != len(frames):
        if len(masks) < len(frames):
            masks = list(masks) + [np.zeros((height, width), dtype=np.float32) for _ in range(len(frames) - len(masks))]
        else:
            masks = list(masks)[: len(frames)]

    blur_k = _odd_kernel(int(min(height, width) * float(config.blur_kernel_fraction)))
    carrier_frames: List[np.ndarray] = []
    masked_frames: List[np.ndarray] = []
    for frame, mask in zip(frames, masks):
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        mask3 = np.repeat(np.clip(mask, 0.0, 1.0)[:, :, None], 3, axis=2).astype(np.float32)
        blurred = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
        dim_context = np.clip(blurred.astype(np.float32) * float(config.carrier_background_alpha), 0, 255)
        carrier = frame.astype(np.float32) * mask3 + dim_context * (1.0 - mask3)

        degraded_region = np.clip(blurred.astype(np.float32) * float(config.masked_region_alpha), 0, 255)
        masked = frame.astype(np.float32) * (1.0 - mask3) + degraded_region * mask3
        carrier_frames.append(np.clip(carrier, 0, 255).astype(np.uint8))
        masked_frames.append(np.clip(masked, 0, 255).astype(np.uint8))

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    sample_dir = output_dir / "visual_tes_counterfactual_videos"
    carrier_path = sample_dir / f"{safe_id}.carrier_only.mp4"
    masked_path = sample_dir / f"{safe_id}.carrier_masked.mp4"
    fps = float(config.output_fps or source_fps or 2.0)
    carrier_ok = _write_video(carrier_path, carrier_frames, fps=fps)
    masked_ok = _write_video(masked_path, masked_frames, fps=fps)
    ready = bool(carrier_ok and masked_ok)
    return {
        "ready": ready,
        "reason": "ok" if ready else "counterfactual_video_write_failed",
        "source_video": source_video,
        "carrier_only_video_path": str(carrier_path) if carrier_ok else "",
        "carrier_masked_video_path": str(masked_path) if masked_ok else "",
        "loaded_original_frame_indices": [int(idx) for idx in original_indices],
        "counterfactual_fps": fps,
        "frame_shape": [int(height), int(width)],
        **mask_info,
    }


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _grounding_detection_to_box(
    detection: Mapping[str, Any],
    *,
    width: int,
    height: int,
    expansion_ratio: float,
) -> Optional[Tuple[int, int, int, int]]:
    box = detection.get("box_xyxy")
    if not isinstance(box, Sequence) or len(box) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    x0 -= bw * float(expansion_ratio)
    x1 += bw * float(expansion_ratio)
    y0 -= bh * float(expansion_ratio)
    y1 += bh * float(expansion_ratio)
    ix0 = max(0, min(width - 1, int(math.floor(x0))))
    iy0 = max(0, min(height - 1, int(math.floor(y0))))
    ix1 = max(ix0 + 1, min(width, int(math.ceil(x1))))
    iy1 = max(iy0 + 1, min(height, int(math.ceil(y1))))
    return ix0, iy0, ix1, iy1


def build_grounding_box_counterfactual_videos(
    *,
    sample_id: str,
    source_video: str,
    grounding_record: Mapping[str, Any],
    output_dir: Path,
    config: Optional[VisualCarrierConfig] = None,
    min_box_score: float = 0.10,
    max_box_area_ratio: float = 0.75,
    top_k_detections: int = 8,
    frame_window: int = 0,
) -> Dict[str, Any]:
    """Build target-box carrier videos from external localization records.

    This is intentionally separate from the Qwen-attention carrier builder above:
    the localization record supplies only where the queried visual target appears,
    while the downstream TES scorer decides whether a candidate claim is supported.
    """
    config = config or VisualCarrierConfig()
    if not grounding_record:
        return {"ready": False, "reason": "missing_target_grounding_record"}
    if not source_video or not Path(source_video).is_file():
        return {"ready": False, "reason": "missing_source_video", "source_video": source_video}

    loaded_original_indices = grounding_record.get("loaded_original_frame_indices")
    if isinstance(loaded_original_indices, Sequence) and loaded_original_indices:
        frames, original_indices, source_fps = load_video_frames_by_original_indices(source_video, loaded_original_indices)
    else:
        try:
            n_loaded = int(grounding_record.get("n_loaded_frames") or 32)
        except (TypeError, ValueError):
            n_loaded = 32
        frames, original_indices, source_fps = load_loaded_video_frames(source_video, frame_count_loaded=max(1, n_loaded))
    if not frames:
        return {"ready": False, "reason": "source_frame_read_failed", "source_video": source_video}

    height, width = frames[0].shape[:2]
    frame_count = len(frames)
    masks = [np.zeros((height, width), dtype=np.float32) for _ in range(frame_count)]
    detections_raw = grounding_record.get("top_grounding_detections")
    if not isinstance(detections_raw, Sequence):
        detections_raw = []

    usable_sources: List[Dict[str, Any]] = []
    for det in detections_raw:
        if not isinstance(det, Mapping):
            continue
        score = _finite_float(det.get("score"), 0.0) or 0.0
        area_ratio = _finite_float(det.get("area_ratio"), None)
        if score < float(min_box_score):
            continue
        if area_ratio is not None and area_ratio > float(max_box_area_ratio):
            continue
        try:
            loaded_idx = int(det.get("frame_index"))
        except (TypeError, ValueError):
            continue
        if loaded_idx < 0 or loaded_idx >= frame_count:
            continue
        box = _grounding_detection_to_box(
            det,
            width=width,
            height=height,
            expansion_ratio=float(config.tube_bbox_expansion_ratio),
        )
        if box is None:
            continue
        x0, y0, x1, y1 = box
        for idx in range(loaded_idx - max(0, int(frame_window)), loaded_idx + max(0, int(frame_window)) + 1):
            if 0 <= idx < frame_count:
                masks[idx][y0:y1, x0:x1] = np.maximum(masks[idx][y0:y1, x0:x1], float(score))
        usable_sources.append(
            {
                "loaded_frame_index": loaded_idx,
                "original_frame_index": int(original_indices[loaded_idx]) if loaded_idx < len(original_indices) else None,
                "label": safe_text(det.get("label")),
                "score": float(score),
                "area_ratio": float(area_ratio) if area_ratio is not None else None,
                "box_xyxy": [int(x0), int(y0), int(x1), int(y1)],
            }
        )
        if len(usable_sources) >= max(1, int(top_k_detections)):
            break

    max_value = max(float(mask.max()) for mask in masks) if masks else 0.0
    if max_value <= 0.0:
        return {
            "ready": False,
            "reason": "no_usable_grounding_boxes",
            "source_video": source_video,
            "target_entity": safe_text(grounding_record.get("target_entity")),
            "min_box_score": float(min_box_score),
            "max_box_area_ratio": float(max_box_area_ratio),
            "n_raw_detections": len(detections_raw),
            "n_usable_detections": 0,
        }

    blur_k = _odd_kernel(int(min(height, width) * 0.025))
    normalized: List[np.ndarray] = []
    for mask in masks:
        m = np.clip(mask / max_value, 0.0, 1.0)
        if blur_k > 1:
            m = cv2.GaussianBlur(m, (blur_k, blur_k), 0)
        if float(m.max()) > 0:
            m = np.clip(m / float(m.max()), 0.0, 1.0)
        normalized.append(m.astype(np.float32))

    blur_k_frame = _odd_kernel(int(min(height, width) * float(config.blur_kernel_fraction)))
    carrier_frames: List[np.ndarray] = []
    masked_frames: List[np.ndarray] = []
    for frame, mask in zip(frames, normalized):
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        mask3 = np.repeat(np.clip(mask, 0.0, 1.0)[:, :, None], 3, axis=2).astype(np.float32)
        blurred = cv2.GaussianBlur(frame, (blur_k_frame, blur_k_frame), 0)
        dim_context = np.clip(blurred.astype(np.float32) * float(config.carrier_background_alpha), 0, 255)
        carrier = frame.astype(np.float32) * mask3 + dim_context * (1.0 - mask3)

        degraded_region = np.clip(blurred.astype(np.float32) * float(config.masked_region_alpha), 0, 255)
        masked = frame.astype(np.float32) * (1.0 - mask3) + degraded_region * mask3
        carrier_frames.append(np.clip(carrier, 0, 255).astype(np.uint8))
        masked_frames.append(np.clip(masked, 0, 255).astype(np.uint8))

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    sample_dir = output_dir / "visual_claim_tes_counterfactual_videos"
    carrier_path = sample_dir / f"{safe_id}.grounding_box_carrier.mp4"
    masked_path = sample_dir / f"{safe_id}.grounding_box_destroyed.mp4"
    fps = float(config.output_fps or source_fps or 2.0)
    carrier_ok = _write_video(carrier_path, carrier_frames, fps=fps)
    masked_ok = _write_video(masked_path, masked_frames, fps=fps)
    active_frames = [idx for idx, mask in enumerate(normalized) if float(mask.max()) > 0.0]
    active_coverages = [float((normalized[idx] > 0.05).mean()) for idx in active_frames]
    mean_coverage = float(sum(float((mask > 0.05).mean()) for mask in normalized) / max(1, len(normalized)))
    active_mean_coverage = float(sum(active_coverages) / max(1, len(active_coverages)))
    ready = bool(carrier_ok and masked_ok)
    return {
        "ready": ready,
        "reason": "ok" if ready else "counterfactual_video_write_failed",
        "source_video": source_video,
        "carrier_only_video_path": str(carrier_path) if carrier_ok else "",
        "carrier_masked_video_path": str(masked_path) if masked_ok else "",
        "target_entity": safe_text(grounding_record.get("target_entity")),
        "strict_aliases": list(grounding_record.get("strict_aliases") or []),
        "localizer_backend": safe_text(grounding_record.get("grounding_backend")) or "grounding_box_record",
        "loaded_original_frame_indices": [int(idx) for idx in original_indices],
        "counterfactual_fps": fps,
        "frame_shape": [int(height), int(width)],
        "active_loaded_frames": active_frames,
        "active_frame_count": len(active_frames),
        "mask_mean_coverage": mean_coverage,
        "mask_active_mean_coverage": active_mean_coverage,
        "mask_temporal_coverage": len(active_frames) / max(1, len(normalized)),
        "n_raw_detections": len(detections_raw),
        "n_usable_detections": len(usable_sources),
        "box_sources": usable_sources[:64],
        "min_box_score": float(min_box_score),
        "max_box_area_ratio": float(max_box_area_ratio),
        "top_k_detections": int(top_k_detections),
        "frame_window": int(frame_window),
    }


@dataclass(frozen=True)
class TESContractConfig:
    min_localized_margin: float = 0.02
    min_necessity_drop: float = 0.02
    min_text_residual_gap: float = 0.02
    min_competitor_residual_gap: float = 0.02
    max_full_against_margin: float = -0.25
    min_direct_localized_margin: float = 0.02
    min_mask_mean_coverage: float = 0.40
    max_mask_mean_coverage: float = 0.90


def evaluate_tes_contract(
    *,
    localized_margin: Optional[float],
    degraded_margin: Optional[float],
    text_margin: Optional[float],
    competitor_margin: Optional[float],
    full_margin: Optional[float],
    direct_localized_margin: Optional[float] = None,
    mask_mean_coverage: Optional[float] = None,
    competitor_available: bool,
    config: Optional[TESContractConfig] = None,
) -> Dict[str, Any]:
    config = config or TESContractConfig()

    def fnum(value: Optional[float]) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    loc = fnum(localized_margin)
    deg = fnum(degraded_margin)
    txt = fnum(text_margin)
    comp = fnum(competitor_margin)
    full = fnum(full_margin)
    direct_loc = fnum(direct_localized_margin)
    coverage = fnum(mask_mean_coverage)
    reasons: List[str] = []
    if loc is None:
        reasons.append("missing_localized_margin")
    elif loc < float(config.min_localized_margin):
        reasons.append("localized_support_too_weak")

    necessity_drop = None if loc is None or deg is None else float(loc - deg)
    if necessity_drop is None:
        reasons.append("missing_necessity_drop")
    elif necessity_drop < float(config.min_necessity_drop):
        reasons.append("counterfactual_necessity_drop_too_small")

    text_residual_gap = None if loc is None or txt is None else float(loc - max(0.0, txt))
    if text_residual_gap is None:
        reasons.append("missing_text_prior_margin")
    elif text_residual_gap < float(config.min_text_residual_gap):
        reasons.append("candidate_explained_by_text_prior")

    competitor_residual_gap = None
    if competitor_available:
        competitor_residual_gap = None if loc is None or comp is None else float(loc - max(0.0, comp))
        if competitor_residual_gap is None:
            reasons.append("missing_competitor_margin")
        elif competitor_residual_gap < float(config.min_competitor_residual_gap):
            reasons.append("candidate_explained_by_competitor_modality")

    if full is not None and full < float(config.max_full_against_margin):
        reasons.append("full_context_strongly_against_candidate")

    if direct_loc is None:
        reasons.append("missing_direct_query_alignment_margin")
    elif direct_loc < float(config.min_direct_localized_margin):
        reasons.append("direct_query_alignment_too_weak")

    if coverage is None:
        reasons.append("missing_carrier_coverage")
    elif coverage < float(config.min_mask_mean_coverage):
        reasons.append("carrier_too_fragmented")
    elif coverage > float(config.max_mask_mean_coverage):
        reasons.append("carrier_too_global")

    accepted = len(reasons) == 0
    return {
        "accepted": accepted,
        "acceptance_reason": "target_evidence_sufficiency_accept" if accepted else "|".join(reasons),
        "localized_margin": loc,
        "degraded_margin": deg,
        "text_margin": txt,
        "competitor_margin": comp,
        "full_margin": full,
        "direct_localized_margin": direct_loc,
        "mask_mean_coverage": coverage,
        "necessity_drop": necessity_drop,
        "text_residual_gap": text_residual_gap,
        "competitor_residual_gap": competitor_residual_gap,
        "competitor_available": bool(competitor_available),
        "contract": {
            "min_localized_margin": float(config.min_localized_margin),
            "min_necessity_drop": float(config.min_necessity_drop),
            "min_text_residual_gap": float(config.min_text_residual_gap),
            "min_competitor_residual_gap": float(config.min_competitor_residual_gap),
            "max_full_against_margin": float(config.max_full_against_margin),
            "min_direct_localized_margin": float(config.min_direct_localized_margin),
            "min_mask_mean_coverage": float(config.min_mask_mean_coverage),
            "max_mask_mean_coverage": float(config.max_mask_mean_coverage),
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
