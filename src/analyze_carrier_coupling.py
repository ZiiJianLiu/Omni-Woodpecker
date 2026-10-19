#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, ROOT, ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from patch_target_branch import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    QwenOmniAdapter,
    capture_branch,
    safe_text,
    signed_margin_for_answer,
    video_budget,
    write_json,
    write_jsonl,
    yes_no_token_ids,
)
from pilot_hidden_state_editing import (  # noqa: E402
    DEFAULT_AUDIO_FINAL,
    DEFAULT_AUDIO_MANIFEST,
    DEFAULT_AVH_QA,
    DEFAULT_AVH_VIDEO_DIR,
    capture_prompt_row,
    joined_audio_rows,
    normalize_binary_label,
    visual_rows,
)


DEFAULT_MANIFEST = None
DEFAULT_OUTPUT_DIR = ROOT / "results" / "qwen_owp_carrier_audit"

OFFLINE_EVAL_FIELD_NAMES = {
    "answer",
    "category",
    "eval_reference_answer",
    "eval_task_family",
    "ground_truth",
    "label",
    "mad_protocol_category",
    "mad_protocol_reference_answer",
    "mad_protocol_sub_category",
    "mad_protocol_task",
    "paper_faithful_scope",
    "reference_answer",
    "reference_answer_offline",
    "sub_category",
    "task",
    "task_category",
    "task_family",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture-only audit for Qwen OWP evidence-prior decomposition. "
            "It constructs target-evidence and prior-carrier directions from full/target/non-target/text "
            "hidden states, orthogonalizes prior against evidence, and reports whether prior suppression "
            "would preserve evidence."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--audio-final", type=Path, required=True)
    parser.add_argument("--avh-qa", type=Path, default=DEFAULT_AVH_QA)
    parser.add_argument("--avh-video-dir", type=Path, default=DEFAULT_AVH_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument(
        "--roles",
        nargs="+",
        default=[],
        help="Optional manifest roles to retain. By default, retain all valid rows.",
    )
    parser.add_argument("--modalities", nargs="+", default=["visual", "audio"])
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 24, 26])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument(
        "--evidence-direction",
        choices=[
            "target_minus_text",
            "text_minus_target",
            "target_minus_nontarget",
            "full_minus_nontarget",
            "nontarget_minus_text",
        ],
        default="target_minus_text",
    )
    parser.add_argument(
        "--prior-direction",
        choices=["nontarget_minus_target", "nontarget_minus_text", "full_minus_target"],
        default="nontarget_minus_target",
    )
    parser.add_argument(
        "--target-capture-prompt",
        choices=["original", "evidence_focus"],
        default="evidence_focus",
    )
    parser.add_argument(
        "--non-target-capture-prompt",
        choices=["original", "evidence_focus"],
        default="original",
    )
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-max-frames", type=int, default=32)
    parser.add_argument("--video-max-pixels", type=int, default=602112)
    parser.add_argument("--video-min-pixels", type=int, default=100352)
    parser.add_argument(
        "--include-offline-eval-fields",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Debug only. By default runtime rows are stripped of benchmark truth/category fields; "
            "offline scoring must be done by a separate join script."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_unique_floats(values: Sequence[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite beta: {value!r}")
        if number not in out:
            out.append(number)
    return out


def build_runtime_lookup(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in joined_audio_rows(manifest_path=args.audio_manifest, final_path=args.audio_final):
        item = dict(row)
        item["runtime_source"] = "audio_manifest"
        lookup[safe_text(item.get("sample_id"))] = item
    for row in visual_rows(visual_replay_path=args.manifest_visual_replay, qa_path=args.avh_qa, video_dir=args.avh_video_dir):
        item = dict(row)
        item["runtime_source"] = "visual_avh_qa"
        lookup[safe_text(item.get("sample_id"))] = item
    return lookup


def build_visual_runtime_lookup(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        safe_text(row.get("sample_id")): {**dict(row), "runtime_source": "visual_avh_qa"}
        for row in visual_rows(visual_replay_path=args.manifest_visual_replay, qa_path=args.avh_qa, video_dir=args.avh_video_dir)
    }


def include_offline_eval_fields(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "include_offline_eval_fields", False))


def strip_offline_eval_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(row).items() if key not in OFFLINE_EVAL_FIELD_NAMES}


def target_branch_for(row: Mapping[str, Any]) -> str:
    modality = safe_text(row.get("target_modality")).lower()
    if modality == "audio":
        return "audio_only"
    if modality == "visual":
        return "visual_only"
    raise ValueError(f"Unsupported target_modality={modality!r} for sample={row.get('sample_id')}")


def non_target_branch_for(row: Mapping[str, Any]) -> str:
    return "visual_only" if target_branch_for(row) == "audio_only" else "audio_only"


def manifest_branch_answer(row: Mapping[str, Any], branch: str) -> str | None:
    branches = row.get("branches")
    if not isinstance(branches, Mapping):
        return None
    payload = branches.get(branch)
    if not isinstance(payload, Mapping):
        return None
    return normalize_binary_label(payload.get("answer"))


def manifest_current_answer(row: Mapping[str, Any]) -> str | None:
    return (
        normalize_binary_label(row.get("current_answer_y0"))
        or normalize_binary_label(row.get("probe_full_answer"))
        or normalize_binary_label(row.get("baseline_answer"))
        or manifest_branch_answer(row, "full")
    )


def manifest_candidate_answer(row: Mapping[str, Any], modality: str) -> str | None:
    explicit = normalize_binary_label(row.get("candidate_answer_y"))
    if explicit is not None:
        return explicit
    target_branch = "audio_only" if modality == "audio" else "visual_only"
    return manifest_branch_answer(row, target_branch)


def manifest_runtime_fields(row: Mapping[str, Any], modality: str) -> dict[str, Any]:
    current = manifest_current_answer(row)
    candidate = manifest_candidate_answer(row, modality)
    conflict = current is not None and candidate is not None and current != candidate
    return {
        "current_answer_y0": current,
        "candidate_answer_y": candidate,
        "evidence_certificate": safe_text(row.get("evidence_certificate"))
        or ("target_branch_disagreement" if conflict else ""),
        "current_best_accepts": bool(row.get("current_best_accepts")) or conflict,
    }


def runtime_rows(args: argparse.Namespace, manifest_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keep_offline = include_offline_eval_fields(args)
    explicit_runtime_rows = getattr(args, "runtime_rows", None)
    if explicit_runtime_rows is not None:
        runtime_path = Path(explicit_runtime_rows)
        explicit_lookup = {
            safe_text(row.get("sample_id")): {**dict(row), "runtime_source": safe_text(row.get("runtime_source")) or "explicit_runtime_rows"}
            for row in read_jsonl(runtime_path)
            if safe_text(row.get("sample_id"))
        }
        out: list[dict[str, Any]] = []
        allowed_roles = {safe_text(role) for role in args.roles}
        allowed_modalities = {safe_text(modality).lower() for modality in args.modalities}
        for manifest in manifest_rows:
            role = safe_text(manifest.get("manifest_role")) or "owp_candidate"
            modality = safe_text(manifest.get("target_modality")).lower()
            if allowed_roles and role not in allowed_roles:
                continue
            if allowed_modalities and modality not in allowed_modalities:
                continue
            sample_id = safe_text(manifest.get("sample_id"))
            runtime = explicit_lookup.get(sample_id)
            if runtime is None:
                continue
            row = dict(runtime) if keep_offline else strip_offline_eval_fields(runtime)
            row.update(
                {
                    "manifest_role": role,
                    "source_dataset": safe_text(manifest.get("source_dataset")) or safe_text(row.get("source_dataset")),
                    "target_modality": modality,
                    **manifest_runtime_fields(manifest, modality),
                }
            )
            if keep_offline:
                reference = (
                    manifest.get("reference_answer_offline")
                    or manifest.get("reference_answer")
                    or manifest.get("mad_protocol_reference_answer")
                    or row.get("reference_answer")
                )
                if safe_text(reference) == "__OTHER__":
                    row["reference_answer"] = "__OTHER__"
                else:
                    row["reference_answer"] = normalize_binary_label(reference)
                row["task_family"] = safe_text(manifest.get("task_family") or row.get("task_family"))
            for field in (
                "mad_protocol_baseline_answer",
                "mad_protocol_comparator_present",
                "base_protocol_answer",
                "manifest_acceptance_reason",
                "evidence_state",
                "prior_state",
                "conflict_state",
                "conflict_strength",
                "full_branch_answer",
                "target_branch_answer",
                "non_target_branch_answer",
                "text_branch_answer",
                "soft_conflict_score",
                "soft_conflict_cur_support",
                "soft_conflict_alt_support",
                "soft_conflict_disagree_count",
                "soft_conflict_full_margin",
                "target_only_risk_type",
                "target_only_policy_answer",
                "target_only_reference_answer",
                "target_only_baseline_answer",
                "target_only_mad_protocol_answer",
                "answer_axis_enabled",
                "answer_axis_kind",
                "answer_axis_yes_label",
                "answer_axis_no_label",
                "answer_axis_reference_label",
                "answer_axis_reference_outside",
                "answer_axis_current_label",
                "answer_axis_candidate_label",
                "answer_axis_candidate_source_branch",
                "answer_axis_candidate_source_modality",
                "answer_axis_yes_token",
                "answer_axis_no_token",
            ):
                if field in manifest:
                    row[field] = manifest.get(field)
            if keep_offline:
                for field in (
                    "benchmark",
                    "mad_protocol_scope",
                    "paper_faithful_scope",
                    "mad_protocol_answer",
                    "mad_protocol_reference_answer",
                    "mad_protocol_task",
                    "mad_protocol_category",
                    "mad_protocol_sub_category",
                ):
                    if field in manifest:
                        row[field] = manifest.get(field)
            out.append(row)
        return out

    audio_lookup = {
        safe_text(row.get("sample_id")): {**dict(row), "runtime_source": "audio_manifest"}
        for row in joined_audio_rows(manifest_path=args.audio_manifest, final_path=args.audio_final)
    }
    visual_lookup = build_visual_runtime_lookup(args)
    out: list[dict[str, Any]] = []
    allowed_roles = {safe_text(role) for role in args.roles}
    allowed_modalities = {safe_text(modality).lower() for modality in args.modalities}
    for manifest in manifest_rows:
        role = safe_text(manifest.get("manifest_role")) or "owp_candidate"
        modality = safe_text(manifest.get("target_modality")).lower()
        if allowed_roles and role not in allowed_roles:
            continue
        if allowed_modalities and modality not in allowed_modalities:
            continue
        sample_id = safe_text(manifest.get("sample_id"))
        runtime = visual_lookup.get(sample_id) if safe_text(manifest.get("source_dataset")) == "visual_fixed1000" else audio_lookup.get(sample_id)
        if runtime is None:
            continue
        row = dict(runtime) if keep_offline else strip_offline_eval_fields(runtime)
        row.update(
            {
                "manifest_role": role,
                "source_dataset": safe_text(manifest.get("source_dataset")),
                "target_modality": modality,
                **manifest_runtime_fields(manifest, modality),
            }
        )
        for field in (
            "manifest_acceptance_reason",
            "evidence_state",
            "prior_state",
            "conflict_state",
            "conflict_strength",
            "full_branch_answer",
            "target_branch_answer",
            "non_target_branch_answer",
            "text_branch_answer",
            "soft_conflict_score",
            "soft_conflict_cur_support",
            "soft_conflict_alt_support",
            "soft_conflict_disagree_count",
            "soft_conflict_full_margin",
            "target_only_risk_type",
            "target_only_policy_answer",
            "target_only_reference_answer",
            "target_only_baseline_answer",
            "target_only_mad_protocol_answer",
        ):
            if field in manifest:
                row[field] = manifest.get(field)
        if keep_offline:
            reference = manifest.get("reference_answer_offline")
            if safe_text(reference) == "__OTHER__":
                row["reference_answer"] = "__OTHER__"
            else:
                row["reference_answer"] = normalize_binary_label(reference)
            row["task_family"] = "visual_grounded_presence" if modality == "visual" else "audio_grounded_presence"
        out.append(row)
    return out


def choose_rows(rows: Sequence[dict[str, Any]], *, max_rows: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(safe_text(row.get("target_modality")), safe_text(row.get("manifest_role")))].append(dict(row))
    rng = random.Random(int(seed))
    for bucket in grouped.values():
        bucket.sort(key=lambda item: safe_text(item.get("sample_id")))
        rng.shuffle(bucket)
        # Small runs should exercise the conflict-conditioned path first. The
        # certificate is derived from model-view disagreement, not benchmark labels.
        bucket.sort(key=lambda item: bool(safe_text(item.get("evidence_certificate"))))
    keys = sorted(grouped)
    selected: list[dict[str, Any]] = []
    while keys and (int(max_rows) <= 0 or len(selected) < int(max_rows)):
        progressed = False
        for key in keys:
            bucket = grouped[key]
            if not bucket:
                continue
            selected.append(bucket.pop())
            progressed = True
            if int(max_rows) > 0 and len(selected) >= int(max_rows):
                break
        if not progressed:
            break
    selected.sort(key=lambda item: (safe_text(item.get("target_modality")), safe_text(item.get("manifest_role")), safe_text(item.get("sample_id"))))
    return selected


def unit(vec: torch.Tensor) -> torch.Tensor:
    return vec.float() / vec.float().norm().clamp_min(1.0e-12)


def scalar_projection(vec: torch.Tensor, direction: torch.Tensor) -> float:
    return float(torch.dot(vec.float(), unit(direction)).item())


def vector_projection(vec: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    d = unit(direction)
    return torch.dot(vec.float(), d) * d


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    an = a.float().norm()
    bn = b.float().norm()
    if float(an.item()) <= 1.0e-12 or float(bn.item()) <= 1.0e-12:
        return None
    return float(torch.dot(a.float(), b.float()).item() / (an.item() * bn.item()))


def direction_from_captures(
    captures: Mapping[str, Mapping[str, Any]],
    *,
    layer: int,
    mode: str,
) -> torch.Tensor:
    full = captures["full"]["layer_vectors"][int(layer)].float()
    target = captures["target"]["layer_vectors"][int(layer)].float()
    non_target = captures["non_target"]["layer_vectors"][int(layer)].float()
    text = captures["text"]["layer_vectors"][int(layer)].float()
    if mode == "target_minus_text":
        return target - text
    if mode == "text_minus_target":
        return text - target
    if mode == "target_minus_nontarget":
        return target - non_target
    if mode == "full_minus_nontarget":
        return full - non_target
    if mode == "nontarget_minus_target":
        return non_target - target
    if mode == "nontarget_minus_text":
        return non_target - text
    if mode == "full_minus_target":
        return full - target
    raise ValueError(f"unknown direction mode: {mode}")


def audit_layer(
    captures: Mapping[str, Mapping[str, Any]],
    *,
    layer: int,
    evidence_direction_mode: str,
    prior_direction_mode: str,
    betas: Sequence[float],
) -> dict[str, Any]:
    full = captures["full"]["layer_vectors"][int(layer)].float()
    target = captures["target"]["layer_vectors"][int(layer)].float()
    non_target = captures["non_target"]["layer_vectors"][int(layer)].float()
    text = captures["text"]["layer_vectors"][int(layer)].float()
    u_e = direction_from_captures(captures, layer=int(layer), mode=evidence_direction_mode)
    u_p_raw = direction_from_captures(captures, layer=int(layer), mode=prior_direction_mode)
    u_p_perp = u_p_raw - vector_projection(u_p_raw, u_e)
    e_full = scalar_projection(full, u_e)
    p_full_raw = scalar_projection(full, u_p_raw)
    p_full_perp = scalar_projection(full, u_p_perp)
    e_target = scalar_projection(target, u_e)
    e_non_target = scalar_projection(non_target, u_e)
    p_non_target = scalar_projection(non_target, u_p_perp)
    p_target = scalar_projection(target, u_p_perp)
    suppressions: dict[str, Any] = {}
    for beta in betas:
        h_supp = full - float(beta) * vector_projection(full, u_p_perp)
        e_after = scalar_projection(h_supp, u_e)
        p_after = scalar_projection(h_supp, u_p_perp)
        suppressions[f"beta={float(beta):.3f}"] = {
            "evidence_projection_after": e_after,
            "prior_projection_after": p_after,
            "evidence_damage": e_full - e_after,
            "prior_reduction": abs(p_full_perp) - abs(p_after),
        }
    return {
        "layer": int(layer),
        "evidence_direction_mode": evidence_direction_mode,
        "prior_direction_mode": prior_direction_mode,
        "norms": {
            "full": float(full.norm().item()),
            "target": float(target.norm().item()),
            "non_target": float(non_target.norm().item()),
            "text": float(text.norm().item()),
            "u_e": float(u_e.norm().item()),
            "u_p_raw": float(u_p_raw.norm().item()),
            "u_p_perp_e": float(u_p_perp.norm().item()),
        },
        "cosines": {
            "evidence_vs_prior_raw": cosine(u_e, u_p_raw),
            "evidence_vs_prior_perp": cosine(u_e, u_p_perp),
        },
        "projections": {
            "evidence_on_full": e_full,
            "prior_raw_on_full": p_full_raw,
            "prior_perp_on_full": p_full_perp,
            "evidence_on_target": e_target,
            "evidence_on_non_target": e_non_target,
            "prior_perp_on_non_target": p_non_target,
            "prior_perp_on_target": p_target,
            "target_evidence_advantage": e_target - e_non_target,
            "non_target_prior_advantage": p_non_target - p_target,
            "full_evidence_prior_gap": e_full - abs(p_full_perp),
        },
        "suppression_simulation": suppressions,
    }


def mean(values: Iterable[Any]) -> float | None:
    clean: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    return float(sum(clean) / len(clean)) if clean else None


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_records": len(records),
        "by_modality_role": {},
        "by_layer": {},
    }
    groupers = {
        "by_modality_role": lambda row: (safe_text(row.get("target_modality")), safe_text(row.get("manifest_role"))),
        "by_layer": lambda row: (str(row.get("layer")),),
    }
    for out_key, key_fn in groupers.items():
        grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[key_fn(record)].append(record)
        for key, bucket in sorted(grouped.items()):
            text = " | ".join(key)
            summary[out_key][text] = {
                "n": len(bucket),
                "mean_cos_e_p_raw": mean(row.get("cos_e_p_raw") for row in bucket),
                "mean_abs_cos_e_p_perp": mean(abs(float(row.get("cos_e_p_perp"))) for row in bucket if row.get("cos_e_p_perp") is not None),
                "mean_target_evidence_advantage": mean(row.get("target_evidence_advantage") for row in bucket),
                "mean_non_target_prior_advantage": mean(row.get("non_target_prior_advantage") for row in bucket),
                "mean_full_evidence_prior_gap": mean(row.get("full_evidence_prior_gap") for row in bucket),
                "mean_prior_reduction_beta1": mean(row.get("prior_reduction_beta1") for row in bucket),
                "mean_abs_evidence_damage_beta1": mean(abs(float(row.get("evidence_damage_beta1"))) for row in bucket if row.get("evidence_damage_beta1") is not None),
            }
    return summary


def flatten_record(row: Mapping[str, Any], layer_audit: Mapping[str, Any]) -> dict[str, Any]:
    suppress = layer_audit.get("suppression_simulation") if isinstance(layer_audit.get("suppression_simulation"), Mapping) else {}
    beta1 = suppress.get("beta=1.000") if isinstance(suppress.get("beta=1.000"), Mapping) else {}
    projections = layer_audit.get("projections") if isinstance(layer_audit.get("projections"), Mapping) else {}
    cosines = layer_audit.get("cosines") if isinstance(layer_audit.get("cosines"), Mapping) else {}
    out = {
        "sample_id": row.get("sample_id"),
        "source_dataset": row.get("source_dataset"),
        "manifest_role": row.get("manifest_role"),
        "target_modality": row.get("target_modality"),
        "current_answer_y0": row.get("current_answer_y0"),
        "candidate_answer_y": row.get("candidate_answer_y"),
        "evidence_certificate": row.get("evidence_certificate"),
        "current_best_accepts": row.get("current_best_accepts"),
        "layer": layer_audit.get("layer"),
        "cos_e_p_raw": cosines.get("evidence_vs_prior_raw"),
        "cos_e_p_perp": cosines.get("evidence_vs_prior_perp"),
        "target_evidence_advantage": projections.get("target_evidence_advantage"),
        "non_target_prior_advantage": projections.get("non_target_prior_advantage"),
        "full_evidence_prior_gap": projections.get("full_evidence_prior_gap"),
        "evidence_damage_beta1": beta1.get("evidence_damage"),
        "prior_reduction_beta1": beta1.get("prior_reduction"),
        "layer_audit": layer_audit,
    }
    if row.get("reference_answer") is not None:
        out["reference_answer_offline"] = row.get("reference_answer")
    return out


def run() -> None:
    args = parse_args()
    # Keep visual replay explicit so this audit remains tied to the same fixed-1000 visual policy.
    args.manifest_visual_replay = (
        ROOT
        / "results/visual_multiview_consistency_proposal_1000_20260602"
        / "replay_visual_prior_arbitration_v1_temporal_compact031_generated_crop_bounded_temporal_color_attr_nli_midlocal_pe_claim_temporal_fullnonlocal_1000_20260603"
        / "rows.jsonl"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_jsonl(args.manifest)
    rows = runtime_rows(args, manifest_rows)
    selected = choose_rows(rows, max_rows=int(args.max_rows), seed=int(args.seed))
    layers = sorted({int(layer) for layer in args.layers})
    betas = stable_unique_floats(args.betas)
    write_jsonl(args.output_dir / "selected_rows.jsonl", selected)
    run_config = {
        "manifest": str(args.manifest),
        "audio_manifest": str(args.audio_manifest),
        "audio_final": str(args.audio_final),
        "avh_qa": str(args.avh_qa),
        "avh_video_dir": str(args.avh_video_dir),
        "output_dir": str(args.output_dir),
        "model_path": args.model_path,
        "device": args.device,
        "seed": int(args.seed),
        "max_rows": int(args.max_rows),
        "roles": list(args.roles),
        "modalities": list(args.modalities),
        "layers": layers,
        "betas": betas,
        "evidence_direction": args.evidence_direction,
        "prior_direction": args.prior_direction,
        "target_capture_prompt": args.target_capture_prompt,
        "non_target_capture_prompt": args.non_target_capture_prompt,
        "video_budget": video_budget(args),
        "n_manifest_rows": len(manifest_rows),
        "n_runtime_joined_rows": len(rows),
        "n_selected_rows": len(selected),
        "runtime_uses_reference_answer": False,
        "runtime_uses_task_family": False,
        "offline_eval_fields_in_runtime": include_offline_eval_fields(args),
        "selected_by_modality_role": dict(Counter(f"{row.get('target_modality')}|{row.get('manifest_role')}" for row in selected)),
    }
    write_json(args.output_dir / "run_config.json", run_config)
    if args.plan_only:
        print(f"[qwen-owp-audit] plan-only selected={len(selected)} output_dir={args.output_dir}")
        return

    adapter = QwenOmniAdapter(model_path=args.model_path, device=args.device, max_new_tokens=4)
    yes_id, no_id = yes_no_token_ids(adapter)
    budget = video_budget(args)
    records: list[dict[str, Any]] = []
    branch_records: list[dict[str, Any]] = []
    iterator = tqdm(
        selected,
        total=len(selected),
        desc="qwen_owp_audit",
        unit="sample",
        disable=bool(args.no_progress),
        dynamic_ncols=True,
    )
    try:
        for row in iterator:
            target_branch = target_branch_for(row)
            non_target_branch = non_target_branch_for(row)
            capture_specs = {
                "full": ("full", row),
                "target": (target_branch, capture_prompt_row(row, prompt_mode=args.target_capture_prompt)),
                "non_target": (non_target_branch, capture_prompt_row(row, prompt_mode=args.non_target_capture_prompt)),
                "text": ("text_only", row),
            }
            captures: dict[str, dict[str, Any]] = {}
            for role, (branch, capture_row) in capture_specs.items():
                captured = capture_branch(
                    adapter,
                    row=capture_row,
                    branch=branch,
                    layers=layers,
                    yes_id=yes_id,
                    no_id=no_id,
                    budget=budget,
                )
                captures[role] = captured
                branch_records.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "target_modality": row.get("target_modality"),
                        "manifest_role": row.get("manifest_role"),
                        "capture_role": role,
                        "branch": branch,
                        "prediction": captured.get("final_prediction"),
                        "margin_yes_minus_no": captured.get("final_yes_no_margin"),
                        "candidate_aligned_margin": signed_margin_for_answer(
                            float(captured.get("final_yes_no_margin")),
                            row.get("candidate_answer_y"),
                        ),
                        "current_aligned_margin": signed_margin_for_answer(
                            float(captured.get("final_yes_no_margin")),
                            row.get("current_answer_y0"),
                        ),
                        "token_count": captured.get("token_count"),
                    }
                )
            for layer in layers:
                layer_audit = audit_layer(
                    captures,
                    layer=int(layer),
                    evidence_direction_mode=args.evidence_direction,
                    prior_direction_mode=args.prior_direction,
                    betas=betas,
                )
                records.append(flatten_record(row, layer_audit))
            iterator.set_postfix(mod=row.get("target_modality"), role=row.get("manifest_role"))
    finally:
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "kind": "qwen_owp_carrier_audit",
        "run_config": run_config,
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "branch_summary": {
            "n": len(branch_records),
            "by_modality_role_branch": {
                key: {
                    "n": len(bucket),
                    "mean_candidate_aligned_margin": mean(row.get("candidate_aligned_margin") for row in bucket),
                    "mean_current_aligned_margin": mean(row.get("current_aligned_margin") for row in bucket),
                }
                for key, bucket in sorted(
                    (
                        key,
                        [
                            row
                            for row in branch_records
                            if f"{row.get('target_modality')}|{row.get('manifest_role')}|{row.get('capture_role')}" == key
                        ],
                    )
                    for key in sorted(
                        {
                            f"{row.get('target_modality')}|{row.get('manifest_role')}|{row.get('capture_role')}"
                            for row in branch_records
                        }
                    )
                )
            },
        },
        "coupling_summary": summarize(records),
    }
    write_jsonl(args.output_dir / "branch_records.jsonl", branch_records)
    write_jsonl(args.output_dir / "coupling_records.jsonl", records)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["coupling_summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
