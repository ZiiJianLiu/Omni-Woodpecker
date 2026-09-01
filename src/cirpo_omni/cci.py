from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cirpo_omni.runtime import (
    DEFAULT_CIRPO_VIDEO_BUDGET,
    compute_sequence_logp_stats,
    get_cirpo_train_model,
    prepare_generated_branch_inputs,
    resolve_episode_contexts,
    resolve_model_input_device,
)
from omni_dpo.media import MediaResolver
from omni_dpo.modeling import compose_question_text, messages_to_inputs
from omni_dpo.preference_pairs import ABSTAIN_TEMPLATE, safe_text, write_json


CCI_CONFLICT_STATE_NO_CONFLICT = "no_conflict"
CCI_CONFLICT_STATE_RETARGET = "retarget_required"
CCI_CONFLICT_STATE_MISMATCH = "mismatch_required"
CCI_CONFLICT_STATE_ABSTAIN = "abstain_legal"
CCI_TRAINING_BUCKET_RETARGET = "retarget_positive"
CCI_TRAINING_BUCKET_MISMATCH = "mismatch_legal"
CCI_TRAINING_BUCKET_PRESERVE = "preserve_control"

CCI_CONFLICT_STATES = (
    CCI_CONFLICT_STATE_NO_CONFLICT,
    CCI_CONFLICT_STATE_RETARGET,
    CCI_CONFLICT_STATE_MISMATCH,
    CCI_CONFLICT_STATE_ABSTAIN,
)
CCI_DECISIVE_MODALITIES = (
    "audio",
    "visual",
    "cross_modal_relation",
    "unknown",
)
CCI_BRANCH_ROLES = (
    "full",
    "target_corruption",
    "control_corruption",
    "donor",
)
CCI_MODALITY_VALUES = (
    "audio",
    "visual",
    "cross_modal_relation",
    "none",
)
CCI_TRANSITION_VALUES = (
    "preserve",
    "retarget",
    "mismatch",
    "abstain",
)


def _one_hot(value: str, choices: Sequence[str], *, device: torch.device) -> torch.Tensor:
    out = torch.zeros(len(choices), device=device, dtype=torch.float32)
    try:
        out[choices.index(value)] = 1.0
    except ValueError:
        if choices:
            out[-1] = 1.0
    return out


def _replace_layer_output(output: Any, hidden_states: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden_states, *output[1:])
    if isinstance(output, list):
        return [hidden_states, *output[1:]]
    return hidden_states


def candidate_layer_ids_from_model(model, *, num_last_layers: int = 8) -> List[int]:
    thinker = get_cirpo_train_model(model)
    layers = getattr(getattr(thinker, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("CCI requires thinker.model.layers to be available on the loaded model.")
    total_layers = len(layers)
    num_last_layers = max(1, min(total_layers, int(num_last_layers)))
    start = max(0, total_layers - num_last_layers)
    return list(range(start, total_layers))


def encode_branch_metadata(
    quartet: Dict[str, Any],
    *,
    branch_role: str,
    device: torch.device,
) -> torch.Tensor:
    decisive_modality = safe_text(quartet.get("decisive_modality")) or "unknown"
    corrupted_modality = safe_text(quartet.get("corrupted_modality")) or "none"
    preserved_modality = safe_text(quartet.get("preserved_modality")) or "none"
    expected_transition = safe_text(quartet.get("expected_transition")) or "preserve"
    donor_available = 1.0 if bool(quartet.get("donor_available")) else 0.0
    return torch.cat(
        [
            _one_hot(decisive_modality, CCI_DECISIVE_MODALITIES, device=device),
            _one_hot(corrupted_modality, CCI_MODALITY_VALUES, device=device),
            _one_hot(preserved_modality, CCI_MODALITY_VALUES, device=device),
            _one_hot(expected_transition, CCI_TRANSITION_VALUES, device=device),
            _one_hot(branch_role, CCI_BRANCH_ROLES, device=device),
            torch.tensor([donor_available], device=device, dtype=torch.float32),
        ],
        dim=0,
    ).unsqueeze(0)


def target_conflict_state(quartet: Dict[str, Any], *, branch_role: str) -> str:
    if branch_role != "target_corruption":
        return CCI_CONFLICT_STATE_NO_CONFLICT
    expected_transition = safe_text(quartet.get("expected_transition"))
    if expected_transition == "retarget":
        return CCI_CONFLICT_STATE_RETARGET
    if expected_transition == "mismatch":
        return CCI_CONFLICT_STATE_MISMATCH
    if expected_transition == "abstain":
        return CCI_CONFLICT_STATE_ABSTAIN
    return CCI_CONFLICT_STATE_NO_CONFLICT


def target_decisive_modality(quartet: Dict[str, Any]) -> str:
    decisive = safe_text(quartet.get("decisive_modality"))
    return decisive if decisive in CCI_DECISIVE_MODALITIES[:-1] else "unknown"


def branch_action_target(quartet: Dict[str, Any], *, branch_role: str) -> str:
    if branch_role == "target_corruption":
        action = safe_text(quartet.get("expected_transition"))
    else:
        action = "preserve"
    return action if action in CCI_TRANSITION_VALUES else "preserve"


def decode_action_logits(action_logits: torch.Tensor) -> str:
    return CCI_TRANSITION_VALUES[int(action_logits.argmax(dim=-1).detach().item())]


def action_hint_text_for_branch(action: str, *, branch_role: str) -> str:
    if branch_role != "target_corruption":
        return ""
    normalized = action if action in CCI_TRANSITION_VALUES else "preserve"
    if normalized == "retarget":
        return (
            "Conflict-routing instruction: the decisive evidence changed. "
            "Update the answer to reflect the changed evidence rather than copying the original answer."
        )
    if normalized == "mismatch":
        return (
            "Conflict-routing instruction: the available evidence no longer supports the original alignment. "
            "Prefer the explicit mismatch option over copying the original answer."
        )
    if normalized == "abstain":
        return (
            "Conflict-routing instruction: the decisive evidence is not recoverable. "
            "Abstain conservatively instead of inventing a new answer."
        )
    return (
        "Conflict-routing instruction: the preserved evidence remains sufficient. "
        "Keep the supported conclusion and do not introduce a new conflict."
    )


def with_action_hint(context: Dict[str, Any], *, action: str, branch_role: str) -> Dict[str, Any]:
    hint = action_hint_text_for_branch(action, branch_role=branch_role)
    if not hint:
        return dict(context)
    question_text = safe_text(context.get("question_text"))
    if not question_text:
        return {**context, "question_text": hint}
    return {**context, "question_text": f"{question_text}\n\n{hint}".strip()}


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, list):
        return output[0]
    return output


@dataclass
class BranchTrace:
    layer_records: Dict[str, Dict[str, torch.Tensor]]
    decisive_logits: torch.Tensor
    conflict_logits: torch.Tensor
    action_logits: torch.Tensor


class LayerArbitrationBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        metadata_dim: int,
        state_dim: int = 32,
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.state_proj = nn.Linear(hidden_size, state_dim * 3)
        self.down_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_size)
        self.gate_proj = nn.Linear(state_dim * 3 + metadata_dim, 1)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: torch.Tensor,
        intervene: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        block_dtype = self.norm.weight.dtype
        block_device = self.norm.weight.device
        hidden_dtype = hidden_states.dtype
        pooled_pre = hidden_states.mean(dim=1)
        pooled_norm = self.norm(pooled_pre.to(device=block_device, dtype=block_dtype))
        state_tensor = self.state_proj(pooled_norm)
        audio_state, visual_state, relation_state = torch.chunk(state_tensor, 3, dim=-1)
        metadata = metadata.to(device=block_device, dtype=block_dtype)
        gate_input = torch.cat([audio_state, visual_state, relation_state, metadata], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        correction = self.up_proj(torch.tanh(self.down_proj(pooled_norm)))
        if intervene:
            updated_hidden = hidden_states + gate.to(dtype=hidden_dtype).unsqueeze(1) * correction.to(dtype=hidden_dtype).unsqueeze(1)
        else:
            updated_hidden = hidden_states
        pooled_post = updated_hidden.mean(dim=1)
        record = {
            "pooled_pre": pooled_pre,
            "pooled_post": pooled_post,
            "audio_state": audio_state,
            "visual_state": visual_state,
            "relation_state": relation_state,
            "gate": gate,
            "delta_norm": correction.norm(dim=-1, keepdim=True),
        }
        return updated_hidden, record


class ConflictArbitrationAdapter(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        layer_ids: Sequence[int],
        metadata_dim: int,
        state_dim: int = 32,
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.layer_ids = [int(layer_id) for layer_id in layer_ids]
        self.metadata_dim = int(metadata_dim)
        self.blocks = nn.ModuleDict(
            {
                str(layer_id): LayerArbitrationBlock(
                    hidden_size=hidden_size,
                    metadata_dim=metadata_dim,
                    state_dim=state_dim,
                    bottleneck_dim=bottleneck_dim,
                )
                for layer_id in self.layer_ids
            }
        )
        combined_dim = state_dim * 3 + metadata_dim
        self.decisive_head = nn.Linear(combined_dim, len(CCI_DECISIVE_MODALITIES))
        self.conflict_head = nn.Linear(combined_dim, len(CCI_CONFLICT_STATES))
        self.action_head = nn.Linear(combined_dim, len(CCI_TRANSITION_VALUES))

    def apply_to_layer(
        self,
        *,
        layer_id: int,
        hidden_states: torch.Tensor,
        metadata: torch.Tensor,
        intervene: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        block = self.blocks[str(int(layer_id))]
        if next(block.parameters()).device != hidden_states.device:
            block.to(hidden_states.device)
        return block(hidden_states, metadata=metadata.to(hidden_states.device), intervene=intervene)

    def classify(
        self,
        layer_records: Dict[str, Dict[str, torch.Tensor]],
        *,
        metadata: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        head_device = self.decisive_head.weight.device
        head_dtype = self.decisive_head.weight.dtype
        audio_states = []
        visual_states = []
        relation_states = []
        for layer_id in self.layer_ids:
            record = layer_records.get(str(layer_id))
            if record is None:
                continue
            audio_states.append(record["audio_state"].to(device=head_device, dtype=head_dtype))
            visual_states.append(record["visual_state"].to(device=head_device, dtype=head_dtype))
            relation_states.append(record["relation_state"].to(device=head_device, dtype=head_dtype))
        if not audio_states:
            raise RuntimeError("CCI classification requires at least one collected layer record.")
        pooled_audio = torch.stack(audio_states).mean(dim=0)
        pooled_visual = torch.stack(visual_states).mean(dim=0)
        pooled_relation = torch.stack(relation_states).mean(dim=0)
        summary = torch.cat([pooled_audio, pooled_visual, pooled_relation, metadata.to(device=head_device, dtype=head_dtype)], dim=-1)
        decisive_logits = self.decisive_head(summary)
        conflict_logits = self.conflict_head(summary)
        action_logits = self.action_head(summary)
        return {
            "decisive_logits": decisive_logits,
            "conflict_logits": conflict_logits,
            "action_logits": action_logits,
        }

    def identity_regularization(self, layer_records: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        head_device = self.decisive_head.weight.device
        penalties = [
            record["gate"].to(head_device).pow(2).mean() + 0.05 * record["delta_norm"].to(head_device).pow(2).mean()
            for record in layer_records.values()
        ]
        if not penalties:
            return torch.tensor(0.0, device=head_device)
        return torch.stack(penalties).mean()


class LayerInterventionSession:
    def __init__(
        self,
        *,
        model,
        layer_ids: Sequence[int],
        callback: Callable[[int, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    ) -> None:
        self._model = get_cirpo_train_model(model)
        self._layer_ids = [int(layer_id) for layer_id in layer_ids]
        self._callback = callback
        self._handles: List[Any] = []
        self.records: Dict[str, Dict[str, torch.Tensor]] = {}

    def __enter__(self) -> "LayerInterventionSession":
        layers = getattr(getattr(self._model, "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("CCI layer intervention requires thinker.model.layers.")
        for layer_id in self._layer_ids:
            handle = layers[layer_id].register_forward_hook(self._make_hook(layer_id))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_id: int) -> Callable[[nn.Module, Tuple[Any, ...], Any], Any]:
        def _hook(_module, _inputs, output):
            hidden_states = _first_tensor(output)
            updated_hidden, record = self._callback(int(layer_id), hidden_states)
            self.records[str(layer_id)] = record
            return _replace_layer_output(output, updated_hidden)

        return _hook


@torch.no_grad()
def generate_branch_response_with_cci(
    *,
    model,
    processor,
    context: Dict[str, Any],
    adapter: ConflictArbitrationAdapter,
    metadata: torch.Tensor,
    layer_ids: Sequence[int],
    max_new_tokens: int = 48,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: Optional[torch.device] = None,
    video_budget: Optional[Dict[str, float]] = None,
) -> str:
    if device is None:
        device = resolve_model_input_device(model)
    prompt_inputs = _build_prompt_inputs(
        processor=processor,
        context=context,
        device=device,
        video_budget=video_budget,
    )

    def _callback(layer_id: int, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return adapter.apply_to_layer(
            layer_id=layer_id,
            hidden_states=hidden_states,
            metadata=metadata.to(hidden_states.device),
            intervene=True,
        )

    input_len = prompt_inputs["input_ids"].shape[-1]
    with LayerInterventionSession(model=model, layer_ids=layer_ids, callback=_callback):
        gen_ids = model.generate(
            **prompt_inputs,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            return_audio=False,
            thinker_max_new_tokens=max_new_tokens,
        )
    new_ids = gen_ids[:, input_len:]
    return processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def predict_branch_action(
    *,
    model,
    processor,
    context: Dict[str, Any],
    adapter: ConflictArbitrationAdapter,
    metadata: torch.Tensor,
    layer_ids: Sequence[int],
    device: Optional[torch.device] = None,
    video_budget: Optional[Dict[str, float]] = None,
) -> Tuple[str, BranchTrace]:
    trace = collect_prompt_trace(
        model=model,
        processor=processor,
        context=context,
        layer_ids=layer_ids,
        device=device,
        video_budget=video_budget,
        adapter=adapter,
        metadata=metadata,
        intervene=True,
    )
    return decode_action_logits(trace.action_logits), trace


def _build_prompt_inputs(
    *,
    processor,
    context: Dict[str, Any],
    device: torch.device,
    video_budget: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": (
                ([{"type": "video", "video": context.get("video_path")}] if context.get("video_path") else [])
                + ([{"type": "audio", "audio": "placeholder"}] if context.get("audio_array") is not None else [])
                + [{"type": "text", "text": context["question_text"]}]
            ),
        },
    ]
    return messages_to_inputs(
        processor=processor,
        messages=messages,
        audio_array=context.get("audio_array"),
        device=device,
        video_budget=video_budget or DEFAULT_CIRPO_VIDEO_BUDGET,
    )


def resolve_quartet_contexts(
    *,
    episode: Dict[str, Any],
    quartet: Dict[str, Any],
    resolver: MediaResolver,
) -> Dict[str, Dict[str, Any]]:
    contexts = resolve_episode_contexts(episode=episode, resolver=resolver)
    out = {
        "full": contexts[safe_text((episode.get("full_branch") or {}).get("branch_id"))],
        "target": contexts[safe_text(quartet.get("target_branch_id"))],
    }
    control_branch_id = safe_text(quartet.get("control_branch_id"))
    if control_branch_id:
        out["control"] = contexts[control_branch_id]
    if bool(quartet.get("donor_available")):
        donor_pair = {
            "dataset": safe_text(quartet.get("donor_dataset")),
            "sample_id": safe_text(quartet.get("donor_sample_id")),
            "input": {
                "prompt": quartet.get("donor_prompt") or {},
                "media": quartet.get("donor_media") or {},
                "intervention": None,
            },
        }
        donor_prepared = resolver.prepare_pair_inputs(donor_pair)
        out["donor"] = {
            "question_text": compose_question_text(quartet.get("donor_prompt") or {}),
            "video_path": donor_prepared.get("video_path"),
            "audio_array": donor_prepared.get("audio_array"),
        }
    return out


def target_answer_for_quartet(episode: Dict[str, Any], quartet: Dict[str, Any]) -> str:
    branch_legality = ((episode.get("target_legality_spec") or {}).get("branch_legality") or {}).get(
        safe_text(quartet.get("target_branch_id"))
    ) or {}
    preferred_answer = safe_text(branch_legality.get("preferred_answer")) or safe_text(quartet.get("donor_supported_answer"))
    preferred_action = safe_text(branch_legality.get("preferred_action"))
    if preferred_action == "abstain":
        return ABSTAIN_TEMPLATE
    if preferred_answer:
        return preferred_answer
    if preferred_action == "mismatch":
        mismatch_answer = safe_text(branch_legality.get("mismatch_answer"))
        return mismatch_answer or ABSTAIN_TEMPLATE
    return safe_text(episode.get("clean_reference")) or ABSTAIN_TEMPLATE


def collect_prompt_trace(
    *,
    model,
    processor,
    context: Dict[str, Any],
    layer_ids: Sequence[int],
    device: Optional[torch.device] = None,
    video_budget: Optional[Dict[str, float]] = None,
    adapter: Optional[ConflictArbitrationAdapter] = None,
    metadata: Optional[torch.Tensor] = None,
    intervene: bool = False,
) -> BranchTrace:
    if device is None:
        device = resolve_model_input_device(model)
    prompt_inputs = _build_prompt_inputs(
        processor=processor,
        context=context,
        device=device,
        video_budget=video_budget,
    )
    forward_model = get_cirpo_train_model(model)

    if adapter is None:
        attention_mask = prompt_inputs.get("attention_mask")
        layer_records: Dict[str, Dict[str, torch.Tensor]] = {}

        def _record_only_callback(layer_id: int, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            if attention_mask is not None:
                mask = attention_mask.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            else:
                pooled = hidden_states.mean(dim=1)
            pooled_cpu = pooled.detach().float().cpu()
            zeros = torch.zeros((pooled_cpu.shape[0], 1), dtype=pooled_cpu.dtype)
            record = {
                "pooled_pre": pooled_cpu,
                "pooled_post": pooled_cpu,
                "audio_state": pooled_cpu,
                "visual_state": pooled_cpu,
                "relation_state": pooled_cpu,
                "gate": zeros,
                "delta_norm": zeros,
            }
            return hidden_states, record

        with torch.no_grad():
            with LayerInterventionSession(model=model, layer_ids=layer_ids, callback=_record_only_callback) as session:
                forward_model(
                    **prompt_inputs,
                    use_cache=False,
                    return_dict=True,
                )
                layer_records = dict(session.records)

        decisive_logits = torch.zeros((1, len(CCI_DECISIVE_MODALITIES)), dtype=torch.float32)
        conflict_logits = torch.zeros((1, len(CCI_CONFLICT_STATES)), dtype=torch.float32)
        action_logits = torch.zeros((1, len(CCI_TRANSITION_VALUES)), dtype=torch.float32)
        return BranchTrace(
            layer_records=layer_records,
            decisive_logits=decisive_logits,
            conflict_logits=conflict_logits,
            action_logits=action_logits,
        )

    if metadata is None:
        raise ValueError("CCI prompt tracing with an adapter requires metadata.")

    def _callback(layer_id: int, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return adapter.apply_to_layer(
            layer_id=layer_id,
            hidden_states=hidden_states,
            metadata=metadata.to(hidden_states.device),
            intervene=intervene,
        )

    with LayerInterventionSession(model=model, layer_ids=layer_ids, callback=_callback) as session:
        forward_model(**prompt_inputs, use_cache=False, return_dict=True)
        logits = adapter.classify(session.records, metadata=metadata.to(device))
    return BranchTrace(
        layer_records=session.records,
        decisive_logits=logits["decisive_logits"],
        conflict_logits=logits["conflict_logits"],
        action_logits=logits["action_logits"],
    )


def compute_answer_mean_logp(
    *,
    model,
    processor,
    context: Dict[str, Any],
    answer_text: str,
    device: Optional[torch.device] = None,
    video_budget: Optional[Dict[str, float]] = None,
    adapter: Optional[ConflictArbitrationAdapter] = None,
    metadata: Optional[torch.Tensor] = None,
    layer_ids: Optional[Sequence[int]] = None,
    intervene: bool = True,
) -> torch.Tensor:
    if device is None:
        device = resolve_model_input_device(model)
    model_inputs = prepare_generated_branch_inputs(
        processor=processor,
        question_text=context["question_text"],
        answer_text=answer_text,
        video_path=context.get("video_path"),
        audio_array=context.get("audio_array"),
        device=device,
        video_budget=video_budget or DEFAULT_CIRPO_VIDEO_BUDGET,
    )
    if adapter is None or not layer_ids:
        stats = compute_sequence_logp_stats(model=model, model_inputs=model_inputs)
        return stats["mean_logp"].mean()
    if metadata is None:
        raise ValueError("CCI answer scoring with an adapter requires metadata.")

    def _callback(layer_id: int, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return adapter.apply_to_layer(
            layer_id=layer_id,
            hidden_states=hidden_states,
            metadata=metadata.to(hidden_states.device),
            intervene=intervene,
        )

    with LayerInterventionSession(model=model, layer_ids=layer_ids, callback=_callback):
        stats = compute_sequence_logp_stats(model=model, model_inputs=model_inputs)
    return stats["mean_logp"].mean()


def compute_branch_losses(
    *,
    adapter: ConflictArbitrationAdapter,
    quartet: Dict[str, Any],
    branch_role: str,
    trace: BranchTrace,
) -> Dict[str, torch.Tensor]:
    decisive_target = torch.tensor(
        [CCI_DECISIVE_MODALITIES.index(target_decisive_modality(quartet))],
        device=trace.decisive_logits.device,
    )
    conflict_target = torch.tensor(
        [CCI_CONFLICT_STATES.index(target_conflict_state(quartet, branch_role=branch_role))],
        device=trace.conflict_logits.device,
    )
    action_target = torch.tensor(
        [CCI_TRANSITION_VALUES.index(branch_action_target(quartet, branch_role=branch_role))],
        device=trace.action_logits.device,
    )
    decisive_loss = F.cross_entropy(trace.decisive_logits, decisive_target)
    conflict_loss = F.cross_entropy(trace.conflict_logits, conflict_target)
    action_loss = F.cross_entropy(trace.action_logits, action_target)
    return {
        "decisive_loss": decisive_loss,
        "conflict_loss": conflict_loss,
        "action_loss": action_loss,
        "decisive_correct": (trace.decisive_logits.argmax(dim=-1) == decisive_target).float().mean(),
        "conflict_correct": (trace.conflict_logits.argmax(dim=-1) == conflict_target).float().mean(),
        "action_correct": (trace.action_logits.argmax(dim=-1) == action_target).float().mean(),
    }


def compute_transport_consistency(
    *,
    quartet: Dict[str, Any],
    clean_trace: BranchTrace,
    target_trace: BranchTrace,
    donor_trace: Optional[BranchTrace],
    control_trace: Optional[BranchTrace],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    per_layer_scores: List[torch.Tensor] = []
    detail: Dict[str, float] = {}
    corrupted_modality = safe_text(quartet.get("corrupted_modality"))
    decisive_modality = safe_text(quartet.get("decisive_modality"))
    expected_transition = safe_text(quartet.get("expected_transition"))

    for layer_id, target_record in target_trace.layer_records.items():
        clean_record = clean_trace.layer_records.get(layer_id)
        if clean_record is None:
            continue
        layer_loss = torch.tensor(0.0, device=target_record["pooled_post"].device)
        if corrupted_modality == "audio":
            if donor_trace is not None and layer_id in donor_trace.layer_records and expected_transition == "retarget":
                layer_loss = layer_loss + F.mse_loss(
                    target_record["audio_state"],
                    donor_trace.layer_records[layer_id]["audio_state"],
                )
            elif expected_transition == "abstain":
                layer_loss = layer_loss + target_record["audio_state"].pow(2).mean()
            layer_loss = layer_loss + F.mse_loss(
                target_record["visual_state"],
                clean_record["visual_state"],
            )
        elif corrupted_modality == "visual":
            if donor_trace is not None and layer_id in donor_trace.layer_records and expected_transition == "retarget":
                layer_loss = layer_loss + F.mse_loss(
                    target_record["visual_state"],
                    donor_trace.layer_records[layer_id]["visual_state"],
                )
            elif expected_transition == "abstain":
                layer_loss = layer_loss + target_record["visual_state"].pow(2).mean()
            layer_loss = layer_loss + F.mse_loss(
                target_record["audio_state"],
                clean_record["audio_state"],
            )
        elif decisive_modality == "cross_modal_relation":
            relation_delta = torch.norm(target_record["relation_state"] - clean_record["relation_state"], dim=-1).mean()
            layer_loss = layer_loss + F.relu(0.5 - relation_delta)
        if control_trace is not None and layer_id in control_trace.layer_records:
            control_record = control_trace.layer_records[layer_id]
            layer_loss = layer_loss + 0.5 * F.mse_loss(control_record["pooled_post"], clean_record["pooled_post"])
        per_layer_scores.append(layer_loss)
        detail[f"layer_{layer_id}_transport_loss"] = float(layer_loss.detach().item())
    if not per_layer_scores:
        zero = torch.tensor(0.0)
        return zero, {"transport_consistency_score": 0.0}
    transport_loss = torch.stack(per_layer_scores).mean()
    detail["transport_consistency_score"] = float(torch.exp(-transport_loss.detach()).item())
    return transport_loss, detail


def compute_patch_gain(
    *,
    model,
    processor,
    context: Dict[str, Any],
    answer_text: str,
    patch_layer_id: int,
    patch_delta: torch.Tensor,
    device: Optional[torch.device] = None,
    video_budget: Optional[Dict[str, float]] = None,
    adapter: Optional[ConflictArbitrationAdapter] = None,
    metadata: Optional[torch.Tensor] = None,
) -> float:
    if device is None:
        device = resolve_model_input_device(model)
    base_logp = compute_answer_mean_logp(
        model=model,
        processor=processor,
        context=context,
        answer_text=answer_text,
        device=device,
        video_budget=video_budget,
        adapter=adapter,
        metadata=metadata,
        layer_ids=[patch_layer_id] if adapter is not None else None,
    )

    def _callback(_layer_id: int, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        updated = hidden_states
        pooled = hidden_states.mean(dim=1)
        if adapter is not None and metadata is not None:
            updated, record = adapter.apply_to_layer(
                layer_id=patch_layer_id,
                hidden_states=updated,
                metadata=metadata.to(hidden_states.device),
                intervene=True,
            )
            pooled = record["pooled_pre"]
        delta = patch_delta.to(hidden_states.device, dtype=hidden_states.dtype).unsqueeze(1)
        updated = updated + delta
        pooled = hidden_states.mean(dim=1)
        return updated, {
            "pooled_pre": pooled,
            "pooled_post": updated.mean(dim=1),
            "audio_state": pooled,
            "visual_state": pooled,
            "relation_state": pooled,
            "gate": torch.ones((pooled.shape[0], 1), device=pooled.device, dtype=pooled.dtype),
            "delta_norm": patch_delta.to(pooled.device, dtype=pooled.dtype).norm().reshape(1, 1),
        }

    model_inputs = prepare_generated_branch_inputs(
        processor=processor,
        question_text=context["question_text"],
        answer_text=answer_text,
        video_path=context.get("video_path"),
        audio_array=context.get("audio_array"),
        device=device,
        video_budget=video_budget or DEFAULT_CIRPO_VIDEO_BUDGET,
    )
    with LayerInterventionSession(model=model, layer_ids=[patch_layer_id], callback=_callback):
        patched_logp = compute_sequence_logp_stats(model=model, model_inputs=model_inputs)["mean_logp"].mean()
    return float((patched_logp - base_logp).detach().item())


def select_conflict_layers_from_ranking(
    ranking: Sequence[Dict[str, Any]],
    *,
    top_k: Optional[int] = None,
    require_positive_patch_gain: bool = True,
) -> List[int]:
    ranked_rows = list(ranking)
    if not ranked_rows:
        return []
    if require_positive_patch_gain:
        selected = [int(row["layer_id"]) for row in ranked_rows if float(row.get("patch_gain_mean", 0.0)) > 0.0]
    else:
        selected = [int(row["layer_id"]) for row in ranked_rows]
    if not selected:
        selected = [int(ranked_rows[0]["layer_id"])]
    if top_k is not None:
        return selected[: max(1, int(top_k))]
    return selected


def discover_conflict_layers(
    *,
    model,
    processor,
    resolver: MediaResolver,
    episodes: Sequence[Dict[str, Any]],
    limit: int = 24,
    num_last_layers: int = 8,
    device: Optional[torch.device] = None,
    video_budget: Optional[Dict[str, float]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if device is None:
        device = resolve_model_input_device(model)
    layer_ids = candidate_layer_ids_from_model(model, num_last_layers=num_last_layers)
    layer_stats: Dict[int, Dict[str, float]] = {
        layer_id: {
            "patch_gain_sum": 0.0,
            "patch_gain_count": 0.0,
            "clean_control_distance_sum": 0.0,
            "clean_control_distance_count": 0.0,
        }
        for layer_id in layer_ids
    }
    analyzed = 0
    for episode in episodes:
        quartets = ((episode.get("quartet_spec") or {}).get("target_quartets") or [])
        if not quartets:
            continue
        for quartet in quartets:
            contexts = resolve_quartet_contexts(episode=episode, quartet=quartet, resolver=resolver)
            clean_trace = collect_prompt_trace(
                model=model,
                processor=processor,
                context=contexts["full"],
                layer_ids=layer_ids,
                device=device,
                video_budget=video_budget,
            )
            target_trace = collect_prompt_trace(
                model=model,
                processor=processor,
                context=contexts["target"],
                layer_ids=layer_ids,
                device=device,
                video_budget=video_budget,
            )
            donor_trace = (
                collect_prompt_trace(
                    model=model,
                    processor=processor,
                    context=contexts["donor"],
                    layer_ids=layer_ids,
                    device=device,
                    video_budget=video_budget,
                )
                if "donor" in contexts
                else None
            )
            control_trace = (
                collect_prompt_trace(
                    model=model,
                    processor=processor,
                    context=contexts["control"],
                    layer_ids=layer_ids,
                    device=device,
                    video_budget=video_budget,
                )
                if "control" in contexts
                else None
            )
            target_answer = target_answer_for_quartet(episode, quartet)
            for layer_id in layer_ids:
                target_record = target_trace.layer_records.get(str(layer_id))
                clean_record = clean_trace.layer_records.get(str(layer_id))
                if target_record is None or clean_record is None:
                    continue
                source_record = None
                if donor_trace is not None and safe_text(quartet.get("expected_transition")) == "retarget":
                    source_record = donor_trace.layer_records.get(str(layer_id))
                elif control_trace is not None:
                    source_record = clean_record
                else:
                    source_record = clean_record
                if source_record is None:
                    continue
                patch_delta = source_record["pooled_post"] - target_record["pooled_post"]
                patch_gain = compute_patch_gain(
                    model=model,
                    processor=processor,
                    context=contexts["target"],
                    answer_text=target_answer,
                    patch_layer_id=layer_id,
                    patch_delta=patch_delta,
                    device=device,
                    video_budget=video_budget,
                )
                layer_stats[layer_id]["patch_gain_sum"] += float(patch_gain)
                layer_stats[layer_id]["patch_gain_count"] += 1.0
                if control_trace is not None and str(layer_id) in control_trace.layer_records:
                    control_distance = torch.norm(
                        clean_record["pooled_post"] - control_trace.layer_records[str(layer_id)]["pooled_post"],
                        dim=-1,
                    ).mean()
                    layer_stats[layer_id]["clean_control_distance_sum"] += float(control_distance.detach().item())
                    layer_stats[layer_id]["clean_control_distance_count"] += 1.0
            analyzed += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "analyzed": int(analyzed),
                        "limit": int(limit),
                        "episode_id": safe_text(episode.get("episode_id")),
                        "quartet_id": safe_text(quartet.get("quartet_id")),
                    }
                )
            if analyzed >= limit:
                break
        if analyzed >= limit:
            break

    ranking: List[Dict[str, Any]] = []
    for layer_id in layer_ids:
        stats = layer_stats[layer_id]
        patch_gain_mean = stats["patch_gain_sum"] / max(1.0, stats["patch_gain_count"])
        control_distance_mean = stats["clean_control_distance_sum"] / max(1.0, stats["clean_control_distance_count"])
        ranking.append(
            {
                "layer_id": int(layer_id),
                "patch_gain_mean": float(patch_gain_mean),
                "clean_control_distance_mean": float(control_distance_mean),
                "conflict_score": float(patch_gain_mean - 0.1 * control_distance_mean),
            }
        )
    ranking = sorted(ranking, key=lambda row: (-float(row["conflict_score"]), int(row["layer_id"])))
    return {
        "analyzed_quartets": int(analyzed),
        "candidate_layer_ids": [int(layer_id) for layer_id in layer_ids],
        "ranked_layers": ranking,
        "selected_layers": select_conflict_layers_from_ranking(ranking),
    }


def summarize_mechanism_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    if not records:
        return {
            "n_records": 0,
            "conflict_state_accuracy": 0.0,
            "selected_decisive_modality_accuracy": 0.0,
            "action_accuracy": 0.0,
            "transport_consistency_score": 0.0,
            "causal_patch_gain": 0.0,
            "conflict_sensitive_layer_stability": 0.0,
        }
    return {
        "n_records": len(records),
        "conflict_state_accuracy": float(sum(float(row.get("conflict_state_correct", 0.0)) for row in records) / len(records)),
        "selected_decisive_modality_accuracy": float(
            sum(float(row.get("decisive_modality_correct", 0.0)) for row in records) / len(records)
        ),
        "action_accuracy": float(sum(float(row.get("action_correct", 0.0)) for row in records) / len(records)),
        "transport_consistency_score": float(
            sum(float(row.get("transport_consistency_score", 0.0)) for row in records) / len(records)
        ),
        "causal_patch_gain": float(sum(float(row.get("causal_patch_gain", 0.0)) for row in records) / len(records)),
        "conflict_sensitive_layer_stability": float(
            sum(float(row.get("layer_stability", 0.0)) for row in records) / len(records)
        ),
    }


def save_cci_payload(
    *,
    output_path: Path,
    adapter: ConflictArbitrationAdapter,
    discovery: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "discovery": discovery,
            "config": config,
        },
        output_path,
    )


def load_cci_payload(path: Path, *, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"CCI checkpoint at '{path}' is not a dictionary payload.")
    return payload


def write_discovery_report(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)
