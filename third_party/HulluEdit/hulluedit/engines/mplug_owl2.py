"""
mPLUG-Owl2 Inference Engine (adapted for Hulluedit method)
Reference: sjc/Hulluedit/hulluedit/engines/mplug_owl2_engine.py and sjc/Nullu/mplug_owl2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import sys
from pathlib import Path
from PIL import Image
import torch

# Optional Nullu dependency. Keep the default repository-relative and allow
# users to provide another checkout through NULLU_ROOT.
NULLU_ROOT = os.environ.get(
    "NULLU_ROOT",
    str(Path(__file__).resolve().parents[3] / "third_party" / "Nullu"),
)
if NULLU_ROOT not in sys.path:
    sys.path.insert(0, NULLU_ROOT)

from mplug_owl2.model.builder import load_pretrained_model  # type: ignore
from mplug_owl2.mm_utils import tokenizer_image_token, process_images  # type: ignore
from mplug_owl2.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX  # type: ignore
from hulluedit.steer import HullueditSteerer, HullueditConfig  # type: ignore


@dataclass
class MplugOwl2EngineConfig:
    """mPLUG-Owl2 Engine Configuration"""
    model_path: str          # HF or local path
    model_name: str = "mplug_owl2"
    anchor_layer: int = 26
    max_new_tokens: int = 128
    top_p: float = 0.9
    temperature: float = 0.2
    precision: str = "fp16"


class MplugOwl2Engine:
    """mPLUG-Owl2 + Hulluedit Inference Engine (step-by-step generation with implicit representation editing)"""

    def __init__(self, eng_cfg: MplugOwl2EngineConfig, hulluedit_cfg: HullueditConfig, device: str = "cuda") -> None:
        self.device = device
        self.eng_cfg = eng_cfg
        self.hulluedit_cfg = hulluedit_cfg
        self.steerer = HullueditSteerer(hulluedit_cfg)

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            eng_cfg.model_path,
            model_base=None,
            model_name=eng_cfg.model_name,
            device=device,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        image_path: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Construct prompt with image token
        prompt_with_image = f"{DEFAULT_IMAGE_TOKEN} {prompt}" if DEFAULT_IMAGE_TOKEN not in prompt else prompt

        # Estimate visual token count: compare sequence length difference with/without image
        image = Image.open(image_path).convert("RGB")
        images = process_images([image], self.image_processor, getattr(self.model, 'config', None))
        images = images.to(self.device, dtype=torch.float16) if not isinstance(images, list) else images[0].unsqueeze(0).to(self.device, dtype=torch.float16)

        input_ids_img = tokenizer_image_token(prompt_with_image, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        input_ids_noimg = tokenizer_image_token(prompt_with_image.replace(DEFAULT_IMAGE_TOKEN, ""), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        with torch.no_grad():
            out_with = self.model(
                input_ids=input_ids_img,
                images=images,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            out_without = self.model(
                input_ids=input_ids_noimg,
                images=None,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        seq_with = out_with.hidden_states[self.eng_cfg.anchor_layer].shape[1]
        seq_without = out_without.hidden_states[self.eng_cfg.anchor_layer].shape[1]
        n_vis = max(seq_with - seq_without, 0)

        # Start step-by-step generation (first step recompute to get past_kv)
        past_key_values = None
        generated_ids = []
        certs = []
        first_pass = True
        cached_vis_states = None
        cached_txt_states = None
        max_steps = max_new_tokens or self.eng_cfg.max_new_tokens

        current_input_ids = input_ids_img
        current_images = images

        for _ in range(max_steps):
            outputs = self.model(
                input_ids=current_input_ids,
                images=current_images if first_pass else None,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.hidden_states

            anchor_hidden = hidden_states[self.eng_cfg.anchor_layer]
            last_hidden = hidden_states[-1]

            if first_pass:
                seq_len = anchor_hidden.shape[1]
                vision_end_idx = min(n_vis, seq_len - 1)
                
                # Cache visual hidden states from anchor layer (consistent with LLaVA engine)
                cached_vis_states = anchor_hidden[0, :vision_end_idx, :].clone()
                
                if vision_end_idx < seq_len - 1:
                    cached_txt_states = anchor_hidden[0, vision_end_idx:-1, :].clone()
                else:
                    hidden_dim = anchor_hidden.shape[2]
                    cached_txt_states = torch.empty(0, hidden_dim, dtype=anchor_hidden.dtype, device=self.device)
                first_pass = False
            else:
                new_txt_hidden = anchor_hidden[0, -1, :].unsqueeze(0)
                cached_txt_states = torch.cat([cached_txt_states, new_txt_hidden], dim=0)

            h_last = last_hidden[0, -1, :]

            # Hulluedit editing (consistent with LLaVA engine)
            U = self.steerer.compute_evidence_subspace(cached_vis_states, h_last)
            P = self.steerer.compute_anti_prior_subspace(cached_txt_states, U)
            edit_result = self.steerer.edit_text_hidden(h_last, U, P)

            logits = self.model.lm_head(edit_result.h_edited.unsqueeze(0))

            if self.eng_cfg.temperature < 0.01:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                safe_temp = max(self.eng_cfg.temperature, 0.01)
                probs = torch.softmax(logits / safe_temp, dim=-1)
                if self.eng_cfg.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.eng_cfg.top_p
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                probs = torch.clamp(probs, min=1e-10)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token_id = torch.multinomial(probs, num_samples=1)

            generated_ids.append(next_token_id.item())
            certs.append({
                "vcr": float(edit_result.vcr.cpu()),
                "pcr": float(edit_result.pcr.cpu()),
                "gate": float(edit_result.gate.cpu()),
            })

            current_input_ids = next_token_id
            current_images = None

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {"text": text, "certs": certs, "tokens": generated_ids}
