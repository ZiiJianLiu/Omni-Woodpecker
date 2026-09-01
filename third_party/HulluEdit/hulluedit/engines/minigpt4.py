"""
MiniGPT-4 Inference Engine (for POPE/CHAIR evaluation, integrated with Hulluedit editing)

Uses official implementation under MODELS/MiniGPT-4, reference from DeCo project.
Does not depend on sjc/Nullu folder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import sys
from pathlib import Path
import torch
from PIL import Image

# Add MODELS/MiniGPT-4 to path (must be at the beginning to avoid conflicts with other versions)
_MINIGPT4_ROOT = os.environ.get(
    "MINIGPT4_ROOT",
    str(Path(__file__).resolve().parents[3] / "third_party" / "MiniGPT-4"),
)
if _MINIGPT4_ROOT not in sys.path:
    sys.path.insert(0, _MINIGPT4_ROOT)

try:
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import (
        Chat as MiniChat,
        CONV_VISION_LLama2,
        CONV_VISION_Vicuna0,
    )
    # Import modules to register models, processors, etc.
    from minigpt4.datasets.builders import *
    from minigpt4.models import *
    from minigpt4.processors import *
except Exception as e:
    raise ImportError(
        f"MiniGPT-4 dependencies not found, please ensure MINIGPT4_ROOT points to the minigpt4 repository."
        f"Current: {_MINIGPT4_ROOT}. Error: {e}"
    )

from hulluedit.steer import HullueditSteerer, HullueditConfig


@dataclass
class MiniGPT4EngineConfig:
    """MiniGPT-4 Engine Configuration"""
    cfg_path: str  # MiniGPT-4 evaluation YAML config file path
    anchor_layer: int = 26
    max_new_tokens: int = 128
    top_p: float = 0.9
    temperature: float = 0.2
    repetition_penalty: float = 1.2  # Repetition penalty coefficient to reduce duplicate generation
    gpu_id: int = 0


class MiniGPT4HullueditEngine:
    """MiniGPT-4 + Hulluedit Inference Engine (explicit step-by-step generation with implicit representation editing)"""

    def __init__(self, eng_cfg: MiniGPT4EngineConfig, hulluedit_cfg: HullueditConfig, device: str = "cuda:0") -> None:
        self.device = device
        self.eng_cfg = eng_cfg
        self.hulluedit_cfg = hulluedit_cfg
        self.steerer = HullueditSteerer(hulluedit_cfg)

        # Use Config class to load config (reference DeCo implementation)
        class Args:
            def __init__(self, cfg_path, options=None):
                self.cfg_path = cfg_path
                self.options = options or []
        
        args = Args(eng_cfg.cfg_path, [])
        cfg = Config(args)
        
        # Initialize model
        model_config = cfg.model_cfg
        model_config.device_8bit = eng_cfg.gpu_id
        
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(self.device)
        self.model.eval()
        
        # Get conversation template (default use CONV_VISION_Vicuna0 to match DeCo)
        conv_dict = {
            'pretrain_vicuna0': CONV_VISION_Vicuna0,
            'pretrain_llama2': CONV_VISION_LLama2
        }
        self.model_type = model_config.model_type  # Save model type for post-processing
        self.conv_template = conv_dict.get(self.model_type, CONV_VISION_Vicuna0).copy()
        
        # Get stop symbol (end_sym) for stop condition check
        self.end_sym = model_config.get("end_sym", "###")
        # Encode stop symbol to token IDs
        if self.end_sym:
            end_sym_ids = self.model.llama_tokenizer.encode(self.end_sym, add_special_tokens=False)
            # If stop symbol is multiple tokens, only check the last one
            self.end_sym_token_id = end_sym_ids[-1] if end_sym_ids else None
            # Save full stop symbol token sequence (for more precise detection)
            self.end_sym_token_ids = end_sym_ids if end_sym_ids else []
        else:
            self.end_sym_token_id = None
            self.end_sym_token_ids = []
        
        # Initialize visual processor
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        
        # Initialize Chat object (for conversation handling)
        self.chat = MiniChat(self.model, self.vis_processor, device=self.device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        image_path: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate response, integrated with Hulluedit editing
        
        Args:
            prompt: Text prompt
            image_path: Image path
            max_new_tokens: Max tokens to generate
            
        Returns:
            Dictionary containing text, certs, tokens
        """
        # Build conversation with image context
        conv = self.conv_template.copy()
        img_list = []
        _ = self.chat.upload_img(image_path, conv, img_list)
        self.chat.encode_img(img_list)  # This encodes image and puts it in img_list[0]
        self.chat.ask(prompt, conv)

        # Prepare input embeddings and sampling parameters
        gen_kwargs = self.chat.answer_prepare(
            conv,
            img_list,
            max_new_tokens=max_new_tokens or self.eng_cfg.max_new_tokens,
            top_p=self.eng_cfg.top_p,
            temperature=self.eng_cfg.temperature,
        )
        inputs_embeds: torch.Tensor = gen_kwargs.pop("inputs_embeds")  # [1, seq, hidden]

        # Get visual embedding length from img_list (avoid redundant computation)
        # img_list[0] is the result of encode_img, shape [1, n_vis, hidden_dim]
        if len(img_list) > 0 and isinstance(img_list[0], torch.Tensor):
            n_vis = int(img_list[0].shape[1])
        else:
            # If img_list is empty or wrong format, fallback to recomputing (shouldn't happen)
            raw_image = Image.open(image_path).convert("RGB")
            image_tensor = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            image_emb, _ = self.model.encode_img(image_tensor)
            n_vis = int(image_emb.shape[1])

        past_key_values = None
        generated_ids = []
        certs = []
        first_pass = True
        cached_vis_states = None
        cached_txt_states = None
        max_steps = max_new_tokens or self.eng_cfg.max_new_tokens

        for _ in range(max_steps):
            if first_pass:
                outputs = self.model.llama_model(
                    inputs_embeds=inputs_embeds,
                    past_key_values=None,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                # Subsequent steps: use input_ids and past_key_values
                # Note: next_token_id shape should be [1, 1]
                if next_token_id.dim() == 1:
                    next_token_id = next_token_id.unsqueeze(0)
                outputs = self.model.llama_model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values
            hidden_states = outputs.hidden_states  # Tuple[Tensors]

            # Check if anchor_layer is in valid range
            num_layers = len(hidden_states)
            if self.eng_cfg.anchor_layer >= num_layers:
                raise ValueError(
                    f"anchor_layer {self.eng_cfg.anchor_layer} out of range "
                    f"(model has {num_layers} layers, index range 0-{num_layers-1})"
                )
            
            anchor_hidden = hidden_states[self.eng_cfg.anchor_layer]  # [1, seq, d]
            last_hidden = hidden_states[-1]

            if first_pass:
                seq_len = anchor_hidden.shape[1]
                # Compute visual embedding end position
                # inputs_embeds structure: [text_before_image, image_emb, text_after_image]
                # Need to find <ImageHere> position in prompt to determine visual embedding boundary
                # Since answer_prepare uses get_context_emb, visual embeddings are already embedded in inputs_embeds
                # We use n_vis as the number of visual tokens
                vision_end_idx = min(n_vis, seq_len)
                
                # Ensure index is valid
                if vision_end_idx > seq_len:
                    vision_end_idx = seq_len
                
                # Cache visual hidden states (extract visual part from anchor_hidden)
                if vision_end_idx > 0:
                    cached_vis_states = anchor_hidden[0, :vision_end_idx, :].clone()
                else:
                    hidden_dim = anchor_hidden.shape[2]
                    cached_vis_states = torch.empty(0, hidden_dim, dtype=anchor_hidden.dtype, device=anchor_hidden.device)
                
                # Cache text hidden states (part after visual to last position before)
                # Last position is current generation position, not included in text states
                if vision_end_idx < seq_len - 1:
                    cached_txt_states = anchor_hidden[0, vision_end_idx:-1, :].clone()
                else:
                    hidden_dim = anchor_hidden.shape[2]
                    cached_txt_states = torch.empty(0, hidden_dim, dtype=anchor_hidden.dtype, device=anchor_hidden.device)
                
                first_pass = False
            else:
                new_txt_hidden = anchor_hidden[0, -1, :].unsqueeze(0)
                cached_txt_states = torch.cat([cached_txt_states, new_txt_hidden], dim=0)

            h_last = last_hidden[0, -1, :]

            # Hulluedit editing
            U = self.steerer.compute_evidence_subspace(cached_vis_states, h_last)
            P = self.steerer.compute_anti_prior_subspace(cached_txt_states, U)
            edit_result = self.steerer.edit_text_hidden(h_last, U, P)

            logits = self.model.llama_model.lm_head(edit_result.h_edited.unsqueeze(0))

            # Apply repetition_penalty: penalize recently generated tokens to reduce duplication
            if self.eng_cfg.repetition_penalty > 1.0 and len(generated_ids) > 0:
                # Only penalize recent 50 tokens to avoid over-penalization
                recent_tokens = set(generated_ids[-50:])
                for token_id in recent_tokens:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= self.eng_cfg.repetition_penalty
                    else:
                        logits[0, token_id] *= self.eng_cfg.repetition_penalty

            # Sampling/greedy
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
                "vcr": float(edit_result.vcr.detach().cpu()),
                "pcr": float(edit_result.pcr.detach().cpu()),
                "gate": float(edit_result.gate.detach().cpu()),
            })

            # Check stop condition: EOS token or end_sym (e.g., "###")
            if next_token_id.item() == self.model.llama_tokenizer.eos_token_id:
                break
            if self.end_sym_token_id is not None and next_token_id.item() == self.end_sym_token_id:
                break

            # Detect duplicate generation of "Image Content" pattern (prevent model from getting stuck in repetition loop)
            if len(generated_ids) >= 20:  # Only detect after generating at least 20 tokens
                recent_text = self.model.llama_tokenizer.decode(generated_ids[-30:], skip_special_tokens=True)
                # Check if repeating "Image Content" or "</Img>" followed by "Image Content"
                if "</Img>" in recent_text:
                    # If "</Img>" is generated, check if "Image Content" repeats after it
                    parts = recent_text.split("</Img>")
                    if len(parts) > 1:
                        after_img_tag = parts[-1].strip()
                        # If "Image Content" appears multiple times after "</Img>", stop generation
                        if after_img_tag.count("Image Content") >= 2:
                            break
                # Check if "Image Content" appears consecutively multiple times (even without "</Img>")
                if recent_text.count("Image Content") >= 3:
                    break

        generated_text = self.model.llama_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Post-processing: different cleanup methods based on model type
        if self.end_sym:
            generated_text = generated_text.split(self.end_sym)[0]
        
        # Remove different prefixes based on model type
        if self.model_type == 'pretrain_llama2':
            # Llama2 format: remove content before [/INST]
            if "[/INST]" in generated_text:
                generated_text = generated_text.split("[/INST]")[-1].strip()
            # Remove possible <s> prefix
            if generated_text.startswith("<s>"):
                generated_text = generated_text[3:].strip()
            # Remove possible [INST] tag (if model generated)
            if generated_text.startswith("[INST]"):
                generated_text = generated_text[6:].strip()
        else:
            # Vicuna0 format: remove "Assistant:" prefix
            if "Assistant:" in generated_text:
                generated_text = generated_text.split("Assistant:")[-1].strip()
        
        # Additional post-processing: remove </Img> tag and duplicate "Image Content" after it
        if "</Img>" in generated_text:
            # Find position of "</Img>"
            img_tag_idx = generated_text.find("</Img>")
            if img_tag_idx >= 0:
                # Keep content before "</Img>", remove "</Img>" and content after
                # But if normal response after "</Img>", keep it
                after_img = generated_text[img_tag_idx + 6:].strip()
                # If only duplicate "Image Content" after "</Img>", remove it
                if after_img and "Image Content" in after_img:
                    # Check if it's mainly duplicate "Image Content"
                    cleaned_after = after_img.replace("Image Content", "").strip()
                    if len(cleaned_after) < len(after_img) * 0.3:  # If 70%+ is "Image Content"
                        generated_text = generated_text[:img_tag_idx].strip()
                    else:
                        # Keep normal content after "</Img>", but remove duplicate "Image Content"
                        # Only keep content after first "Image Content"
                        parts = after_img.split("Image Content")
                        if len(parts) > 2:  # If multiple "Image Content"
                            # Keep content before first and after last "Image Content"
                            generated_text = generated_text[:img_tag_idx] + "</Img>" + parts[0] + "Image Content" + parts[-1]
        
        return {"text": generated_text, "certs": certs, "tokens": generated_ids}
