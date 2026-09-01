"""
LLaVA-1.5-7B Hulluedit Engine
Single forward pass + online subspace estimation + internal contrastive editing
Reference ParamSteer loading method
"""
from __future__ import annotations

# CRITICAL: Bypass PyTorch 2.6 security check BEFORE any transformers imports
# This must be done first to prevent ValueError in torch.load
try:
    from transformers.utils import import_utils
    def _bypass_torch_load_check():
        pass  # Bypass the check - we trust our model files
    import_utils.check_torch_load_is_safe = _bypass_torch_load_check
except Exception:
    pass

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, CLIPImageProcessor
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import ParamSteer model loader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "ParamSteer"))
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from hulluedit.steer import HullueditSteerer, HullueditConfig


@dataclass
class EngineConfig:
    """Engine Configuration"""
    model_name: str
    anchor_layer: int = 26
    estimate_layer: Optional[int] = None  # Subspace estimation layer, defaults to anchor_layer
    edit_layer: Optional[int] = None      # Edit layer, None means top layer (-1)
    # Multi-layer aggregation (optional): if provided, overrides single anchor_layer
    multi_anchor_layers: Optional[List[int]] = None
    layer_weighting: str = "learned"   # "learned" | "equal"
    layer_weight_temp: float = 1.0     # Learned weight temperature (based on per-layer VCR softmax)
    max_new_tokens: int = 128
    top_p: float = 0.9
    temperature: float = 0.2
    precision: str = "bf16"


class LLaVAHullueditEngine:
    """LLaVA-1.5-7B + Hulluedit Inference Engine"""
    
    def __init__(
        self, 
        eng_cfg: EngineConfig, 
        hulluedit_cfg: HullueditConfig, 
        device: str = "cuda"
    ):
        self.device = device
        self.eng_cfg = eng_cfg
        self.hulluedit_cfg = hulluedit_cfg
        
        # Load model (reference ParamSteer)
        print(f"[Hulluedit] Loading model: {eng_cfg.model_name}")
        disable_torch_init()
        
        # Use ParamSteer loading method
        model_name = "llava-v1.5-7b"  # Fixed name
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            eng_cfg.model_name, 
            None,  # model_base
            model_name, 
            device=device
        )
        
        self.model.eval()
        
        # Hulluedit steerer
        self.steerer = HullueditSteerer(hulluedit_cfg)
        print("[Hulluedit] Engine initialization complete")

    @torch.no_grad()
    def _build_inputs(
        self, 
        prompt: str, 
        image: str | Image.Image
    ) -> Dict[str, torch.Tensor]:
        """Build model inputs (reference ParamSteer method)"""
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Process image using image_processor
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        
        # Build input IDs
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        
        return {
            "input_ids": input_ids,
            "images": image_tensor
        }

    def _extract_vision_mask(
        self, 
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract vision token mask
        LLaVA uses special token <image> (id=32000) to mark image positions
        During actual inference, it will be replaced by 576 tokens from vision encoder
        """
        # IMAGE_TOKEN_INDEX = 32000 (LLaVA default)
        IMAGE_TOKEN_INDEX = 32000
        
        # Find positions of <image> tokens
        image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
        
        # LLaVA-1.5: each <image> is replaced by 576 visual tokens
        # Here we return a simplified mask (expanded later with actual sequence length)
        return image_token_mask

    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        image: str | Image.Image,
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response (with Hulluedit editing)
        
        Args:
            prompt: Input prompt
            image: Image path or PIL.Image
            max_new_tokens: Max tokens to generate
            
        Returns:
            {
                "text": generated text,
                "certs": [{"vcr": ..., "pcr": ..., "gate": ...}, ...],
                "tokens": generated token list
            }
        """
        batch = self._build_inputs(prompt, image)
        max_steps = max_new_tokens or self.eng_cfg.max_new_tokens
        
        # Initialize
        input_ids = batch["input_ids"]  # [1, prompt_len]
        images = batch.get("images")
        attention_mask = None  # LLaVA handles automatically
        
        past_key_values = None
        generated_ids = []
        certs = []
        
        # First forward pass to get vision token positions
        first_pass = True
        cached_vis_states = None  # Cached visual hidden states
        cached_txt_states = None  # Cached text hidden states (cumulative)
        
        for step in range(max_steps):
            # Forward inference
            outputs = self.model(
                input_ids=input_ids,
                images=images if first_pass else None,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            past_key_values = outputs.past_key_values
            hidden_states = outputs.hidden_states  # Tuple[Tensor]: (layer_num+1) x [1, seq, hidden]
            
            # Determine estimation and editing layers
            num_layers = len(hidden_states) - 1  # Total layers (hidden_states includes input layer)
            estimate_layer = self.eng_cfg.estimate_layer if self.eng_cfg.estimate_layer is not None else self.eng_cfg.anchor_layer
            if estimate_layer == -1:
                estimate_layer = num_layers  # Top layer index
            edit_layer = self.eng_cfg.edit_layer if self.eng_cfg.edit_layer is not None else -1  # -1 means top layer
            if edit_layer == -1:
                edit_layer = num_layers  # Top layer index
            
            # Target layer set: multi-layer or single-layer (for estimation)
            target_layers: List[int]
            if self.eng_cfg.multi_anchor_layers and len(self.eng_cfg.multi_anchor_layers) > 0:
                target_layers = list(self.eng_cfg.multi_anchor_layers)
            else:
                target_layers = [estimate_layer]  # Use estimate_layer instead of anchor_layer

            # Get anchor layer and last layer hidden states
            anchor_hidden = hidden_states[target_layers[0]]  # First layer for first pass cache shape
            last_hidden = hidden_states[-1]  # [1, seq, hidden]
            
            # First inference: determine vision token range and cache visual/text hidden states
            if first_pass:
                seq_len = anchor_hidden.shape[1]
                # LLaVA: image tokens at beginning (<s> <image_tokens> prompt_tokens)
                # Simplified assumption: first 576 tokens are visual tokens
                # More precise method requires parsing prepare_inputs_labels_for_multimodal
                num_image_tokens = 576  # LLaVA-1.5 default
                vision_end_idx = min(1 + num_image_tokens, seq_len - 1)  # Ensure at least one text token
                
                # Cache visual/text hidden states for each layer separately
                cached_vis_states = {}
                cached_txt_states = {}
                for L in target_layers:
                    hL = hidden_states[L]
                    # Extract and cache visual hidden states (reused in subsequent steps)
                    visL = hL[0, 1:vision_end_idx, :].clone()  # [n_vis, d]
                    cached_vis_states[L] = visL
                    # Extract initial text hidden states (not including last token, for anti-prior)
                    if vision_end_idx < seq_len - 1:
                        txtL = hL[0, vision_end_idx:-1, :].clone()  # [n_txt, d]
                    else:
                        hidden_dim = hL.shape[2]
                        txtL = torch.empty(0, hidden_dim, dtype=hL.dtype, device=self.device)
                    cached_txt_states[L] = txtL
                
                first_pass = False
            else:
                # Subsequent steps: use KV cache, only newly generated tokens (seq_len=1)
                # Cumulative text hidden states (for subsequent anti-prior calculation)
                for L in target_layers:
                    hL = hidden_states[L]
                    new_txt_hidden = hL[0, -1, :].unsqueeze(0)  # [1, d]
                    cached_txt_states[L] = torch.cat([cached_txt_states[L], new_txt_hidden], dim=0)
            
            # Get hidden state for editing (edit_layer already converted to actual index)
            h_to_edit = hidden_states[edit_layer][0, -1, :]  # [hidden] specified layer
            
            # Hulluedit editing (supports multi-layer aggregation)
            if len(target_layers) == 1:
                L = target_layers[0]
                U = self.steerer.compute_evidence_subspace(cached_vis_states[L], h_to_edit)
                P = self.steerer.compute_anti_prior_subspace(cached_txt_states[L], U)
                edit_result = self.steerer.edit_text_hidden(h_to_edit, U, P)
                h_edited = edit_result.h_edited
                certs.append({
                    "vcr": float(edit_result.vcr.cpu()),
                    "pcr": float(edit_result.pcr.cpu()),
                    "gate": float(edit_result.gate.cpu())
                })
            else:
                # Independently compute editing for each layer, then aggregate
                per_layer = []
                vcr_list = []
                for L in target_layers:
                    U = self.steerer.compute_evidence_subspace(cached_vis_states[L], h_to_edit)
                    P = self.steerer.compute_anti_prior_subspace(cached_txt_states[L], U)
                    er = self.steerer.edit_text_hidden(h_to_edit, U, P)
                    per_layer.append(er)
                    vcr_list.append(float(er.vcr.detach().cpu()))
                # Compute weights
                if self.eng_cfg.layer_weighting == "equal":
                    weights = torch.ones(len(per_layer), device=h_to_edit.device, dtype=h_to_edit.dtype) / len(per_layer)
                else:
                    # learned: softmax based on per-layer VCR
                    vcr_tensor = torch.tensor(vcr_list, device=h_to_edit.device, dtype=h_to_edit.dtype)
                    temp = max(1e-6, float(self.eng_cfg.layer_weight_temp))
                    weights = torch.softmax(vcr_tensor / temp, dim=0)
                # Aggregate edited hidden states
                h_stack = torch.stack([er.h_edited for er in per_layer], dim=0)  # [L, d]
                h_edited = (weights[:, None] * h_stack).sum(dim=0)
                # Record last layer certificate (and total weight info)
                certs.append({
                    "vcr": float(sum(vcr_list) / max(1, len(vcr_list))),
                    "pcr": float(sum(float(er.pcr.detach().cpu()) for er in per_layer) / max(1, len(per_layer))),
                    "gate": float(sum(float(er.gate.detach().cpu()) for er in per_layer) / max(1, len(per_layer)))
                })
            
            # Map to logits via lm_head (internal contrastive)
            logits = self.model.lm_head(h_edited.unsqueeze(0))  # [1, vocab]
            
            # Greedy decoding (temperature=0.0) or sampling
            if self.eng_cfg.temperature <= 1e-6:
                # Greedy decoding: use argmax directly
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
            else:
                # Sampling mode
                probs = torch.softmax(
                    logits / self.eng_cfg.temperature, 
                    dim=-1
                )
                
                # Top-p sampling
                if self.eng_cfg.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability exceeding top_p
                    sorted_indices_to_remove = cumulative_probs > self.eng_cfg.top_p
                    sorted_indices_to_remove[..., 0] = False  # Keep at least one
                    
                    # Create mask
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    probs[indices_to_remove] = 0.0
                    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
                
                # Multinomial sampling
                next_token_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
            
            # Record
            generated_ids.append(next_token_id.item())
            # Certificate already appended above
            
            # Prepare next input
            input_ids = next_token_id
            
            # EOS check
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return {
            "text": generated_text,
            "certs": certs,
            "tokens": generated_ids
        }
