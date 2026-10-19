# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import json
import os
import warnings
import shutil

import torch
from transformers import PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.utils.hub import cached_file

from .projector import load_mm_projector
from .videollama2_llama import Videollama2LlamaForCausalLM, Videollama2LlamaConfig
from .videollama2_mistral import Videollama2MistralForCausalLM, Videollama2MistralConfig
from .videollama2_mixtral import Videollama2MixtralForCausalLM, Videollama2MixtralConfig
from .videollama2_qwen2 import Videollama2Qwen2ForCausalLM, Videollama2Qwen2Config
from .videollama2_gemma2 import Videollama2Gemma2ForCausalLM, Videollama2Gemma2Config
from .videollama2_phi3 import Videollama2Phi3ForCausalLM, Videollama2Phi3Config


VLLMs = {
    "videollama2": Videollama2MistralForCausalLM,
    "videollama2_llama": Videollama2LlamaForCausalLM,
    "videollama2_mistral": Videollama2MistralForCausalLM,
    "videollama2_mixtral": Videollama2MixtralForCausalLM,
    "videollama2_qwen2": Videollama2Qwen2ForCausalLM,
    "videollama2_gemma2": Videollama2Gemma2ForCausalLM,
    "videollama2_phi3": Videollama2Phi3ForCausalLM,
}

VLLMConfigs = {
    "videollama2": Videollama2MistralConfig,
    "videollama2_llama": Videollama2LlamaConfig,
    "videollama2_mistral": Videollama2MistralConfig,
    "videollama2_mixtral": Videollama2MixtralConfig,
    "videollama2_qwen2": Videollama2Qwen2Config,
    "videollama2_gemma2": Videollama2Gemma2Config,
    "videollama2_phi3": Videollama2Phi3Config,
}


def _offline_mode_enabled():
    return str(os.environ.get("HF_HUB_OFFLINE", "")).upper() in {"1", "ON", "YES", "TRUE"} or str(
        os.environ.get("TRANSFORMERS_OFFLINE", "")
    ).upper() in {"1", "ON", "YES", "TRUE"}


def _repair_beats_legacy_weight_norm(model, model_path, token=None):
    """Restore BEATs old-style weight_norm tensors after Transformers key conversion."""
    try:
        audio_tower = model.get_model().get_audio_tower()
        conv = audio_tower.encoder.pos_conv[0]
    except Exception:
        return
    if not hasattr(conv, "weight_g") or not hasattr(conv, "weight_v"):
        return

    g_key = "model.audio_tower.encoder.pos_conv.0.weight_g"
    v_key = "model.audio_tower.encoder.pos_conv.0.weight_v"
    try:
        index_path = cached_file(
            model_path,
            "model.safetensors.index.json",
            token=token,
            local_files_only=_offline_mode_enabled(),
        )
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", {})
        shard_name = weight_map.get(g_key)
        if not shard_name or weight_map.get(v_key) != shard_name:
            return
        shard_path = cached_file(
            model_path,
            shard_name,
            token=token,
            local_files_only=_offline_mode_enabled(),
        )
        from safetensors import safe_open

        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            weight_g = handle.get_tensor(g_key)
            weight_v = handle.get_tensor(v_key)
    except Exception as exc:
        warnings.warn(f"Could not repair BEATs legacy weight_norm tensors: {exc}")
        return

    with torch.no_grad():
        conv.weight_g.copy_(weight_g.to(device=conv.weight_g.device, dtype=conv.weight_g.dtype))
        conv.weight_v.copy_(weight_v.to(device=conv.weight_v.device, dtype=conv.weight_v.dtype))
        for hook in conv._forward_pre_hooks.values():
            if hasattr(hook, "name") and hook.name == "weight" and hasattr(hook, "compute_weight"):
                setattr(conv, "weight", hook.compute_weight(conv))
                break


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    if 'token' in kwargs:
        token = kwargs['token']
    else:
        token = None
    
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # NOTE: High-version Transformers will report: """ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time."""
        # kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    if kwargs['torch_dtype'] is None:
        kwargs['torch_dtype'] = torch.bfloat16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    config = AutoConfig.from_pretrained(model_path)

    # judge model type
    model_type = config.model_type

    # judge pretrain/finetune
    try:
        is_pretraining = config.tune_mm_mlp_adapter
    except:
        is_pretraining = False

    # NOTE: lora/qlora model loading
    if 'lora' in model_name.lower() or 'qlora' in model_name.lower():
        cfg_pretrained = PretrainedConfig.from_pretrained(model_path, token=token)
        # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
        # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
        model_base = model_base if model_base is not None else cfg_pretrained._name_or_path

        # NOTE: remove qlora training quantization config 
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, token=token)
        print('Loading VideoLLaMA from base model...')

        if 'vicuna' in model_base.lower():
            model = Videollama2LlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        elif 'mistral' in model_base.lower():
            model = Videollama2MistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            model = Videollama2MistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional VideoLLaMA weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            # this is probably from HF Hub
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')
            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None or '-base' in model_name.lower() or is_pretraining:
        # NOTE: Base/Pretrain model loading
        print('Loading VideoLLaMA 2 from base model...')
        cfg_pretrained = PretrainedConfig.from_pretrained(model_path, token=token)
        # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
        # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
        model_base = model_base if model_base is not None else cfg_pretrained._name_or_path

        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, token=token)

        if model_type in ['videollama2', 'videollama2_mistral']:
            model = Videollama2MistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        elif model_type in ['videollama2_mixtral']:
            model = Videollama2MixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        elif model_type in ['videollama2_qwen2']:
            model = Videollama2Qwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        elif model_type in ['videollama2_gemma2']:
            model = Videollama2Gemma2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        elif model_type in ['videollama2_phi3']:
            model = Videollama2Phi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            model = Videollama2MistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)

        # NOTE; loading vision-language projector
        # * old codes for loading local mm_projector.bin
        # mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        # mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        # model.load_state_dict(mm_projector_weights, strict=False)
        # * new codes which supports loading mm_projector.bin both offline and online 
        mm_projector_weights = load_mm_projector(model_path, token=token)
        model.load_state_dict(mm_projector_weights, strict=False)
    elif 'videollama2' in model_type:
        # NOTE: SFT model loading
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)

        if model_type in ['videollama2', 'videollama2_mistral']:
            model = Videollama2MistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
        elif model_type in ['videollama2_mixtral']:
            model = Videollama2MixtralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
        elif model_type in ['videollama2_qwen2']:
            model = Videollama2Qwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
        elif model_type in ['videollama2_gemma2']:
            model = Videollama2Gemma2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
        elif model_type in ['videollama2_phi3']:
            model = Videollama2Phi3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            model = Videollama2MistralForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **kwargs)

    processor = None

    if "videollama" in model_type:
        _repair_beats_legacy_weight_norm(model, model_path, token=token)

    if "videollama" in model_type:
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        # NOTE: videollama2 adopts the same processor for processing image and video.
        processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len
