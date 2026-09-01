import os
import copy
import warnings
import shutil
import re
from pathlib import Path
from functools import partial

import torch
from transformers import StoppingCriteria

from .model import load_pretrained_model
from .mm_utils import process_image, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria, process_audio_file
from .constants import NUM_FRAMES, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, MODAL_INDEX_MAP, DEFAULT_AUDIO_TOKEN
from torch.nn.utils.rnn import pad_sequence
import math

try:
    from OMhallucination.asset_manager import model_repo, resolve_model_path
except ImportError:  # pragma: no cover - preserves standalone upstream usage
    model_repo = lambda _kind: "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    resolve_model_path = lambda value, **_kwargs: value

_YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _yes_no_early_stop_enabled():
    return os.environ.get("VIDEOLLAMA2_DISABLE_YESNO_EARLY_STOP", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _has_yes_no_answer(tokenizer, token_ids):
    if not token_ids:
        return False
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return bool(_YES_NO_PATTERN.search(text))


class YesNoStoppingCriteria(StoppingCriteria):
    """Stop yes/no benchmark generation once the answer token appears."""

    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[0, self.prompt_length:].tolist()
        return _has_yes_no_answer(self.tokenizer, generated)

def model_init(model_path=None, **kwargs):
    if model_path is None:
        model_path = os.environ.get("OWP_VIDEOLLAMA2_MODEL_ID") or str(
            Path(__file__).resolve().parents[3] / "models" / "VideoLLaMA2.1-7B-AV"
        )
    model_path = resolve_model_path(model_path, kind="videollama2")
    local_siglip = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "models", "siglip-so400m-patch14-384")
    if os.path.isdir(local_siglip):
        os.environ.setdefault("OWP_SIGLIP_PATH", os.path.abspath(local_siglip))
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, **kwargs)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.top_k = 50

    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    num_frames = model.config.num_frames if hasattr(model.config, "num_frames") else NUM_FRAMES
    processor = {
        'image': partial(process_image, processor=processor, aspect_ratio=None),
        'video': partial(process_video, processor=processor, aspect_ratio=None, num_frames=num_frames),
        'audio': process_audio_file,
    }

    return model, processor, tokenizer


def mm_infer(image_or_video, instruct, model, tokenizer, modal='video', add_prompt='', **kwargs):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """

    # 1. text preprocess (tag process & generate prompt).
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'text':
        modal_token = ''
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    # 1. vision preprocess (load & transform image or video).
    if modal == 'text':
        tensor = None
    else:
        # if isinstance(image_or_video, dict):
        #     tensor = {k: v.half().cuda() for k, v in image_or_video.items()}
        # else:
        #     tensor = image_or_video.half().cuda() 
            
        if isinstance(image_or_video, dict):
            tensor = {k: v.to(dtype=torch.bfloat16).cuda() for k, v in image_or_video.items()}
        else:
            tensor = image_or_video.to(dtype=torch.bfloat16).cuda() 
        tensor = [(tensor, modal)]

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    prompt += add_prompt

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
    if _yes_no_early_stop_enabled():
        stopping_criteria.append(YesNoStoppingCriteria(tokenizer, input_ids.shape[-1]))

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.8 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)
    generation_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "stopping_criteria": stopping_criteria,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
        })

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            **generation_kwargs,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def mm_logit(image_or_video, instruct, model, tokenizer, modal='video', add_prompt='', **kwargs):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """

    # 1. text preprocess (tag process & generate prompt).
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'text':
        modal_token = ''
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    # 1. vision preprocess (load & transform image or video).
    if modal == 'text':
        tensor = None
    else:
        # if isinstance(image_or_video, dict):
        #     tensor = {k: v.half().cuda() for k, v in image_or_video.items()}
        # else:
        #     tensor = image_or_video.half().cuda() 
            
        if isinstance(image_or_video, dict):
            tensor = {k: v.to(dtype=torch.bfloat16).cuda() for k, v in image_or_video.items()}
        else:
            tensor = image_or_video.to(dtype=torch.bfloat16).cuda() 
        tensor = [(tensor, modal)]

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    prompt += add_prompt

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    
    head_prompt_len = (input_ids.flatten() == MODAL_INDEX_MAP[modal_token]).nonzero(as_tuple=True)[0].item()
    vid_token_len = model.encode_images_or_videos([(tensor[0][0]['video'],'video')]).shape[1]
    tail_prompt_len = input_ids.flatten().shape[0] - (head_prompt_len) -1 # modal token
    
    # 3. generate response according to visual signals and prompts. 
    keywords = [tokenizer.eos_token]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)
    output_attentions = kwargs.get('output_attentions', False)

    with torch.no_grad():
        outputs = model.forward(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
            output_attentions=output_attentions,
        )
        
    if output_attentions:
        visual_token_range = list(range(head_prompt_len, head_prompt_len + vid_token_len))
        audio_token_range = list(range(head_prompt_len + vid_token_len, outputs.logits.shape[1] - tail_prompt_len + 1))
        question_token_range = list(range(outputs.logits.shape[1] - tail_prompt_len + 1, outputs.logits.shape[1]-5))
        
        results = {k: [] for k in ["video", "audio", "question"]}

        attn_range = range(len(outputs.attentions))
        # attn_range = range(1)

        for i in attn_range:
            attn = outputs.attentions[i][0]
            attn_avg = attn.mean(dim=0)
            
            answer2vision = attn_avg[-1, visual_token_range]
            answer2audio = attn_avg[-1, audio_token_range]
            answer2question = attn_avg[-1, question_token_range]

            results['video'].append(answer2vision.sum().detach().item())
            results['audio'].append(answer2audio.sum().detach().item())
            results['question'].append(answer2question.sum().detach().item())

        # visual_score /= len(attn_range)
        # audio_score /= len(attn_range)
        # question_score /= len(attn_range)
        
        return outputs, results
    else:
        return outputs, {}


def mm_contrast_decode(image_or_videos, instruct, model, tokenizer, modals=['video','video', 'audio','text'], add_prompt='', debug = False, gamma = 0.5, **kwargs):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """

    # 1. vision preprocess (load & transform image or video).
    tensors = []
    
    for image_or_video, modal in zip(image_or_videos, modals):
        if modal == 'text':
            tensors.append(None)
        else:
            if isinstance(image_or_video, dict):
                tensor = {k: v.to(dtype=torch.bfloat16).cuda() for k, v in image_or_video.items()}
            else:
                tensor = image_or_video.to(dtype=torch.bfloat16).cuda() 
            tensors.append([(tensor, modal)])
    
    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message_head = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + "Question: " + instruct + "\n" + add_prompt}]
        message_va = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + instruct}]
        message_v = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + instruct}]
        message_a = [{'role': 'user', 'content': DEFAULT_AUDIO_TOKEN + '\n' + instruct}]
        message_t = [{'role': 'user', 'content': instruct}]
        
    elif isinstance(instruct, list):
        message_va = copy.deepcopy(instruct)
        message_v = copy.deepcopy(instruct)
        message_a = copy.deepcopy(instruct)
        message_t = copy.deepcopy(instruct)
        
        message_va[0]['content'] = DEFAULT_VIDEO_TOKEN + '\n' + message_va[0]['content']
        message_v[0]['content'] = DEFAULT_VIDEO_TOKEN + '\n' + message_v[0]['content']
        message_a[0]['content'] = DEFAULT_AUDIO_TOKEN + '\n' + message_a[0]['content']
        message_t[0]['content'] = instruct
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    # video audio

    message_va = system_message + message_va
    message_v = system_message + message_v
    message_a = system_message + message_a
    message_t = system_message + message_t

    message_head = system_message + message_head

    prompt_head = tokenizer.apply_chat_template(message_head, tokenize=False, add_generation_prompt=True)
    prompt_va = tokenizer.apply_chat_template(message_va, tokenize=False, add_generation_prompt=True)
    prompt_v = tokenizer.apply_chat_template(message_v, tokenize=False, add_generation_prompt=True)
    prompt_a = tokenizer.apply_chat_template(message_a, tokenize=False, add_generation_prompt=True)
    prompt_t = tokenizer.apply_chat_template(message_t, tokenize=False, add_generation_prompt=True)
    
    input_ids_head = tokenizer_multimodal_token(prompt_head, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_va = tokenizer_multimodal_token(prompt_va, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_v = tokenizer_multimodal_token(prompt_v, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_a = tokenizer_multimodal_token(prompt_a, tokenizer, DEFAULT_AUDIO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_t = tokenizer_multimodal_token(prompt_t, tokenizer, None, return_tensors='pt').unsqueeze(0).long().cuda()

    attention_masks_head = input_ids_head.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_va = input_ids_va.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_v = input_ids_v.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_a = input_ids_a.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_t = input_ids_t.ne(tokenizer.pad_token_id).long().cuda()
    
    # attention_mask = (batch_input_ids != tokenizer.pad_token_id).long().cuda()
    input_ids = [input_ids_va, input_ids_v, input_ids_a, input_ids_t]
    attention_masks = [attention_masks_va, attention_masks_v, attention_masks_a, attention_masks_t]

    # 3. generate response according to visual signals and prompts. 

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)
    allowed_token_ids = kwargs.get('allowed_token_ids')

    with torch.inference_mode():
        
        keywords = [tokenizer.eos_token]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids[0])

        do_sample = kwargs.get('do_sample', False)
        temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
        top_p = kwargs.get('top_p', 0.9)
        max_new_tokens = kwargs.get('max_new_tokens', 2048)
        
        # print(prompt_head)
        outputs = model.forward(
            input_ids_head,
            attention_mask=attention_masks_head,
            images=tensors[0],
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        audio_token_idx=tokenizer.encode("audio")[0]
        video_token_idx=tokenizer.encode("video")[0]
        both_token_idx=tokenizer.encode("both")[0]
        
        audio_logit = outputs.logits[0, -1, audio_token_idx].detach().cpu().item()
        video_logit = outputs.logits[0, -1, video_token_idx].detach().cpu().item()
        both_logit = outputs.logits[0, -1, both_token_idx].detach().cpu().item()

        avb_logits = torch.tensor([audio_logit, video_logit, both_logit])
        av_probs = torch.softmax(avb_logits, dim=0)
        audio_prob, video_prob, both_prob = av_probs.tolist()
        # alpha = math.fabs(audio_prob - video_prob)
        if debug:
            print(f"Audio prob: {audio_prob:.4f}, Video prob: {video_prob:.4f}, Both prob: {both_prob:.4f}")

        device = input_ids_va.device
        eos_id = tokenizer.eos_token_id
        max_new_tokens = kwargs.get('max_new_tokens', 256)

        # 분기 상태: 각 분기별 KV 캐시 보관
        branches = []
        step_logits = []

        # 1) 프롬프트 한 번 태워 캐시 warm-up (이미지/비디오는 첫 호출에만 전달)
        for inp_ids, attn_mask, vision in zip(input_ids, attention_masks, tensors):
            out = model(
                inp_ids,
                attention_mask=attn_mask,
                images=vision,          # 첫 스텝만 전달
                use_cache=True,
                pad_token_id=eos_id,
            )
            branches.append({
                "past": out.past_key_values,
                "images": None,         # 이후 스텝엔 보통 None
            })
            step_logits.append(out.logits[:, -1, :])  # [1, vocab]

        # 생성 결과 모음
        generated = []
        alpha_av = 2 * both_prob * gamma
        alpha_v = (both_prob - video_prob) * gamma
        alpha_a = (both_prob - audio_prob) * gamma
        alpha_t = (video_prob + audio_prob) * gamma

        weights = [2 + alpha_av, 1-alpha_v, 1-alpha_a, -alpha_t]
        # weights = [2 + alpha_av, -alpha_v, -alpha_a, -alpha_t]
        # weights = [(2+2*gamma)*both_prob, video_prob, audio_prob, - (video_prob + audio_prob) * gamma]
        
        for _ in range(max_new_tokens):
            # [num_branches, vocab] 평균 후 greedy
            logits_mat = torch.stack([w*lg.squeeze(0) for lg, w in zip(step_logits, weights)], dim=0)   # GPU 상
            avg_logits = logits_mat.sum(dim=0)                                      # [vocab]
            if allowed_token_ids:
                candidate_ids = torch.as_tensor(allowed_token_ids, device=avg_logits.device, dtype=torch.long)
                next_token = int(candidate_ids[torch.argmax(avg_logits[candidate_ids])].item())
            else:
                next_token = int(torch.argmax(avg_logits, dim=-1))
            generated.append(next_token)

            if next_token == eos_id or (_yes_no_early_stop_enabled() and _has_yes_no_answer(tokenizer, generated)):
                break

            # 다음 스텝: 모든 분기에 동일 토큰 공급 + 캐시 갱신
            next_tok_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)
            new_logits = []
            for b in branches:
                out = model(
                    next_tok_tensor,
                    attention_mask=None,        # 캐시 사용 시 보통 불필요
                    images=b["images"],         # 이후 None 유지
                    use_cache=True,
                    past_key_values=b["past"],  # 캐시 재사용
                    pad_token_id=eos_id,
                )
                b["past"] = out.past_key_values
                new_logits.append(out.logits[:, -1, :])  # [1, vocab]
            step_logits = new_logits

        # 텍스트로 변환
        text = tokenizer.decode(generated, skip_special_tokens=True)

    return text


def mm_contrast_decode_multi_gamma(image_or_videos, instruct, model, tokenizer, modals=['video','video', 'audio','text'], add_prompt='', debug=False, gamma_list=[0.5, 1.0, 1.5, 2.0], **kwargs):
    """Multi-gamma version of mm_contrast_decode for VideoLLaMA2.

    Efficiently evaluates multiple gamma values in a single pass.
    Model forward is done only once, then decoding is done for each gamma.

    Args:
        Same as mm_contrast_decode, but gamma is replaced with gamma_list
        gamma_list (list): List of gamma values to evaluate

    Returns:
        dict: {gamma: generated_text} for each gamma value
    """

    # 1. vision preprocess (load & transform image or video).
    tensors = []

    for image_or_video, modal in zip(image_or_videos, modals):
        if modal == 'text':
            tensors.append(None)
        else:
            if isinstance(image_or_video, dict):
                tensor = {k: v.to(dtype=torch.bfloat16).cuda() for k, v in image_or_video.items()}
            else:
                tensor = image_or_video.to(dtype=torch.bfloat16).cuda()
            tensors.append([(tensor, modal)])

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message_head = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + "Question: " + instruct + "\n" + add_prompt}]
        message_va = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + instruct}]
        message_v = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + instruct}]
        message_a = [{'role': 'user', 'content': DEFAULT_AUDIO_TOKEN + '\n' + instruct}]
        message_t = [{'role': 'user', 'content': instruct}]

    elif isinstance(instruct, list):
        message_va = copy.deepcopy(instruct)
        message_v = copy.deepcopy(instruct)
        message_a = copy.deepcopy(instruct)
        message_t = copy.deepcopy(instruct)

        message_va[0]['content'] = DEFAULT_VIDEO_TOKEN + '\n' + message_va[0]['content']
        message_v[0]['content'] = DEFAULT_VIDEO_TOKEN + '\n' + message_v[0]['content']
        message_a[0]['content'] = DEFAULT_AUDIO_TOKEN + '\n' + message_a[0]['content']
        message_t[0]['content'] = instruct
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message_va = system_message + message_va
    message_v = system_message + message_v
    message_a = system_message + message_a
    message_t = system_message + message_t
    message_head = system_message + message_head

    prompt_head = tokenizer.apply_chat_template(message_head, tokenize=False, add_generation_prompt=True)
    prompt_va = tokenizer.apply_chat_template(message_va, tokenize=False, add_generation_prompt=True)
    prompt_v = tokenizer.apply_chat_template(message_v, tokenize=False, add_generation_prompt=True)
    prompt_a = tokenizer.apply_chat_template(message_a, tokenize=False, add_generation_prompt=True)
    prompt_t = tokenizer.apply_chat_template(message_t, tokenize=False, add_generation_prompt=True)

    input_ids_head = tokenizer_multimodal_token(prompt_head, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_va = tokenizer_multimodal_token(prompt_va, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_v = tokenizer_multimodal_token(prompt_v, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_a = tokenizer_multimodal_token(prompt_a, tokenizer, DEFAULT_AUDIO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_t = tokenizer_multimodal_token(prompt_t, tokenizer, None, return_tensors='pt').unsqueeze(0).long().cuda()

    attention_masks_head = input_ids_head.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_va = input_ids_va.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_v = input_ids_v.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_a = input_ids_a.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_t = input_ids_t.ne(tokenizer.pad_token_id).long().cuda()

    input_ids = [input_ids_va, input_ids_v, input_ids_a, input_ids_t]
    attention_masks = [attention_masks_va, attention_masks_v, attention_masks_a, attention_masks_t]

    # 3. generate response according to visual signals and prompts.
    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    with torch.inference_mode():

        keywords = [tokenizer.eos_token]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids[0])

        # Get modality probabilities (done once)
        outputs = model.forward(
            input_ids_head,
            attention_mask=attention_masks_head,
            images=tensors[0],
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id
        )

        audio_token_idx=tokenizer.encode("audio")[0]
        video_token_idx=tokenizer.encode("video")[0]
        both_token_idx=tokenizer.encode("both")[0]

        audio_logit = outputs.logits[0, -1, audio_token_idx].detach().cpu().item()
        video_logit = outputs.logits[0, -1, video_token_idx].detach().cpu().item()
        both_logit = outputs.logits[0, -1, both_token_idx].detach().cpu().item()

        avb_logits = torch.tensor([audio_logit, video_logit, both_logit])
        av_probs = torch.softmax(avb_logits, dim=0)
        audio_prob, video_prob, both_prob = av_probs.tolist()

        if debug:
            print(f"Audio prob: {audio_prob:.4f}, Video prob: {video_prob:.4f}, Both prob: {both_prob:.4f}")

        device = input_ids_va.device
        eos_id = tokenizer.eos_token_id
        max_new_tokens = kwargs.get('max_new_tokens', 256)

        # Initialize branches (done once for all gammas)
        branches_initial = []
        step_logits_initial = []

        for inp_ids, attn_mask, vision in zip(input_ids, attention_masks, tensors):
            out = model(
                inp_ids,
                attention_mask=attn_mask,
                images=vision,
                use_cache=True,
                pad_token_id=eos_id,
            )
            branches_initial.append({
                "past": out.past_key_values,
                "images": None,
            })
            step_logits_initial.append(out.logits[:, -1, :])  # [1, vocab]

        # Generate for each gamma
        results = {}

        for gamma in gamma_list:
            # Deep copy branches for this gamma
            import copy
            branches = copy.deepcopy(branches_initial)
            step_logits = [lg.clone() for lg in step_logits_initial]

            generated = []
            alpha_av = 2 * both_prob * gamma
            alpha_v = (both_prob - video_prob) * gamma
            alpha_a = (both_prob - audio_prob) * gamma
            alpha_t = (video_prob + audio_prob) * gamma

            weights = [2 + alpha_av, 1-alpha_v, 1-alpha_a, -alpha_t]

            for _ in range(max_new_tokens):
                logits_mat = torch.stack([w*lg.squeeze(0) for lg, w in zip(step_logits, weights)], dim=0)
                avg_logits = logits_mat.sum(dim=0)
                next_token = int(torch.argmax(avg_logits, dim=-1))
                generated.append(next_token)

                if next_token == eos_id:
                    break

                next_tok_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)
                new_logits = []
                for b in branches:
                    out = model(
                        next_tok_tensor,
                        attention_mask=None,
                        images=b["images"],
                        use_cache=True,
                        past_key_values=b["past"],
                        pad_token_id=eos_id,
                    )
                    b["past"] = out.past_key_values
                    new_logits.append(out.logits[:, -1, :])
                step_logits = new_logits

            text = tokenizer.decode(generated, skip_special_tokens=True)
            results[gamma] = text

    return results

def mm_contrast_decode_multi_weights(image_or_videos, instruct, model, tokenizer, modals=['video','video', 'audio','text'], add_prompt='', debug=False, gamma=2.5, weights_list=['av', 'a', 'v', 'all'], **kwargs):
    """Multi-gamma version of mm_contrast_decode for VideoLLaMA2.

    Efficiently evaluates multiple gamma values in a single pass.
    Model forward is done only once, then decoding is done for each gamma.

    Args:
        Same as mm_contrast_decode, but gamma is replaced with gamma_list
        gamma_list (list): List of gamma values to evaluate

    Returns:
        dict: {gamma: generated_text} for each gamma value
    """

    # 1. vision preprocess (load & transform image or video).
    tensors = []

    for image_or_video, modal in zip(image_or_videos, modals):
        if modal == 'text':
            tensors.append(None)
        else:
            if isinstance(image_or_video, dict):
                tensor = {k: v.to(dtype=torch.bfloat16).cuda() for k, v in image_or_video.items()}
            else:
                tensor = image_or_video.to(dtype=torch.bfloat16).cuda()
            tensors.append([(tensor, modal)])

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message_head = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + "Question: " + instruct + "\n" + add_prompt}]
        message_va = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + instruct}]
        message_v = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + instruct}]
        message_a = [{'role': 'user', 'content': DEFAULT_AUDIO_TOKEN + '\n' + instruct}]
        message_t = [{'role': 'user', 'content': instruct}]

    elif isinstance(instruct, list):
        message_va = copy.deepcopy(instruct)
        message_v = copy.deepcopy(instruct)
        message_a = copy.deepcopy(instruct)
        message_t = copy.deepcopy(instruct)

        message_va[0]['content'] = DEFAULT_VIDEO_TOKEN + '\n' + message_va[0]['content']
        message_v[0]['content'] = DEFAULT_VIDEO_TOKEN + '\n' + message_v[0]['content']
        message_a[0]['content'] = DEFAULT_AUDIO_TOKEN + '\n' + message_a[0]['content']
        message_t[0]['content'] = instruct
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message_va = system_message + message_va
    message_v = system_message + message_v
    message_a = system_message + message_a
    message_t = system_message + message_t
    message_head = system_message + message_head

    prompt_head = tokenizer.apply_chat_template(message_head, tokenize=False, add_generation_prompt=True)
    prompt_va = tokenizer.apply_chat_template(message_va, tokenize=False, add_generation_prompt=True)
    prompt_v = tokenizer.apply_chat_template(message_v, tokenize=False, add_generation_prompt=True)
    prompt_a = tokenizer.apply_chat_template(message_a, tokenize=False, add_generation_prompt=True)
    prompt_t = tokenizer.apply_chat_template(message_t, tokenize=False, add_generation_prompt=True)

    input_ids_head = tokenizer_multimodal_token(prompt_head, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_va = tokenizer_multimodal_token(prompt_va, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_v = tokenizer_multimodal_token(prompt_v, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_a = tokenizer_multimodal_token(prompt_a, tokenizer, DEFAULT_AUDIO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_t = tokenizer_multimodal_token(prompt_t, tokenizer, None, return_tensors='pt').unsqueeze(0).long().cuda()

    attention_masks_head = input_ids_head.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_va = input_ids_va.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_v = input_ids_v.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_a = input_ids_a.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_t = input_ids_t.ne(tokenizer.pad_token_id).long().cuda()

    input_ids = [input_ids_va, input_ids_v, input_ids_a, input_ids_t]
    attention_masks = [attention_masks_va, attention_masks_v, attention_masks_a, attention_masks_t]

    # 3. generate response according to visual signals and prompts.
    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    with torch.inference_mode():

        keywords = [tokenizer.eos_token]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids[0])

        # Get modality probabilities (done once)
        outputs = model.forward(
            input_ids_head,
            attention_mask=attention_masks_head,
            images=tensors[0],
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id
        )

        audio_token_idx=tokenizer.encode("audio")[0]
        video_token_idx=tokenizer.encode("video")[0]
        both_token_idx=tokenizer.encode("both")[0]

        audio_logit = outputs.logits[0, -1, audio_token_idx].detach().cpu().item()
        video_logit = outputs.logits[0, -1, video_token_idx].detach().cpu().item()
        both_logit = outputs.logits[0, -1, both_token_idx].detach().cpu().item()

        avb_logits = torch.tensor([audio_logit, video_logit, both_logit])
        av_probs = torch.softmax(avb_logits, dim=0)
        audio_prob, video_prob, both_prob = av_probs.tolist()

        if debug:
            print(f"Audio prob: {audio_prob:.4f}, Video prob: {video_prob:.4f}, Both prob: {both_prob:.4f}")

        device = input_ids_va.device
        eos_id = tokenizer.eos_token_id
        max_new_tokens = kwargs.get('max_new_tokens', 256)

        # Initialize branches (done once for all gammas)
        branches_initial = []
        step_logits_initial = []

        for inp_ids, attn_mask, vision in zip(input_ids, attention_masks, tensors):
            out = model(
                inp_ids,
                attention_mask=attn_mask,
                images=vision,
                use_cache=True,
                pad_token_id=eos_id,
            )
            branches_initial.append({
                "past": out.past_key_values,
                "images": None,
            })
            step_logits_initial.append(out.logits[:, -1, :])  # [1, vocab]

        # Generate for each gamma
        results = {}

        for weight in weights_list:
            # Deep copy branches for this gamma
            import copy
            branches = copy.deepcopy(branches_initial)
            step_logits = [lg.clone() for lg in step_logits_initial]

            generated = []
            
            if weight == 'av':
                alpha_av = 0
                alpha_v = (- video_prob) * gamma
                alpha_a = (- audio_prob) * gamma
                alpha_t = (video_prob + audio_prob) * gamma
            elif weight == 'a':
                alpha_av = 2 * both_prob * gamma
                alpha_v = (both_prob - video_prob) * gamma
                alpha_a = (both_prob) * gamma
                alpha_t = (video_prob) * gamma
            elif weight == 'v':
                alpha_av = 2 * both_prob * gamma
                alpha_v = (both_prob ) * gamma
                alpha_a = (both_prob - audio_prob) * gamma
                alpha_t = (audio_prob) * gamma
            elif weight == 'all':
                alpha_av = 2 * both_prob * gamma
                alpha_v = (both_prob - video_prob) * gamma
                alpha_a = (both_prob - audio_prob) * gamma
                alpha_t = (video_prob + audio_prob) * gamma
            
            # alpha_av = 2 * both_prob * gamma
            # alpha_v = (both_prob - video_prob) * gamma
            # alpha_a = (both_prob - audio_prob) * gamma
            # alpha_t = (video_prob + audio_prob) * gamma
            weights = [2 + alpha_av, 1-alpha_v, 1-alpha_a, -alpha_t]

            for _ in range(max_new_tokens):
                logits_mat = torch.stack([w*lg.squeeze(0) for lg, w in zip(step_logits, weights)], dim=0)
                avg_logits = logits_mat.sum(dim=0)
                next_token = int(torch.argmax(avg_logits, dim=-1))
                generated.append(next_token)

                if next_token == eos_id:
                    break

                next_tok_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)
                new_logits = []
                for b in branches:
                    out = model(
                        next_tok_tensor,
                        attention_mask=None,
                        images=b["images"],
                        use_cache=True,
                        past_key_values=b["past"],
                        pad_token_id=eos_id,
                    )
                    b["past"] = out.past_key_values
                    new_logits.append(out.logits[:, -1, :])
                step_logits = new_logits

            text = tokenizer.decode(generated, skip_special_tokens=True)
            results[weight] = text

    return results


def mm_test(image_or_video, instruct, model, tokenizer, modal='video', add_prompt='', **kwargs):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """

    # 1. text preprocess (tag process & generate prompt).
    if modal == 'image':
        modal_token = DEFAULT_IMAGE_TOKEN
    elif modal == 'video':
        modal_token = DEFAULT_VIDEO_TOKEN
    elif modal == 'text':
        modal_token = ''
    elif modal == 'audio':
        modal_token = DEFAULT_AUDIO_TOKEN
    else:
        raise ValueError(f"Unsupported modal: {modal}")

    # 1. vision preprocess (load & transform image or video).
    if modal == 'text':
        tensor = None
    else:
        # if isinstance(image_or_video, dict):
        #     tensor = {k: v.half().cuda() for k, v in image_or_video.items()}
        # else:
        #     tensor = image_or_video.half().cuda() 
            
        if isinstance(image_or_video, dict):
            tensor = {k: v.to(dtype=torch.bfloat16).cuda() for k, v in image_or_video.items()}
        else:
            tensor = image_or_video.to(dtype=torch.bfloat16).cuda() 
        tensor = [(tensor, modal)]

    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message = [{'role': 'user', 'content': modal_token + '\n' + instruct}]
    elif isinstance(instruct, list):
        message = copy.deepcopy(instruct)
        message[0]['content'] = modal_token + '\n' + message[0]['content']
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    message = system_message + message
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    # prompt += add_prompt

    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors='pt').unsqueeze(0).long().cuda()
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    # 3. generate response according to visual signals and prompts. 

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    with torch.inference_mode():
        
        keywords = [tokenizer.eos_token]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        outputs = model.forward(
            input_ids,
            attention_mask=attention_masks,
            images=tensor,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id
        )
        
        audio_token_idx=tokenizer.encode("audio")[0]
        video_token_idx=tokenizer.encode("video")[0]
        both_token_idx=tokenizer.encode("both")[0]
        
        audio_logit = outputs.logits[0, -1, audio_token_idx].detach().cpu().item()
        video_logit = outputs.logits[0, -1, video_token_idx].detach().cpu().item()
        both_logit = outputs.logits[0, -1, both_token_idx].detach().cpu().item()

        avb_logits = torch.tensor([audio_logit, video_logit, both_logit])
        av_probs = torch.softmax(avb_logits, dim=0)
        audio_prob, video_prob, both_prob = av_probs.tolist()

        return {
            "audio": audio_prob,
            "video": video_prob,
            "both": both_prob
        }
def mm_default_cd(image_or_videos, instruct, model, tokenizer, modals=['video','video', 'audio','text'], add_prompt='', debug = False, gamma = 0.5, **kwargs):
    """inference api of VideoLLaMA2 for video understanding.

    Args:
        model: VideoLLaMA2 model.
        image_or_video (torch.Tensor): image tensor (1, C, H, W) / video tensor (T, C, H, W).
        instruct (str): text instruction for understanding video.
        tokenizer: tokenizer.
        do_sample (bool): whether to sample.
        modal (str): inference modality.
    Returns:
        str: response of the model.
    """

    # 1. vision preprocess (load & transform image or video).
    tensors = []
    
    for image_or_video, modal in zip(image_or_videos, modals):
        if modal == 'text':
            tensors.append(None)
        else:
            if isinstance(image_or_video, dict):
                tensor = {k: v.to(dtype=torch.bfloat16).cuda() for k, v in image_or_video.items()}
            else:
                tensor = image_or_video.to(dtype=torch.bfloat16).cuda() 
            tensors.append([(tensor, modal)])
    
    # 2. text preprocess (tag process & generate prompt).
    if isinstance(instruct, str):
        message_head = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + "Question: " + instruct + "\n" + add_prompt}]
        message_va = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + instruct}]
        message_v = [{'role': 'user', 'content': DEFAULT_VIDEO_TOKEN + '\n' + instruct}]
        message_a = [{'role': 'user', 'content': DEFAULT_AUDIO_TOKEN + '\n' + instruct}]
        message_t = [{'role': 'user', 'content': instruct}]
        
    elif isinstance(instruct, list):
        message_va = copy.deepcopy(instruct)
        message_v = copy.deepcopy(instruct)
        message_a = copy.deepcopy(instruct)
        message_t = copy.deepcopy(instruct)
        
        message_va[0]['content'] = DEFAULT_VIDEO_TOKEN + '\n' + message_va[0]['content']
        message_v[0]['content'] = DEFAULT_VIDEO_TOKEN + '\n' + message_v[0]['content']
        message_a[0]['content'] = DEFAULT_AUDIO_TOKEN + '\n' + message_a[0]['content']
        message_t[0]['content'] = instruct
    else:
        raise ValueError(f"Unsupported type of instruct: {type(instruct)}")

    if model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
        system_message = [
            {'role': 'system', 'content': (
            """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
            """\n"""
            """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
            }
        ]
    else:
        system_message = []

    # video audio

    message_va = system_message + message_va
    message_v = system_message + message_v
    message_a = system_message + message_a
    message_t = system_message + message_t

    message_head = system_message + message_head

    prompt_head = tokenizer.apply_chat_template(message_head, tokenize=False, add_generation_prompt=True)
    prompt_va = tokenizer.apply_chat_template(message_va, tokenize=False, add_generation_prompt=True)
    prompt_v = tokenizer.apply_chat_template(message_v, tokenize=False, add_generation_prompt=True)
    prompt_a = tokenizer.apply_chat_template(message_a, tokenize=False, add_generation_prompt=True)
    prompt_t = tokenizer.apply_chat_template(message_t, tokenize=False, add_generation_prompt=True)
    
    input_ids_head = tokenizer_multimodal_token(prompt_head, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_va = tokenizer_multimodal_token(prompt_va, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_v = tokenizer_multimodal_token(prompt_v, tokenizer, DEFAULT_VIDEO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_a = tokenizer_multimodal_token(prompt_a, tokenizer, DEFAULT_AUDIO_TOKEN, return_tensors='pt').unsqueeze(0).long().cuda()
    input_ids_t = tokenizer_multimodal_token(prompt_t, tokenizer, None, return_tensors='pt').unsqueeze(0).long().cuda()

    attention_masks_head = input_ids_head.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_va = input_ids_va.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_v = input_ids_v.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_a = input_ids_a.ne(tokenizer.pad_token_id).long().cuda()
    attention_masks_t = input_ids_t.ne(tokenizer.pad_token_id).long().cuda()
    
    # attention_mask = (batch_input_ids != tokenizer.pad_token_id).long().cuda()
    input_ids = [input_ids_va, input_ids_v, input_ids_a, input_ids_t]
    attention_masks = [attention_masks_va, attention_masks_v, attention_masks_a, attention_masks_t]

    # 3. generate response according to visual signals and prompts. 

    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
    top_p = kwargs.get('top_p', 0.9)
    max_new_tokens = kwargs.get('max_new_tokens', 2048)

    with torch.inference_mode():
        
        keywords = [tokenizer.eos_token]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids[0])

        do_sample = kwargs.get('do_sample', False)
        temperature = kwargs.get('temperature', 0.2 if do_sample else 0.0)
        top_p = kwargs.get('top_p', 0.9)
        max_new_tokens = kwargs.get('max_new_tokens', 2048)
        
        # print(prompt_head)
        outputs = model.forward(
            input_ids_head,
            attention_mask=attention_masks_head,
            images=tensors[0],
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id
        )
        
        audio_token_idx=tokenizer.encode("audio")[0]
        video_token_idx=tokenizer.encode("video")[0]
        both_token_idx=tokenizer.encode("both")[0]
        
        audio_logit = outputs.logits[0, -1, audio_token_idx].detach().cpu().item()
        video_logit = outputs.logits[0, -1, video_token_idx].detach().cpu().item()
        both_logit = outputs.logits[0, -1, both_token_idx].detach().cpu().item()

        avb_logits = torch.tensor([audio_logit, video_logit, both_logit])
        av_probs = torch.softmax(avb_logits, dim=0)
        audio_prob, video_prob, both_prob = av_probs.tolist()
        # alpha = math.fabs(audio_prob - video_prob)
        if debug:
            print(f"Audio prob: {audio_prob:.4f}, Video prob: {video_prob:.4f}, Both prob: {both_prob:.4f}")

        device = input_ids_va.device
        eos_id = tokenizer.eos_token_id
        max_new_tokens = kwargs.get('max_new_tokens', 256)

        # 분기 상태: 각 분기별 KV 캐시 보관
        branches = []
        step_logits = []

        # 1) 프롬프트 한 번 태워 캐시 warm-up (이미지/비디오는 첫 호출에만 전달)
        for inp_ids, attn_mask, vision in zip(input_ids, attention_masks, tensors):
            out = model(
                inp_ids,
                attention_mask=attn_mask,
                images=vision,          # 첫 스텝만 전달
                use_cache=True,
                pad_token_id=eos_id,
            )
            branches.append({
                "past": out.past_key_values,
                "images": None,         # 이후 스텝엔 보통 None
            })
            step_logits.append(out.logits[:, -1, :])  # [1, vocab]

        # 생성 결과 모음
        generated = []
        alpha_av = 1 + 3*gamma 
        alpha_v = - gamma
        alpha_a = - gamma
        alpha_t = - gamma

        weights = [alpha_av, alpha_v, alpha_a, alpha_t]
        # weights = [2 + alpha_av, -alpha_v, -alpha_a, -alpha_t]
        # weights = [(2+2*gamma)*both_prob, video_prob, audio_prob, - (video_prob + audio_prob) * gamma]
        
        for _ in range(max_new_tokens):
            # [num_branches, vocab] 평균 후 greedy
            logits_mat = torch.stack([w*lg.squeeze(0) for lg, w in zip(step_logits, weights)], dim=0)   # GPU 상
            avg_logits = logits_mat.sum(dim=0)                                      # [vocab]
            next_token = int(torch.argmax(avg_logits, dim=-1))
            generated.append(next_token)

            if next_token == eos_id:
                break

            # 다음 스텝: 모든 분기에 동일 토큰 공급 + 캐시 갱신
            next_tok_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)
            new_logits = []
            for b in branches:
                out = model(
                    next_tok_tensor,
                    attention_mask=None,        # 캐시 사용 시 보통 불필요
                    images=b["images"],         # 이후 None 유지
                    use_cache=True,
                    past_key_values=b["past"],  # 캐시 재사용
                    pad_token_id=eos_id,
                )
                b["past"] = out.past_key_values
                new_logits.append(out.logits[:, -1, :])  # [1, vocab]
            step_logits = new_logits

        # 텍스트로 변환
        text = tokenizer.decode(generated, skip_special_tokens=True)

    return text
