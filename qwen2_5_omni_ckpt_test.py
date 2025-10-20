#!/usr/bin/env python3
"""
End-to-end test example for Qwen2.5-Omni merged model using real checkpoints.
Modified for deterministic behavior - all randomness removed for debugging.

This script expects a merged checkpoint directory that includes a config.json
with architectures: ["Qwen2_5OmniMergedModel"], and thinker/talker weights.

It can also automatically download the official model repo from Hugging Face,
e.g. Qwen/Qwen2.5-Omni-7B, and prepare the directory for you.

It also requires Code2Wav checkpoints (DiT and BigVGAN), or a code2wav model
folder containing spk_dict.pt. You can provide either individual ckpt files or
a --code2wav-dir that contains spk_dict.pt and (optionally) model files.

Usage (Windows PowerShell):

python examples/offline_inference/qwen2_5_omni_ckpt_test.py \
  --model C:\\path\\to\\qwen2_5_omni_merged_ckpt \
  --prompt "你好，请介绍一下你自己。" \
  --voice-type m02 \
  --dit-ckpt C:\\path\\to\\code2wav\\dit.pt \
  --bigvgan-ckpt C:\\path\\to\\code2wav\\bigvgan.pt \
  --output-wav C:\\path\\to\\output.wav

Or if you have a code2wav folder:

python examples/offline_inference/qwen2_5_omni_ckpt_test.py \
  --model C:\\path\\to\\qwen2_5_omni_merged_ckpt \
  --prompt "请用中文介绍一下人工智能的发展历程。" \
  --voice-type m02 \
  --code2wav-dir C:\\path\\to\\code2wav_dir \
  --output-wav C:\\path\\to\\output.wav

Auto-download from Hugging Face (Qwen/Qwen2.5-Omni-7B):

python examples/offline_inference/qwen2_5_omni_ckpt_test.py \
  --hf-hub-id Qwen/Qwen2.5-Omni-7B \
  --model C:\\models\\Qwen2.5-Omni-7B \
  --prompt "请用中文介绍一下人工智能的发展历程。" \
  --voice-type default \
  --code2wav-dir C:\\models\\Qwen2.5-Omni-7B \
  --output-wav C:\\temp\\omni_out.wav
"""

import os
import random
import numpy as np
import torch
from dataclasses import asdict
# =============================================================================
# DETERMINISTIC SETUP - MUST BE FIRST
# =============================================================================
SEED = 42  # Fixed seed for reproducibility

# Set all random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make PyTorch deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variables for deterministic behavior
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import copy
import json
import queue
import signal
import tempfile
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from urllib.request import urlopen

import librosa
import psutil
import requests
import resampy
import soundfile as sf

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from vllm.inputs import TextPrompt
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.multimodal.processing_omni import fetch_image, fetch_video
from vllm.sampling_params import SamplingParams

import asyncio
try:
    from huggingface_hub import snapshot_download
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

import os as _os_env_toggle
_os_env_toggle.environ["VLLM_USE_V1"] = "1"
from vllm_omni.entrypoints.omni_llm import OmniLLM
from vllm_omni.engine.arg_utils import OmniEngineArgs

import hashlib

def hash_tensor_bytes(t: torch.Tensor, *, algo: str = "sha256") -> str:
    """
    Hash a tensor by its raw bytes, including dtype and shape metadata.

    - Moves to CPU (no copy if already CPU) and makes contiguous for stable bytes.
    - Includes dtype and shape explicitly to avoid collisions across different views.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor")

    # Normalize representation
    t_cpu = t.to(torch.float32).detach().to("cpu").contiguous()
    # Prepare a stable header with dtype and shape
    header = f"{str(t_cpu.dtype)}|{tuple(t_cpu.shape)}|".encode("utf-8")
    h = hashlib.new(algo)
    h.update(header)
    h.update(t_cpu.numpy().tobytes())  # shares memory; no extra copy beyond contiguous
    return h.hexdigest()

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to merged model directory (will be created if downloading).')
parser.add_argument('--thinker-model', type=str, default=None)
parser.add_argument('--talker-model', type=str, default=None)
parser.add_argument('--code2wav-model', type=str, default=None)
parser.add_argument('--hf-hub-id', default='Qwen/Qwen2.5-Omni-7B', help='Hugging Face repo id to download if needed.')
parser.add_argument('--hf-revision', default=None, help='Optional HF revision (branch/tag/commit).')
parser.add_argument('--prompts', required=True, nargs='+', help='Input text prompts.')
parser.add_argument('--voice-type', default='default', help='Voice type, e.g., m02, f030, default.')
parser.add_argument('--code2wav-dir', default=None, help='Path to code2wav folder (contains spk_dict.pt).')
parser.add_argument('--dit-ckpt', default=None, help='Path to DiT checkpoint file (e.g., dit.pt).')
parser.add_argument('--bigvgan-ckpt', default=None, help='Path to BigVGAN checkpoint file.')
parser.add_argument('--dtype', default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
parser.add_argument('--max-model-len', type=int, default=32768)

parser.add_argument("--thinker-only", action="store_true")
parser.add_argument("--text-only", action="store_true")
parser.add_argument("--do-wave", action="store_true")
parser.add_argument('--prompt_type',
                    choices=[
                        'text', 'audio', 'audio-long', 'audio-long-chunks',
                        'audio-long-expand-chunks', 'image', 'video',
                        'video-frames', 'audio-in-video', 'audio-in-video-v2',
                        "audio-multi-round", "badcase-vl", "badcase-text",
                        "badcase-image-early-stop", "badcase-two-audios",
                        "badcase-two-videos", "badcase-multi-round",
                        "badcase-voice-type", "badcase-voice-type-v2",
                        "badcase-audio-tower-1", "badcase-audio-only"
                    ],
                    default='text')
parser.add_argument('--use-torchvision', action='store_true')
parser.add_argument('--tokenize', action='store_true')
parser.add_argument('--output-wav', default="output.wav", help='Output wav dir or file path.')
parser.add_argument('--thinker-hidden-states-dir', default="thinker_hidden_states", help='Path to thinker hidden states directory.')
parser.add_argument('--stats-csv', default="stats_log.csv", help='Path to save per-prompt stats CSV.')
parser.add_argument('--write-csv', action='store_true', help='Enable writing per-prompt stats to CSV.')
args = parser.parse_args()

os.environ["thinker_hidden_states_dir"] = args.thinker_hidden_states_dir
os.makedirs(args.thinker_hidden_states_dir, exist_ok=True)
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def print_nvidia_sim_info(gpu_index = 0, stage = None, model_stage = None):
    # stage: Before_destroy/After_destroy
    # model_stage: thinker/talker/code2wav

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"--------------------------------")
        print(f"{stage} {model_stage} GPU {gpu_index} Memory Usage: {info.used / 1024 ** 3:.2f} GB / {info.total / 1024 ** 3:.2f} GB")
        print(f"--------------------------------")
    except Exception as e:
        print(f"Error getting GPU memory usage: {e}")

def ensure_hf_model(model_dir: str, hf_hub_id: str | None, revision: str | None):
    """Download model repo from HF into model_dir if needed."""
    if not hf_hub_id:
        return
    if not _HF_AVAILABLE:
        raise RuntimeError("huggingface_hub is not installed. Please `pip install -U huggingface_hub`." )
    cfg_path = os.path.join(model_dir, 'config.json')
    if os.path.isdir(model_dir) and os.path.exists(cfg_path):
        return  # already prepared
    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(
        repo_id=hf_hub_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        revision=revision,
        ignore_patterns=['.git/*'],
    )

def ensure_config(model_dir: str, code2wav_dir: str | None, dit_ckpt: str | None, bigvgan_ckpt: str | None):
    """Ensure the merged model's config.json has code2wav_config fields set.

    This function updates/creates config.json in-place to include:
      - architectures: ["Qwen2_5OmniMergedModel"]
      - code2wav_config: { dit_checkpoint, bigvgan_checkpoint, model_path?, frequency }
    It does not modify thinker/talker sub-configs; those should already match your weights.
    """
    cfg_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    else:
        cfg = {}

    archs = cfg.get('architectures') or []
    if 'Qwen2_5OmniMergedModel' not in archs:
        cfg['architectures'] = ['Qwen2_5OmniMergedModel']

    code2wav_cfg = cfg.get('code2wav_config') or {}
    if code2wav_dir:
        code2wav_cfg['model_path'] = code2wav_dir
    if dit_ckpt:
        code2wav_cfg['dit_checkpoint'] = dit_ckpt
    if bigvgan_ckpt:
        code2wav_cfg['bigvgan_checkpoint'] = bigvgan_ckpt
    code2wav_cfg.setdefault('frequency', '50hz')
    cfg['code2wav_config'] = code2wav_cfg

    os.makedirs(model_dir, exist_ok=True)
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def get_model_instance(engine):
    # V1 AsyncLLM 不暴露底层模型句柄；如需底层信息，请通过 EngineCore/collective_rpc 查询
    return None

def resample_wav_to_16khz(input_filepath):
    data, original_sample_rate = sf.read(input_filepath)
    # Only use the first channel
    if len(data.shape) > 1:
        data = data[:, 0]
    # resample to 16kHz
    data_resampled = resampy.resample(data,
                                      sr_orig=original_sample_rate,
                                      sr_new=16000)
    return data_resampled

def fetch_and_read_video(video_url: str, fps=2):
    import torchvision.io

    def read_video_with_torchvision(video_file_name: str):
        video, audio, info = torchvision.io.read_video(
            video_file_name,
            start_pts=0.0,
            end_pts=None,
            pts_unit="sec",
            output_format="TCHW",
        )

        total_frames, video_fps = video.size(0), info["video_fps"]
        total_duration = round(total_frames / video_fps, 3)
        nframes = int(total_frames / video_fps * fps)

        frame_timestamps = total_duration * torch.arange(1,
                                                         nframes + 1) / nframes
        grid_timestamps = frame_timestamps[::2]
        second_per_grid = grid_timestamps[1] - grid_timestamps[0]

        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        video_height, video_width = video.shape[2:]
        video = video[idx]

        if args.legacy_omni_video:
            return [video, total_duration, nframes, second_per_grid.item()]
        else:
            return video

    def read_video_with_transformers(video_file_name: Union[str, List[str]]):
        video, total_duration, nframes, second_per_grid = fetch_video(
            {'video': video_file_name})
        if total_duration is None and nframes is None:
            nframes = len(video)
            total_duration = 0.5 * nframes
            second_per_grid = 1.0
        if args.legacy_omni_video:
            return [video, total_duration, nframes, second_per_grid]
        else:
            return video

    def read_video(video_file_name: str):
        if args.use_torchvision:
            return read_video_with_torchvision(video_file_name)
        else:
            return read_video_with_transformers(video_file_name)

    if isinstance(video_url, str) and video_url.startswith("http"):
        with tempfile.NamedTemporaryFile(delete=True) as temp_video_file:
            resp = requests.get(video_url)
            assert resp.status_code == requests.codes.ok, f"Failed to fetch video from {video_url}, status_code:{resp.status_code}, resp:{resp}"

            temp_video_file.write(urlopen(video_url).read())
            temp_video_file_path = temp_video_file.name
            video_file_name = temp_video_file_path
            return read_video(video_file_name)
    else:
        video_file_name = video_url
        return read_video(video_file_name)

# =============================================================================
# PROMPT CREATION FUNCTIONS
# =============================================================================
def make_inputs_qwen2_omni(
    messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
    use_audio_in_video: Optional[bool] = False,
    tokenize: bool = args.tokenize,
) -> Union[OmniTokensPrompt, TextPrompt]:
    
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    try:
        config = AutoConfig.from_pretrained(args.model)
        if 'Qwen2_5OmniModel' in config.architectures:
            args.legacy_omni_video = False
        else:
            args.legacy_omni_video = True
    except:
        args.legacy_omni_video = True

    audios, images, videos = [], [], []
    for message in messages:
        if not isinstance(message['content'], list):
            message['content'] = [{
                'type': 'text',
                'text': message['content'],
            }]
        index, num_contents = 0, len(message['content'])
        while index < num_contents:
            ele = message['content'][index]
            if 'type' not in ele:
                if 'text' in ele:
                    ele['type'] = 'text'
                elif 'audio' in ele:
                    ele['type'] = 'audio'
                elif 'audio_url' in ele:
                    ele['type'] = 'audio_url'
                elif 'image' in ele:
                    ele['type'] = 'image'
                elif 'image_url' in ele:
                    ele['type'] = 'image_url'
                elif 'video' in ele:
                    ele['type'] = 'video'
                elif 'video_url' in ele:
                    ele['type'] = 'video_url'
                else:
                    raise ValueError(f'Unknown ele: {ele}')

            if ele['type'] == 'audio' or ele['type'] == 'audio_url':
                if 'audio_url' in ele:
                    audio_key = 'audio_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_audio_file:
                        temp_audio_file.write(urlopen(ele[audio_key]).read())
                        temp_audio_file_path = temp_audio_file.name
                        audios.append(
                            resample_wav_to_16khz(temp_audio_file_path))
                        ele['audio'] = temp_audio_file_path
                elif 'audio' in ele:
                    audio_key = 'audio'
                    audios.append(resample_wav_to_16khz(ele[audio_key]))
                else:
                    raise ValueError(f'Unknown ele {ele}')
            elif use_audio_in_video and (ele['type'] == 'video'
                                         or ele['type'] == 'video_url'):
                # use video as audio as well
                if 'video_url' in ele:
                    audio_key = 'video_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_video_file:
                        temp_video_file.write(urlopen(ele[audio_key]).read())
                        temp_video_file_path = temp_video_file.name
                        ele[audio_key] = temp_video_file_path
                        audios.append(
                            librosa.load(temp_video_file_path, sr=16000)[0])
                        videos.append(
                            fetch_and_read_video(temp_video_file_path))
                        ele['video'] = temp_video_file_path
                elif 'video' in ele:
                    audio_key = 'video'
                    audios.append(librosa.load(ele[audio_key], sr=16000)[0])
                    videos.append(fetch_and_read_video(audio_key))
                else:
                    raise ValueError("Unknown ele {}".format(ele))
                # insert a audio after the video
                message['content'].insert(index + 1, {
                    "type": "audio",
                    "audio": ele[audio_key],
                })
                # no need to load the added audio again
                index += 1
            elif ele['type'] == 'video' or ele['type'] == 'video_url':
                if 'video_url' in ele:
                    video_key = 'video_url'
                    with tempfile.NamedTemporaryFile(
                            delete=True) as temp_video_file:
                        temp_video_file.write(urlopen(ele['video_url']).read())
                        temp_video_file_path = temp_video_file.name
                        videos.append(fetch_and_read_video(temp_video_file))
                        ele['video'] = temp_video_file_path
                else:
                    video_key = 'video'
                    videos.append(fetch_and_read_video(ele[video_key]))
            elif ele['type'] == 'image' or ele['type'] == 'image_url':
                images.append(fetch_image(ele))

            # move to the next content
            index += 1

    prompt = processor.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=True,
        add_vision_id=True,
    )

    audios = audios if len(audios) > 0 else None
    images = images if len(images) > 0 else None
    videos = videos if len(videos) > 0 else None

    multi_modal_data = {}
    if audios:
        multi_modal_data["audio"] = audios
    if images:
        multi_modal_data["image"] = images
    if videos:
        multi_modal_data["video"] = videos

    if isinstance(prompt, list) and isinstance(prompt[0], (list, str)):
        prompt = prompt[0]

    if tokenize:
        return OmniTokensPrompt(
            prompt_token_ids=prompt,
            multi_modal_data=multi_modal_data,
        )
    else:
        return TextPrompt(
            prompt=prompt,
            multi_modal_data=multi_modal_data,
        )

def get_system_prompt():
    
    return {
        'role':
        'system',
        'content': [{
            'type':
            'text',
            'text':
            'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
        }]
    }

def make_text_prompt(prompt):
    messages = [
        get_system_prompt(),
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ]
        },
    ]

    prompt = make_inputs_qwen2_omni(messages, )
    return prompt

def make_audio_in_video_v2_prompt():
    messages = [
        {
            'role':
            'system',
            'content': [{
                'type':
                'text',
                'text':
                'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
            }]
        },
        {
            "role":
            "user",
            "content": [
                {
                    "type":
                    "video_url",
                    "video_url":
                    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw_small.mp4"
                },
            ]
        },
    ]
    prompt = make_inputs_qwen2_omni(
        messages,
        use_audio_in_video=True,
    )
    return prompt

def make_omni_prompt(prompt = None) -> Union[OmniTokensPrompt, List[OmniTokensPrompt]]:
    if args.prompt_type == 'text':
        prompt = make_text_prompt(prompt)
    elif args.prompt_type == 'audio-in-video-v2':
        prompt = make_audio_in_video_v2_prompt()
    else:
        raise ValueError(f'Unsupported prompt type: {args.prompt_type}')
    return prompt

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def run_generation(llm, sampling, prompt=None, prompt_id=None, prompt_embeds=None):
    final_outputs = []
    request_id = f"req-omni-deterministic-{SEED}"
    llm_input = prompt if prompt is not None else {
        "prompt_token_ids": prompt_id,
        "prompt_embeds": prompt_embeds,
    }
    for ro in llm.generate(llm_input, sampling):
        final_outputs.append(ro)
    return final_outputs

def thinker2talker(llm:OmniLLM, thinker_outputs, multi_modal_data = None):
    talker_outputs = []
    pad_token_id = 8292
    
    model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.get_model()
    connection_func = model._thinker_to_talker_prefill
    voice_type = "Chelsie"
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.token_ids
        prompt_token_ids_len = len(prompt_token_ids)
        thinker_hidden_states = output.multimodal_output["latent"].clone().detach().cuda()
        talker_outputs.append(
            # {
            #     "input_ids": talker_input_ids,
            #     "input_embeds": talker_input_embeds,
            # }
            OmniTokensPrompt(
                prompt_token_ids=[0] * (len(prompt_token_ids) + 2), # add 2 for codec pad and start token
                additional_information={
                    "thinker_result": thinker_hidden_states[prompt_token_ids_len:].to(torch.float32),
                    "prompt_embeds": thinker_hidden_states[:prompt_token_ids_len].to(torch.float32),
                    "prompt_token_ids": prompt_token_ids,
                    "thinker_output_token_ids": thinker_output_ids,
                },
                multi_modal_data=multi_modal_data[thinker_output.request_id] if multi_modal_data is not None else None,
                mm_processor_kwargs=None,
            )
        )
    return talker_outputs

def extract_text(llm:OmniLLM, thinker_outputs):
    text_outputs = {}
    for output in thinker_outputs:
        text_outputs[output.request_id] = output.outputs[0].text
    return text_outputs

def talker2code(llm:OmniLLM, talker_outputs, multi_modal_data = None):
    code_outputs = []
    for i, output in enumerate(talker_outputs):
        talker_output_ids = output.outputs[0].token_ids
        code_outputs.append(
            {
                "prompt_token_ids": talker_output_ids,
                "multi_modal_data": multi_modal_data[output.request_id] if multi_modal_data is not None else None,
            }
        )
    return code_outputs

def thinker_generate(llm: OmniLLM, prompt: str | None = None):
    print("--------------------------")
    print("thinker_generate")
    print("--------------------------")
    prompt_list = [make_omni_prompt(prompt)]
    
    sampling = SamplingParams(
        temperature=0.0,    # Deterministic - no randomness
        top_p=1.0,          # Disable nucleus sampling
        top_k=-1,           # Disable top-k sampling
        max_tokens=2048,
        seed=SEED,          # Fixed seed for sampling
        detokenize=True,
        repetition_penalty=1.1,
        )
    start_time = time.time()
    final = run_generation(llm, sampling, prompt_list)
    elapsed = time.time() - start_time

    multi_modal_data = {output.request_id: getattr(prompt_obj, 'multi_modal_data', None) for output, prompt_obj in zip(final, prompt_list)}
    
    thinker_outputs = thinker2talker(llm, final, multi_modal_data)
    text_outputs = extract_text(llm, final)
    print_nvidia_sim_info(gpu_index=torch.cuda.current_device(), stage="After_thinker2talker", model_stage="thinker")
    tokens = len(final[0].outputs[0].token_ids)
    tps = tokens / max(elapsed, 1e-9)
    print(f"Thinker: time={elapsed:.3f}s, tokens={tokens}, tps={tps:.2f}")
    return thinker_outputs, text_outputs, multi_modal_data, tokens, elapsed, tps

def talker_generate(llm: OmniLLM, thinker_outputs, multi_modal_data):
    print("--------------------------")
    print("talker_generate")
    print("--------------------------")
    sampling = SamplingParams(
        temperature=0.0,    # Deterministic - no randomness
        top_p=1.0,          # Disable nucleus sampling
        top_k=-1,           # Disable top-k sampling
        max_tokens=2048,
        seed=SEED,          # Fixed seed for sampling
        detokenize=True,
        repetition_penalty=1.1,
        stop_token_ids=[8294]
    )
    start_time = time.time()
    final = run_generation(llm, sampling, thinker_outputs)
    elapsed = time.time() - start_time
    print_nvidia_sim_info(gpu_index=torch.cuda.current_device(), stage="After_talker_generate", model_stage="talker")
    tokens = len(final[0].outputs[0].token_ids)
    tps = tokens / max(elapsed, 1e-9)
    print(f"Talker: time={elapsed:.3f}s, tokens={tokens}, tps={tps:.2f}")
    talker_outputs = talker2code(llm, final, multi_modal_data)
    return talker_outputs, tokens, elapsed, tps

def token2wav_generate(llm: OmniLLM, talker_outputs):
    print("--------------------------")
    print("token2wav_generate")
    print("--------------------------")
    input_tokens = len(talker_outputs[0]["prompt_token_ids"]) if isinstance(talker_outputs, list) and len(talker_outputs) > 0 else 0
    print(f"Code2Wav input num tokens: {input_tokens}")
    sampling = SamplingParams(
        temperature=0.0,    # Deterministic - no randomness
        top_p=1.0,          # Disable nucleus sampling
        top_k=-1,           # Disable top-k sampling
        max_tokens=2048,
        seed=SEED,          # Fixed seed for sampling
        detokenize=True,
        repetition_penalty=1.1,
    )
    start_time = time.time()
    final = run_generation(llm, sampling, talker_outputs)
    elapsed = time.time() - start_time
    print_nvidia_sim_info(gpu_index=torch.cuda.current_device(), stage="After_token2wav_generate", model_stage="code2wav")
    tps = input_tokens / max(elapsed, 1e-9)
    print(f"Code2Wav: time={elapsed:.3f}s, tokens={input_tokens}, tps={tps:.2f}")
    return final, input_tokens, elapsed, tps

def init_thinker_llm() -> OmniLLM:
    engine_args = asdict(OmniEngineArgs(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
        seed=SEED,
        worker_cls="vllm_omni.worker.AR_gpu_worker.ARGPUWorker",
        scheduler_cls="vllm_omni.core.sched.scheduler.OmniScheduler",
        engine_output_type="latent",
        model_stage="thinker",
    ))
    return OmniLLM(**engine_args)

def init_talker_llm() -> OmniLLM:
    engine_args = asdict(OmniEngineArgs(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
        seed=SEED,
        worker_cls="vllm_omni.worker.AR_gpu_worker.ARGPUWorker",
        scheduler_cls="vllm_omni.core.sched.scheduler.OmniScheduler",
        engine_output_type="latent",
        model_stage="talker",
    ))
    return OmniLLM(**engine_args)

def init_diffusion_llm() -> OmniLLM:
    engine_args = asdict(OmniEngineArgs(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        gpu_memory_utilization=0.25,
        enforce_eager=True,
        seed=SEED,
        worker_cls="vllm_omni.worker.diffusion_gpu_worker.DiffusionGPUWorker",
        scheduler_cls="vllm_omni.core.sched.diffusion_scheduler.DiffusionScheduler",
        engine_output_type="audio",
        model_stage="code2wav",
    ))
    return OmniLLM(**engine_args)

def main():
    print(f"Running with deterministic seed: {SEED}")
    print(f"PyTorch deterministic mode: {torch.backends.cudnn.deterministic}")
    print(f"PyTorch benchmark mode: {torch.backends.cudnn.benchmark}")
    
    # 1) Download HF model if needed
    ensure_hf_model(args.model, args.hf_hub_id, args.hf_revision)
    
    # 2) Ensure config for merged model + code2wav
    ensure_config(args.model, args.code2wav_dir, args.dit_ckpt, args.bigvgan_ckpt)
    
    thinker_llm = init_thinker_llm()
    talker_llm = init_talker_llm()
    diffusion_llm = init_diffusion_llm()
    final_outputs = []
    final_text_outputs = []
    thinker_stats = []
    talker_stats = []
    code2wav_stats = []

    total_start_time = time.time()
    total_thinker_time = 0.0
    total_talker_time = 0.0
    total_code2wav_time = 0.0
    total_thinker_tokens = 0
    total_talker_tokens = 0
    total_code2wav_tokens = 0

    csv_file = None
    csv_writer = None
    if args.write_csv:
        os.makedirs(os.path.dirname(args.stats_csv) or ".", exist_ok=True)
        import csv
        csv_file = open(args.stats_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "prompt_id",
            "thinker_time_s","thinker_tokens","thinker_tps",
            "talker_time_s","talker_tokens","talker_tps",
            "code2wav_time_s","code2wav_tokens","code2wav_tps",
            "prompt_text_preview"
        ])

    for prompt_id, prompt in enumerate(args.prompts):
        thinker_outputs, text_outputs, multi_modal_data, th_tokens, th_time, th_tps = thinker_generate(thinker_llm, prompt)
        talker_outputs, tk_tokens, tk_time, tk_tps = talker_generate(talker_llm, thinker_outputs, multi_modal_data)
        token2wav_outputs, cw_tokens, cw_time, cw_tps = token2wav_generate(diffusion_llm, talker_outputs)

        final_outputs += token2wav_outputs
        for ro in token2wav_outputs:
            rid = ro.request_id
            final_text_outputs.append(text_outputs.get(rid, ""))

        thinker_stats.append((th_time, th_tokens, th_tps))
        talker_stats.append((tk_time, tk_tokens, tk_tps))
        code2wav_stats.append((cw_time, cw_tokens, cw_tps))

        total_thinker_time += th_time
        total_talker_time += tk_time
        total_code2wav_time += cw_time
        total_thinker_tokens += th_tokens
        total_talker_tokens += tk_tokens
        total_code2wav_tokens += cw_tokens

        if csv_writer:
            preview = str(prompt).replace("\n", " ")[:120]
            csv_writer.writerow([
                prompt_id,
                f"{th_time:.6f}", th_tokens, f"{th_tps:.6f}",
                f"{tk_time:.6f}", tk_tokens, f"{tk_tps:.6f}",
                f"{cw_time:.6f}", cw_tokens, f"{cw_tps:.6f}",
                preview
            ])

    # Save audio files
    if len(final_outputs) == 1 and not os.path.isdir(args.output_wav) and not args.output_wav.endswith(os.sep):
        os.makedirs(os.path.dirname(args.output_wav) or ".", exist_ok=True)
        audio_tensor = final_outputs[0].multimodal_output["audio"]
        sf.write(args.output_wav, audio_tensor.detach().cpu().numpy(), samplerate=24000)
        print(f"Saved audio to {args.output_wav}")
        print(f"Generated text: {final_text_outputs[0] if final_text_outputs else ''}")
    else:
        os.makedirs(args.output_wav, exist_ok=True)
        for i, output in enumerate(final_outputs):
            output_wav = os.path.join(args.output_wav, f"output_{i}.wav")
            audio_tensor = output.multimodal_output["audio"]
            sf.write(output_wav, audio_tensor.detach().cpu().numpy(), samplerate=24000)
            print(f"Saved audio to {output_wav}")
            print(f"Generated text: {final_text_outputs[i] if i < len(final_text_outputs) else ''}")

    total_end_time = time.time()
    total_wall_time = total_end_time - total_start_time
    grand_total_tokens = total_thinker_tokens + total_talker_tokens + total_code2wav_tokens
    overall_tps = grand_total_tokens / max(total_wall_time, 1e-9)

    print("\n===== Overall Stats =====")
    print(f"Total wall time: {total_wall_time:.3f}s")
    print(f"Total tokens: {grand_total_tokens}")
    print(f"Overall throughput: {overall_tps:.2f} tokens/sec")

    def print_stage(name, total_time, total_tokens, n):
        tps_total = total_tokens / max(total_time, 1e-9)
        avg_time = total_time / max(n, 1)
        avg_tokens = total_tokens / max(n, 1)
        print(f"{name}: total_time={total_time:.3f}s, total_tokens={total_tokens}, "
              f"tps={tps_total:.2f}, avg_time_per_prompt={avg_time:.3f}s, avg_tokens_per_prompt={avg_tokens:.2f}")

    num_prompts = len(args.prompts)
    print_stage("Thinker", total_thinker_time, total_thinker_tokens, num_prompts)
    print_stage("Talker",  total_talker_time,  total_talker_tokens,  num_prompts)
    print_stage("Code2Wav", total_code2wav_time, total_code2wav_tokens, num_prompts)

    if csv_file:
        csv_file.close()
        print(f"Stats CSV saved to: {args.stats_csv}")

if __name__ == '__main__':
    main()