import tempfile
from typing import Dict, List, Optional, Union
from urllib.request import urlopen

import librosa
import requests
import resampy
import soundfile as sf
import torch
import torchvision.io
from processing_omni import fetch_image, fetch_video

from vllm.inputs import TextPrompt
from vllm_omni.inputs.data import OmniTokensPrompt

# Simple caches to avoid repeated heavy HF loads per prompt
_PROCESSOR_CACHE: Dict[str, "AutoProcessor"] = {}
_CONFIG_CACHE: Dict[str, "AutoConfig"] = {}


def get_system_prompt():

    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def resample_wav_to_16khz(input_filepath):
    data, original_sample_rate = sf.read(input_filepath)
    # Only use the first channel
    if len(data.shape) > 1:
        data = data[:, 0]
    # resample to 16kHz
    data_resampled = resampy.resample(data, sr_orig=original_sample_rate, sr_new=16000)
    return data_resampled


def fetch_and_read_video(args, video_url: str, fps=2):

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

        frame_timestamps = total_duration * torch.arange(1, nframes + 1) / nframes
        grid_timestamps = frame_timestamps[::2]
        second_per_grid = grid_timestamps[1] - grid_timestamps[0]

        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        video = video[idx]

        if args.legacy_omni_video:
            return [video, total_duration, nframes, second_per_grid.item()]
        else:
            return video

    def read_video_with_transformers(video_file_name: Union[str, List[str]]):
        video, total_duration, nframes, second_per_grid = fetch_video(
            {"video": video_file_name}
        )
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
            assert resp.status_code == requests.codes.ok, (
                f"Failed to fetch video from {video_url}, "
                f"status_code:{resp.status_code}, resp:{resp}"
            )

            temp_video_file.write(urlopen(video_url).read())
            temp_video_file_path = temp_video_file.name
            video_file_name = temp_video_file_path
            return read_video(video_file_name)
    else:
        video_file_name = video_url
        return read_video(video_file_name)


def make_inputs_qwen2_omni(
    args,
    messages: List[Dict[str, Union[str, List[Dict[str, str]]]]],
    use_audio_in_video: Optional[bool] = False,
    tokenize: bool = False,
) -> Union[OmniTokensPrompt, TextPrompt]:

    from transformers import AutoConfig, AutoProcessor

    # Cached processor/config to prevent per-prompt reloading and repeated warnings
    if args.model not in _PROCESSOR_CACHE:
        _PROCESSOR_CACHE[args.model] = AutoProcessor.from_pretrained(args.model)
    processor = _PROCESSOR_CACHE[args.model]

    config = _CONFIG_CACHE.get(args.model)
    if config is None:
        try:
            config = AutoConfig.from_pretrained(args.model)
        except Exception:
            config = None
        _CONFIG_CACHE[args.model] = config  # cache even if None to avoid retry storms

    # Decide legacy flag only once based on config (default True if unknown)
    if getattr(args, "legacy_omni_video", None) is None:
        if config is not None and hasattr(config, "architectures"):
            args.legacy_omni_video = not ("Qwen2_5OmniModel" in config.architectures)
        else:
            args.legacy_omni_video = True

    audios, images, videos = [], [], []
    for message in messages:
        if not isinstance(message["content"], list):
            message["content"] = [
                {
                    "type": "text",
                    "text": message["content"],
                }
            ]
        index, num_contents = 0, len(message["content"])
        while index < num_contents:
            ele = message["content"][index]
            if "type" not in ele:
                if "text" in ele:
                    ele["type"] = "text"
                elif "audio" in ele:
                    ele["type"] = "audio"
                elif "audio_url" in ele:
                    ele["type"] = "audio_url"
                elif "image" in ele:
                    ele["type"] = "image"
                elif "image_url" in ele:
                    ele["type"] = "image_url"
                elif "video" in ele:
                    ele["type"] = "video"
                elif "video_url" in ele:
                    ele["type"] = "video_url"
                else:
                    raise ValueError(f"Unknown ele: {ele}")

            if ele["type"] == "audio" or ele["type"] == "audio_url":
                if "audio_url" in ele:
                    audio_key = "audio_url"
                    with tempfile.NamedTemporaryFile(delete=True) as temp_audio_file:
                        temp_audio_file.write(urlopen(ele[audio_key]).read())
                        temp_audio_file_path = temp_audio_file.name
                        audios.append(resample_wav_to_16khz(temp_audio_file_path))
                        ele["audio"] = temp_audio_file_path
                elif "audio" in ele:
                    audio_key = "audio"
                    audios.append(resample_wav_to_16khz(ele[audio_key]))
                else:
                    raise ValueError(f"Unknown ele {ele}")
            elif use_audio_in_video and (
                ele["type"] == "video" or ele["type"] == "video_url"
            ):
                # use video as audio as well
                if "video_url" in ele:
                    audio_key = "video_url"
                    with tempfile.NamedTemporaryFile(delete=True) as temp_video_file:
                        temp_video_file.write(urlopen(ele[audio_key]).read())
                        temp_video_file_path = temp_video_file.name
                        ele[audio_key] = temp_video_file_path
                        audios.append(librosa.load(temp_video_file_path, sr=16000)[0])
                        videos.append(fetch_and_read_video(args, temp_video_file_path))
                        ele["video"] = temp_video_file_path
                elif "video" in ele:
                    audio_key = "video"
                    audios.append(librosa.load(ele[audio_key], sr=16000)[0])
                    videos.append(fetch_and_read_video(args, audio_key))
                else:
                    raise ValueError("Unknown ele {}".format(ele))
                # insert a audio after the video
                message["content"].insert(
                    index + 1,
                    {
                        "type": "audio",
                        "audio": ele[audio_key],
                    },
                )
                # no need to load the added audio again
                index += 1
            elif ele["type"] == "video" or ele["type"] == "video_url":
                if "video_url" in ele:
                    video_key = "video_url"
                    with tempfile.NamedTemporaryFile(delete=True) as temp_video_file:
                        temp_video_file.write(urlopen(ele["video_url"]).read())
                        temp_video_file_path = temp_video_file.name
                        videos.append(fetch_and_read_video(args, temp_video_file))
                        ele["video"] = temp_video_file_path
                else:
                    video_key = "video"
                    videos.append(fetch_and_read_video(args, ele[video_key]))
            elif ele["type"] == "image" or ele["type"] == "image_url":
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


def make_text_prompt(args, prompt):
    messages = [
        get_system_prompt(),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    ]

    prompt = make_inputs_qwen2_omni(args, messages, tokenize=args.tokenize)
    return prompt


def make_audio_in_video_v2_prompt(args):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are Qwen, a virtual human developed by the Qwen Team, "
                        "Alibaba Group, capable of perceiving auditory and visual "
                        "inputs, as well as generating text and speech."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": (
                        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/"
                        "Qwen2.5-Omni/draw_small.mp4"
                    ),
                },
            ],
        },
    ]
    prompt = make_inputs_qwen2_omni(
        args,
        messages,
        use_audio_in_video=True,
        tokenize=args.tokenize,
    )
    return prompt


def make_omni_prompt(
    args, prompt=None
) -> Union[OmniTokensPrompt, List[OmniTokensPrompt]]:
    if args.prompt_type == "text":
        prompt = make_text_prompt(args, prompt)
    elif args.prompt_type == "audio-in-video-v2":
        prompt = make_audio_in_video_v2_prompt(args)
    else:
        raise ValueError(f"Unsupported prompt type: {args.prompt_type}")
    return prompt
