import dataclasses
import io
import json
import logging
import math
import os
import random
import string
import tempfile
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torchaudio

logger = logging.getLogger(__name__)

"""
system_prompt
"""

AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL = """Generate audio with the following timbre, prosody and speaking style

[speaker_start]
speaker name: {speaker}
speaker prompt text:
{prompt_text}
speaker audio tokens:
{prompt_wav_tokens}
[speaker_end]
"""

AUDIO_EDIT_SYSTEM_PROMPT = (
    "As a highly skilled audio editing and tuning specialist, you excel in interpreting user instructions and applying "
    "precise adjustments to meet their needs. Your expertise spans a wide range of enhancement capabilities, including "
    "but not limited to:\n"
    "# Emotional Enhancement\n"
    "# Speaking Style Transfer\n"
    "# Non-linguistic Adjustments\n"
    "# Audio Tuning & Editing\n"
    "Note: You will receive instructions in natural language and are expected to accurately interpret and execute the "
    "most suitable audio edits and enhancements.\n"
)


def resample_audio(wav, original_sample_rate, target_sample_rate):
    if original_sample_rate != target_sample_rate:
        assert original_sample_rate > target_sample_rate, (
            f"wav sample rate {original_sample_rate} must be greater than {target_sample_rate}"
        )
        wav = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)(wav)
    return wav


def energy_norm_fn(wav):
    if type(wav) is np.ndarray:
        max_data = np.max(np.abs(wav))
        wav = wav / max(max_data, 0.01) * 0.999
    else:
        max_data = torch.max(torch.abs(wav))
        wav = wav / max(max_data, 0.01) * 0.999
    return wav


def _trim_silence_indices(audio: np.ndarray, top_db: float, frame_length: int, hop_length: int) -> tuple[int, int]:
    """Return librosa-like non-silent sample bounds for mono audio."""
    if audio.ndim != 1:
        audio = np.asarray(audio).reshape(-1)
    if audio.size == 0:
        return 0, 0

    audio = np.asarray(audio, dtype=np.float32)
    pad = frame_length // 2
    padded = np.pad(audio, (pad, pad), mode="constant")
    if padded.size < frame_length:
        padded = np.pad(padded, (0, frame_length - padded.size), mode="constant")

    frame_count = 1 + max(0, (padded.size - frame_length) // hop_length)
    shape = (frame_count, frame_length)
    strides = (padded.strides[0] * hop_length, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(frames * frames, axis=1))

    ref = float(np.max(rms))
    if ref <= 0.0:
        return 0, audio.size

    db = 20.0 * np.log10(np.maximum(rms, 1e-10) / ref)
    non_silent = np.flatnonzero(db > -top_db)
    if non_silent.size == 0:
        return 0, audio.size

    start = max(0, int(non_silent[0] * hop_length))
    end = min(audio.size, int((non_silent[-1] + 1) * hop_length))
    return start, end


def trim_silence(audio, sr, keep_left_time=0.05, keep_right_time=0.22, hop_size=240):
    index = _trim_silence_indices(audio, top_db=20, frame_length=512, hop_length=128)
    num_frames = int(math.ceil((index[1] - index[0]) / hop_size))  # 300

    left_sil_samples = int(keep_left_time * sr)

    wav_len = len(audio)
    start_idx = index[0] - left_sil_samples
    trim_wav = audio

    if start_idx > 0:
        trim_wav = trim_wav[start_idx:]
    else:
        trim_wav = np.pad(trim_wav, (abs(start_idx), 0), mode="constant", constant_values=0.0)
    wav_len = len(trim_wav)
    out_len = int(num_frames * hop_size + (keep_left_time + keep_right_time) * sr)

    if out_len < wav_len:
        trim_wav = trim_wav[:out_len]
    else:
        trim_wav = np.pad(trim_wav, (0, (out_len - wav_len)), mode="constant", constant_values=0.0)
    return trim_wav


def load_bytes(input):
    middle_data = np.frombuffer(input, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    if middle_data.dtype.kind not in "iu":
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype("float32")
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(middle_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
    return array


def load_audio_text_image_video(
    data_or_path_or_list, fs: int = 16000, audio_fs: int = 16000, data_type="sound", tokenizer=None, **kwargs
):
    if isinstance(data_or_path_or_list, (list, tuple)):
        if data_type is not None and isinstance(data_type, (list, tuple)):
            data_types = [data_type] * len(data_or_path_or_list)
            data_or_path_or_list_ret = [[] for d in data_type]
            for i, (data_type_i, data_or_path_or_list_i) in enumerate(zip(data_types, data_or_path_or_list)):
                for j, (data_type_j, data_or_path_or_list_j) in enumerate(zip(data_type_i, data_or_path_or_list_i)):
                    data_or_path_or_list_j = load_audio_text_image_video(
                        data_or_path_or_list_j,
                        fs=fs,
                        audio_fs=audio_fs,
                        data_type=data_type_j,
                        tokenizer=tokenizer,
                        **kwargs,
                    )
                    data_or_path_or_list_ret[j].append(data_or_path_or_list_j)

            return data_or_path_or_list_ret
        else:
            return [
                load_audio_text_image_video(audio, fs=fs, audio_fs=audio_fs, data_type=data_type, **kwargs)
                for audio in data_or_path_or_list
            ]

    if isinstance(data_or_path_or_list, str) and data_or_path_or_list.startswith("http"):  # download url to local file
        data_or_path_or_list = download_from_url(data_or_path_or_list)

    if isinstance(data_or_path_or_list, io.BytesIO):
        data_or_path_or_list, audio_fs = torchaudio.load(data_or_path_or_list)
        if kwargs.get("reduce_channels", True):
            data_or_path_or_list = data_or_path_or_list.mean(0)
    elif isinstance(data_or_path_or_list, str) and os.path.exists(data_or_path_or_list):  # local file
        if data_type is None or data_type == "sound":
            data_or_path_or_list, audio_fs = torchaudio.load(data_or_path_or_list)
            if kwargs.get("reduce_channels", True):
                data_or_path_or_list = data_or_path_or_list.mean(0)
        elif data_type == "text" and tokenizer is not None:
            data_or_path_or_list = tokenizer.encode(data_or_path_or_list)
        elif data_type == "image":  # undo
            pass
        elif data_type == "video":  # undo
            pass

        # if data_in is a file or url, set is_final=True
        if "cache" in kwargs:
            kwargs["cache"]["is_final"] = True
            kwargs["cache"]["is_streaming_input"] = False
    elif isinstance(data_or_path_or_list, str) and data_type == "text" and tokenizer is not None:
        data_or_path_or_list = tokenizer.encode(data_or_path_or_list)
    elif isinstance(data_or_path_or_list, np.ndarray):  # audio sample point
        data_or_path_or_list = torch.from_numpy(data_or_path_or_list).squeeze()  # [n_samples,]
    else:
        pass
        # print(f"unsupported data type: {data_or_path_or_list}, return raw data")

    if audio_fs != fs and data_type != "text":
        resampler = torchaudio.transforms.Resample(audio_fs, fs)
        data_or_path_or_list = resampler(data_or_path_or_list[None, :])[0, :]
    return data_or_path_or_list


def prepare_data_iterator(data_in, data_type=None, key=None):
    """

    :param input:
    :param input_len:
    :param data_type:
    :param frontend:
    :return:
    """
    data_list = []
    key_list = []
    filelist = [".scp", ".txt", ".json", ".jsonl"]

    chars = string.ascii_letters + string.digits
    if isinstance(data_in, str) and data_in.startswith("http"):  # url
        data_in = download_from_url(data_in)
    if isinstance(data_in, str) and os.path.exists(data_in):  # wav_path; filelist: wav.scp, file.jsonl;text.txt;
        _, file_extension = os.path.splitext(data_in)
        file_extension = file_extension.lower()
        if file_extension in filelist:  # filelist: wav.scp, file.jsonl;text.txt;
            with open(data_in, encoding="utf-8") as fin:
                for line in fin:
                    key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
                    if data_in.endswith(".jsonl"):  # file.jsonl: json.dumps({"source": data})
                        lines = json.loads(line.strip())
                        data = lines["source"]
                        key = data["key"] if "key" in data else key
                    else:  # filelist, wav.scp, text.txt: id \t data or data
                        lines = line.strip().split(maxsplit=1)
                        data = lines[1] if len(lines) > 1 else lines[0]
                        key = lines[0] if len(lines) > 1 else key

                    data_list.append(data)
                    key_list.append(key)
        else:
            key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
            data_list = [data_in]
            key_list = [key]
    elif isinstance(data_in, (list, tuple)):
        if data_type is not None and isinstance(data_type, (list, tuple)):  # multiple inputs
            data_list_tmp = []
            for data_in_i, data_type_i in zip(data_in, data_type):
                key_list, data_list_i = prepare_data_iterator(data_in=data_in_i, data_type=data_type_i)
                data_list_tmp.append(data_list_i)
            data_list = []
            for item in zip(*data_list_tmp):
                data_list.append(item)
        else:
            # [audio sample point, fbank, text]
            data_list = data_in
            key_list = ["rand_key_" + "".join(random.choice(chars) for _ in range(13)) for _ in range(len(data_in))]
    else:  # raw text; audio sample point, fbank; bytes
        if isinstance(data_in, bytes):  # audio bytes
            data_in = load_bytes(data_in)
        if key is None:
            key = "rand_key_" + "".join(random.choice(chars) for _ in range(13))
        data_list = [data_in]
        key_list = [key]

    return key_list, data_list


class HTTPStorage:
    def read(self, url):
        r = requests.get(url)
        r.raise_for_status()
        return r.content


def download_from_url(url):
    result = urlparse(url)
    file_path = None
    if result.scheme is not None and len(result.scheme) > 0:
        storage = HTTPStorage()
        # bytes
        data = storage.read(url)
        work_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        file_path = os.path.join(work_dir, os.path.basename(url))
        with open(file_path, "wb") as fb:
            fb.write(data)
    assert file_path is not None, f"failed to download: {url}"
    return file_path


def to_device(data, device=None, dtype=None, non_blocking=False, copy=False):
    """Change the device of object recursively"""
    if isinstance(data, dict):
        return {k: to_device(v, device, dtype, non_blocking, copy) for k, v in data.items()}
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        return type(data)(*[to_device(v, device, dtype, non_blocking, copy) for v in dataclasses.astuple(data)])
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[to_device(o, device, dtype, non_blocking, copy) for o in data])
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, dtype, non_blocking, copy) for v in data)
    elif isinstance(data, np.ndarray):
        return to_device(torch.from_numpy(data), device, dtype, non_blocking, copy)
    elif isinstance(data, torch.Tensor):
        return data.to(device, dtype, non_blocking, copy)
    else:
        return data
