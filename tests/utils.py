# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import os
import tempfile
import time
from contextlib import contextmanager
from typing import Optional

import requests
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode
from vllm.platforms import current_platform

if current_platform.is_rocm():
    from amdsmi import (
        amdsmi_get_gpu_vram_usage,
        amdsmi_get_processor_handles,
        amdsmi_init,
        amdsmi_shut_down,
    )

    @contextmanager
    def _nvml():
        try:
            amdsmi_init()
            yield
        finally:
            amdsmi_shut_down()
elif current_platform.is_cuda():
    from vllm.third_party.pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
        nvmlShutdown,
    )

    @contextmanager
    def _nvml():
        try:
            nvmlInit()
            yield
        finally:
            nvmlShutdown()
else:

    @contextmanager
    def _nvml():
        yield


def get_physical_device_indices(devices):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        return devices

    visible_indices = [int(x) for x in visible_devices.split(",")]
    index_mapping = {i: physical for i, physical in enumerate(visible_indices)}
    return [index_mapping[i] for i in devices if i in index_mapping]


@_nvml()
def wait_for_gpu_memory_to_clear(
    *,
    devices: list[int],
    threshold_bytes: int | None = None,
    threshold_ratio: float | None = None,
    timeout_s: float = 120,
) -> None:
    assert threshold_bytes is not None or threshold_ratio is not None
    # Use nvml instead of pytorch to reduce measurement error from torch cuda
    # context.
    devices = get_physical_device_indices(devices)
    start_time = time.time()
    while True:
        output: dict[int, str] = {}
        output_raw: dict[int, tuple[float, float]] = {}
        for device in devices:
            if current_platform.is_rocm():
                dev_handle = amdsmi_get_processor_handles()[device]
                mem_info = amdsmi_get_gpu_vram_usage(dev_handle)
                gb_used = mem_info["vram_used"] / 2**10
                gb_total = mem_info["vram_total"] / 2**10
            else:
                dev_handle = nvmlDeviceGetHandleByIndex(device)
                mem_info = nvmlDeviceGetMemoryInfo(dev_handle)
                gb_used = mem_info.used / 2**30
                gb_total = mem_info.total / 2**30
            output_raw[device] = (gb_used, gb_total)
            output[device] = f"{gb_used:.02f}/{gb_total:.02f}"

        print("gpu memory used/total (GiB): ", end="")
        for k, v in output.items():
            print(f"{k}={v}; ", end="")
        print("")

        if threshold_bytes is not None:
            is_free = lambda used, total: used <= threshold_bytes / 2**30  # noqa E731
            threshold = f"{threshold_bytes / 2**30} GiB"
        else:
            is_free = lambda used, total: used / total <= threshold_ratio  # noqa E731
            threshold = f"{threshold_ratio:.2f}"

        dur_s = time.time() - start_time
        if all(is_free(used, total) for used, total in output_raw.values()):
            print(f"Done waiting for free GPU memory on devices {devices=} ({threshold=}) {dur_s=:.02f}")
            break

        if dur_s >= timeout_s:
            raise ValueError(f"Memory of devices {devices=} not free after {dur_s=:.02f} ({threshold=})")

        time.sleep(5)


def _check_model_available(model_name: str) -> bool:
    """Check if model weights are available locally or allowed to download."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        try:
            snapshot_download(model_name, local_files_only=True)
            return True
        except LocalEntryNotFoundError:
            return os.environ.get("VLLM_OMNI_DOWNLOAD_MODEL", "0") == "1"
    except ImportError:
        return False


def _create_local_128_image_path() -> str:
    """Create a temporary 128x128 PNG from built-in asset and return its path."""
    image = convert_image_mode(ImageAsset("cherry_blossom").pil_image.resize((128, 128)), "RGB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
        image.save(f, format="PNG")
        return f.name


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode remote content to base64."""
    with requests.get(content_url) as response:
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_video_url_from_path(video_path: Optional[str]) -> str:
    """Convert a video path (local or remote) to a URL or data URL."""
    if not video_path:
        return "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

    if video_path.startswith(("http://", "https://")):
        return video_path

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_path_lower = video_path.lower()
    if video_path_lower.endswith(".mp4"):
        mime_type = "video/mp4"
    elif video_path_lower.endswith(".webm"):
        mime_type = "video/webm"
    elif video_path_lower.endswith(".mov"):
        mime_type = "video/quicktime"
    elif video_path_lower.endswith(".avi"):
        mime_type = "video/x-msvideo"
    elif video_path_lower.endswith(".mkv"):
        mime_type = "video/x-matroska"
    else:
        mime_type = "video/mp4"

    video_base64 = encode_base64_content_from_file(video_path)
    return f"data:{mime_type};base64,{video_base64}"


def get_image_url_from_path(image_path: Optional[str]) -> str:
    """Convert an image path (local or remote) to a URL or data URL."""
    if not image_path:
        return "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"

    if image_path.startswith(("http://", "https://")):
        return image_path

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_path_lower = image_path.lower()
    if image_path_lower.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif image_path_lower.endswith(".png"):
        mime_type = "image/png"
    elif image_path_lower.endswith(".gif"):
        mime_type = "image/gif"
    elif image_path_lower.endswith(".webp"):
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"

    image_base64 = encode_base64_content_from_file(image_path)
    return f"data:{mime_type};base64,{image_base64}"


def get_audio_url_from_path(audio_path: Optional[str]) -> str:
    """Convert an audio path (local or remote) to a URL or data URL."""
    if not audio_path:
        return AudioAsset("mary_had_lamb").url

    if audio_path.startswith(("http://", "https://")):
        return audio_path

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".m4a"):
        mime_type = "audio/mp4"
    else:
        mime_type = "audio/wav"

    audio_base64 = encode_base64_content_from_file(audio_path)
    return f"data:{mime_type};base64,{audio_base64}"
