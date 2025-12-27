import os

import pytest
import torch
import socket
import subprocess
import sys
import time
import yaml
import base64
import psutil
import tempfile
import cv2
import io
import numpy as np
import soundfile as sf
from typing import Dict, Any
from pathlib import Path
from vllm.logger import init_logger
from vllm.utils import get_open_port

from vllm.assets.audio import AudioAsset
from vllm.assets.video import VideoAsset
from vllm.assets.image import ImageAsset

logger = init_logger(__name__)


@pytest.fixture(autouse=True)
def clean_gpu_memory_between_tests():
    if os.getenv("VLLM_TEST_CLEAN_GPU_MEMORY", "0") != "1":
        yield
        return

    # Wait for GPU memory to be cleared before starting the test
    import gc

    from tests.utils import wait_for_gpu_memory_to_clear

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        try:
            wait_for_gpu_memory_to_clear(
                devices=list(range(num_gpus)),
                threshold_ratio=0.1,
            )
        except ValueError as e:
            logger.info("Failed to clean GPU memory: %s", e)

    yield

    # Clean up GPU memory after the test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def dummy_messages_from_mix_data(
    system_prompt: Dict[str, Any],
    video_data_url: str = None,
    audio_data_url: str = None,
    image_data_url: str = None,
    content_text:
    str = "What is recited in the audio? What is in this image? Describe the video briefly.",
):
    """Create messages with video、image、audio data URL for OpenAI API."""
    content = [{"type": "text", "text": content_text}]
    media_items = [
        (video_data_url, "video"),
        (image_data_url, "image"),
        (audio_data_url, "audio"),
    ]
    content.extend({
        "type": f"{media_type}_url",
        f"{media_type}_url": {
            "url": url
        }
    } for url, media_type in media_items if url is not None)
    return [
        system_prompt,
        {
            "role": "user",
            "content": content
        },
    ]


def generate_synthetic_audio(
    duration: int,  # seconds
    num_channels: int  # 1：Mono，2：Stereo 5：5.1 surround sound
) -> Dict[str, Any]:
    """Generate synthetic audio with random values.
       Default use 48000Hz.
    """
    sample_rate = 48000
    num_samples = int(sample_rate * duration)
    audio_data = np.random.uniform(-0.5, 0.5, (num_samples, num_channels))
    audio_data = np.clip(audio_data, -1.0, 1.0)
    audio_tensor = torch.FloatTensor(audio_data.T)
    audio_np = audio_tensor.numpy()

    buffer = io.BytesIO()

    sf.write(buffer, audio_np.T, sample_rate, format="mp3")

    buffer.seek(0)
    audio_bytes = buffer.read()
    buffer.close()
    return base64.b64encode(audio_bytes).decode("utf-8")


def generate_synthetic_video(width: int, height: int, num_frames: int) -> Any:
    """Generate synthetic video with random values.
    """
    video_data = np.random.randint(
        0,
        256,
        (num_frames, height, width, 3),
        dtype=np.uint8,
    )
    video_tensor = torch.from_numpy(video_data)
    with tempfile.NamedTemporaryFile(suffix=f".mp4", delete=False) as tmp:
        temp_path = tmp.name
    frames, height, width, channels = video_tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, 30, (width, height))

    for i in range(frames):
        frame = video_tensor[i].numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        out.release()

    with open(temp_path, "rb") as f:
        content = f.read()
    os.unlink(temp_path)

    return base64.b64encode(content).decode("utf-8")


def generate_synthetic_image(width: int, height: int) -> Any:
    """Generate synthetic image with random values.
       """
    from PIL import Image
    image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    pil_image = Image.fromarray(image)
    pil_image = pil_image.convert('RGB')
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85, optimize=True)
    buffer.seek(0)
    image_bytes = buffer.read()

    return base64.b64encode(image_bytes).decode('utf-8')


def modify_stage_config(
    yaml_path: str,
    stage_updates: Dict[int, Dict[str, Any]],
) -> str:
    """
    Batch modify configurations for multiple stages in a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.
        stage_updates: Dictionary where keys are stage IDs and values are dictionaries of
                      modifications for that stage. Each modification dictionary uses
                      dot-separated paths as keys and new configuration values as values.
                      Example: {
                          0: {'engine_args.max_model_len': 5800},
                          1: {'runtime.max_batch_size': 2}
                      }

    Returns:
        str: Path to the newly created modified YAML file with timestamp suffix.

    Example:
        >>> output_file = modify_stage_config(
        ...     'config.yaml',
        ...     {
        ...         0: {'engine_args.max_model_len': 5800},
        ...         1: {'runtime.max_batch_size': 2}
        ...     }
        ... )
        >>> print(f"Modified configuration saved to: {output_file}")
        Modified configuration saved to: config_1698765432.yaml
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"yaml does not exist: {path}")
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Cannot parse YAML file: {e}")

    stage_args = config.get('stage_args', [])
    if not stage_args:
        raise ValueError("the stage_args does not exist")

    for stage_id, config_dict in stage_updates.items():
        target_stage = None
        for stage in stage_args:
            if stage.get('stage_id') == stage_id:
                target_stage = stage
                break

        if target_stage is None:
            available_ids = [
                s.get('stage_id') for s in stage_args if 'stage_id' in s
            ]
            raise KeyError(
                f"Stage ID {stage_id} is not exist, available IDs: {available_ids}"
            )

        for key_path, value in config_dict.items():
            current = target_stage
            keys = key_path.split(".")
            for i in range(len(keys) - 1):
                key = keys[i]
                if key not in current:
                    raise KeyError(
                        f"the {'.'.join(keys[:i+1])} does not exist")

                elif not isinstance(current[key], dict) and i < len(keys) - 2:
                    raise ValueError(
                        f"{'.'.join(keys[:i+1])}' cannot continue deeper because it's not a dict"
                    )
                current = current[key]
            current[keys[-1]] = value

    output_path = f"{yaml_path.split('.')[0]}_{int(time.time())}.yaml"
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config,
                  f,
                  default_flow_style=False,
                  sort_keys=False,
                  allow_unicode=True,
                  indent=2)

    return output_path


class OmniServer:
    """Omniserver for vLLM-Omni tests."""

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        env_dict: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.serve_args = serve_args
        self.env_dict = env_dict
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        self.port = get_open_port()

    def _start_server(self) -> None:
        """Start the vLLM-Omni server subprocess."""
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ] + self.serve_args

        print(f"Launching OmniServer with: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(
                __file__))),  # Set working directory to vllm-omni root
        )

        # Wait for server to be ready
        max_wait = 600  # 10 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    result = sock.connect_ex((self.host, self.port))
                    if result == 0:
                        print(f"Server ready on {self.host}:{self.port}")
                        return
            except Exception:
                pass
            time.sleep(2)

        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    def _kill_process_tree(self, pid):
        """kill process and its children"""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            gone, still_alive = psutil.wait_procs(children, timeout=10)

            for child in still_alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            try:
                parent.terminate()
                parent.wait(timeout=10)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

        except psutil.NoSuchProcess:
            pass

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            try:
                parent = psutil.Process(self.proc.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass

                gone, still_alive = psutil.wait_procs(children, timeout=10)

                for child in still_alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass

                try:
                    parent.terminate()
                    parent.wait(timeout=10)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    try:
                        parent.kill()
                    except psutil.NoSuchProcess:
                        pass

            except psutil.NoSuchProcess:
                pass
