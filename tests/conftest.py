import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Set CPU device for CI environments without GPU
if "VLLM_TARGET_DEVICE" not in os.environ:
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"

import base64
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pytest
import torch
import whisper
import yaml
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port

logger = init_logger(__name__)


@pytest.fixture(autouse=True)
def default_vllm_config():
    """Set a default VllmConfig for all tests.

    This fixture is auto-used for all tests to ensure that any test
    that directly instantiates vLLM CustomOps (e.g., RMSNorm, LayerNorm)
    or model components has the required VllmConfig context.

    This fixture is required for vLLM 0.14.0+ where CustomOp initialization
    requires a VllmConfig context set via set_current_vllm_config().
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    # Use CPU device if no GPU is available (e.g., in CI environments)
    has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = "cuda" if has_gpu else "cpu"
    device_config = DeviceConfig(device=device)

    with set_current_vllm_config(VllmConfig(device_config=device_config)):
        yield


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
    system_prompt: dict[str, Any] = None,
    video_data_url: Any = None,
    audio_data_url: Any = None,
    image_data_url: Any = None,
    content_text: str = None,
):
    """Create messages with video、image、audio data URL for OpenAI API."""

    if content_text is not None:
        content = [{"type": "text", "text": content_text}]
    else:
        content = []

    media_items = []
    if isinstance(video_data_url, list):
        for video_url in video_data_url:
            media_items.append((video_url, "video"))
    else:
        media_items.append((video_data_url, "video"))

    if isinstance(image_data_url, list):
        for url in image_data_url:
            media_items.append((url, "image"))
    else:
        media_items.append((image_data_url, "image"))

    if isinstance(audio_data_url, list):
        for url in audio_data_url:
            media_items.append((url, "audio"))
    else:
        media_items.append((audio_data_url, "audio"))

    content.extend(
        {"type": f"{media_type}_url", f"{media_type}_url": {"url": url}}
        for url, media_type in media_items
        if url is not None
    )
    messages = [{"role": "user", "content": content}]
    if system_prompt is not None:
        messages = [system_prompt] + messages
    return messages


def cosine_similarity_text(s1, s2):
    """
        Calculate cosine similarity between two text strings.
        Notes:
    ------
    - Higher score means more similar texts
    - Score of 1.0 means identical word composition (bag-of-words)
    - Score of 0.0 means completely different vocabulary
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = CountVectorizer().fit_transform([s1, s2])
    vectors = vectorizer.toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]


def convert_audio_to_text(audio_data):
    """
    Convert base64 encoded audio data to text using speech recognition.
    """

    audio_data = base64.b64decode(audio_data)
    output_path = f"./test_{int(time.time())}"
    with open(output_path, "wb") as audio_file:
        audio_file.write(audio_data)

    print(f"audio data is saved: {output_path}")
    model = whisper.load_model("base")
    text = model.transcribe(output_path)["text"]
    if text:
        return text
    else:
        return ""


def modify_stage_config(
    yaml_path: str,
    stage_updates: dict[int, dict[str, Any]],
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
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Cannot parse YAML file: {e}")

    stage_args = config.get("stage_args", [])
    if not stage_args:
        raise ValueError("the stage_args does not exist")

    for stage_id, config_dict in stage_updates.items():
        target_stage = None
        for stage in stage_args:
            if stage.get("stage_id") == stage_id:
                target_stage = stage
                break

        if target_stage is None:
            available_ids = [s.get("stage_id") for s in stage_args if "stage_id" in s]
            raise KeyError(f"Stage ID {stage_id} is not exist, available IDs: {available_ids}")

        for key_path, value in config_dict.items():
            current = target_stage
            keys = key_path.split(".")
            for i in range(len(keys) - 1):
                key = keys[i]
                if key not in current:
                    raise KeyError(f"the {'.'.join(keys[: i + 1])} does not exist")

                elif not isinstance(current[key], dict) and i < len(keys) - 2:
                    raise ValueError(f"{'.'.join(keys[: i + 1])}' cannot continue deeper because it's not a dict")
                current = current[key]
            current[keys[-1]] = value

    output_path = f"{yaml_path.split('.')[0]}_{int(time.time())}.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True, indent=2)

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
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Set working directory to vllm-omni root
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
