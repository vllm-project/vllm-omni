# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pytest configuration and fixtures for vllm-omni tests.
"""

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any

import pytest
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.omni import Omni

PromptAudioInput = list[tuple[Any, int]] | tuple[Any, int] | None
PromptImageInput = list[Any] | Any | None
PromptVideoInput = list[Any] | Any | None


class OmniRunner:
    """
    Test runner for Omni models.
    """

    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        init_sleep_seconds: int = 20,
        batch_timeout: int = 10,
        init_timeout: int = 300,
        shm_threshold_bytes: int = 65536,
        log_stats: bool = False,
        stage_configs_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize an OmniRunner for testing.

        Args:
            model_name: The model name or path
            seed: Random seed for reproducibility
            init_sleep_seconds: Sleep time after starting each stage
            batch_timeout: Timeout for batching in seconds
            init_timeout: Timeout for initializing stages in seconds
            shm_threshold_bytes: Threshold for using shared memory
            log_stats: Enable detailed statistics logging
            stage_configs_path: Optional path to YAML stage config file
            **kwargs: Additional arguments passed to Omni
        """
        self.model_name = model_name
        self.seed = seed

        self.omni = Omni(
            model=model_name,
            log_stats=log_stats,
            init_sleep_seconds=init_sleep_seconds,
            batch_timeout=batch_timeout,
            init_timeout=init_timeout,
            shm_threshold_bytes=shm_threshold_bytes,
            stage_configs_path=stage_configs_path,
            **kwargs,
        )

    def get_default_sampling_params_list(self) -> list[SamplingParams]:
        """
        Get a list of default sampling parameters for all stages.

        Returns:
            List of SamplingParams with default decoding for each stage
        """
        return [st.default_sampling_params for st in self.omni.instance.stage_list]

    def get_omni_inputs(
        self,
        prompts: list[str] | str,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Construct Omni input format from prompts and multimodal data.

        Args:
            prompts: Text prompt(s) - either a single string or list of strings
            system_prompt: Optional system prompt (defaults to Qwen system prompt)
            audios: Audio input(s) - tuple of (audio_array, sample_rate) or list of tuples
            images: Image input(s) - PIL Image or list of PIL Images
            videos: Video input(s) - numpy array or list of numpy arrays
            mm_processor_kwargs: Optional processor kwargs (e.g., use_audio_in_video)

        Returns:
            List of prompt dictionaries suitable for Omni.generate()
        """
        if system_prompt is None:
            system_prompt = (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                "Group, capable of perceiving auditory and visual inputs, as well as "
                "generating text and speech."
            )

        video_padding_token = "<|VIDEO|>"
        image_padding_token = "<|IMAGE|>"
        audio_padding_token = "<|AUDIO|>"

        if self.model_name == "Qwen/Qwen3-Omni-30B-A3B-Instruct":
            video_padding_token = "<|video_pad|>"
            image_padding_token = "<|image_pad|>"
            audio_padding_token = "<|audio_pad|>"

        if isinstance(prompts, str):
            prompts = [prompts]

        def _normalize_mm_input(mm_input, num_prompts):
            if mm_input is None:
                return [None] * num_prompts
            if isinstance(mm_input, list):
                if len(mm_input) != num_prompts:
                    raise ValueError(
                        f"Multimodal input list length ({len(mm_input)}) must match prompts length ({num_prompts})"
                    )
                return mm_input
            return [mm_input] * num_prompts

        num_prompts = len(prompts)
        audios_list = _normalize_mm_input(audios, num_prompts)
        images_list = _normalize_mm_input(images, num_prompts)
        videos_list = _normalize_mm_input(videos, num_prompts)

        omni_inputs = []
        for i, prompt_text in enumerate(prompts):
            user_content = ""
            multi_modal_data = {}

            audio = audios_list[i]
            if audio is not None:
                if isinstance(audio, list):
                    for _ in audio:
                        user_content += f"<|audio_bos|>{audio_padding_token}<|audio_eos|>"
                    multi_modal_data["audio"] = audio
                else:
                    user_content += f"<|audio_bos|>{audio_padding_token}<|audio_eos|>"
                    multi_modal_data["audio"] = audio

            image = images_list[i]
            if image is not None:
                if isinstance(image, list):
                    for _ in image:
                        user_content += f"<|vision_bos|>{image_padding_token}<|vision_eos|>"
                    multi_modal_data["image"] = image
                else:
                    user_content += f"<|vision_bos|>{image_padding_token}<|vision_eos|>"
                    multi_modal_data["image"] = image

            video = videos_list[i]
            if video is not None:
                if isinstance(video, list):
                    for _ in video:
                        user_content += f"<|vision_bos|>{video_padding_token}<|vision_eos|>"
                    multi_modal_data["video"] = video
                else:
                    user_content += f"<|vision_bos|>{video_padding_token}<|vision_eos|>"
                    multi_modal_data["video"] = video

            user_content += prompt_text

            full_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            input_dict: dict[str, Any] = {"prompt": full_prompt}
            if multi_modal_data:
                input_dict["multi_modal_data"] = multi_modal_data
            if mm_processor_kwargs:
                input_dict["mm_processor_kwargs"] = mm_processor_kwargs

            omni_inputs.append(input_dict)

        return omni_inputs

    def generate(
        self,
        prompts: list[dict[str, Any]],
        sampling_params_list: list[SamplingParams] | None = None,
    ) -> list[Any]:
        """
        Generate outputs for the given prompts.

        Args:
            prompts: List of prompt dictionaries with 'prompt' and optionally
                    'multi_modal_data' keys
            sampling_params_list: List of sampling parameters for each stage.
                                 If None, uses default parameters.

        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        if sampling_params_list is None:
            sampling_params_list = self.get_default_sampling_params_list()

        return self.omni.generate(prompts, sampling_params_list)

    def generate_multimodal(
        self,
        prompts: list[str] | str,
        sampling_params_list: list[SamplingParams] | None = None,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Convenience method to generate with multimodal inputs.

        Args:
            prompts: Text prompt(s)
            sampling_params_list: List of sampling parameters for each stage
            system_prompt: Optional system prompt
            audios: Audio input(s)
            images: Image input(s)
            videos: Video input(s)
            mm_processor_kwargs: Optional processor kwargs

        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            audios=audios,
            images=images,
            videos=videos,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return self.generate(omni_inputs, sampling_params_list)

    def generate_audio(
        self,
        prompts: list[str] | str,
        sampling_params_list: list[SamplingParams] | None = None,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Convenience method to generate with multimodal inputs.
        Args:
            prompts: Text prompt(s)
            sampling_params_list: List of sampling parameters for each stage
            system_prompt: Optional system prompt
            audios: Audio input(s)
            mm_processor_kwargs: Optional processor kwargs
        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return self.generate(omni_inputs, sampling_params_list)

    def generate_video(
        self,
        prompts: list[str] | str,
        sampling_params_list: list[SamplingParams] | None = None,
        system_prompt: str | None = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Convenience method to generate with multimodal inputs.
        Args:
            prompts: Text prompt(s)
            sampling_params_list: List of sampling parameters for each stage
            system_prompt: Optional system prompt
            videos: Video input(s)
            mm_processor_kwargs: Optional processor kwargs
        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            videos=videos,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return self.generate(omni_inputs, sampling_params_list)

    def generate_image(
        self,
        prompts: list[str] | str,
        sampling_params_list: list[SamplingParams] | None = None,
        system_prompt: str | None = None,
        images: PromptImageInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Convenience method to generate with multimodal inputs.
        Args:
            prompts: Text prompt(s)
            sampling_params_list: List of sampling parameters for each stage
            system_prompt: Optional system prompt
            images: Image input(s)
            mm_processor_kwargs: Optional processor kwargs
        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            images=images,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return self.generate(omni_inputs, sampling_params_list)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        del self.omni
        cleanup_dist_env_and_memory()

    def close(self):
        """Close and cleanup the Omni instance."""
        if hasattr(self.omni.instance, "close"):
            self.omni.instance.close()


@pytest.fixture(scope="session")
def omni_runner():
    return OmniRunner


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def _terminate_process_group(process: subprocess.Popen, wait_timeout: float = 15) -> None:
    """Gracefully terminate a subprocess group, then force kill if it hangs."""
    if process.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=wait_timeout)
        return
    except subprocess.TimeoutExpired:
        print(f"[TEST] Process did not exit in {wait_timeout}s; sending SIGKILL")
    except Exception as e:  # pragma: no cover - defensive logging
        print(f"[TEST] Error while waiting for process to exit: {e}")

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=5)
    except Exception as e:  # pragma: no cover - defensive logging
        print(f"[TEST] Error while force-killing process: {e}")


class _OutputStreamer:
    """Stream subprocess output in a background thread."""

    def __init__(self, process: subprocess.Popen, max_lines: int = 500):
        self.process = process
        self.lines: deque = deque(maxlen=max_lines)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._stream_output, daemon=True)
        self._thread.start()

    def _stream_output(self):
        """Read and print output lines from the process."""
        try:
            for line in iter(self.process.stdout.readline, b""):
                if self._stop_event.is_set():
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                self.lines.append(decoded)
                # Print server logs with prefix
                print(f"[SERVER] {decoded}")
        except Exception as e:
            print(f"[SERVER] Output streaming error: {e}")

    def stop(self):
        """Stop the output streaming."""
        self._stop_event.set()
        self._thread.join(timeout=2)

    def get_output(self) -> str:
        """Get all captured output."""
        return "\n".join(self.lines)


def _wait_for_server(host: str, port: int, timeout: float = 1800, process: subprocess.Popen | None = None) -> bool:
    """Wait for the server to be ready, fail fast if the process has died.

    Args:
        timeout: Maximum time to wait in seconds. Default is 1800 (30 minutes)
                 because vLLM-Omni multi-stage pipeline takes a long time to initialize.
    """
    import requests

    start_time = time.time()
    url = f"http://{host}:{port}/health"

    while time.time() - start_time < timeout:
        elapsed = int(time.time() - start_time)
        # Exit early if subprocess already crashed
        if process is not None and process.poll() is not None:
            print(f"\n[TEST] Server process exited early with code {process.returncode}")
            return False
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"\n[TEST] Server ready after {elapsed}s")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    print(f"\n[TEST] Server failed to start after {timeout}s")
    return False


def _create_server_cmd(host: str, port: int, model_name: str) -> list[str]:
    """Build the command used to launch the omni server."""
    return [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        model_name,
        "--omni",
        "--host",
        host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.8",
        "--load-format",
        "dummy",
    ]


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with actual model weights.

    Uses module scope so the server starts only once for all tests.
    Multi-stage initialization can take 10-20+ minutes.
    """
    if not hasattr(request, "param"):
        raise ValueError("omni_server fixture requires a model name via @pytest.mark.parametrize")

    model_name = request.param
    port = _find_free_port()
    host = "127.0.0.1"
    cmd = _create_server_cmd(host, port, model_name)

    print(f"\n{'=' * 60}")
    print(f"Starting vLLM-Omni server on {host}:{port}")
    print("Multi-stage init may take 10-20+ minutes...")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        preexec_fn=os.setsid,  # Create new process group for cleanup
    )

    streamer = _OutputStreamer(process)

    try:
        if not _wait_for_server(host, port, timeout=1800, process=process):
            streamer.stop()
            process.terminate()
            process.wait(timeout=10)
            raise RuntimeError("Server failed to start within 30 minute timeout.\nSee server logs above for details.")

        print(f"\n{'=' * 60}")
        print(f"Server ready at http://{host}:{port}")
        print(f"{'=' * 60}\n")

        yield {
            "host": host,
            "port": port,
            "base_url": f"http://{host}:{port}",
            "model": model_name,
            "process": process,
            "streamer": streamer,
        }

    finally:
        print("\nShutting down server...")
        streamer.stop()
        _terminate_process_group(process)
        if process.poll() is None:
            print("[TEST] Server shutdown forcefully; check logs above for details")
        else:
            print("Server shut down successfully")
