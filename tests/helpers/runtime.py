"""Server/client/runner runtime primitives for tests."""

import base64
import concurrent.futures
import errno
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple

import psutil
import requests
import soundfile as sf
import torch
import yaml
from openai import OpenAI, omit
from PIL import Image
from vllm import TextPrompt
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.logger import init_logger

from tests.helpers.assertions import (
    assert_audio_speech_response,
    assert_diffusion_response,
    assert_omni_response,
)
from tests.helpers.env import run_forced_gpu_cleanup_round
from tests.helpers.media import (
    _merge_base64_audio_to_segment,
    convert_audio_bytes_to_text,
    decode_b64_image,
)
from vllm_omni.config.stage_config import resolve_deploy_yaml
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

PromptAudioInput = list[tuple[Any, int]] | tuple[Any, int] | None
PromptImageInput = list[Any] | Any | None
PromptVideoInput = list[Any] | Any | None

try:
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory  # type: ignore
except Exception:  # pragma: no cover

    def cleanup_dist_env_and_memory() -> None:
        return None


def get_open_port(host: str = "127.0.0.1", *, max_attempts: int = 128) -> int:
    """Return a local TCP port that is suitable for binding a new listener.

    A single ``bind(host, 0)`` / close cycle leaves a race where another process can
    take the same port number before PyTorch/vLLM bind it, yielding
    ``EADDRINUSE`` / ``DistNetworkError``. We therefore:

    #. Allocate an ephemeral port on *host*.
    #. Immediately attempt ``bind(host, port)`` again. If that fails with
       ``errno.EADDRINUSE``, retry from step 1.

    Raises ``RuntimeError`` if no free port is found after *max_attempts* (e.g. port
    exhaustion under heavy parallel tests).
    """
    last_exc: OSError | None = None
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            port = int(s.getsockname()[1])
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((host, port))
        except OSError as exc:
            last_exc = exc
            if exc.errno == errno.EADDRINUSE:
                continue
            raise
        return port
    raise RuntimeError(
        f"Could not obtain a free TCP port on {host!r} after {max_attempts} attempts (last error: {last_exc!r})"
    ) from last_exc


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


def _omni_subprocess_cwd() -> str:
    """Repo root for ``python -m vllm_omni...`` (legacy conftest lived under ``tests/``; helpers under ``tests/helpers/``)."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


class OmniServerParams(NamedTuple):
    model: str
    port: int | None = None
    stage_config_path: str | None = None
    server_args: list[str] | None = None
    env_dict: dict[str, str] | None = None
    use_omni: bool = True
    use_stage_cli: bool = False
    init_timeout: int | None = None
    stage_init_timeout: int | None = None  # None: fixture supplies default (600 s)


def run_omni_server(
    request: Any,
    run_level: str,
    model_prefix: str,
    omni_fixture_lock: threading.Lock,
) -> Generator["OmniServer", Any, None]:
    from tests.helpers.stage_config import modify_stage_config

    with omni_fixture_lock:
        params: OmniServerParams = request.param
        model = model_prefix + params.model
        port = params.port
        stage_config_path = params.stage_config_path
        if run_level in {"advanced_model", "full_model"} and stage_config_path is not None:
            with open(stage_config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            # Strip ``load_format: dummy`` (CI overlay default) so advanced_model
            # tests use real weights. New schema (``stages:``) writes the field
            # flat at stage level; legacy schema (``stage_args:``) nests it as
            # ``engine_args.load_format``. Handle both.
            new_schema_stages = cfg.get("stages")
            stage_key = "stages" if new_schema_stages is not None else "stage_args"
            delete_path = "load_format" if new_schema_stages is not None else "engine_args.load_format"
            stage_entries = cfg.get(stage_key, [])
            stage_ids = [stage["stage_id"] for stage in stage_entries if "stage_id" in stage]
            stage_config_path = modify_stage_config(
                stage_config_path,
                deletes={stage_key: {stage_id: [delete_path] for stage_id in stage_ids}},
            )

        server_args = params.server_args or []
        if params.use_omni and params.stage_init_timeout is not None:
            server_args = [*server_args, "--stage-init-timeout", str(params.stage_init_timeout)]
        else:
            server_args = [*server_args, "--stage-init-timeout", "600"]
        if params.init_timeout is not None:
            server_args = [*server_args, "--init-timeout", str(params.init_timeout)]
        else:
            server_args = [*server_args, "--init-timeout", "900"]
        if params.use_stage_cli:
            if not params.use_omni:
                raise ValueError("omni_server with use_stage_cli=True requires use_omni=True")
            if stage_config_path is None:
                raise ValueError("omni_server with use_stage_cli=True requires a stage_config_path")
            server_args += ["--stage-configs-path", stage_config_path]

            with OmniServerStageCli(
                model,
                stage_config_path,
                server_args,
                port=port,
                env_dict=params.env_dict,
            ) as server:
                print("OmniServer started successfully")
                yield server
                print("OmniServer stopping...")
        else:
            if stage_config_path is not None:
                server_args += ["--stage-configs-path", stage_config_path]

            with (
                OmniServer(
                    model,
                    server_args,
                    port=port,
                    env_dict=params.env_dict,
                    use_omni=params.use_omni,
                )
                if port
                else OmniServer(
                    model,
                    server_args,
                    env_dict=params.env_dict,
                    use_omni=params.use_omni,
                )
            ) as server:
                print("OmniServer started successfully")
                yield server
                print("OmniServer stopping...")

        print("OmniServer stopped")


class OmniServer:
    """Omniserver for vLLM-Omni tests."""

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        port: int | None = None,
        env_dict: dict[str, str] | None = None,
        use_omni: bool = True,
    ) -> None:
        run_forced_gpu_cleanup_round()
        cleanup_dist_env_and_memory()
        self.model = model
        self.serve_args = serve_args
        self.env_dict = env_dict
        self.use_omni = use_omni
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        self.port = get_open_port() if port is None else port

    def _start_server(self) -> None:
        env = os.environ.copy()
        env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.use_omni:
            cmd.append("--omni")
        cmd += self.serve_args

        print(f"Launching OmniServer with: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=_omni_subprocess_cwd(),
        )

        max_wait = 1200
        start_time = time.time()
        while time.time() - start_time < max_wait:
            ret = self.proc.poll()
            if ret is not None:
                raise RuntimeError(f"Server processes exited with code {ret} before becoming ready.")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                if sock.connect_ex((self.host, self.port)) == 0:
                    print(f"Server ready on {self.host}:{self.port}")
                    return
            time.sleep(2)
        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    def _kill_process_tree(self, pid):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            all_pids = [pid] + [child.pid for child in children]

            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            _, still_alive = psutil.wait_procs(children, timeout=10)

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

            time.sleep(1)
            alive_processes = []
            for check_pid in all_pids:
                if psutil.pid_exists(check_pid):
                    alive_processes.append(check_pid)

            if alive_processes:
                print(f"Warning: Processes still alive: {alive_processes}")
                for alive_pid in alive_processes:
                    try:
                        subprocess.run(["kill", "-9", str(alive_pid)], timeout=2)
                    except Exception as e:
                        print(f"Cleanup failed: {e}")

        except psutil.NoSuchProcess:
            pass

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            self._kill_process_tree(self.proc.pid)
        run_forced_gpu_cleanup_round()
        cleanup_dist_env_and_memory()


class OmniServerStageCli(OmniServer):
    """Omni server harness that exercises the stage CLI flow."""

    def __init__(
        self,
        model: str,
        stage_config_path: str,
        serve_args: list[str] | None = None,
        *,
        stage_ids: list[int] | None = None,
        port: int | None = None,
        env_dict: dict[str, str] | None = None,
    ) -> None:
        super().__init__(model, serve_args or [], port=port, env_dict=env_dict, use_omni=True)
        self.stage_config_path = stage_config_path
        self.master_port = get_open_port()
        self.visible_device_list = self._load_visible_device_list(env_dict)
        resolved_cfg = resolve_deploy_yaml(stage_config_path)
        # Dump the resolved deploy config so CI logs show each stage's
        # gpu_memory_utilization / max_model_len / max_num_seqs after
        # base_config inheritance and overlay merge — essential when
        # diagnosing OOMs that depend on the merged values.
        print(
            f"[OmniServerStageCli] Resolved deploy config from {stage_config_path}:\n"
            f"{yaml.safe_dump(resolved_cfg, sort_keys=False, default_flow_style=False)}",
            flush=True,
        )
        self.stage_runtime_devices = self._load_stage_runtime_devices(resolved_cfg)
        self.stage_ids = stage_ids or self._load_stage_ids(resolved_cfg)
        if 0 not in self.stage_ids:
            raise ValueError(f"Stage CLI test requires stage_id=0 in config: {stage_config_path}")
        self.stage_procs: dict[int, subprocess.Popen] = {}
        self.proc = None

    @staticmethod
    def _stage_entries(cfg: dict) -> list[dict]:
        """Return the list of stage entries from either legacy (``stage_args``)
        or new-schema (``stages``) deploy YAMLs."""
        return cfg.get("stage_args") or cfg.get("stages") or []

    @staticmethod
    def _load_stage_ids(resolved_config: dict) -> list[int]:
        stage_ids = [
            stage["stage_id"] for stage in OmniServerStageCli._stage_entries(resolved_config) if "stage_id" in stage
        ]
        if not stage_ids:
            raise ValueError("No stage IDs found in resolved config")
        return stage_ids

    @staticmethod
    def _load_stage_runtime_devices(resolved_config: dict) -> dict[int, str]:
        runtime_devices: dict[int, str] = {}
        for stage in OmniServerStageCli._stage_entries(resolved_config):
            stage_id = stage.get("stage_id")
            # New schema: stage.devices is flat at stage level.
            # Legacy schema: stage.runtime.devices is nested.
            devices = stage.get("devices") or stage.get("runtime", {}).get("devices")
            if stage_id is not None and devices:
                runtime_devices[int(stage_id)] = str(devices)
        return runtime_devices

    @classmethod
    def _parse_device_list(cls, devices: str | int) -> list[str]:
        if isinstance(devices, int):
            if devices < 0:
                raise ValueError("Device IDs must be non-negative integers")
            return [str(devices)]
        return [token.strip() for token in str(devices).split(",") if token.strip()]

    @classmethod
    def _load_visible_device_list(cls, env_dict: dict[str, str] | None) -> list[str] | None:
        env = os.environ.copy()
        if env_dict is not None:
            env.update(env_dict)

        env_var = getattr(current_omni_platform, "device_control_env_var", None)
        if env_var and env_var in env:
            return [token.strip() for token in env[env_var].split(",") if token.strip()]
        return None

    @classmethod
    def _map_stage_devices(cls, stage_id: int, visible_device_list: list[str] | None, devices: str) -> str:
        device_list = cls._parse_device_list(devices)

        if visible_device_list is None:
            return ",".join(device_list)

        if not all(device.isdigit() for device in device_list):
            raise ValueError("Logical devices must be non-negative integers")

        logical_ids = [int(device) for device in device_list]
        if logical_ids and max(logical_ids) >= len(visible_device_list):
            raise ValueError(
                f"Stage {stage_id} has logical IDs {device_list}, one or more of which exceed the number of visible devices"
            )

        return ",".join(visible_device_list[idx] for idx in logical_ids)

    def _set_stage_device_env(self, stage_id: int, env: dict[str, str], devices: str) -> None:
        mapped_devices = self._map_stage_devices(stage_id, self.visible_device_list, devices)
        env_var = getattr(current_omni_platform, "device_control_env_var", None)
        if env_var:
            env[env_var] = mapped_devices

    def _build_stage_cmd(self, stage_id: int, *, headless: bool) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--stage-configs-path",
            self.stage_config_path,
            "--stage-id",
            str(stage_id),
            "--omni-master-address",
            self.host,
            "--omni-master-port",
            str(self.master_port),
        ]

        if headless:
            cmd.append("--headless")
        else:
            cmd += ["--host", self.host, "--port", str(self.port)]

        cmd += self.serve_args
        return cmd

    def _launch_stage(self, stage_id: int, *, headless: bool) -> None:
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.env_dict is not None:
            env.update(self.env_dict)

        devices = self.stage_runtime_devices.get(stage_id)
        if devices:
            self._set_stage_device_env(stage_id, env, devices)

        cmd = self._build_stage_cmd(stage_id, headless=headless)
        print(f"Launching OmniServerStageCli stage {stage_id}: {' '.join(cmd)}")
        # Capture each subprocess's stdout+stderr to a per-stage log file so
        # debugging "Stage N exited before API server ready" doesn't rely on
        # guessing; the file is surfaced in the RuntimeError message.
        log_path = Path(tempfile.gettempdir()) / f"omni_stage_{stage_id}_{self.master_port}.log"
        self._stage_log_paths = getattr(self, "_stage_log_paths", {})
        self._stage_log_paths[stage_id] = log_path
        log_fh = open(log_path, "w", buffering=1)  # noqa: SIM115 - closed in __exit__
        self._stage_log_files = getattr(self, "_stage_log_files", {})
        self._stage_log_files[stage_id] = log_fh
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        self.stage_procs[stage_id] = proc
        if stage_id == 0:
            self.proc = proc

    def _ensure_stage_processes_alive(self) -> None:
        for stage_id, proc in self.stage_procs.items():
            ret = proc.poll()
            if ret is not None:
                log_path = getattr(self, "_stage_log_paths", {}).get(stage_id)
                tail = ""
                if log_path and log_path.exists():
                    try:
                        with open(log_path, encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                        tail = "\n=== Last 60 lines of stage {} log ({}) ===\n{}".format(
                            stage_id, log_path, "".join(lines[-60:]) or "<empty>"
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic only
                        tail = f"\n<failed to read stage log {log_path}: {exc}>"
                raise RuntimeError(f"Stage {stage_id} exited with code {ret} before API server became ready.{tail}")

    def _start_server(self) -> None:
        ordered_stage_ids = [0, *[stage_id for stage_id in self.stage_ids if stage_id != 0]]

        self._launch_stage(0, headless=False)
        time.sleep(2)
        self._ensure_stage_processes_alive()

        for stage_id in ordered_stage_ids[1:]:
            self._launch_stage(stage_id, headless=True)

        max_wait = 1200
        start_time = time.time()
        while time.time() - start_time < max_wait:
            self._ensure_stage_processes_alive()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((self.host, self.port))
                if result == 0:
                    print(f"OmniServerStageCli ready on {self.host}:{self.port}")
                    return
            time.sleep(2)

        raise RuntimeError(f"OmniServerStageCli failed to start within {max_wait} seconds")

    def _dump_stage_logs_for_debug(self, head_lines: int = 300, tail_lines: int = 500) -> None:
        """Tail each stage's subprocess log back to stdout on teardown.

        Stage subprocesses redirect stdout/stderr to ``/tmp/omni_stage_*.log``
        so we don't spam the main CI stream while tests run; but that also
        hides engine init (KV cache size, Available KV cache memory, vLLM
        engine config) when things go wrong. Dump them here so buildkite
        captures them post-run. Head covers engine init; tail covers
        whatever state the stage was in when it was torn down.
        """
        log_paths = getattr(self, "_stage_log_paths", {}) or {}
        for stage_id in sorted(log_paths):
            log_path = log_paths[stage_id]
            if not log_path or not log_path.exists():
                continue
            try:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception as exc:  # pragma: no cover - diagnostic only
                print(f"[OmniServerStageCli] stage {stage_id} log read failed: {exc}", flush=True)
                continue
            total = len(lines)
            if total <= head_lines + tail_lines:
                head_chunk = lines
                tail_chunk = []
                elided = 0
            else:
                head_chunk = lines[:head_lines]
                tail_chunk = lines[-tail_lines:]
                elided = total - head_lines - tail_lines
            print(f"\n=== stage {stage_id} log HEAD ({log_path}) ===", flush=True)
            print("".join(head_chunk).rstrip("\n"), flush=True)
            if tail_chunk:
                print(f"\n... [{elided} lines elided] ...", flush=True)
                print(f"\n=== stage {stage_id} log TAIL ({log_path}) ===", flush=True)
                print("".join(tail_chunk).rstrip("\n"), flush=True)
            print(f"=== end stage {stage_id} log ===\n", flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dump_stage_logs_for_debug()
        for stage_id in sorted(self.stage_procs, reverse=True):
            proc = self.stage_procs[stage_id]
            if proc.poll() is None:
                self._kill_process_tree(proc.pid)
        run_forced_gpu_cleanup_round()
        cleanup_dist_env_and_memory()


@dataclass
class OmniResponse:
    text_content: str | None = None
    audio_data: list[str] | None = None
    audio_content: str | None = None
    audio_format: str | None = None
    audio_bytes: bytes | None = None
    e2e_latency: float | None = None
    success: bool = False
    error_message: str | None = None
    cached_tokens: int | None = None


@dataclass
class DiffusionResponse:
    text_content: str | None = None
    images: list[Image.Image] | None = None
    audios: list[Any] | None = None
    videos: list[Any] | None = None
    e2e_latency: float | None = None
    success: bool = False
    error_message: str | None = None


class OpenAIClientHandler:
    def __init__(self, host: str = "127.0.0.1", port: int = None, api_key: str = "EMPTY", run_level: str = None):
        if port is None:
            port = get_open_port()
        self.base_url = f"http://{host}:{port}"
        self.client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)
        self.run_level = run_level

    def _process_stream_omni_response(self, chat_completion) -> OmniResponse:
        result = OmniResponse()
        start_time = time.perf_counter()
        try:
            text_content = ""
            audio_data = []
            for chunk in chat_completion:
                for choice in chunk.choices:
                    content = getattr(getattr(choice, "delta", None), "content", None)
                    modality = getattr(chunk, "modality", None)
                    if modality == "audio" and content:
                        audio_data.append(content)
                    elif modality == "text" and content:
                        text_content += content
            result.e2e_latency = time.perf_counter() - start_time
            audio_content = None
            if audio_data:
                merged_seg = _merge_base64_audio_to_segment(audio_data)
                wav_buf = BytesIO()
                merged_seg.export(wav_buf, format="wav")
                result.audio_bytes = wav_buf.getvalue()
                audio_content = convert_audio_bytes_to_text(result.audio_bytes)
            result.text_content = text_content
            result.audio_data = audio_data
            result.audio_content = audio_content
            result.success = True
        except Exception as e:
            result.error_message = f"Stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")
        return result

    def _process_non_stream_omni_response(self, chat_completion) -> OmniResponse:
        result = OmniResponse()
        start_time = time.perf_counter()
        try:
            audio_data = None
            text_content = None
            for choice in chat_completion.choices:
                if hasattr(choice.message, "audio") and choice.message.audio is not None:
                    audio_data = choice.message.audio.data
                if hasattr(choice.message, "content") and choice.message.content is not None:
                    text_content = choice.message.content
            # Extract cached_tokens for prefix caching tests
            usage = getattr(chat_completion, "usage", None)
            if usage and (details := getattr(usage, "prompt_tokens_details", None)):
                result.cached_tokens = details.cached_tokens
            result.e2e_latency = time.perf_counter() - start_time
            audio_content = None
            if audio_data:
                result.audio_bytes = base64.b64decode(audio_data)
                audio_content = convert_audio_bytes_to_text(result.audio_bytes)
            result.text_content = text_content
            result.audio_content = audio_content
            result.success = True
        except Exception as e:
            result.error_message = f"Non-stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")
        return result

    def _process_diffusion_response(self, chat_completion) -> DiffusionResponse:
        result = DiffusionResponse()
        start_time = time.perf_counter()
        try:
            images = []
            for choice in chat_completion.choices:
                content = getattr(choice.message, "content", None)
                if isinstance(content, list):
                    for item in content:
                        image_url = None
                        if isinstance(item, dict):
                            image_url = item.get("image_url", {}).get("url")
                        else:
                            image_url_obj = getattr(item, "image_url", None)
                            image_url = getattr(image_url_obj, "url", None) if image_url_obj else None
                        if image_url and image_url.startswith("data:image"):
                            b64_data = image_url.split(",", 1)[1]
                            images.append(decode_b64_image(b64_data))
            result.e2e_latency = time.perf_counter() - start_time
            result.images = images if images else None
            result.success = True
        except Exception as e:
            result.error_message = f"Diffusion response processing error: {str(e)}"
            print(f"Error: {result.error_message}")
        return result

    def send_omni_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        responses: list[OmniResponse] = []
        stream = request_config.get("stream", False)
        modalities = request_config.get("modalities", ["text", "audio"])
        extra_body: dict[str, Any] = {}
        if "speaker" in request_config:
            extra_body["speaker"] = request_config["speaker"]
        if request_config.get("use_audio_in_video"):
            mm = dict(extra_body.get("mm_processor_kwargs") or {})
            mm["use_audio_in_video"] = True
            extra_body["mm_processor_kwargs"] = mm
        if "sampling_params_list" in request_config:
            extra_body["sampling_params_list"] = request_config["sampling_params_list"]

        create_kwargs: dict[str, Any] = {
            "model": request_config.get("model"),
            "messages": request_config.get("messages"),
            "stream": stream,
            "modalities": modalities,
        }
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        if request_num == 1:
            chat_completion = self.client.chat.completions.create(**create_kwargs)
            resp = (
                self._process_stream_omni_response(chat_completion)
                if stream
                else self._process_non_stream_omni_response(chat_completion)
            )
            assert_omni_response(resp, request_config, run_level=self.run_level)
            responses.append(resp)
            return responses

        def _one():
            chat_completion = self.client.chat.completions.create(**create_kwargs)
            return (
                self._process_stream_omni_response(chat_completion)
                if stream
                else self._process_non_stream_omni_response(chat_completion)
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
            futures = [executor.submit(_one) for _ in range(request_num)]
            for future in concurrent.futures.as_completed(futures):
                resp = future.result()
                assert_omni_response(resp, request_config, run_level=self.run_level)
                responses.append(resp)
        return responses

    def _process_stream_audio_speech_response(self, response, *, response_format: str | None = None) -> OmniResponse:
        """
        Process streaming /v1/audio/speech responses into an OmniResponse.

        This mirrors _process_stream_omni_response but operates on low-level
        audio bytes and produces an OmniResponse with audio_content filled
        from Whisper transcription.
        """
        result = OmniResponse()
        start_time = time.perf_counter()

        try:
            # Aggregate all audio bytes from the streaming response.
            data = bytearray()

            # Preferred OpenAI helper.
            if hasattr(response, "iter_bytes") and callable(getattr(response, "iter_bytes")):
                for chunk in response.iter_bytes():
                    if chunk:
                        data.extend(chunk)
            else:
                # Generic iterable-of-bytes fallback (e.g., generator or list of chunks).
                try:
                    iterator = iter(response)
                except TypeError:
                    iterator = None

                if iterator is not None:
                    for chunk in iterator:
                        if not chunk:
                            continue
                        if isinstance(chunk, (bytes, bytearray)):
                            data.extend(chunk)
                        elif hasattr(chunk, "data"):
                            data.extend(chunk.data)  # type: ignore[arg-type]
                        elif hasattr(chunk, "content"):
                            data.extend(chunk.content)  # type: ignore[arg-type]
                        else:
                            raise TypeError(f"Unsupported stream chunk type: {type(chunk)}")
                else:
                    raise TypeError(f"Unsupported audio speech streaming response type: {type(response)}")

            raw_bytes = bytes(data)
            if response_format == "pcm":
                transcript = None
            else:
                transcript = convert_audio_bytes_to_text(raw_bytes)

            # Populate OmniResponse.
            result.audio_bytes = raw_bytes
            result.audio_content = transcript
            result.e2e_latency = time.perf_counter() - start_time
            result.success = True
            result.audio_format = getattr(response, "response", None)
            if result.audio_format is not None:
                result.audio_format = result.audio_format.headers.get("content-type", "")

        except Exception as e:
            result.error_message = f"Audio speech stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def _process_non_stream_audio_speech_response(
        self, response, *, response_format: str | None = None
    ) -> OmniResponse:
        """
        Process non-streaming /v1/audio/speech responses into an OmniResponse.

        This mirrors _process_non_stream_omni_response but for the binary
        audio payload returned by audio.speech.create.
        """
        result = OmniResponse()
        start_time = time.perf_counter()

        try:
            # OpenAI non-streaming audio.speech.create returns HttpxBinaryResponseContent (.read() or .content)
            if hasattr(response, "read") and callable(getattr(response, "read")):
                raw_bytes = response.read()
            elif hasattr(response, "content"):
                raw_bytes = response.content  # type: ignore[assignment]
            else:
                raise TypeError(f"Unsupported audio speech response type: {type(response)}")

            if response_format == "pcm":
                transcript = None
            else:
                transcript = convert_audio_bytes_to_text(raw_bytes)

            result.audio_bytes = raw_bytes
            result.audio_content = transcript
            result.e2e_latency = time.perf_counter() - start_time
            result.success = True
            result.audio_format = getattr(response, "response", None)
            if result.audio_format is not None:
                result.audio_format = result.audio_format.headers.get("content-type", "")

        except Exception as e:
            result.error_message = f"Audio speech non-stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def send_audio_speech_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """
        Call the /v1/audio/speech endpoint using the same configuration-dict
        style as send_omni_request, but via the OpenAI Python client's
        audio.speech APIs.

        Expected keys in request_config:
          - model: model name/path (required)
          - input: text to synthesize (required)
          - response_format: audio format such as "wav" or "pcm" (optional)
          - task_type, ref_text, ref_audio: TTS-specific extras (optional, passed via extra_body)
          - timeout: request timeout in seconds (float, optional, default 120.0)
          - stream: whether to use streaming API (bool, optional, default False)
        """
        timeout = float(request_config.get("timeout", 120.0))

        model = request_config["model"]
        text_input = request_config["input"]
        stream = bool(request_config.get("stream", False))
        voice = request_config.get("voice", None)

        # Standard OpenAI param: use omit when not provided to keep default behavior.
        response_format = request_config.get("response_format", omit)

        # Qwen3-TTS custom fields, forwarded via extra_body.
        extra_body: dict[str, Any] = {}
        # Keep this list aligned with vllm_omni.entrypoints.openai.protocol.audio params.
        for key in ("task_type", "ref_text", "ref_audio", "language", "max_new_tokens"):
            if key in request_config:
                extra_body[key] = request_config[key]

        responses: list[OmniResponse] = []

        speech_fmt: str | None = None if response_format is omit else str(response_format).lower()

        if request_num == 1:
            if stream:
                # Use streaming response helper.
                with self.client.audio.speech.with_streaming_response.create(
                    model=model,
                    input=text_input,
                    response_format=response_format,
                    extra_body=extra_body or None,
                    timeout=timeout,
                    voice=voice,
                ) as resp:
                    omni_resp = self._process_stream_audio_speech_response(resp, response_format=speech_fmt)
            else:
                # Non-streaming response.
                resp = self.client.audio.speech.create(
                    model=model,
                    input=text_input,
                    response_format=response_format,
                    extra_body=extra_body or None,
                    timeout=timeout,
                    voice=voice,
                )
                omni_resp = self._process_non_stream_audio_speech_response(resp, response_format=speech_fmt)

            assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
            responses.append(omni_resp)
            return responses
        else:
            # request_num > 1: concurrent requests (use same params as single-request path)

            if stream:

                def _stream_task():
                    with self.client.audio.speech.with_streaming_response.create(
                        model=model,
                        input=text_input,
                        response_format=response_format,
                        extra_body=extra_body or None,
                        timeout=timeout,
                        voice=voice,
                    ) as resp:
                        return self._process_stream_audio_speech_response(resp, response_format=speech_fmt)

                with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                    futures = [executor.submit(_stream_task) for _ in range(request_num)]
                    for future in concurrent.futures.as_completed(futures):
                        omni_resp = future.result()
                        assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
                        responses.append(omni_resp)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                    futures = []
                    for _ in range(request_num):
                        future = executor.submit(
                            self.client.audio.speech.create,
                            model=model,
                            input=text_input,
                            response_format=response_format,
                            extra_body=extra_body or None,
                            timeout=timeout,
                            voice=voice,
                        )
                        futures.append(future)

                    for future in concurrent.futures.as_completed(futures):
                        resp = future.result()
                        omni_resp = self._process_non_stream_audio_speech_response(resp, response_format=speech_fmt)
                        assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
                        responses.append(omni_resp)

        return responses

    def send_diffusion_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[DiffusionResponse]:
        """
        Send OpenAI requests for diffusion models.
        Args:
            request_config: Request configuration dictionary containing parameters like model, messages
            request_num: Number of requests to send concurrently, defaults to 1 (single request)
        Returns:
            list[DiffusionResponse]: List of DiffusionResponse objects containing the response data
        """
        responses: list[DiffusionResponse] = []
        stream = request_config.get("stream", False)
        modalities = request_config.get("modalities", omit)  # Most diffusion models don't require modalities param
        extra_body = request_config.get("extra_body", None)
        if stream:
            raise NotImplementedError("Streaming is not currently implemented for diffusion model e2e test")
        if request_num == 1:
            # Send single request
            chat_completion = self.client.chat.completions.create(
                model=request_config.get("model"),
                messages=request_config.get("messages"),
                extra_body=extra_body,
                modalities=modalities,
            )
            response = self._process_diffusion_response(chat_completion)
            assert_diffusion_response(response, request_config, run_level=self.run_level)
            responses.append(response)
        else:
            # Send concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                futures = []
                # Submit all request tasks
                for _ in range(request_num):
                    future = executor.submit(
                        self.client.chat.completions.create,
                        model=request_config.get("model"),
                        messages=request_config.get("messages"),
                        modalities=modalities,
                        extra_body=extra_body,
                    )
                    futures.append(future)
                # Process completed tasks
                for future in concurrent.futures.as_completed(futures):
                    chat_completion = future.result()
                    response = self._process_diffusion_response(chat_completion)
                    assert_diffusion_response(response, request_config, run_level=self.run_level)
                    responses.append(response)
        return responses

    def send_video_diffusion_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """
        Send native /v1/videos requests.
        """
        if request_num != 1:
            raise NotImplementedError("Concurrent video diffusion requests are not currently implemented")
        form_data = request_config.get("form_data")
        if not isinstance(form_data, dict):
            raise ValueError("Video request_config must contain 'form_data'")
        normalized_form_data = {key: str(value) for key, value in form_data.items() if value is not None}
        files: dict[str, tuple[str, BytesIO, str]] = {}
        image_reference = request_config.get("image_reference")
        if image_reference:
            if image_reference.startswith("data:image"):
                header, encoded = image_reference.split(",", 1)
                content_type = header.split(";")[0].removeprefix("data:")
                extension = content_type.split("/")[-1]
                file_data = base64.b64decode(encoded)
                files["input_reference"] = (f"reference.{extension}", BytesIO(file_data), content_type)
            else:
                normalized_form_data["image_reference"] = json.dumps({"image_url": image_reference})

        result = DiffusionResponse()
        start_time = time.perf_counter()
        create_url = self._build_url("/v1/videos")
        response = requests.post(
            create_url,
            data=normalized_form_data,
            files=files,
            headers={"Accept": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        job_data = response.json()
        video_id = job_data["id"]
        self._wait_until_video_completed(video_id)
        video_content = self._download_video_content(video_id)
        result.success = True
        result.videos = [video_content]
        result.e2e_latency = time.perf_counter() - start_time
        assert_diffusion_response(result, request_config, run_level=self.run_level)
        return [result]

    def _wait_until_video_completed(
        self, video_id: str, poll_interval_seconds: int = 2, timeout_seconds: int = 300
    ) -> None:
        status_url = self._build_url(f"/v1/videos/{video_id}")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status_resp = requests.get(status_url, headers={"Accept": "application/json"}, timeout=30)
            status_resp.raise_for_status()
            status_data = status_resp.json()
            current_status = status_data["status"]
            if current_status == "completed":
                return
            if current_status == "failed":
                error_msg = status_data.get("last_error", "Unknown error")
                raise RuntimeError(f"Job failed: {error_msg}")
            time.sleep(poll_interval_seconds)
        raise TimeoutError(f"Video job {video_id} did not complete within {timeout_seconds}s")

    def _download_video_content(self, video_id: str) -> bytes:
        download_url = self._build_url(f"/v1/videos/{video_id}/content")
        video_resp = requests.get(download_url, stream=True, timeout=60)
        video_resp.raise_for_status()
        video_bytes = BytesIO()
        for chunk in video_resp.iter_content(chunk_size=8192):
            if chunk:
                video_bytes.write(chunk)
        return video_bytes.getvalue()

    def _build_url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


class OmniRunner:
    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        stage_init_timeout: int = 600,
        batch_timeout: int = 10,
        # Bumped from 900s -> 1800s to give CI cold-cache loads of large
        # diffusion models enough headroom (Buildkite #8418 hit a 6-second
        # overrun loading Tongyi-MAI/Z-Image-Turbo: weights alone took 690s,
        # the full stage was ready at ~896s, but the orchestrator wrapper
        # finished at ~906s, just past the previous 900s ceiling). Engine
        # production default in AsyncOmniEngine remains 600s; this only
        # affects the test runner wrapper.
        init_timeout: int = 1800,
        shm_threshold_bytes: int = 65536,
        log_stats: bool = False,
        stage_configs_path: str | None = None,
        **kwargs,
    ) -> None:
        cleanup_dist_env_and_memory()
        run_forced_gpu_cleanup_round()
        self.model_name = model_name
        self.seed = seed
        self._prompt_len_estimate_cache: dict[str, Any] = {}
        from vllm_omni.entrypoints.omni import Omni

        self.omni = Omni(
            model=model_name,
            log_stats=log_stats,
            stage_init_timeout=stage_init_timeout,
            batch_timeout=batch_timeout,
            init_timeout=init_timeout,
            shm_threshold_bytes=shm_threshold_bytes,
            stage_configs_path=stage_configs_path,
            **kwargs,
        )

    def get_default_sampling_params_list(self) -> list[Any]:
        if not hasattr(self.omni, "default_sampling_params_list"):
            raise AttributeError("Omni.default_sampling_params_list is not available")
        return list(self.omni.default_sampling_params_list)

    def _estimate_prompt_len(
        self,
        additional_information: dict[str, Any],
        model_name: str,
    ) -> int:
        """Estimate prompt_token_ids placeholder length for the Talker stage.

        The AR Talker replaces all input embeddings via ``preprocess``, so the
        placeholder values are irrelevant but the **length** must match the
        embeddings that ``preprocess`` will produce.
        """
        _cache = self._prompt_len_estimate_cache
        try:
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
                Qwen3TTSTalkerForConditionalGeneration,
            )

            if model_name not in _cache:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
                cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)
                _cache[model_name] = (tok, getattr(cfg, "talker_config", None))

            tok, tcfg = _cache[model_name]
            task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]
            return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
                additional_information=additional_information,
                task_type=task_type,
                tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
                codec_language_id=getattr(tcfg, "codec_language_id", None),
                spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
            )
        except Exception as exc:
            logger.warning("Failed to estimate prompt length, using fallback 2048: %s", exc)
            return 2048

    def get_omni_inputs(
        self,
        prompts: list[str] | str,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        modalities: list[str] | None = None,
    ) -> list[TextPrompt]:
        if system_prompt is None:
            system_prompt = (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                "Group, capable of perceiving auditory and visual inputs, as well as "
                "generating text and speech."
            )
        video_padding_token = "<|VIDEO|>"
        image_padding_token = "<|IMAGE|>"
        audio_padding_token = "<|AUDIO|>"
        if "Qwen3-Omni-30B-A3B-Instruct" in self.model_name:
            video_padding_token = "<|video_pad|>"
            image_padding_token = "<|image_pad|>"
            audio_padding_token = "<|audio_pad|>"
        elif "Ming-flash-omni" in self.model_name:
            video_padding_token = "<VIDEO>"
            image_padding_token = "<IMAGE>"
            audio_padding_token = "<AUDIO>"
        if isinstance(prompts, str):
            prompts = [prompts]

        # Qwen-TTS: follow examples/offline_inference/qwen3_tts/end2end.py style.
        # Stage 0 expects token placeholders + additional_information (text/speaker/task_type/...),
        # and Talker replaces embeddings in preprocess based on additional_information only.
        is_tts_model = "Qwen3-TTS" in self.model_name or "qwen3_tts" in self.model_name.lower()
        if is_tts_model and modalities == ["audio"]:
            tts_kw = mm_processor_kwargs or {}
            task_type = tts_kw.get("task_type", "CustomVoice")
            speaker = tts_kw.get("speaker", "Vivian")
            language = tts_kw.get("language", "Auto")
            max_new_tokens = int(tts_kw.get("max_new_tokens", 2048))
            ref_audio = tts_kw.get("ref_audio", None)
            ref_text = tts_kw.get("ref_text", None)

            omni_inputs: list[TextPrompt] = []
            for prompt_text in prompts:
                text_str = str(prompt_text).strip() or " "
                additional_information: dict[str, Any] = {
                    "task_type": [task_type],
                    "text": [text_str],
                    "language": [language],
                    "speaker": [speaker],
                    "max_new_tokens": [max_new_tokens],
                }
                if ref_audio is not None:
                    additional_information["ref_audio"] = [ref_audio]
                if ref_text is not None:
                    additional_information["ref_text"] = [ref_text]
                plen = self._estimate_prompt_len(additional_information, self.model_name)
                input_dict: TextPrompt = {
                    "prompt_token_ids": [0] * plen,
                    "additional_information": additional_information,
                }
                omni_inputs.append(input_dict)
            return omni_inputs

        def _normalize(mm_input, num_prompts):
            if mm_input is None:
                return [None] * num_prompts
            if isinstance(mm_input, list):
                if len(mm_input) != num_prompts:
                    raise ValueError("Multimodal input list length must match prompts length")
                return mm_input
            return [mm_input] * num_prompts

        num_prompts = len(prompts)
        audios_list = _normalize(audios, num_prompts)
        images_list = _normalize(images, num_prompts)
        videos_list = _normalize(videos, num_prompts)

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
            if modalities:
                input_dict["modalities"] = modalities
            if mm_processor_kwargs:
                input_dict["mm_processor_kwargs"] = mm_processor_kwargs
            omni_inputs.append(input_dict)
        return omni_inputs

    def generate(
        self,
        prompts: list[Any],
        sampling_params_list: list[Any] | None = None,
    ) -> list[Any]:
        if sampling_params_list is None:
            sampling_params_list = self.get_default_sampling_params_list()
        return self.omni.generate(prompts, sampling_params_list)

    def generate_multimodal(
        self,
        prompts: list[str] | str,
        sampling_params_list: list[Any] | None = None,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        modalities: list[str] | None = None,
    ) -> list[Any]:
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            audios=audios,
            images=images,
            videos=videos,
            mm_processor_kwargs=mm_processor_kwargs,
            modalities=modalities,
        )
        return self.generate(omni_inputs, sampling_params_list)

    def start_profile(self, profile_prefix: str | None = None, stages: list[int] | None = None) -> list[Any]:
        return self.omni.start_profile(profile_prefix=profile_prefix, stages=stages)

    def stop_profile(self, stages: list[int] | None = None) -> list[Any]:
        return self.omni.stop_profile(stages=stages)

    def _cleanup_process(self):
        try:
            keywords = ["enginecore"]
            matched = []
            for proc in psutil.process_iter(["pid", "name", "cmdline", "username"]):
                try:
                    cmdline = " ".join(proc.cmdline()).lower() if proc.cmdline() else ""
                    name = proc.name().lower()
                    if any(k in cmdline for k in keywords) or any(k in name for k in keywords):
                        print(f"Found vllm process: PID={proc.pid}, cmd={cmdline[:100]}")
                        matched.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            for proc in matched:
                try:
                    proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            _, still_alive = psutil.wait_procs(matched, timeout=5)
            for proc in still_alive:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if still_alive:
                _, stubborn = psutil.wait_procs(still_alive, timeout=3)
                if stubborn:
                    print(f"Warning: failed to kill residual vllm pids: {[p.pid for p in stubborn]}")
                else:
                    print(f"Force-killed residual vllm pids: {[p.pid for p in still_alive]}")
            elif matched:
                print(f"Terminated vllm pids: {[p.pid for p in matched]}")
        except Exception as e:
            print(f"Error in psutil vllm cleanup: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.omni, "close"):
            self.omni.close()
        self._cleanup_process()
        run_forced_gpu_cleanup_round()
        cleanup_dist_env_and_memory()


class OmniRunnerHandler:
    def __init__(self, omni_runner):
        self.runner = omni_runner

    def _process_output(self, outputs: list[Any]) -> OmniResponse:
        result = OmniResponse()
        try:
            text_content = None
            audio_content = None
            for stage_output in outputs:
                if getattr(stage_output, "final_output_type", None) == "text":
                    text_content = stage_output.request_output.outputs[0].text
                if getattr(stage_output, "final_output_type", None) == "audio":
                    audio_content = stage_output.request_output.outputs[0].multimodal_output["audio"]
            result.audio_content = audio_content
            result.text_content = text_content
            result.success = True
        except Exception as e:
            result.error_message = f"Output processing error: {str(e)}"
            result.success = False
            print(f"Error: {result.error_message}")
        return result

    def send_request(self, request_config: dict[str, Any] | None = None) -> OmniResponse:
        if request_config is None:
            request_config = {}
        prompts = request_config.get("prompts")
        videos = request_config.get("videos")
        images = request_config.get("images")
        audios = request_config.get("audios")
        modalities = request_config.get("modalities", ["text", "audio"])
        outputs = self.runner.generate_multimodal(
            prompts=prompts, videos=videos, images=images, audios=audios, modalities=modalities
        )
        response = self._process_output(outputs)
        assert_omni_response(response, request_config, run_level="core_model")
        return response

    def send_audio_speech_request(self, request_config: dict[str, Any]) -> OmniResponse:
        """
        Offline TTS: text -> audio via generate_multimodal, then validate with assert_audio_speech_response.

        request_config must contain:
          - 'input' or 'prompts': text to synthesize.
        Optional keys:
          - 'voice'       -> speaker (CustomVoice)
          - 'task_type'   -> task_type in additional_information (default: "CustomVoice")
          - 'language'    -> language in additional_information (default: "Auto")
          - 'max_new_tokens' -> max_new_tokens in additional_information (default: 2048)
          - 'response_format' -> desired audio format (used only for assertion)
        """
        input_text = request_config.get("input") or request_config.get("prompts")
        if input_text is None:
            raise ValueError("request_config must contain 'input' or 'prompts' for TTS")
        if isinstance(input_text, list):
            input_text = input_text[0] if input_text else ""

        mm_processor_kwargs: dict[str, Any] = {}
        if "voice" in request_config:
            mm_processor_kwargs["speaker"] = request_config["voice"]
        if "task_type" in request_config:
            mm_processor_kwargs["task_type"] = request_config["task_type"]
        if "ref_audio" in request_config:
            mm_processor_kwargs["ref_audio"] = request_config["ref_audio"]
        if "ref_text" in request_config:
            mm_processor_kwargs["ref_text"] = request_config["ref_text"]
        if "language" in request_config:
            mm_processor_kwargs["language"] = request_config["language"]
        if "max_new_tokens" in request_config:
            mm_processor_kwargs["max_new_tokens"] = request_config["max_new_tokens"]

        outputs = self.runner.generate_multimodal(
            prompts=input_text,
            modalities=["audio"],
            mm_processor_kwargs=mm_processor_kwargs or None,
        )
        mm_out: dict[str, Any] | None = None
        for stage_out in outputs:
            if getattr(stage_out, "final_output_type", None) == "audio":
                mm_out = stage_out.request_output.outputs[0].multimodal_output
                break
        if mm_out is None:
            result = OmniResponse(success=False, error_message="No audio output from pipeline")
            assert result.success, result.error_message
            return result

        audio_data = mm_out.get("audio")
        if audio_data is None:
            result = OmniResponse(success=False, error_message="No audio tensor in multimodal output")
            assert result.success, result.error_message
            return result

        sr_raw = mm_out.get("sr")
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sr = int(sr_val.item() if hasattr(sr_val, "item") else sr_val)
        wav_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
        wav_buf = io.BytesIO()
        sf.write(
            wav_buf,
            wav_tensor.float().cpu().numpy().reshape(-1),
            samplerate=sr,
            format="WAV",
            subtype="PCM_16",
        )
        result = OmniResponse(success=True, audio_bytes=wav_buf.getvalue(), audio_format="audio/wav")
        assert_audio_speech_response(result, request_config, run_level="core_model")
        return result

    def start_profile(self, profile_prefix: str | None = None, stages: list[int] | None = None) -> list[Any]:
        return self.runner.start_profile(profile_prefix=profile_prefix, stages=stages)

    def stop_profile(self, stages: list[int] | None = None) -> list[Any]:
        return self.runner.stop_profile(stages=stages)


__all__ = [
    "DiffusionResponse",
    "OmniResponse",
    "OmniRunner",
    "OmniRunnerHandler",
    "OmniServer",
    "OmniServerParams",
    "OmniServerStageCli",
    "OpenAIClientHandler",
    "get_open_port",
    "run_forced_gpu_cleanup_round",
    "dummy_messages_from_mix_data",
]
