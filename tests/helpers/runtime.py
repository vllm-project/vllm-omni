"""Server/client/runner runtime primitives for tests."""

import asyncio
import base64
import concurrent.futures
import copy
import errno
import gc
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
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple, cast
from urllib.parse import quote

import psutil
import requests
import soundfile as sf
import torch
import yaml
from openai import APIError, OpenAI, omit
from PIL import Image
from vllm import TextPrompt, envs
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.logger import init_logger

from tests.helpers.assertions import (
    assert_audio_speech_response,
    assert_diffusion_response,
    assert_http_error,
    assert_omni_response,
)
from tests.helpers.env import run_post_test_cleanup, run_pre_test_cleanup
from tests.helpers.media import (
    _merge_base64_audio_to_segment,
    decode_b64_image,
)
from tests.model_tests.diffusion.utils import resolve_tiny_model_path
from vllm_omni.config.stage_config import resolve_deploy_yaml
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    # Reset environment variable cache
    envs.disable_envs_cache()

    # Reset rocm_aiter_ops class variables to match current os.environ.
    # These are class-level attributes that persist across tests and are
    # NOT restored by monkeypatch (which only restores os.environ).
    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.is_rocm():
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()

    # Ensure all objects are not frozen before cleanup
    gc.unfreeze()

    destroy_model_parallel()
    destroy_distributed_environment()
    if shutdown_ray:
        import ray  # Lazy import Ray

        ray.shutdown()
    gc.collect()

    if not current_omni_platform.is_cpu():
        current_omni_platform.empty_cache()
        try:
            torch._C._host_emptyCache()
        except AttributeError:
            logger.warning("torch._C._host_emptyCache() only available in Pytorch >=2.5")


def _parse_response_json(r: requests.Response) -> dict[str, Any] | list[Any] | None:
    try:
        data = r.json()
        if isinstance(data, (dict, list)):
            return data
    except Exception:
        pass
    return None


def _split_request_config_by_per_output_sizes(cfg: dict[str, Any]) -> list[dict[str, Any]] | None:
    """If ``extra_body`` has list ``height``/``width``, return one config per index (scalar h/w, ``num_outputs_per_prompt=1``)."""
    eb = cfg.get("extra_body")
    if not eb:
        return None
    h, w = eb.get("height"), eb.get("width")
    if (isinstance(h, (list, tuple)) or isinstance(w, (list, tuple))) and not (
        isinstance(h, (list, tuple)) and isinstance(w, (list, tuple))
    ):
        raise ValueError("extra_body height and width must both be lists or both be scalars")
    if not (isinstance(h, (list, tuple)) and isinstance(w, (list, tuple))):
        return None
    if len(h) != len(w):
        raise ValueError(f"height and width lists must have equal length; got {len(h)=} {len(w)=}")
    n = len(h)
    n_out = eb.get("num_outputs_per_prompt")
    if n_out is not None:
        n_out = int(n_out)
        if n_out != n:
            raise ValueError(
                "When height/width are lists, num_outputs_per_prompt must equal their length; "
                f"got num_outputs_per_prompt={n_out}, len(lists)={n}"
            )
    splits: list[dict[str, Any]] = []
    for i in range(n):
        sub = copy.deepcopy(cfg)
        sub_eb = dict(sub.get("extra_body") or {})
        sub_eb["height"] = int(h[i])
        sub_eb["width"] = int(w[i])
        sub_eb["num_outputs_per_prompt"] = 1
        sub["extra_body"] = sub_eb
        splits.append(sub)
    return splits


PromptAudioInput = list[tuple[Any, int]] | tuple[Any, int] | None
PromptImageInput = list[Any] | Any | None
PromptVideoInput = list[Any] | Any | None


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
        run_pre_test_cleanup()
        run_post_test_cleanup()
        cleanup_dist_env_and_memory()
        self.model = model
        args = list(serve_args)
        self.serve_args = args
        self.log_stats = "--disable-log-stats" not in args and "--log-stats" in args
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
        startup_t0 = time.perf_counter()
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
                    startup_s = time.perf_counter() - startup_t0
                    if self.log_stats:
                        print(
                            f"Server ready on {self.host}:{self.port} (OmniServer startup took {startup_s:.3f}s)",
                            flush=True,
                        )
                    return
            time.sleep(2)
        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    @staticmethod
    def _reap_zombie(proc: "psutil.Process") -> bool:
        """Reap a zombie child process via ``os.waitpid``.

        ``psutil.Process.wait()`` uses ``pidfd_open`` + ``poll()``, which
        never fires for a zombie (the zombie is already dead and will not
        change state).  Since the test process is the parent, we can reap
        the zombie directly with ``os.waitpid(pid, os.WNOHANG)`` and
        retrieve its exit code.

        Returns True if the process was a zombie and was reaped.
        """
        if proc.status() != psutil.STATUS_ZOMBIE:
            return False
        try:
            os.waitpid(proc.pid, os.WNOHANG)
        except ChildProcessError:
            pass
        return True

    @staticmethod
    def _wait_or_reap(proc: "psutil.Process", timeout: int) -> None:
        """Wait for *proc* to exit, handling zombie state transparently.

        When the process is already a zombie (e.g. orchestrator thread
        hung during shutdown and the main process exited), ``pidfd_open``
        + ``poll()`` inside ``psutil`` will never see a state change.
        Fall back to ``os.waitpid`` to reap the zombie.
        """
        if OmniServer._reap_zombie(proc):
            return
        try:
            proc.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            OmniServer._reap_zombie(proc)

    def _kill_process_tree(self, pid):
        """Kill the process tree rooted at *pid*.

        Terminate the parent **first** so the OmniServer can gracefully shut
        down its stage-engine children through the orchestrator.  This avoids
        the ``subprocess died unexpectedly`` ERROR that the APIServer monitor
        thread logs when children are killed before the parent, which in turn
        can cause CI watchdogs to false-trigger on the upstream ``Shutdown
        initiated`` message.

        When the parent does not exit within the grace period (e.g. CPU-
        offloaded workers stuck in CUDA D-state), the method falls back to
        killing children first so the parent can be reaped cleanly.
        """
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # 1. Terminate the parent first — let it run its graceful
            #    shutdown cascade (orchestrator → stage pools → engine cores).
            try:
                parent.terminate()
            except psutil.NoSuchProcess:
                pass

            # 2. Give the parent time to shut down its children cleanly.
            parent_exited = False
            try:
                parent.wait(timeout=15)
                parent_exited = True
            except psutil.NoSuchProcess:
                parent_exited = True
            except psutil.TimeoutExpired:
                parent_exited = OmniServer._reap_zombie(parent)

            if not parent_exited:
                # Parent is stuck — children (e.g. CPU-offloaded CFG workers)
                # are likely in uninterruptible sleep.  Kill children first
                # so the parent can be reaped without lingering as a zombie.
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(children, timeout=5)
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
                OmniServer._wait_or_reap(parent, timeout=5)
            else:
                # Parent exited cleanly — clean up any remaining children.
                for child in children:
                    try:
                        if child.is_running():
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
                    if parent.is_running() and not OmniServer._reap_zombie(parent):
                        parent.kill()
                        parent.wait(timeout=10)
                except psutil.NoSuchProcess:
                    pass

            # 3. Final sweep — ``kill -9`` anything that escaped.
            time.sleep(1)
            alive_processes: list[int] = []
            for child in children:
                try:
                    if child.is_running():
                        alive_processes.append(child.pid)
                except psutil.NoSuchProcess:
                    pass
            # Only count the parent as alive if it is NOT a zombie
            # (zombies are already dead — just waiting to be reaped).
            try:
                if parent.is_running() and parent.status() != psutil.STATUS_ZOMBIE:
                    alive_processes.append(parent.pid)
            except psutil.NoSuchProcess:
                pass

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
        run_pre_test_cleanup()
        run_post_test_cleanup()
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
        self.stage_replica_counts = self._load_stage_replica_counts(resolved_cfg)
        self.stage_procs: dict[tuple[int, int], subprocess.Popen] = {}
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

    @staticmethod
    def _load_stage_replica_counts(resolved_config: dict) -> dict[int, int]:
        replica_counts: dict[int, int] = {}
        for stage in OmniServerStageCli._stage_entries(resolved_config):
            stage_id = stage.get("stage_id")
            if stage_id is None:
                continue
            replica_counts[int(stage_id)] = max(
                1,
                int(stage.get("num_replicas") or stage.get("runtime", {}).get("num_replicas", 1)),
            )
        return replica_counts

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

    def _devices_for_replica(self, stage_id: int, devices: str, replica_id: int) -> str:
        replica_count = self.stage_replica_counts.get(stage_id, 1)
        if replica_count == 1:
            return devices

        device_list = self._parse_device_list(devices)
        if len(device_list) % replica_count != 0:
            raise ValueError(
                f"Stage {stage_id} has {len(device_list)} device(s) for {replica_count} replica(s); "
                "device count must be divisible by replica count"
            )
        devices_per_replica = len(device_list) // replica_count
        start = replica_id * devices_per_replica
        return ",".join(device_list[start : start + devices_per_replica])

    def _set_stage_device_env(self, stage_id: int, env: dict[str, str], devices: str, replica_id: int = 0) -> None:
        replica_devices = self._devices_for_replica(stage_id, devices, replica_id)
        mapped_devices = self._map_stage_devices(stage_id, self.visible_device_list, replica_devices)
        env_var = getattr(current_omni_platform, "device_control_env_var", None)
        if env_var:
            env[env_var] = mapped_devices

    def _build_stage_cmd(self, stage_id: int, *, headless: bool, replica_id: int = 0) -> list[str]:
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
            "--replica-id",
            str(replica_id),
        ]

        if headless:
            cmd.append("--headless")
        else:
            cmd += ["--host", self.host, "--port", str(self.port)]

        cmd += self.serve_args
        return cmd

    def _launch_stage(self, stage_id: int, *, headless: bool, replica_id: int = 0) -> None:
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = self._build_stage_cmd(stage_id, headless=headless, replica_id=replica_id)
        print(f"Launching OmniServerStageCli stage {stage_id} replica {replica_id}: {' '.join(cmd)}")
        # Capture each subprocess's stdout+stderr to a per-stage log file so
        # debugging "Stage N exited before API server ready" doesn't rely on
        # guessing; the file is surfaced in the RuntimeError message.
        log_path = Path(tempfile.gettempdir()) / f"omni_stage_{stage_id}_replica_{replica_id}_{self.master_port}.log"
        self._stage_log_paths = getattr(self, "_stage_log_paths", {})
        stage_key = (stage_id, replica_id)
        self._stage_log_paths[stage_key] = log_path
        log_fh = open(log_path, "w", buffering=1)  # noqa: SIM115 - closed in __exit__
        self._stage_log_files = getattr(self, "_stage_log_files", {})
        self._stage_log_files[stage_key] = log_fh
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        self.stage_procs[stage_key] = proc
        if stage_id == 0 and replica_id == 0:
            self.proc = proc

    def _ensure_stage_processes_alive(self) -> None:
        for (stage_id, replica_id), proc in self.stage_procs.items():
            ret = proc.poll()
            if ret is not None:
                log_path = getattr(self, "_stage_log_paths", {}).get((stage_id, replica_id))
                tail = ""
                if log_path and log_path.exists():
                    try:
                        with open(log_path, encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                        tail = "\n=== Last 60 lines of stage {} replica {} log ({}) ===\n{}".format(
                            stage_id, replica_id, log_path, "".join(lines[-60:]) or "<empty>"
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic only
                        tail = f"\n<failed to read stage log {log_path}: {exc}>"
                raise RuntimeError(
                    f"Stage {stage_id} replica {replica_id} exited with code {ret} before API server became ready.{tail}"
                )

    def _start_server(self) -> None:
        startup_t0 = time.perf_counter()
        ordered_stage_ids = [0, *[stage_id for stage_id in self.stage_ids if stage_id != 0]]

        self._launch_stage(0, headless=False, replica_id=0)
        time.sleep(2)
        self._ensure_stage_processes_alive()

        for stage_id in ordered_stage_ids[1:]:
            for replica_id in range(self.stage_replica_counts.get(stage_id, 1)):
                self._launch_stage(stage_id, headless=True, replica_id=replica_id)

        max_wait = 1200
        start_time = time.time()
        while time.time() - start_time < max_wait:
            self._ensure_stage_processes_alive()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((self.host, self.port))
                if result == 0:
                    startup_s = time.perf_counter() - startup_t0
                    if self.log_stats:
                        print(
                            f"OmniServerStageCli ready on {self.host}:{self.port} "
                            f"(stage-CLI startup took {startup_s:.3f}s)",
                            flush=True,
                        )
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
        for stage_id, replica_id in sorted(log_paths):
            log_path = log_paths[(stage_id, replica_id)]
            if not log_path or not log_path.exists():
                continue
            try:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except Exception as exc:  # pragma: no cover - diagnostic only
                print(f"[OmniServerStageCli] stage {stage_id} replica {replica_id} log read failed: {exc}", flush=True)
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
            print(f"\n=== stage {stage_id} replica {replica_id} log HEAD ({log_path}) ===", flush=True)
            print("".join(head_chunk).rstrip("\n"), flush=True)
            if tail_chunk:
                print(f"\n... [{elided} lines elided] ...", flush=True)
                print(f"\n=== stage {stage_id} replica {replica_id} log TAIL ({log_path}) ===", flush=True)
                print("".join(tail_chunk).rstrip("\n"), flush=True)
            print(f"=== end stage {stage_id} replica {replica_id} log ===\n", flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dump_stage_logs_for_debug()
        for stage_key in sorted(self.stage_procs, reverse=True):
            proc = self.stage_procs[stage_key]
            if proc.poll() is None:
                self._kill_process_tree(proc.pid)
        run_pre_test_cleanup()
        run_post_test_cleanup()
        cleanup_dist_env_and_memory()


@dataclass
class OmniResponse:
    """Decoded multimodal / chat output from the OpenAI SDK or offline runner (not raw ``requests``)."""

    text_content: str | None = None
    audio_data: list[str] | None = None
    audio_content: str | None = None
    audio_format: str | None = None
    audio_bytes: bytes | None = None
    #: End-to-end wall time in **seconds** (``perf_counter`` delta), from just before the
    #: OpenAI client call through response parsing and local post-process (e.g. audio decode).
    e2e_latency: float | None = None
    success: bool = False
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    logprobs: list | None = None
    #: HTTP status + error text for the error-handling path (e.g. validator
    #: rejections); populated when the OpenAI client raises an APIError.
    status_code: int | None = None
    error_message: str | None = None


@dataclass
class DiffusionResponse:
    """Decoded diffusion output from chat completions or offline runner (not raw ``requests``)."""

    text_content: str | None = None
    images: list[Image.Image] | None = None
    audios: list[Any] | None = None
    videos: list[Any] | None = None
    #: End-to-end wall time in **seconds** (``perf_counter`` delta), from just before
    #: ``chat.completions.create`` through local image / audio decode.
    e2e_latency: float | None = None
    success: bool = False


@dataclass
class HttpResponse:
    """Normalized view of a ``requests`` response from :class:`OpenAIClientHandler` HTTP helpers."""

    status_code: int
    success: bool
    error_message: str | None = None
    json_body: dict[str, Any] | list[Any] | None = None


@dataclass
class WebSocketJsonResponse:
    """First JSON object delivered as a text WebSocket frame (streaming endpoints)."""

    json_body: dict[str, Any]


def _merge_http_expectation_kwargs(
    base: dict[str, Any] | None,
    *,
    err_code: int | tuple[int, ...] | list[int] | None = None,
    err_message: str | tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    cfg = dict(base or {})
    if err_code is not None:
        cfg["err_code"] = err_code
    if err_message is not None:
        cfg["err_message"] = err_message
    return cfg


def _merge_ws_expectation_kwargs(
    base: dict[str, Any] | None,
    *,
    err_message: str | tuple[str, ...] | list[str] | None = None,
    ws_json_type: str | None = None,
    ws_error_code: str | None = None,
) -> dict[str, Any]:
    cfg = dict(base or {})
    if err_message is not None:
        cfg["err_message"] = err_message
    if ws_json_type is not None:
        cfg["ws_json_type"] = ws_json_type
    if ws_error_code is not None:
        cfg["ws_error_code"] = ws_error_code
    return cfg


def _run_ws_expectations_from_request_config(cfg: dict[str, Any], resp: WebSocketJsonResponse) -> None:
    jb = resp.json_body
    want_type = cfg.get("ws_json_type")
    if want_type is not None:
        assert jb.get("type") == want_type, (jb, want_type)
    want_code = cfg.get("ws_error_code")
    if want_code is not None:
        assert jb.get("code") == want_code, (jb, want_code)
    err_message = cfg.get("err_message")
    if err_message is not None:
        assert_http_error(resp, err_message=err_message, websocket_json_message=True)


def _merge_diffusion_responses(parts: list[DiffusionResponse]) -> DiffusionResponse:
    """Concatenate images in order; ``e2e_latency`` is wall-clock of the batch (set by caller) or max of parts."""
    merged = DiffusionResponse()
    merged.success = all(p.success for p in parts) and len(parts) > 0
    imgs: list[Image.Image] = []
    for p in parts:
        if p.images:
            imgs.extend(p.images)
    merged.images = imgs if imgs else None
    latencies = [p.e2e_latency for p in parts if p.e2e_latency is not None]
    merged.e2e_latency = max(latencies) if latencies else None
    return merged


class OpenAIClientHandler:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = None,
        api_key: str = "EMPTY",
        run_level: str = None,
        *,
        log_stats: bool = True,
    ):
        if port is None:
            port = get_open_port()
        self.base_url = f"http://{host}:{port}"
        self.client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)
        self.run_level = run_level
        self.log_stats = log_stats

    def _print_client_stat(self, message: str) -> None:
        if self.log_stats:
            print(message, flush=True)

    def _process_stream_omni_response(self, chat_completion, *, wall_start: float) -> OmniResponse:
        """Wall clock from *before* ``chat.completions.create`` through stream drain + local decode."""
        result = OmniResponse()
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
                # Usage is yielded after the last token
                if chunk.usage:
                    result.prompt_tokens = chunk.usage.prompt_tokens
                    if details := getattr(chunk.usage, "prompt_tokens_details", None):
                        result.cached_tokens = details.cached_tokens

            if audio_data:
                merged_seg = _merge_base64_audio_to_segment(audio_data)
                wav_buf = BytesIO()
                merged_seg.export(wav_buf, format="wav")
                result.audio_bytes = wav_buf.getvalue()
            result.text_content = text_content
            result.audio_data = audio_data
            result.e2e_latency = time.perf_counter() - wall_start
            result.success = True
        except Exception as e:
            msg = f"Stream processing error: {str(e)}"
            print(f"Error: {msg}")
        return result

    def _process_non_stream_omni_response(self, chat_completion, *, wall_start: float) -> OmniResponse:
        """Wall clock from *before* ``chat.completions.create`` through response parse + local decode."""
        result = OmniResponse()
        try:
            audio_data = None
            text_content = None
            for choice in chat_completion.choices:
                if hasattr(choice.message, "audio") and choice.message.audio is not None:
                    audio_data = choice.message.audio.data
                if hasattr(choice.message, "content") and choice.message.content is not None:
                    text_content = choice.message.content
            # Extract cached & prompt token counts for prefix caching tests
            usage = getattr(chat_completion, "usage", None)
            if usage:
                result.prompt_tokens = usage.prompt_tokens
                if details := getattr(usage, "prompt_tokens_details", None):
                    result.cached_tokens = details.cached_tokens
            if audio_data:
                result.audio_bytes = base64.b64decode(audio_data)
            result.text_content = text_content
            result.e2e_latency = time.perf_counter() - wall_start
            if chat_completion.choices and chat_completion.choices[0].logprobs is not None:
                result.logprobs = chat_completion.choices[0].logprobs.content
            result.success = True
        except Exception as e:
            msg = f"Non-stream processing error: {str(e)}"
            print(f"Error: {msg}")
        return result

    def _process_diffusion_response(self, chat_completion, *, wall_start: float) -> DiffusionResponse:
        """Wall clock from *before* ``chat.completions.create`` through image decode."""
        result = DiffusionResponse()
        try:
            images = []
            audios = []
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

                # OpenAI audio responses (e.g. AudioX text-to-audio) populate `message.audio`.
                audio_obj = getattr(choice.message, "audio", None)
                audio_b64 = getattr(audio_obj, "data", None) if audio_obj is not None else None
                if audio_b64:
                    audios.append(
                        {
                            "wav_bytes": base64.b64decode(audio_b64),
                            "id": getattr(audio_obj, "id", None),
                            "expires_at": getattr(audio_obj, "expires_at", None),
                        }
                    )
            result.images = images if images else None
            result.audios = audios if audios else None
            result.e2e_latency = time.perf_counter() - wall_start
            result.success = True
        except Exception as e:
            msg = f"Diffusion response processing error: {str(e)}"
            print(f"Error: {msg}")
        return result

    def _http_response_from_requests(self, r: requests.Response) -> HttpResponse:
        payload = _parse_response_json(r)
        ok = 200 <= r.status_code < 300
        return HttpResponse(
            status_code=r.status_code,
            success=ok,
            error_message=None if ok else (r.text[:8000] if r.text else None),
            json_body=payload,
        )

    def send_health_http_request(
        self,
        request_config: dict[str, Any] | None = None,
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/health`` (raw ``requests``).

        ``request_config``: optional ``timeout`` plus optional ``err_code`` / ``err_message`` for
        :func:`~tests.helpers.assertions.assert_http_error` (also as keyword-only args).
        """
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = requests.get(self._build_url("/health"), timeout=float(cfg.get("timeout", 120.0)))
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_models_http_request(
        self,
        request_config: dict[str, Any] | None = None,
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/v1/models``. Optional ``timeout`` and HTTP assertions (see :func:`~tests.helpers.assertions.assert_http_error`)."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = requests.get(
            self._build_url("/v1/models"),
            headers={"Accept": "application/json"},
            timeout=float(cfg.get("timeout", 120.0)),
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_chat_completions_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/chat/completions`` with ``json`` or ``raw_body`` (malformed-body / contract tests)."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/chat/completions", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_completions_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/completions`` with ``json`` or ``raw_body``."""
        # TODO (Alex): A lot of these helpers should be consolidated as they differ only by endpoint
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/completions", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_omni_sleep_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/omni/sleep`` — ``json`` or ``raw_body``, ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/omni/sleep", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_omni_wakeup_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/omni/wakeup``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/omni/wakeup", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_voices_list_http_request(
        self,
        request_config: dict[str, Any] | None = None,
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/v1/audio/voices``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = requests.get(
            self._build_url("/v1/audio/voices"),
            headers={"Accept": "application/json"},
            timeout=float(cfg.get("timeout", 120.0)),
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_voices_create_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/audio/voices`` (multipart): ``data`` / ``files`` / ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_form_endpoint("/v1/audio/voices", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_voices_delete_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """DELETE ``/v1/audio/voices/{name}`` — requires ``name``, optional ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        name = cfg["name"]
        timeout = float(cfg.get("timeout", 120.0))
        path = f"/v1/audio/voices/{quote(str(name), safe='')}"
        r = requests.delete(
            self._build_url(path),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_speech_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/audio/speech`` with ``json`` or ``raw_body``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/audio/speech", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_speech_batch_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/audio/speech/batch``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/audio/speech/batch", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_audio_generate_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/audio/generate``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/audio/generate", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_images_generations_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/images/generations`` — ``json`` or ``raw_body``, ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_json_endpoint("/v1/images/generations", cfg, default_timeout=300.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_images_edits_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/images/edits`` — ``data`` / ``files`` / ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_form_endpoint("/v1/images/edits", cfg, default_timeout=300.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_videos_create_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/videos`` (async job) — multipart ``data`` / ``files``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_form_endpoint("/v1/videos", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_videos_sync_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """POST ``/v1/videos/sync``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = self._post_form_endpoint("/v1/videos/sync", cfg, default_timeout=120.0)
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_videos_list_http_request(
        self,
        request_config: dict[str, Any] | None = None,
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/v1/videos`` — optional ``params``, ``timeout``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        r = requests.get(
            self._build_url("/v1/videos"),
            params=cfg.get("params"),
            headers={"Accept": "application/json"},
            timeout=float(cfg.get("timeout", 120.0)),
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_video_retrieve_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/v1/videos/{video_id}``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        video_id = cfg["video_id"]
        timeout = float(cfg.get("timeout", 120.0))
        r = requests.get(
            self._build_url(f"/v1/videos/{quote(str(video_id), safe='')}"),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_video_delete_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """DELETE ``/v1/videos/{video_id}``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        video_id = cfg["video_id"]
        timeout = float(cfg.get("timeout", 120.0))
        r = requests.delete(
            self._build_url(f"/v1/videos/{quote(str(video_id), safe='')}"),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def send_video_content_http_request(
        self,
        request_config: dict[str, Any],
        *,
        err_code: int | tuple[int, ...] | list[int] | None = None,
        err_message: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[HttpResponse]:
        """GET ``/v1/videos/{video_id}/content``."""
        cfg = _merge_http_expectation_kwargs(
            request_config,
            err_code=err_code,
            err_message=err_message,
        )
        video_id = cfg["video_id"]
        timeout = float(cfg.get("timeout", 120.0))
        r = requests.get(
            self._build_url(f"/v1/videos/{quote(str(video_id), safe='')}/content"),
            timeout=timeout,
        )
        resp = self._http_response_from_requests(r)
        assert_http_error(
            resp,
            err_code=cfg.get("err_code"),
            err_message=cfg.get("err_message"),
        )
        return [resp]

    def _build_ws_url(self, path: str) -> str:
        """Turn HTTP ``base_url`` into ``ws`` / ``wss`` for WebSocket helpers."""
        base = self.base_url.rstrip("/")
        suffix = "/" + path.lstrip("/")
        if base.startswith("http://"):
            return "ws://" + base.removeprefix("http://") + suffix
        if base.startswith("https://"):
            return "wss://" + base.removeprefix("https://") + suffix
        raise ValueError(f"Unsupported base_url for WebSocket: {base!r}")

    def _send_websocket_first_json_request(
        self,
        path: str,
        cfg: dict[str, Any],
    ) -> list[WebSocketJsonResponse]:
        """Connect, optionally send text frames, return first JSON text frame as :class:`WebSocketJsonResponse`.

        ``request_config`` keys:

        - ``send_frames``: optional ``str`` or sequence of ``str`` raw WebSocket text frames (omit when the server
          speaks first, e.g. ``/v1/realtime`` rejection path).
        - ``ws_skip_types``: optional event ``type`` strings to ignore while waiting for the first matching frame
          (e.g. ``["session.created"]`` on ``/v1/realtime``).
        - ``timeout``: seconds to wait for the first inbound text frame (default ``120``).
        - ``ws_max_size``: passed through as ``max_size`` to :func:`websockets.connect` when the key is present.
        """
        send_frames_raw = cfg.get("send_frames")
        if send_frames_raw is None:
            frames: list[str] = []
        elif isinstance(send_frames_raw, str):
            frames = [send_frames_raw]
        else:
            frames = list(send_frames_raw)

        timeout = float(cfg.get("timeout", 120.0))
        uri = self._build_ws_url(path)
        skip_types = set(cfg.get("ws_skip_types") or [])

        connect_kw: dict[str, Any] = {}
        if "ws_max_size" in cfg:
            connect_kw["max_size"] = cfg["ws_max_size"]

        async def _recv_first_json_object() -> WebSocketJsonResponse:
            import websockets

            async with websockets.connect(uri, **connect_kw) as ws:
                for frame in frames:
                    await ws.send(frame)
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    if not isinstance(raw, str):
                        raise AssertionError(f"Expected JSON text frame from {uri}, got {type(raw).__name__}")
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise AssertionError(f"Expected JSON text frame from {uri}, body={raw[:500]!r}") from exc
                    if not isinstance(data, dict):
                        raise AssertionError(f"Expected JSON object from {uri}, got {type(data).__name__}")
                    if skip_types and data.get("type") in skip_types:
                        continue
                    return WebSocketJsonResponse(json_body=data)

        resp = asyncio.run(_recv_first_json_object())
        _run_ws_expectations_from_request_config(cfg, resp)
        return [resp]

    def send_audio_speech_stream_ws_request(
        self,
        request_config: dict[str, Any],
        *,
        err_message: str | tuple[str, ...] | list[str] | None = None,
        ws_json_type: str | None = None,
        ws_error_code: str | None = None,
    ) -> list[WebSocketJsonResponse]:
        """WebSocket ``/v1/audio/speech/stream`` — send ``send_frames`` then read first JSON text frame."""
        cfg = _merge_ws_expectation_kwargs(
            request_config,
            err_message=err_message,
            ws_json_type=ws_json_type,
            ws_error_code=ws_error_code,
        )
        return self._send_websocket_first_json_request("/v1/audio/speech/stream", cfg)

    def send_video_chat_stream_ws_request(
        self,
        request_config: dict[str, Any],
        *,
        err_message: str | tuple[str, ...] | list[str] | None = None,
        ws_json_type: str | None = None,
        ws_error_code: str | None = None,
    ) -> list[WebSocketJsonResponse]:
        """WebSocket ``/v1/video/chat/stream`` — send ``send_frames`` then read first JSON text frame."""
        cfg = _merge_ws_expectation_kwargs(
            request_config,
            err_message=err_message,
            ws_json_type=ws_json_type,
            ws_error_code=ws_error_code,
        )
        return self._send_websocket_first_json_request("/v1/video/chat/stream", cfg)

    def send_realtime_ws_request(
        self,
        request_config: dict[str, Any] | None = None,
        *,
        err_message: str | tuple[str, ...] | list[str] | None = None,
        ws_json_type: str | None = None,
        ws_error_code: str | None = None,
    ) -> list[WebSocketJsonResponse]:
        """WebSocket ``/v1/realtime`` — optional outbound frames, then first JSON text frame (often server-initiated)."""
        cfg = _merge_ws_expectation_kwargs(
            request_config,
            err_message=err_message,
            ws_json_type=ws_json_type,
            ws_error_code=ws_error_code,
        )
        return self._send_websocket_first_json_request("/v1/realtime", cfg)

    def send_omni_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """Chat completions via the OpenAI Python SDK (not raw HTTP)."""
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
        if request_config.get("extra_body"):
            extra_body.update(request_config["extra_body"])

        create_kwargs: dict[str, Any] = {
            "model": request_config.get("model"),
            "messages": request_config.get("messages"),
            "stream": stream,
            "modalities": modalities,
        }
        if "logprobs" in request_config:
            create_kwargs["logprobs"] = request_config["logprobs"]
        if "top_logprobs" in request_config:
            create_kwargs["top_logprobs"] = request_config["top_logprobs"]
        if "stream_options" in request_config:
            create_kwargs["stream_options"] = request_config["stream_options"]
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        if request_num == 1:
            wall_start = time.perf_counter()
            chat_completion = self.client.chat.completions.create(**create_kwargs)
            resp = (
                self._process_stream_omni_response(chat_completion, wall_start=wall_start)
                if stream
                else self._process_non_stream_omni_response(chat_completion, wall_start=wall_start)
            )
            assert_omni_response(resp, request_config, run_level=self.run_level)
            if resp.e2e_latency is not None:
                self._print_client_stat(f"[omni] request#1 success in {resp.e2e_latency:.3f}s")
            else:
                self._print_client_stat("[omni] request#1 completed")
            responses.append(resp)
            return responses

        def _one():
            wall_start = time.perf_counter()
            chat_completion = self.client.chat.completions.create(**create_kwargs)
            return (
                self._process_stream_omni_response(chat_completion, wall_start=wall_start)
                if stream
                else self._process_non_stream_omni_response(chat_completion, wall_start=wall_start)
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
            futures = {executor.submit(_one): i + 1 for i in range(request_num)}
            for future in concurrent.futures.as_completed(futures):
                request_idx = futures[future]
                resp = future.result()
                assert_omni_response(resp, request_config, run_level=self.run_level)
                if resp.e2e_latency is not None:
                    self._print_client_stat(f"[omni] request#{request_idx} success in {resp.e2e_latency:.3f}s")
                else:
                    self._print_client_stat(f"[omni] request#{request_idx} completed")
                responses.append(resp)
        return responses

    def _process_stream_audio_speech_response(
        self, response, *, response_format: str | None = None, wall_start: float
    ) -> OmniResponse:
        """
        Process streaming /v1/audio/speech responses into an OmniResponse.

        This mirrors _process_stream_omni_response but operates on low-level
        audio bytes. Whisper transcription runs in assert_audio_speech_response
        when the run_level requires it.
        """
        result = OmniResponse()

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

            # Populate OmniResponse.
            result.audio_bytes = raw_bytes
            result.e2e_latency = time.perf_counter() - wall_start
            result.success = True
            result.audio_format = getattr(response, "response", None)
            if result.audio_format is not None:
                result.audio_format = result.audio_format.headers.get("content-type", "")

        except Exception as e:
            msg = f"Audio speech stream processing error: {str(e)}"
            print(f"Error: {msg}")

        return result

    def _process_non_stream_audio_speech_response(
        self, response, *, response_format: str | None = None, wall_start: float
    ) -> OmniResponse:
        """
        Process non-streaming /v1/audio/speech responses into an OmniResponse.

        This mirrors _process_non_stream_omni_response but for the binary
        audio payload returned by audio.speech.create.
        """
        result = OmniResponse()

        try:
            # OpenAI non-streaming audio.speech.create returns HttpxBinaryResponseContent (.read() or .content)
            if hasattr(response, "read") and callable(getattr(response, "read")):
                raw_bytes = response.read()
            elif hasattr(response, "content"):
                raw_bytes = response.content  # type: ignore[assignment]
            else:
                raise TypeError(f"Unsupported audio speech response type: {type(response)}")

            result.audio_bytes = raw_bytes
            result.e2e_latency = time.perf_counter() - wall_start
            result.success = True
            result.audio_format = getattr(response, "response", None)
            if result.audio_format is not None:
                result.audio_format = result.audio_format.headers.get("content-type", "")

        except Exception as e:
            msg = f"Audio speech non-stream processing error: {str(e)}"
            print(f"Error: {msg}")

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
          - min_audio_bytes: optional minimum ``len(audio_bytes)`` checked in ``assert_audio_speech_response``
          - status_code: if set, HTTP status is asserted (int or e.g. ``(400, 422)``); uses APIError handling
          - err_message: optional substring(s) to match against error text (``str`` or list/tuple of alternatives;
            see ``assert_audio_speech_response``). If set, uses the same APIError path as ``status_code``.
          - When both ``status_code`` and ``err_message`` are absent (or each is ``None``), the normal request path
            is used (no try/except around ``APIError``).
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
        for key in (
            "task_type",
            "ref_text",
            "ref_audio",
            "language",
            "max_new_tokens",
            "seed",
            "instructions",
            "speed",
            "stream_format",
            "x_vector_only_mode",
        ):
            if key in request_config:
                extra_body[key] = request_config[key]

        responses: list[OmniResponse] = []

        speech_fmt: str | None = None if response_format is omit else str(response_format).lower()

        print(f"[audio.speech] start model={model}, stream={stream}, request_num={request_num}, timeout={timeout:.1f}s")

        # Error validation path: only when at least one of these is set to a non-``None`` value.
        expect_error_handling = (request_config.get("status_code") is not None) or (
            request_config.get("err_message") is not None
        )
        if expect_error_handling and request_num != 1:
            raise ValueError(
                "request_config error validation (status_code / err_message) is only supported when request_num=1"
            )

        if request_num == 1:
            if expect_error_handling:
                # ``status`` and/or ``err_message`` requested: catch APIError (4xx) and (optionally) assert body text;
                # HTTP 200 with JSON error body is handled in ``assert_audio_speech_response``.
                try:
                    if stream:
                        wall_start = time.perf_counter()
                        with self.client.audio.speech.with_streaming_response.create(
                            model=model,
                            input=text_input,
                            response_format=response_format,
                            extra_body=extra_body or None,
                            timeout=timeout,
                            voice=voice,
                        ) as resp:
                            omni_resp = self._process_stream_audio_speech_response(
                                resp, response_format=speech_fmt, wall_start=wall_start
                            )
                    else:
                        wall_start = time.perf_counter()
                        resp = self.client.audio.speech.create(
                            model=model,
                            input=text_input,
                            response_format=response_format,
                            extra_body=extra_body or None,
                            timeout=timeout,
                            voice=voice,
                        )
                        omni_resp = self._process_non_stream_audio_speech_response(
                            resp, response_format=speech_fmt, wall_start=wall_start
                        )
                except APIError as e:
                    sc = getattr(e, "status_code", None)
                    if sc is None:
                        raise
                    omni_resp = OmniResponse(
                        success=False,
                        status_code=sc,
                        error_message=str(e),
                    )
                else:
                    if getattr(omni_resp, "status_code", None) is None:
                        omni_resp.status_code = 200
            elif stream:
                # Use streaming response helper.
                wall_start = time.perf_counter()
                with self.client.audio.speech.with_streaming_response.create(
                    model=model,
                    input=text_input,
                    response_format=response_format,
                    extra_body=extra_body or None,
                    timeout=timeout,
                    voice=voice,
                ) as resp:
                    omni_resp = self._process_stream_audio_speech_response(
                        resp, response_format=speech_fmt, wall_start=wall_start
                    )
            else:
                # Non-streaming response.
                wall_start = time.perf_counter()
                resp = self.client.audio.speech.create(
                    model=model,
                    input=text_input,
                    response_format=response_format,
                    extra_body=extra_body or None,
                    timeout=timeout,
                    voice=voice,
                )
                omni_resp = self._process_non_stream_audio_speech_response(
                    resp, response_format=speech_fmt, wall_start=wall_start
                )

            assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
            if omni_resp.e2e_latency is not None:
                self._print_client_stat(f"[audio.speech] request#1 success in {omni_resp.e2e_latency:.3f}s")
            else:
                self._print_client_stat("[audio.speech] request#1 completed")
            responses.append(omni_resp)
            return responses
        else:
            # request_num > 1: concurrent requests (use same params as single-request path)

            if stream:

                def _stream_task(request_idx: int):
                    wall_start = time.perf_counter()
                    with self.client.audio.speech.with_streaming_response.create(
                        model=model,
                        input=text_input,
                        response_format=response_format,
                        extra_body=extra_body or None,
                        timeout=timeout,
                        voice=voice,
                    ) as resp:
                        result = self._process_stream_audio_speech_response(
                            resp, response_format=speech_fmt, wall_start=wall_start
                        )
                    if result.e2e_latency is not None:
                        self._print_client_stat(
                            f"[audio.speech] request#{request_idx} success in {result.e2e_latency:.3f}s"
                        )
                    else:
                        self._print_client_stat(f"[audio.speech] request#{request_idx} completed")
                    return result

                with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                    futures = {executor.submit(_stream_task, i + 1): i + 1 for i in range(request_num)}
                    for future in concurrent.futures.as_completed(futures):
                        request_idx = futures[future]
                        try:
                            omni_resp = future.result()
                        except Exception as e:
                            print(
                                f"[audio.speech] request#{request_idx} failed "
                                f"(stream={stream}, timeout={timeout:.1f}s): {e!r}"
                            )
                            raise
                        assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
                        responses.append(omni_resp)
            else:

                def _non_stream_task(request_idx: int):
                    wall_start = time.perf_counter()
                    r = self.client.audio.speech.create(
                        model=model,
                        input=text_input,
                        response_format=response_format,
                        extra_body=extra_body or None,
                        timeout=timeout,
                        voice=voice,
                    )
                    result = self._process_non_stream_audio_speech_response(
                        r, response_format=speech_fmt, wall_start=wall_start
                    )
                    if result.e2e_latency is not None:
                        self._print_client_stat(
                            f"[audio.speech] request#{request_idx} success in {result.e2e_latency:.3f}s"
                        )
                    else:
                        self._print_client_stat(f"[audio.speech] request#{request_idx} completed")
                    return result

                with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                    futures = {executor.submit(_non_stream_task, i + 1): i + 1 for i in range(request_num)}
                    for future in concurrent.futures.as_completed(futures):
                        request_idx = futures[future]
                        try:
                            omni_resp = future.result()
                        except Exception as e:
                            print(
                                f"[audio.speech] request#{request_idx} failed "
                                f"(stream={stream}, timeout={timeout:.1f}s): {e!r}"
                            )
                            raise
                        assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
                        responses.append(omni_resp)

        return responses

    def send_diffusion_request(
        self, request_config: dict[str, Any] | list[dict[str, Any]], request_num: int = 1
    ) -> list[DiffusionResponse]:
        """
        Send OpenAI requests for diffusion models.
        If ``extra_body`` has list ``height``/``width``, sends one chat completion per index in parallel
        (scalar h/w, ``num_outputs_per_prompt=1`` each) and merges images in list order.

        Args:
            request_config: A single request configuration dict, or a list of
                request configuration dicts (one request per element)
            request_num: Number of requests to send concurrently, defaults to 1 (single request)
        Returns:
            list[DiffusionResponse]: List of DiffusionResponse objects containing the response data
        """
        responses: list[DiffusionResponse] = []

        def _create_from_config(cfg: dict[str, Any]) -> tuple[Any, float]:
            stream = cfg.get("stream", False)
            if stream:
                raise NotImplementedError("Streaming is not currently implemented for diffusion model e2e test")
            modalities = cfg.get("modalities", omit)  # Most diffusion models don't require modalities param
            eb = cfg.get("extra_body")
            extra = copy.deepcopy(eb) if eb else None
            wall_start = time.perf_counter()
            chat_completion = self.client.chat.completions.create(
                model=cfg.get("model"),
                messages=cfg.get("messages"),
                extra_body=extra,
                modalities=modalities,
            )
            return chat_completion, wall_start

        if isinstance(request_config, list):
            if not request_config:
                raise ValueError("request_config list must not be empty")
            if request_num != 1:
                raise ValueError("request_num is not supported when request_config is a list")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(request_config)) as executor:
                futures = {
                    executor.submit(_create_from_config, cfg): (i + 1, cfg) for i, cfg in enumerate(request_config)
                }
                for future in concurrent.futures.as_completed(futures):
                    request_idx, cfg = futures[future]
                    chat_completion, wall_start = future.result()
                    response = self._process_diffusion_response(chat_completion, wall_start=wall_start)
                    assert_diffusion_response(response, cfg, run_level=self.run_level)
                    if response.e2e_latency is not None:
                        self._print_client_stat(
                            f"[diffusion] request#{request_idx} success in {response.e2e_latency:.3f}s"
                        )
                    else:
                        self._print_client_stat(f"[diffusion] request#{request_idx} completed")
                    responses.append(response)
            return responses

        size_splits = _split_request_config_by_per_output_sizes(request_config)
        if size_splits is not None:
            if request_num != 1:
                raise ValueError(
                    "request_num must be 1 when extra_body height/width are lists (split into concurrent per-size calls)"
                )
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(size_splits)) as executor:
                futures = [executor.submit(_create_from_config, sub) for sub in size_splits]
                chat_completions = [f.result() for f in futures]
            parts = [self._process_diffusion_response(cc, wall_start=ws) for cc, ws in chat_completions]
            merged = _merge_diffusion_responses(parts)
            merged.e2e_latency = time.perf_counter() - t0
            assert_diffusion_response(merged, request_config, run_level=self.run_level)
            if merged.e2e_latency is not None:
                self._print_client_stat(f"[diffusion] request#1 success in {merged.e2e_latency:.3f}s")
            else:
                self._print_client_stat("[diffusion] request#1 completed")
            return [merged]

        if request_num == 1:
            # Send single request
            chat_completion, wall_start = _create_from_config(request_config)
            response = self._process_diffusion_response(chat_completion, wall_start=wall_start)
            assert_diffusion_response(response, request_config, run_level=self.run_level)
            if response.e2e_latency is not None:
                self._print_client_stat(f"[diffusion] request#1 success in {response.e2e_latency:.3f}s")
            else:
                self._print_client_stat("[diffusion] request#1 completed")
            responses.append(response)
            return responses

        # Send concurrent requests for the same request_config
        with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
            futures = {executor.submit(_create_from_config, request_config): i + 1 for i in range(request_num)}
            for future in concurrent.futures.as_completed(futures):
                request_idx = futures[future]
                chat_completion, wall_start = future.result()
                response = self._process_diffusion_response(chat_completion, wall_start=wall_start)
                assert_diffusion_response(response, request_config, run_level=self.run_level)
                if response.e2e_latency is not None:
                    self._print_client_stat(f"[diffusion] request#{request_idx} success in {response.e2e_latency:.3f}s")
                else:
                    self._print_client_stat(f"[diffusion] request#{request_idx} completed")
                responses.append(response)
        return responses

    def send_video_diffusion_request(
        self, request_config: dict[str, Any], request_num: int = 1
    ) -> list[DiffusionResponse]:
        """
        Send native /v1/videos requests: multipart ``form_data`` job create, poll until done, download content.

        For raw HTTP to video routes without polling, use ``send_videos_create_http_request``, etc.
        """
        if request_num != 1:
            raise NotImplementedError("Concurrent video diffusion requests are not currently implemented")

        form_data = request_config.get("form_data")
        if not isinstance(form_data, dict):
            raise ValueError("Video request_config must contain 'form_data'")
        normalized_form_data = {key: str(value) for key, value in form_data.items() if value is not None}
        files: dict[str, tuple[str, BytesIO, str]] = {}
        image_reference = request_config.get("image_reference")
        video_reference = request_config.get("video_reference")
        if image_reference and video_reference:
            raise ValueError("Only one of image_reference or video_reference can be provided")
        if image_reference:
            if image_reference.startswith("data:image"):
                header, encoded = image_reference.split(",", 1)
                content_type = header.split(";")[0].removeprefix("data:")
                extension = content_type.split("/")[-1]
                file_data = base64.b64decode(encoded)
                files["input_reference"] = (f"reference.{extension}", BytesIO(file_data), content_type)
            else:
                normalized_form_data["image_reference"] = json.dumps({"image_url": image_reference})
        if video_reference:
            if video_reference.startswith("data:video"):
                header, encoded = video_reference.split(",", 1)
                content_type = header.split(";")[0].removeprefix("data:")
                extension = content_type.split("/")[-1]
                file_data = base64.b64decode(encoded)
                files["input_reference"] = (f"reference.{extension}", BytesIO(file_data), content_type)
            else:
                normalized_form_data["video_reference"] = json.dumps({"video_url": video_reference})

        result = DiffusionResponse()
        create_url = self._build_url("/v1/videos")
        response = requests.post(
            create_url,
            data=normalized_form_data,
            files=files,
            headers={"Accept": "application/json"},
            timeout=60,
        )
        start_time = time.perf_counter()
        response.raise_for_status()
        job_data = response.json()
        video_id = job_data["id"]
        self._wait_until_video_completed(video_id)
        end_time = time.perf_counter()
        video_content = self._download_video_content(video_id)
        result.success = True
        result.videos = [video_content]
        result.e2e_latency = end_time - start_time
        assert_diffusion_response(result, request_config, run_level=self.run_level)
        if result.e2e_latency is not None:
            self._print_client_stat(f"[diffusion] request#1 success in {result.e2e_latency:.3f}s")
        else:
            self._print_client_stat("[diffusion] request#1 completed")
        return [result]

    def _post_json_endpoint(
        self,
        path: str,
        request_config: dict[str, Any],
        *,
        default_timeout: float,
    ) -> requests.Response:
        url = self._build_url(path)
        timeout = float(request_config.get("timeout", default_timeout))
        if "raw_body" in request_config:
            raw = request_config["raw_body"]
            payload = raw.encode("utf-8") if isinstance(raw, str) else raw
            return requests.post(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=timeout,
            )
        if "json" not in request_config:
            raise ValueError(f"{path} request_config must include 'json' or 'raw_body'")
        return requests.post(
            url,
            json=request_config["json"],
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def _post_form_endpoint(
        self,
        path: str,
        request_config: dict[str, Any],
        *,
        default_timeout: float = 120.0,
    ) -> requests.Response:
        url = self._build_url(path)
        timeout = float(request_config.get("timeout", default_timeout))
        data = request_config.get("data")
        files = request_config.get("files")
        if data is None and not files:
            data = {}
        return requests.post(
            url,
            data=data,
            files=files,
            headers={"Accept": "application/json"} if not files else {"Accept": "application/json"},
            timeout=timeout,
        )

    def send_streaming_video_diffusion_request(
        self,
        request_config: dict[str, Any],
        request_num: int = 1,
        *,
        timeout_seconds: float = 600.0,
    ) -> list[DiffusionResponse]:
        """
        Send a native ``/v1/realtime/video`` WebSocket request and return one
        finalized MP4 artifact assembled from the streamed binary fragments.
        """
        if request_num != 1:
            raise NotImplementedError("Concurrent streaming video diffusion requests are not currently implemented")

        response = asyncio.run(
            self._send_streaming_video_diffusion_request_once(
                request_config,
                timeout_seconds=timeout_seconds,
            )
        )
        assert_diffusion_response(response, request_config, run_level=self.run_level)
        if response.e2e_latency is not None:
            self._print_client_stat(f"[diffusion.stream] request#1 success in {response.e2e_latency:.3f}s")
        else:
            self._print_client_stat("[diffusion.stream] request#1 completed")
        return [response]

    async def _send_streaming_video_diffusion_request_once(
        self,
        request_config: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> DiffusionResponse:
        form_data = request_config.get("form_data")
        if not isinstance(form_data, dict):
            raise ValueError("Video request_config must contain 'form_data'")
        payload: dict[str, Any] = {
            "type": "session.start",
            **{key: value for key, value in form_data.items() if value is not None},
        }
        model = request_config.get("model")
        if model is not None:
            payload["model"] = model
        payload.setdefault("format", "m4s")

        fps = float(payload.get("fps") or 16)
        stream_format = payload["format"]
        url = self._build_ws_url("/v1/realtime/video")

        result = DiffusionResponse()
        chunks: list[bytes] = []
        start_time = time.perf_counter()
        deadline = start_time + timeout_seconds

        import websockets

        async with websockets.connect(url, max_size=None) as websocket:
            await websocket.send(json.dumps(payload))

            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f"Streaming video request did not complete within {timeout_seconds}s")

                message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
                if isinstance(message, bytes):
                    chunks.append(message)
                    continue

                msg = json.loads(message)
                msg_type = msg.get("type")
                if msg_type == "video.start":
                    stream_format = msg.get("format") or stream_format
                    continue
                if msg_type == "session.done":
                    break
                if msg_type == "error":
                    raise RuntimeError(str(msg.get("message", msg)))

        from vllm_omni.diffusion.utils.media_utils import finalize_streaming_video_bytes
        from vllm_omni.entrypoints.openai.video_api_utils import StreamingVideoFormat

        streamed_bytes = b"".join(chunks)
        if not streamed_bytes:
            raise RuntimeError("Streaming video request completed without binary video chunks")
        result.videos = [
            finalize_streaming_video_bytes(
                streamed_bytes,
                input_format=cast(StreamingVideoFormat, stream_format),
                fps=fps,
            )
        ]
        result.e2e_latency = time.perf_counter() - start_time
        result.success = True
        return result

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
        startup_t0 = time.perf_counter()
        cleanup_dist_env_and_memory()
        run_pre_test_cleanup()
        run_post_test_cleanup()
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
        startup_s = time.perf_counter() - startup_t0
        if log_stats:
            print(f"OmniRunner startup took {startup_s:.3f}s (model={model_name})", flush=True)

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
            from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
                Qwen3TTSPromptEmbedsBuilder,
            )

            if model_name not in _cache:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
                cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)
                _cache[model_name] = (tok, getattr(cfg, "talker_config", None))

            tok, tcfg = _cache[model_name]
            task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]
            return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
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

        # Qwen-TTS: follow examples/offline_inference/text_to_speech/qwen3_tts/end2end.py style.
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
    ) -> list[OmniRequestOutput]:
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
    ) -> list[OmniRequestOutput]:
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
        run_pre_test_cleanup()
        run_post_test_cleanup()
        cleanup_dist_env_and_memory()


class OmniRunnerHandler:
    def __init__(self, omni_runner: OmniRunner):
        self.runner = omni_runner

    def _process_omni_output(self, outputs: list[OmniRequestOutput]) -> OmniResponse:
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
            msg = f"Output processing error: {str(e)}"
            result.success = False
            print(f"Error: {msg}")
        return result

    def _process_diffusion_output(self, outputs: list[OmniRequestOutput]) -> DiffusionResponse:
        result = DiffusionResponse()
        output = outputs[0]
        if isinstance(output.images[0], list):
            # Returning frames of images as a video
            result.videos = output.images
        else:
            # Returning actual images
            result.images = output.images
        # [TODO] Add audio processing when tests are introduced
        result.success = True
        return result

    def send_omni_request(self, request_config: dict[str, Any] | None = None) -> OmniResponse:
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
        response = self._process_omni_output(outputs)
        assert_omni_response(response, request_config, run_level="core_model")
        return response

    def send_diffusion_request(self, request_config: dict[str, Any]) -> DiffusionResponse:
        prompt = request_config.get("prompt")
        if prompt is None:
            prompts = request_config.get("prompts")
            if not prompts:
                raise ValueError("request_config must contain a prompt")
            if len(prompts) > 1:
                raise ValueError(
                    "In the current internal data structure, "
                    "only one prompt is supported for diffusion requests. "
                    "Because one prompt can contain multiple images or videos, "
                    "the current internal data structure is ambiguous when multiple prompts are provided."
                )
            prompt = prompts[0]

        # Sync previous `extra_body` field with sampling params object
        sampling_params: OmniDiffusionSamplingParams | None = request_config.get("sampling_params")
        if sampling_params is None:
            extra_body = request_config.get("extra_body", {})
            if extra_body:
                sampling_params = OmniDiffusionSamplingParams(**extra_body)
        else:
            extra_body = asdict(sampling_params)
            request_config["extra_body"] = extra_body
        if not extra_body:
            logger.warning("No sampling params provided in request_config, will skip output assertion")

        negative_prompt = extra_body.get("negative_prompt") or request_config.get("negative_prompt")
        videos = request_config.get("videos")
        images = request_config.get("images")
        audios = request_config.get("audios")
        # Full dict (e.g. image + mask_image for inpainting) or partial; merged with top-level image/video keys.
        extra_multi_modal = request_config.get("multi_modal_data")
        modalities = request_config.get("modalities")  # only used by limited models. Do not add default value here

        prompt_object = OmniTextPrompt(prompt=prompt)
        if negative_prompt:
            prompt_object["negative_prompt"] = negative_prompt
        multi_modal: dict = {}
        if extra_multi_modal is not None:
            multi_modal.update(dict(extra_multi_modal))
        if videos is not None:
            multi_modal["video"] = videos
        if images is not None:
            multi_modal["image"] = images
        if audios is not None:
            multi_modal["audio"] = audios
        if multi_modal:
            prompt_object["multi_modal_data"] = multi_modal
        if modalities:
            prompt_object["modalities"] = modalities  # pyright: ignore[reportGeneralTypeIssues]

        start_time = time.perf_counter()
        response = self.runner.generate([prompt_object], [sampling_params] if sampling_params else None)
        end_time = time.perf_counter()

        response = self._process_diffusion_output(response)
        response.e2e_latency = end_time - start_time
        assert_diffusion_response(response, request_config, run_level="core_model")
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
            raise AssertionError("No audio output from pipeline")

        audio_data = mm_out.get("audio")
        if audio_data is None:
            raise AssertionError("No audio tensor in multimodal output")

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


# ---------------------------------------------------------------------------
# Pytest fixture helpers (used from ``tests.helpers.fixtures.runtime``; live here
# to avoid importing ``tests.helpers.runtime`` from the plugin module at import time).
# ---------------------------------------------------------------------------


def iter_omni_server(
    request: Any,
    run_level: str,
    model_prefix: str,
    omni_fixture_lock: threading.Lock,
) -> Generator[Any, Any, None]:
    """Start/stop an Omni HTTP server; used by ``omni_server`` / ``omni_server_function`` fixtures."""
    from tests.helpers.stage_config import stage_config_path_for_run_level

    with omni_fixture_lock:
        params: OmniServerParams = request.param
        # For now, when a tiny model is substituted, we preserve the original model
        # name via --served-model-name (so that the server still accepts requests with
        # the original name). We also do the same for server.model so that tests reading
        # server.model send the correct name in requests.
        #
        # TODO: core models on this path currently do not clean up tiny models, although
        # tiny model paths are deterministic, so it's not a huge footprint. Still, it would
        # be ideal to cleanup consistently everywhere.
        original_model = model_prefix + params.model
        model = original_model
        if run_level == "core_model" and request.node.get_closest_marker("diffusion"):
            model = resolve_tiny_model_path(model)
        port = params.port
        stage_config_path = stage_config_path_for_run_level(params.stage_config_path, run_level)

        server_args = params.server_args or []
        if model != original_model:
            server_args = [*server_args, "--served-model-name", original_model]
        if params.use_omni and params.stage_init_timeout is not None:
            server_args = [*server_args, "--stage-init-timeout", str(params.stage_init_timeout)]
        else:
            server_args = [*server_args, "--stage-init-timeout", "600"]
        if params.init_timeout is not None:
            server_args = [*server_args, "--init-timeout", str(params.init_timeout)]
        else:
            server_args = [*server_args, "--init-timeout", "900"]
        # ``omni_server`` / ``omni_server_function``: match ``serve`` (``--disable-log-stats`` wins).
        if "--disable-log-stats" not in server_args and "--log-stats" not in server_args:
            server_args = [*server_args, "--log-stats"]
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
                if model != original_model:
                    server.model = original_model
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
                if model != original_model:
                    server.model = original_model
                print("OmniServer started successfully")
                yield server
                print("OmniServer stopping...")

        print("OmniServer stopped")


def iter_omni_runner(
    request: Any,
    model_prefix: str,
    run_level: str,
    omni_fixture_lock: threading.Lock,
) -> Generator[Any, None, None]:
    """Yield an :class:`OmniRunner`; used by ``omni_runner`` / ``omni_runner_function`` fixtures."""
    from tests.helpers.stage_config import stage_config_path_for_run_level

    with omni_fixture_lock:
        param = request.param
        if not isinstance(param, (tuple, list)) or len(param) not in (2, 3):
            raise ValueError(
                "omni_runner param must be (model, stage_config_path) or "
                "(model, stage_config_path, extra_omni_kwargs_dict)"
            )
        if len(param) == 2:
            model, stage_config_path = param[0], param[1]
            extra_omni_kwargs: dict = {}
        else:
            model, stage_config_path, extra = param[0], param[1], param[2]
            extra_omni_kwargs = dict(extra) if extra is not None else {}
        stage_config_path = stage_config_path_for_run_level(stage_config_path, run_level)
        model = model_prefix + model
        if run_level == "core_model" and request.node.get_closest_marker("diffusion"):
            model = resolve_tiny_model_path(model)
        with OmniRunner(model, seed=42, stage_configs_path=stage_config_path, **extra_omni_kwargs) as runner:
            print("OmniRunner started successfully")
            yield runner
            print("OmniRunner stopping...")

        print("OmniRunner stopped")


__all__ = [
    "DiffusionResponse",
    "HttpResponse",
    "WebSocketJsonResponse",
    "OmniResponse",
    "OmniRunner",
    "OmniRunnerHandler",
    "OmniServer",
    "OmniServerParams",
    "OmniServerStageCli",
    "OpenAIClientHandler",
    "get_open_port",
    "dummy_messages_from_mix_data",
]
