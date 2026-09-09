# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Server and offline runner runtime primitives for tests.

Request clients and response types live in ``tests.helpers.client``;
this module re-exports them for backward-compatible imports.
"""

import errno
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import psutil
import yaml
from vllm import TextPrompt
from vllm.logger import init_logger

from tests.helpers.clean import cleanup_test_environment
from tests.helpers.media import (
    release_audio_transcriber,
)
from tests.helpers.stage_config import load_stage_ids, load_stage_replica_counts
from tests.model_tests.diffusion.utils import resolve_tiny_model_path
from vllm_omni.config.stage_config import resolve_deploy_yaml
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

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


def get_distributed_init_method(prefix: str = "torch_dist_init_") -> str:
    """Return a ``file://`` init_method for a ``torch.distributed`` process group.

    Args:
        prefix: Prefix for the temporary rendezvous filename. Defaults to
            ``torch_dist_init_``.

    Returns:
        An init_method string for ``torch.distributed.init_process_group``.
    """
    with tempfile.NamedTemporaryFile(prefix=prefix) as f:
        return f"file://{f.name}"


def dummy_messages_from_mix_data(
    system_prompt: dict[str, Any] | None = None,
    video_data_url: Any = None,
    audio_data_url: Any = None,
    image_data_url: Any = None,
    content_text: str | None = None,
):
    """Create messages with video、image、audio data URL for OpenAI API."""
    if content_text is not None:
        content: list[dict[str, Any]] = [{"type": "text", "text": content_text}]
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
        cleanup_test_environment()
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
            os.waitpid(proc.pid, getattr(os, "WNOHANG", 0))
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
        try:
            self._start_server()
        except BaseException:
            # ``__exit__`` is not invoked when ``__enter__`` raises; roll back
            # any processes / log handles already launched.
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            self._kill_process_tree(self.proc.pid)
        cleanup_test_environment()


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
        self.stage_ids = stage_ids or load_stage_ids(resolved_cfg)
        if 0 not in self.stage_ids:
            raise ValueError(f"Stage CLI test requires stage_id=0 in config: {stage_config_path}")
        self.stage_replica_counts = load_stage_replica_counts(resolved_cfg)
        self.stage_procs: dict[tuple[int, int], subprocess.Popen] = {}
        self._stage_log_paths: dict[tuple[int, int], Path] = {}
        self._stage_log_files: dict[tuple[int, int], Any] = {}
        self.proc = None

    def _build_stage_cmd(self, stage_id: int, *, headless: bool, replica_id: int = 0) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--omni",
            "--deploy-config",
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
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = self._build_stage_cmd(stage_id, headless=headless, replica_id=replica_id)
        print(f"Launching OmniServerStageCli stage {stage_id} replica {replica_id}: {' '.join(cmd)}")
        # Capture each subprocess's stdout+stderr to a per-stage log file so
        # debugging "Stage N exited before API server ready" doesn't rely on
        # guessing; the file is surfaced in the RuntimeError message.
        log_path = Path(tempfile.gettempdir()) / f"omni_stage_{stage_id}_replica_{replica_id}_{self.master_port}.log"
        stage_key = (stage_id, replica_id)
        self._stage_log_paths[stage_key] = log_path
        log_fh = open(log_path, "w", buffering=1)  # noqa: SIM115 - closed in ``__exit__``
        self._stage_log_files[stage_key] = log_fh
        proc = subprocess.Popen(
            cmd,
            env=env,
            # Must be repo root (not tests/): after Docker deps-cache installs an empty
            # vllm_omni stub into site-packages, cwd on sys.path[0] is what makes
            # ``python -m vllm_omni.entrypoints...`` resolve the real package.
            cwd=_omni_subprocess_cwd(),
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
                log_path = self._stage_log_paths.get((stage_id, replica_id))
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
        log_paths = self._stage_log_paths
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

        # Close per-stage log handles; delete temp files unless VLLM_OMNI_KEEP_LOG is set.
        keep_logs = os.environ.get("VLLM_OMNI_KEEP_LOG", "").lower() in ("1", "true", "yes")
        log_files = self._stage_log_files
        log_paths = self._stage_log_paths
        for stage_key in set(log_files) | set(log_paths):
            log_fh = log_files.get(stage_key)
            if log_fh is not None:
                try:
                    log_fh.close()
                except Exception:
                    pass
            if not keep_logs:
                log_path = log_paths.get(stage_key)
                if log_path is not None:
                    try:
                        Path(log_path).unlink(missing_ok=True)
                    except Exception:
                        pass
        self._stage_log_files = {}
        if not keep_logs:
            self._stage_log_paths = {}

        cleanup_test_environment()


class OmniRunner:
    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        stage_init_timeout: int = 600,
        # Bumped from 900s -> 1800s to give CI cold-cache loads of large
        # diffusion models enough headroom (Buildkite #8418 hit a 6-second
        # overrun loading Tongyi-MAI/Z-Image-Turbo: weights alone took 690s,
        # the full stage was ready at ~896s, but the orchestrator wrapper
        # finished at ~906s, just past the previous 900s ceiling). Engine
        # production default in AsyncOmniEngine remains 600s; this only
        # affects the test runner wrapper.
        init_timeout: int = 1800,
        log_stats: bool = False,
        deploy_config: str | None = None,
        **kwargs,
    ) -> None:
        startup_t0 = time.perf_counter()
        cleanup_test_environment()
        self.model_name = model_name
        self.seed = seed
        self._prompt_len_estimate_cache: dict[str, Any] = {}
        self.omni: Any = None
        try:
            from vllm_omni.entrypoints.omni import Omni

            self.omni = Omni(
                model=model_name,
                log_stats=log_stats,
                stage_init_timeout=stage_init_timeout,
                init_timeout=init_timeout,
                deploy_config=deploy_config,
                **kwargs,
            )
            startup_s = time.perf_counter() - startup_t0
            if log_stats:
                print(f"OmniRunner startup took {startup_s:.3f}s (model={model_name})", flush=True)
        except BaseException:
            # ``with OmniRunner(...)`` never reaches ``__enter__``/``__exit__``
            # when construction fails after worker processes have started.
            self.__exit__(None, None, None)
            raise

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
        # Default wrapping for Qwen-style models (bos/eos around the placeholder).
        audio_fmt = "<|audio_bos|>{p}<|audio_eos|>"
        image_fmt = "<|vision_bos|>{p}<|vision_eos|>"
        video_fmt = "<|vision_bos|>{p}<|vision_eos|>"
        if "Qwen3-Omni-30B-A3B-Instruct" in self.model_name:
            video_padding_token = "<|video_pad|>"
            image_padding_token = "<|image_pad|>"
            audio_padding_token = "<|audio_pad|>"
        elif "Ming-flash-omni" in self.model_name:
            video_padding_token = "<VIDEO>"
            image_padding_token = "<IMAGE>"
            audio_padding_token = "<AUDIO>"
        elif "MiniCPM" in self.model_name:
            # MiniCPM-o expects the bare placeholder literals (with parens and ./),
            # not Qwen-style bos/eos wrapping.
            video_padding_token = "(<video>./</video>)"
            image_padding_token = "(<image>./</image>)"
            audio_padding_token = "(<audio>./</audio>)"
            audio_fmt = "{p}"
            image_fmt = "{p}"
            video_fmt = "{p}"
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
                        user_content += audio_fmt.format(p=audio_padding_token)
                    multi_modal_data["audio"] = audio
                else:
                    user_content += audio_fmt.format(p=audio_padding_token)
                    multi_modal_data["audio"] = audio
            image = images_list[i]
            if image is not None:
                if isinstance(image, list):
                    for _ in image:
                        user_content += image_fmt.format(p=image_padding_token)
                    multi_modal_data["image"] = image
                else:
                    user_content += image_fmt.format(p=image_padding_token)
                    multi_modal_data["image"] = image
            video = videos_list[i]
            if video is not None:
                if isinstance(video, list):
                    for _ in video:
                        user_content += video_fmt.format(p=video_padding_token)
                    multi_modal_data["video"] = video
                else:
                    user_content += video_fmt.format(p=video_padding_token)
                    multi_modal_data["video"] = video
            user_content += prompt_text

            full_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            prompt_dict: dict[str, Any] = {"prompt": full_prompt}
            if multi_modal_data:
                prompt_dict["multi_modal_data"] = multi_modal_data
            if modalities:
                prompt_dict["modalities"] = modalities
            if mm_processor_kwargs:
                prompt_dict["mm_processor_kwargs"] = mm_processor_kwargs
            omni_inputs.append(prompt_dict)
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
        omni = getattr(self, "omni", None)
        if omni is not None and hasattr(omni, "close"):
            omni.close()
        self._cleanup_process()
        cleanup_test_environment()


# ---------------------------------------------------------------------------
# Pytest fixture helpers (used from ``tests.helpers.fixtures.runtime``; live here
# to avoid importing ``tests.helpers.runtime`` from the plugin module at import time).
# ---------------------------------------------------------------------------


@contextmanager
def _whisper_device_free_around():
    """Keep the Whisper judge off the accelerator around a server/runner lifecycle.

    Released on entry so the next instance -- e.g. the next parametrization in
    the same module -- initializes on a clean device, and again on exit
    (including when a test raises) so it does not linger for the instance after.
    """
    release_audio_transcriber()
    try:
        yield
    finally:
        release_audio_transcriber()


def get_model_prefix() -> str:
    """Return ``MODEL_PREFIX`` with a trailing slash, or ``""`` when unset."""
    prefix = os.environ.get("MODEL_PREFIX", "")
    return f"{prefix.rstrip('/')}/" if prefix else ""


def iter_omni_server(
    request: Any,
    run_level: str,
    omni_fixture_lock: threading.Lock,
) -> Generator[Any, Any, None]:
    """Start/stop an Omni HTTP server; used by ``omni_server`` / ``omni_server_function`` fixtures."""
    from tests.helpers.stage_config import stage_config_path_for_run_level

    model_prefix = get_model_prefix()
    with omni_fixture_lock, _whisper_device_free_around():
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
            server_args += ["--deploy-config", stage_config_path]

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
                server_args += ["--deploy-config", stage_config_path]

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
    run_level: str,
    omni_fixture_lock: threading.Lock,
) -> Generator[Any, None, None]:
    """Yield an :class:`OmniRunner`; used by ``omni_runner`` / ``omni_runner_function`` fixtures."""
    from tests.helpers.stage_config import stage_config_path_for_run_level

    model_prefix = get_model_prefix()
    with omni_fixture_lock, _whisper_device_free_around():
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
        with OmniRunner(model, seed=42, deploy_config=stage_config_path, **extra_omni_kwargs) as runner:
            print("OmniRunner started successfully")
            yield runner
            print("OmniRunner stopping...")

        print("OmniRunner stopped")


# ─────────────────────────────────────────────────────────────────────
# π0 (Pi-Zero) OpenPI websocket policy client
#
# Minimal client for the OpenPI realtime robot protocol
# (``/v1/realtime/robot/openpi``): connect → read handshake metadata → send
# observation → receive an action chunk. Used by the π0 online-serving e2e
# (tests/e2e/online_serving/test_pi0_expansion.py). π0's observation is 3 RGB
# cameras + proprioceptive state + a language prompt, built as synthetic numpy
# frames here (no video assets).
# ─────────────────────────────────────────────────────────────────────
PI0_OPENPI_PATH = "/v1/realtime/robot/openpi"
PI0_OPENPI_DEFAULT_PROMPT = "pick up the red block and place it in the bin"
# π0 / pi0_base camera identities (must match the server's image_feature_keys,
# i.e. vllm_omni/deploy/pi0.yaml, whose image_key_map is empty).
PI0_CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
PI0_ACTION_HORIZON = 50
PI0_ACTION_DIM = 32
PI0_STATE_DIM = 32
PI0_IMAGE_SIZE = 224


def pi0_openpi_require_dependencies() -> None:
    """Raise ModuleNotFoundError if the OpenPI websocket client deps are missing."""
    missing = []
    try:
        import websockets.sync.client  # noqa: F401
    except ImportError:
        missing.append("websockets")
    try:
        from openpi_client import msgpack_numpy  # noqa: F401
    except ImportError:
        missing.append("openpi-client")
    if missing:
        raise ModuleNotFoundError(f"π0 OpenPI test dependencies are missing: {', '.join(missing)}")


def _pi0_decode_action_response(response: bytes | str) -> np.ndarray:
    from openpi_client import msgpack_numpy

    if isinstance(response, str):
        raise RuntimeError(f"Inference failed: {response}")
    decoded = msgpack_numpy.unpackb(response)
    if isinstance(decoded, dict) and decoded.get("type") == "error":
        raise RuntimeError(f"Inference failed: {decoded.get('message', decoded)}")
    return np.asarray(decoded, dtype=np.float32)


def pi0_make_dummy_obs(*, prompt: str, session_id: str, image_size: int = PI0_IMAGE_SIZE) -> dict[str, Any]:
    """A single π0 observation: 3 blank cameras (HWC uint8) + zero state + prompt."""
    obs: dict[str, Any] = {cam: np.zeros((image_size, image_size, 3), dtype=np.uint8) for cam in PI0_CAMERA_KEYS}
    obs["state"] = np.zeros(PI0_STATE_DIM, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def pi0_openpi_run_policy_session(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = PI0_OPENPI_PATH,
    prompt: str = PI0_OPENPI_DEFAULT_PROMPT,
    session_id: str | None = None,
    num_steps: int = 2,
) -> dict[str, Any]:
    """Connect, read handshake metadata, send ``num_steps`` observations."""
    import uuid

    import websockets.sync.client as websockets_client
    from openpi_client import msgpack_numpy

    session_id = session_id or str(uuid.uuid4())
    uri = f"ws://{host}:{port}{path}"
    packer = msgpack_numpy.Packer()
    conn = websockets_client.connect(uri, compression=None, max_size=None, ping_interval=300, ping_timeout=3600)
    try:
        metadata = msgpack_numpy.unpackb(conn.recv())
        if not isinstance(metadata, dict):
            raise TypeError(f"Expected dict metadata from server, got {type(metadata)!r}")
        actions = []
        for _ in range(num_steps):
            payload = pi0_make_dummy_obs(prompt=prompt, session_id=session_id)
            payload["endpoint"] = "infer"
            conn.send(packer.pack(payload))
            actions.append(_pi0_decode_action_response(conn.recv()))
        return {"metadata": dict(metadata), "actions": actions, "session_id": session_id}
    finally:
        conn.close()


def pi0_openpi_validate_session_result(
    result: dict[str, Any],
    *,
    expected_action_horizon: int = PI0_ACTION_HORIZON,
    expected_action_dim: int = PI0_ACTION_DIM,
) -> None:
    """Assert the handshake metadata + every returned action chunk for π0."""
    metadata = result["metadata"]
    required_keys = (
        "image_resolution",
        "needs_wrist_camera",
        "needs_session_id",
        "action_space",
        "action_horizon",
        "action_dim",
    )
    missing = [key for key in required_keys if key not in metadata]
    if missing:
        raise AssertionError(f"Missing π0 metadata keys: {missing}")

    if tuple(metadata["image_resolution"]) != (PI0_IMAGE_SIZE, PI0_IMAGE_SIZE):
        raise AssertionError(f"Unexpected image_resolution: {metadata['image_resolution']!r}")
    if not metadata["needs_wrist_camera"]:
        raise AssertionError("π0 test expects needs_wrist_camera=True")
    if metadata["needs_session_id"]:
        raise AssertionError("π0 is stateless; needs_session_id must be False")
    if metadata["action_space"] != "joint_position":
        raise AssertionError(f"Unexpected action_space: {metadata['action_space']!r}")
    if int(metadata["action_horizon"]) != expected_action_horizon:
        raise AssertionError(f"Unexpected action_horizon: {metadata['action_horizon']!r}")
    if int(metadata["action_dim"]) != expected_action_dim:
        raise AssertionError(f"Unexpected action_dim: {metadata['action_dim']!r}")

    actions = result["actions"]
    if not actions:
        raise AssertionError("No actions returned from the server")
    for index, action in enumerate(actions):
        if action.shape != (expected_action_horizon, expected_action_dim):
            raise AssertionError(
                f"Action {index} shape mismatch: expected "
                f"{(expected_action_horizon, expected_action_dim)}, got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise AssertionError(f"Action {index} contains non-finite values")


# Re-export client types for backward-compatible imports.
from tests.helpers.client import (  # noqa: E402
    DREAMZERO_ACTION_DIM,
    DREAMZERO_ACTION_HORIZON,
    DREAMZERO_CAMERA_FILES,
    DiffusionResponse,
    HttpResponse,
    OfflineOmniClient,
    OmniResponse,
    OmniRunnerHandler,
    OnlineOmniClient,
    OpenAIClientHandler,
    OpenPIWebSocketResponse,
    OpenPIWebSocketSession,
    WebSocketJsonResponse,
    build_dreamzero_demo_observations,
    build_openpi_droid_observation,
    load_dreamzero_camera_frames,
)

__all__ = [
    "DiffusionResponse",
    "HttpResponse",
    "WebSocketJsonResponse",
    "OpenPIWebSocketResponse",
    "build_openpi_droid_observation",
    "build_dreamzero_demo_observations",
    "DREAMZERO_ACTION_DIM",
    "DREAMZERO_ACTION_HORIZON",
    "DREAMZERO_CAMERA_FILES",
    "load_dreamzero_camera_frames",
    "OpenPIWebSocketSession",
    "OmniResponse",
    "OmniRunner",
    "OfflineOmniClient",
    "OmniServer",
    "OmniServerParams",
    "OmniServerStageCli",
    "OnlineOmniClient",
    "OpenAIClientHandler",
    "OmniRunnerHandler",
    "get_model_prefix",
    "get_open_port",
    "dummy_messages_from_mix_data",
    "pi0_openpi_require_dependencies",
    "pi0_openpi_run_policy_session",
    "pi0_openpi_validate_session_result",
    "pi0_make_dummy_obs",
]
