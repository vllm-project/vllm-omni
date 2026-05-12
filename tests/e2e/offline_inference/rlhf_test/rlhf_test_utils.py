# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");

"""Self-contained inlined replica of every ``verl`` / ``verl_omni`` symbol
that the reference test
``verl-omni/tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py``
transitively touches. No imports from ``verl`` or ``verl_omni``.

Top-level classes / helpers (verbatim where it matters):

* ``RolloutMode``                  — verl.workers.rollout.replica
* ``DiffusionOutput``              — verl_omni.workers.rollout.replica
* ``normalize_token_ids``          — verl.utils.tokenizer
* ``omega_conf_to_dataclass``      — verl.utils.config
* ``build_cli_args_from_config``   — verl.workers.rollout.vllm_rollout.utils
* ``import_external_libs``         — verl.utils.import_utils
* ``get_free_port``                — verl.utils.net_utils
* ``DistProfiler`` (no-op stub)    — verl.utils.profiler
* ``BaseConfig``                   — verl.base_config
* ``DiffusionPipelineConfig``      — verl_omni.workers.config.diffusion.rollout
* ``DiffusionRolloutConfig``       — verl_omni.workers.config.diffusion.rollout
* ``DiffusionModelConfig``         — verl_omni.workers.config.diffusion.model
* ``VllmOmniPipelineBase``         — verl_omni.pipelines.model_base
* ``vLLMHttpServer``               — verl.workers.rollout.vllm_rollout.vllm_async_server
* ``vLLMOmniHttpServer``           — verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pprint import pprint
from typing import Any, Optional, Union

import ray
import torch
import torchvision.transforms as T
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

import vllm_omni.entrypoints.cli.serve
from vllm.entrypoints.openai.api_server import build_app
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# In-repo test pipeline / worker extension dotted paths
# (the production reads these via ``import_external_libs(external_lib)``
#  side-effecting ``@VllmOmniPipelineBase.register(...)``. Our in-repo
#  test pipeline is not decorated, so we pre-seed the registry below.)
# ---------------------------------------------------------------------

CUSTOM_PIPELINE_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline."
    "qwen_image_pipeline_with_logprob.QwenImagePipelineWithLogProbForTest"
)
WORKER_EXTENSION_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline."
    "worker_extension.vLLMOmniColocateWorkerExtensionForTest"
)


# =====================================================================
# verl.utils.tokenizer.normalize_token_ids
# =====================================================================


def normalize_token_ids(tokenized_output) -> list[int]:
    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], (list, tuple)):
        token_ids = list(token_ids[0])
    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like, got {type(token_ids).__name__}")

    out: list[int] = []
    for tid in token_ids:
        if hasattr(tid, "item"):
            tid = tid.item()
        out.append(int(tid))
    return out


# =====================================================================
# verl.utils.import_utils.import_external_libs
# =====================================================================


def import_external_libs(external_libs=None) -> None:
    if external_libs is None:
        return
    import importlib

    if not isinstance(external_libs, list):
        external_libs = [external_libs]
    for lib in external_libs:
        importlib.import_module(lib)


# =====================================================================
# verl.utils.net_utils.get_free_port (slim variant; no with_alive_sock)
# =====================================================================


def get_free_port(host: str = "127.0.0.1", with_alive_sock: bool = False):
    """Pick an ephemeral free TCP port. When ``with_alive_sock`` is True the
    underlying socket is returned along with the port so the caller can hold
    it open until the consumer binds (this is what verl's real
    ``get_free_port`` does to avoid TOCTOU). The test never reads the port
    back \u2014 single-node DP=1 \u2014 so the live socket is just stored on the
    server instance to be closed on shutdown / GC.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, 0))
    port = s.getsockname()[1]
    if with_alive_sock:
        return port, s
    s.close()
    return port


# =====================================================================
# verl.utils.profiler.DistProfiler (no-op stub: test passes no profiler)
# =====================================================================


class DistProfiler:
    def __init__(self, replica_rank: int, config=None, tool_config=None):
        self.replica_rank = replica_rank
        self.config = config
        self.tool_config = tool_config

    def start(self, *a, **kw):
        pass

    def stop(self, *a, **kw):
        pass


def build_vllm_profiler_args(config, tool_config, replica_rank: int) -> dict:
    """Verbatim no-op: when ``config`` is None (test never enables profiler)
    the production helper also returns an empty dict."""
    return {}


# =====================================================================
# verl.workers.rollout.replica.RolloutMode
# =====================================================================


class RolloutMode(Enum):
    HYBRID = "hybrid"
    COLOCATED = "colocated"
    STANDALONE = "standalone"


# =====================================================================
# verl_omni.workers.rollout.replica.DiffusionOutput
# =====================================================================


class DiffusionOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    diffusion_output: Any
    log_probs: Optional[Any] = None
    stop_reason: Optional[str] = None
    num_preempted: Optional[int] = None
    extra_fields: dict[str, Any] = {}


# =====================================================================
# verl.utils.config.omega_conf_to_dataclass
# (tolerant variant: ignores extra keys like ``_target_`` / ``free_cache_engine``)
# =====================================================================


def omega_conf_to_dataclass(config, dataclass_type: Optional[type[Any]] = None) -> Any:
    if not config:
        return dataclass_type() if dataclass_type is not None else None
    if not isinstance(config, (DictConfig, ListConfig, dict, list)):
        return config

    assert dataclass_type is not None and is_dataclass(dataclass_type)

    raw = (
        OmegaConf.to_container(config, resolve=True)
        if isinstance(config, (DictConfig, ListConfig))
        else config
    )

    valid = {f.name: f for f in fields(dataclass_type)}
    kwargs: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in valid:
            continue
        f_def = valid[k]
        # Recurse into nested dataclass fields (e.g. ``pipeline``).
        f_type = f_def.type
        if isinstance(f_type, str):
            # Forward-ref string; resolve via the enclosing dataclass module
            # globals. The only two we use are DiffusionPipelineConfig in
            # DiffusionRolloutConfig.
            f_type = globals().get(f_type, f_type)
        if isinstance(f_type, type) and is_dataclass(f_type) and isinstance(v, dict):
            v = omega_conf_to_dataclass(v, dataclass_type=f_type)
        kwargs[k] = v
    return dataclass_type(**kwargs)


# =====================================================================
# verl.workers.rollout.vllm_rollout.utils.build_cli_args_from_config
# =====================================================================


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                continue
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args


# =====================================================================
# verl.base_config.BaseConfig
# =====================================================================


@dataclass
class BaseConfig:
    _target_: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return any(f.name == key for f in fields(self))


# =====================================================================
# verl_omni.workers.config.diffusion.{rollout,model}
# =====================================================================


@dataclass
class DiffusionPipelineConfig(BaseConfig):
    height: int = 512
    width: int = 512
    num_inference_steps: int = 10
    true_cfg_scale: float = 1.0
    max_sequence_length: int = 512
    guidance_scale: Optional[float] = None


@dataclass
class DiffusionRolloutConfig(BaseConfig):
    name: Optional[str] = None
    mode: str = "async"
    n: int = 1
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    enforce_eager: bool = False
    cudagraph_capture_sizes: Optional[list] = None
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    logprobs_mode: Optional[str] = "processed_logprobs"
    scheduling_policy: Optional[str] = "fcfs"
    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024
    disable_log_stats: bool = True
    engine_kwargs: dict = field(default_factory=dict)
    pipeline: DiffusionPipelineConfig = field(default_factory=DiffusionPipelineConfig)
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    load_format: str = "dummy"
    skip_tokenizer_init: bool = True
    quantization: Optional[str] = None
    enable_rollout_routing_replay: bool = False
    enable_sleep_mode: bool = True
    external_lib: Optional[str] = None
    seed: int = 0
    limit_images: Optional[int] = None
    # parent vLLMHttpServer reads these; we keep them None/disabled.
    profiler: Optional[Any] = None
    prometheus: Optional[Any] = None
    mtp: Optional[Any] = None


@dataclass
class DiffusionModelConfig(BaseConfig):
    path: str = ""
    architecture: Optional[str] = None
    local_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None
    model_type: str = "diffusion_model"
    load_tokenizer: bool = True
    trust_remote_code: bool = False
    custom_chat_template: Optional[str] = None
    external_lib: Optional[str] = None
    lora_rank: int = 0
    lora_alpha: int = 64
    lora_init_weights: str = "gaussian"
    target_modules: Optional[Any] = "all-linear"
    target_parameters: Optional[list[str]] = None
    exclude_modules: Optional[str] = None
    lora: dict[str, Any] = field(default_factory=dict)
    lora_adapter_path: Optional[str] = None

    def __post_init__(self):
        if self.local_path is None:
            self.local_path = self.path
        if self.tokenizer_path is None:
            self.tokenizer_path = os.path.join(str(self.local_path), "tokenizer")
        if self.local_tokenizer_path is None:
            self.local_tokenizer_path = self.tokenizer_path
        if self.architecture is None:
            mi = os.path.join(str(self.local_path), "model_index.json")
            if os.path.exists(mi):
                try:
                    with open(mi, encoding="utf-8") as f:
                        self.architecture = json.load(f).get("_class_name")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read model_index.json: %s", exc)


# =====================================================================
# verl_omni.pipelines.model_base.VllmOmniPipelineBase
# =====================================================================


class VllmOmniPipelineBase:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type) -> type:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get_class(cls, architecture: str):
        return cls._registry.get(architecture)

    @classmethod
    def get_pipeline_path(cls, architecture: Optional[str]) -> Optional[str]:
        pipeline_cls = cls._registry.get(architecture) if architecture else None
        if pipeline_cls is None:
            return None
        return f"{pipeline_cls.__module__}.{pipeline_cls.__qualname__}"


# =====================================================================
# verl.workers.rollout.vllm_rollout.vllm_async_server.vLLMHttpServer
# (full replication; branches the test never enters are kept guarded
#  so behavioural equivalence is maintained for any input.)
# =====================================================================


class vLLMHttpServer:
    """Inlined vLLMHttpServer parent (verbatim flow)."""

    def __init__(
        self,
        config,
        model_config,
        rollout_mode: RolloutMode,
        workers: list,
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        os.environ["VERL_REPLICA_RANK"] = str(replica_rank)

        self.config = self._init_config(config)
        self.model_config = self._init_model_config(model_config)
        self._validate_configs()

        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        self.global_steps = None

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port: Optional[int] = None

        # Profiler controller (test passes no profiler config -> None).
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None and getattr(profiler_config, "tool", None) in ("torch", "npu"):
            tool_config = omega_conf_to_dataclass(
                (getattr(profiler_config, "tool_config", None) or {}).get(profiler_config.tool)
            )
        self.profiler_controller = DistProfiler(
            self.replica_rank, config=profiler_config, tool_config=tool_config
        )

        # Port allocation (single-node DP=1 -> none of these are consumed).
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address, with_alive_sock=True)
            self._dp_rpc_port, self._dp_rpc_sock = get_free_port(self._server_address, with_alive_sock=True)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address, with_alive_sock=True)
        else:
            self._master_address = None
            self._master_port = None
            self._dp_rpc_port = None
            self._dp_master_port = None

        self._post_init(cuda_visible_devices)

    # ------------------- accessors ------------------------------------
    def get_master_address(self):
        return self._master_address, self._master_port, self._dp_rpc_port

    def get_server_address(self):
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    @property
    def lora_as_adapter(self) -> bool:
        return (
            self.model_config.lora_rank > 0 or self.model_config.lora.get("rank", 0) > 0
        ) and not self.model_config.lora.get("merge", False)

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        await self.engine.collective_rpc(method=method, timeout=timeout, args=args, kwargs=kwargs)

    # ------------------- launch_server (verbatim) ---------------------
    async def launch_server(self, master_address=None, master_port=None, dp_rpc_port=None):
        if self.node_rank != 0:
            assert master_address and master_port and dp_rpc_port, (
                "non-master node should provide master_address, master_port and dp_rpc_port"
            )
            self._master_address = master_address
            self._master_port = master_port
            self._dp_rpc_port = dp_rpc_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get(self._get_engine_kwargs_key(), {}) or {}
        if isinstance(engine_kwargs, DictConfig):
            engine_kwargs = OmegaConf.to_container(engine_kwargs, resolve=True)
        engine_kwargs = {key: val for key, val in dict(engine_kwargs).items() if val is not None}
        if self.config.get("limit_images", None):
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        self._preprocess_engine_kwargs(engine_kwargs)

        override_generation_config = self._get_override_generation_config()
        logger.info(f"override_generation_config: {override_generation_config}")

        logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            # verbatim ``set_expandable_segments(True)`` \u2014 best-effort
            try:
                if torch.cuda.is_available():
                    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            except Exception as exc:  # noqa: BLE001
                logger.warning("set_expandable_segments(True) failed: %s", exc)

        quantization, hf_overrides = self._apply_quantization()

        compilation_config = engine_kwargs.pop("compilation_config", None) or {}
        if isinstance(compilation_config, str):
            compilation_config = json.loads(compilation_config)
        if isinstance(compilation_config, DictConfig):
            compilation_config = OmegaConf.to_container(compilation_config, resolve=True)
        compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")

        # FULL cuda graph is not yet supported with DCP, downgrade to PIECEWISE
        dcp_size = engine_kwargs.get("decode_context_parallel_size", 1) or 1
        if dcp_size > 1 and compilation_config["cudagraph_mode"] == "FULL_AND_PIECEWISE":
            logger.warning(
                "FULL cuda graph is not supported with DCP (decode_context_parallel_size=%d), "
                "downgrading cudagraph_mode to PIECEWISE.",
                dcp_size,
            )
            compilation_config["cudagraph_mode"] = "PIECEWISE"

        compilation_config = json.dumps(compilation_config)
        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "distributed_executor_backend": "mp",
            "worker_extension_cls": self._get_worker_extension_cls(),
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.replica_rank + self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            "scheduling_policy": self.config.scheduling_policy,
            "compilation_config": compilation_config,
            **engine_kwargs,
        }

        # profiler / prometheus / MTP / DP / EP / multi-node / LoRA wiring
        # (test config disables all of these; we keep the guards verbatim).
        profiler_args = build_vllm_profiler_args(
            self.profiler_controller.config, self.profiler_controller.tool_config, self.replica_rank
        )
        args.update(profiler_args)

        prom = self.config.prometheus
        if prom is not None and getattr(prom, "enable", False):
            served = getattr(prom, "served_model_name", None)
            if served:
                if "/" in served:
                    served = served.split("/")[-1]
                args["served_model_name"] = served

        mtp = self.config.mtp
        if mtp is not None and getattr(mtp, "enable", False) and getattr(mtp, "enable_rollout", False):
            args["speculative_config"] = {
                "method": mtp.method,
                "num_speculative_tokens": mtp.num_speculative_tokens,
            }

        if self.config.data_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size
            args.update(
                {
                    "data_parallel_size": self.config.data_parallel_size,
                    "data_parallel_size_local": data_parallel_size_local,
                    "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        args.update({"enable_expert_parallel": self.config.expert_parallel_size > 1})

        if self.nnodes > 1:
            args.update(
                {
                    "master_addr": self._master_address,
                    "master_port": self._master_port,
                    "node_rank": self.node_rank,
                    "nnodes": self.nnodes,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        lora_rank = self.model_config.lora.get("rank", 0)
        if lora_rank <= 0:
            lora_rank = self.model_config.lora_rank
        if self.model_config.lora.get("merge", False):
            lora_rank = 0
        if lora_rank > 0:
            args.update(
                {
                    "enable_lora": True,
                    "max_loras": 1,
                    "max_lora_rank": lora_rank,
                }
            )
            if self.model_config.lora.get("fully_sharded_loras", False):
                args["fully_sharded_loras"] = True

        if self.config.enable_rollout_routing_replay:
            args["enable_return_routed_experts"] = True

        server_args = ["serve", str(self.model_config.local_path)] + build_cli_args_from_config(args)
        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = self._get_cli_modules()
        parser = FlexibleArgumentParser(description=self._get_cli_description())
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds: dict[str, Any] = {}
        for cmd_module in CMD_MODULES:
            for cmd in cmd_module.cmd_init():
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        parsed = parser.parse_args(args=server_args)
        parsed.model = parsed.model_tag
        if parsed.subparser in cmds:
            cmds[parsed.subparser].validate(parsed)

        # 3. launch server
        if self.node_rank == 0:
            # Release the held-open ports so the engine can rebind them.
            for sock_name in ("_master_sock", "_dp_rpc_sock", "_dp_master_sock"):
                sock = getattr(self, sock_name, None)
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:  # noqa: BLE001
                        pass
            await self.run_server(parsed)
        else:
            await self.run_headless(parsed)

    async def run_server(self, args: argparse.Namespace):
        raise NotImplementedError  # overridden by vLLMOmniHttpServer

    async def run_headless(self, args: argparse.Namespace):
        raise NotImplementedError("headless mode not used by the diffusion test")

    # ------------------- override hooks (verbatim from parent) --------
    def _init_config(self, config):
        # Production: omega_conf_to_dataclass(config) without a dataclass_type
        # uses Hydra's instantiate(_target_). We sidestep Hydra by typing the
        # merge against DiffusionRolloutConfig (vLLMOmniHttpServer override
        # path) \u2014 done by the child override below.
        return omega_conf_to_dataclass(config, dataclass_type=DiffusionRolloutConfig)

    def _init_model_config(self, model_config):
        return omega_conf_to_dataclass(model_config, dataclass_type=DiffusionModelConfig)

    def _validate_configs(self) -> None:
        # Default parent reads ``model_config.hf_config.max_position_embeddings``
        # \u2014 diffusion model has no such attribute. The vLLMOmniHttpServer
        # override no-ops in diffusion mode; we inherit that behaviour by
        # making the parent a no-op here too (the child won't add anything).
        pass

    def _post_init(self, cuda_visible_devices: str) -> None:
        logger.info(
            f"{self.__class__.__name__}, replica_rank: {self.replica_rank}, "
            f"node_rank: {self.node_rank}, CUDA_VISIBLE_DEVICES: {cuda_visible_devices}, "
            f"master_address: {self._master_address}, master_port: {self._master_port}, "
            f"data_parallel_rpc_port: {self._dp_rpc_port}, data_parallel_master_port: {self._dp_master_port}"
        )

    def _get_engine_kwargs_key(self) -> str:
        return "vllm"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        pass

    def _get_override_generation_config(self) -> dict:
        return {}

    def _apply_quantization(self) -> tuple[Optional[str], dict]:
        # Test never enables quantization \u2014 verbatim default return path.
        quantization = self.config.get("quantization", None)
        return quantization, {}

    def _get_worker_extension_cls(self) -> str:
        # Parent default for non-Omni servers (overridden by child).
        return WORKER_EXTENSION_CLASS

    def _get_cli_modules(self) -> list:
        return [vllm_omni.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        return "vLLM CLI"

    def _get_wake_up_tags(self) -> list[str]:
        return ["weights", "kv_cache"]


# =====================================================================
# verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server.vLLMOmniHttpServer
# (diffusion happy path \u2014 verbatim where the test enters; AR / LoRA-as-adapter
#  branches retained as guards but inactive given the test config.)
# =====================================================================


class vLLMOmniHttpServer(vLLMHttpServer):
    """Inlined vLLMOmniHttpServer (diffusion + AR routing; AR is unreachable
    in this test because ``engine_kwargs.vllm_omni.output_mode`` defaults to
    ``"diffusion"``)."""

    # -------------- initialisation overrides --------------------------
    def _init_model_config(self, model_config):
        engine_kwargs = getattr(self.config, "engine_kwargs", None) or {}
        omni_kwargs = engine_kwargs.get("vllm_omni", {}) or {}
        self._ar_mode = omni_kwargs.get("output_mode", "diffusion") == "ar"
        # AR branch would dispatch HFModelConfig; we only need diffusion.
        assert not self._ar_mode, "this inlined test replica covers diffusion mode only"
        return omega_conf_to_dataclass(model_config, dataclass_type=DiffusionModelConfig)

    def _validate_configs(self) -> None:
        # diffusion override: no-op (no max_position_embeddings).
        if getattr(self, "_ar_mode", False):
            if self.config.max_model_len is None:
                self.config.max_model_len = self.config.prompt_length + self.config.response_length

    def _post_init(self, cuda_visible_devices: str) -> None:
        if not getattr(self, "_ar_mode", False):
            self._to_tensor = T.PILToTensor()
        super()._post_init(cuda_visible_devices)

    # -------------- launch_server hooks -------------------------------
    def _get_override_generation_config(self) -> dict:
        return {}

    def _get_engine_kwargs_key(self) -> str:
        return "vllm_omni"

    def _get_worker_extension_cls(self) -> str:
        return WORKER_EXTENSION_CLASS

    def _get_cli_modules(self) -> list:
        return [vllm_omni.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        return "vLLM-Omni CLI"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        engine_kwargs.pop("output_mode", None)
        # AR-only key fixups omitted (we asserted diffusion mode above).

    # -------------- run_server (diffusion branch, verbatim) -----------
    async def run_server(self, args: argparse.Namespace):
        engine_args = OmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)

        import_external_libs(self.config.external_lib)
        pipeline_path = VllmOmniPipelineBase.get_pipeline_path(self.model_config.architecture)
        if pipeline_path is not None:
            engine_args["enable_dummy_pipeline"] = True
            engine_args["custom_pipeline_args"] = {"pipeline_class": pipeline_path}

        engine_client = AsyncOmni(**engine_args)
        app = build_app(args)
        await omni_init_app_state(engine_client, app.state, args)

        self.engine = engine_client
        # ``run_uvicorn`` skipped \u2014 the test invokes generate() via Ray RPC,
        # never opens the HTTP port (the verl-omni reference test doesn't
        # either, by virtue of using the same Ray-actor entry point).

    async def run_headless(self, args: argparse.Namespace):
        raise NotImplementedError("vLLM-Omni headless mode is not implemented yet.")

    def _get_wake_up_tags(self) -> list[str]:
        return ["weights"]

    # -------------- generate dispatch ---------------------------------
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        priority: int = 0,
    ) -> Union[DiffusionOutput]:
        return await self._generate_diffusion(
            prompt_ids, sampling_params, request_id, image_data, video_data, negative_prompt_ids, priority
        )

    # -------------- _generate_diffusion (verbatim) --------------------
    async def _generate_diffusion(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        priority: int = 0,  # noqa: ARG002
    ) -> DiffusionOutput:
        prompt_ids = normalize_token_ids(prompt_ids)

        multi_modal_data: dict[str, Any] = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        # LoRA-as-adapter branch elided: model_config has no LoRA -> property False.

        custom_prompt: OmniCustomPrompt = {"prompt_ids": prompt_ids}
        if negative_prompt_ids is not None:
            custom_prompt["negative_prompt_ids"] = negative_prompt_ids
        if multi_modal_data:
            custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}

        sampling_kwargs: dict[str, Any] = {}
        extra_args: dict[str, Any] = {}
        for k, v in sampling_params.items():
            if hasattr(OmniDiffusionSamplingParams, k):
                sampling_kwargs[k] = v
            else:
                extra_args[k] = v
        sampling_kwargs["extra_args"] = extra_args
        diffusion_sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)

        generator = self.engine.generate(
            prompt=custom_prompt,
            request_id=request_id,
            sampling_params_list=[diffusion_sampling_params],
        )

        final_res: Optional[OmniRequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        diffusion_output = self._to_tensor(final_res.images[0]).float() / 255.0

        mm_output = final_res.custom_output or {}

        if sampling_params.get("logprobs", False):
            all_log_probs = mm_output.get("all_log_probs")
            log_probs = all_log_probs[0] if all_log_probs is not None else None
        else:
            log_probs = None

        all_latents = mm_output.get("all_latents")
        all_timesteps = mm_output.get("all_timesteps")
        prompt_embeds = mm_output.get("prompt_embeds")
        prompt_embeds_mask = mm_output.get("prompt_embeds_mask")
        negative_prompt_embeds = mm_output.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = mm_output.get("negative_prompt_embeds_mask")

        extra_fields = {
            "all_latents": all_latents[0] if all_latents is not None else None,
            "all_timesteps": all_timesteps[0] if all_timesteps is not None else None,
            "prompt_embeds": prompt_embeds[0] if prompt_embeds is not None else None,
            "prompt_embeds_mask": prompt_embeds_mask[0] if prompt_embeds_mask is not None else None,
            "negative_prompt_embeds": negative_prompt_embeds[0] if negative_prompt_embeds is not None else None,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask[0]
            if negative_prompt_embeds_mask is not None
            else None,
            "global_steps": self.global_steps,
        }

        if final_res.request_output is not None and hasattr(final_res.request_output, "finish_reason"):
            finish_reason = final_res.request_output.finish_reason or "stop"
        else:
            finish_reason = "stop"

        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason

        num_preempted = None
        if final_res.request_output is not None and hasattr(final_res.request_output, "num_preempted"):
            num_preempted = final_res.request_output.num_preempted

        return DiffusionOutput(
            diffusion_output=diffusion_output,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )