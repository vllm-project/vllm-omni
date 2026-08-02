# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ray-serializable verl-omni HTTP server test double for RLHF E2E tests.

The actor class must live in an importable module (not a pytest test file) so
Ray workers can unpickle ``vLLMOmniHttpServerLocal`` via
``tests.e2e.features.helpers.verl_omni_server``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from enum import Enum
from typing import Any

import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from vllm.entrypoints.openai.api_server import build_app
from vllm.utils.argparse_utils import FlexibleArgumentParser

import vllm_omni.entrypoints.cli.serve
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

logger = logging.getLogger(__name__)

CUSTOM_PIPELINE_CLASS = "tests.e2e.features.helpers.custom_pipeline.QwenImagePipelineWithLogProbForTest"
WORKER_EXTENSION_CLASS = "tests.e2e.features.helpers.custom_pipeline.vLLMOmniColocateWorkerExtensionForTest"


class RolloutMode(Enum):
    HYBRID = "hybrid"
    COLOCATED = "colocated"
    STANDALONE = "standalone"


class DiffusionOutput(BaseModel):
    """Pydantic mirror of ``verl_omni.workers.rollout.replica.DiffusionOutput``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    diffusion_output: Any
    log_probs: Any | None = None
    stop_reason: str | None = None
    num_preempted: int | None = None
    extra_fields: dict[str, Any] = {}


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


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Uniform getter that works for both DictConfig and dataclass-like objects."""
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class vLLMOmniHttpServerLocal:
    """Self-contained simulation of ``vLLMOmniHttpServer`` for diffusion mode."""

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
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
        os.environ["VERL_REPLICA_RANK"] = str(replica_rank)

        self.config = OmegaConf.create(config) if not isinstance(config, DictConfig) else config
        self.model_config = OmegaConf.create(model_config) if not isinstance(model_config, DictConfig) else model_config

        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        self.global_steps = 0
        self.engine: AsyncOmni | None = None
        self._to_tensor = T.PILToTensor()

        logger.info(
            "%s, replica_rank: %d, node_rank: %d, CUDA_VISIBLE_DEVICES: %s",
            self.__class__.__name__,
            replica_rank,
            node_rank,
            cuda_visible_devices,
        )

    async def launch_server(self) -> None:
        engine_kwargs_root = _cfg_get(self.config, "engine_kwargs", {}) or {}
        if isinstance(engine_kwargs_root, (dict, DictConfig)):
            engine_kwargs = engine_kwargs_root.get("vllm_omni", {}) or {}
        else:
            engine_kwargs = {}
        engine_kwargs = {k: v for k, v in dict(engine_kwargs).items() if v is not None}

        engine_kwargs.pop("output_mode", None)

        compilation_config = engine_kwargs.pop("compilation_config", None) or {}
        if isinstance(compilation_config, str):
            compilation_config = json.loads(compilation_config)
        if isinstance(compilation_config, DictConfig):
            compilation_config = OmegaConf.to_container(compilation_config, resolve=True)
        compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")
        compilation_config = json.dumps(compilation_config)

        args = {
            "dtype": _cfg_get(self.config, "dtype", "bfloat16"),
            "load_format": _cfg_get(self.config, "load_format", "auto"),
            "skip_tokenizer_init": False,
            "distributed_executor_backend": "mp",
            "worker_extension_cls": WORKER_EXTENSION_CLASS,
            "trust_remote_code": _cfg_get(self.model_config, "trust_remote_code", True),
            "max_model_len": _cfg_get(self.config, "max_model_len", 1058),
            "max_num_seqs": _cfg_get(self.config, "max_num_seqs", 1),
            "enable_chunked_prefill": _cfg_get(self.config, "enable_chunked_prefill", False),
            "max_num_batched_tokens": _cfg_get(self.config, "max_num_batched_tokens", 8192),
            "enable_prefix_caching": _cfg_get(self.config, "enable_prefix_caching", False),
            "enable_sleep_mode": _cfg_get(self.config, "enable_sleep_mode", False),
            "logprobs_mode": _cfg_get(self.config, "logprobs_mode", "processed_logprobs"),
            "enforce_eager": _cfg_get(self.config, "enforce_eager", True),
            "gpu_memory_utilization": _cfg_get(self.config, "gpu_memory_utilization", 0.8),
            "disable_log_stats": _cfg_get(self.config, "disable_log_stats", True),
            "tensor_parallel_size": _cfg_get(self.config, "tensor_model_parallel_size", 1),
            "seed": self.replica_rank + int(_cfg_get(self.config, "seed", 0) or 0),
            "override_generation_config": json.dumps({}),
            "scheduling_policy": _cfg_get(self.config, "scheduling_policy", "fcfs"),
            "compilation_config": compilation_config,
            **engine_kwargs,
        }

        model_path = str(_cfg_get(self.model_config, "local_path", None) or _cfg_get(self.model_config, "path"))
        server_args_list = ["serve", model_path] + build_cli_args_from_config(args)

        parser = FlexibleArgumentParser(description="vLLM-Omni CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds: dict[str, Any] = {}
        for cmd in vllm_omni.entrypoints.cli.serve.cmd_init():
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
        parsed = parser.parse_args(args=server_args_list)
        parsed.model = parsed.model_tag
        if parsed.subparser in cmds:
            cmds[parsed.subparser].validate(parsed)

        await self.run_server(parsed)

    async def run_server(self, args: argparse.Namespace) -> None:
        engine_args = OmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)

        engine_args["enable_dummy_pipeline"] = True
        engine_args["custom_pipeline_args"] = {"pipeline_class": CUSTOM_PIPELINE_CLASS}

        engine_client = AsyncOmni(**engine_args)
        app = build_app(args)
        await omni_init_app_state(engine_client, app.state, args)

        self.engine = engine_client

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        negative_prompt_ids: list[int] | None = None,
        priority: int = 0,  # noqa: ARG002 (signature parity)
    ) -> DiffusionOutput:
        prompt_ids = normalize_token_ids(prompt_ids)

        multi_modal_data: dict[str, Any] = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

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

        final_res: OmniRequestOutput | None = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        diffusion_output = self._to_tensor(final_res.images[0]).float() / 255.0

        metadata = (
            final_res.multimodal_output.get("metadata", {}) if isinstance(final_res.multimodal_output, dict) else {}
        )
        prompt_embeddings = metadata.get("prompt_embeddings", {}) if isinstance(metadata, dict) else {}

        if sampling_params.get("logprobs", False):
            all_log_probs = final_res.trajectory_log_probs
            log_probs = all_log_probs[0] if all_log_probs is not None else None
        else:
            log_probs = None

        all_latents = final_res.trajectory_latents
        all_timesteps = final_res.trajectory_timesteps
        prompt_embeds = prompt_embeddings.get("prompt_embeds")
        prompt_embeds_mask = prompt_embeddings.get("prompt_embeds_mask")
        negative_prompt_embeds = prompt_embeddings.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = prompt_embeddings.get("negative_prompt_embeds_mask")

        extra_fields = {
            "all_latents": all_latents[0] if all_latents is not None else None,
            "all_timesteps": all_timesteps[0] if all_timesteps is not None else None,
            "prompt_embeds": prompt_embeds[0] if prompt_embeds is not None else None,
            "prompt_embeds_mask": prompt_embeds_mask[0] if prompt_embeds_mask is not None else None,
            "negative_prompt_embeds": negative_prompt_embeds[0] if negative_prompt_embeds is not None else None,
            "negative_prompt_embeds_mask": (
                negative_prompt_embeds_mask[0] if negative_prompt_embeds_mask is not None else None
            ),
            "global_steps": self.global_steps,
        }

        req_output = final_res.request_output
        if req_output is not None and hasattr(req_output, "finish_reason"):
            finish_reason = req_output.finish_reason or "stop"
        else:
            finish_reason = "stop"

        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason

        num_preempted = None
        if req_output is not None and hasattr(req_output, "num_preempted"):
            num_preempted = req_output.num_preempted

        return DiffusionOutput(
            diffusion_output=diffusion_output,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )
