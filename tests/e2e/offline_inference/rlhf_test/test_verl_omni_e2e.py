# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E test that follows the EXACT same flow as
``verl-omni/tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py``
but inlines all the ``verl`` / ``verl_omni`` symbols the flow depends on
(stripped down to only what the diffusion-mode test actually exercises).

Flow (identical to the verl-omni test):

1. Build two ``OmegaConf`` configs (``DiffusionRolloutConfig`` /
   ``DiffusionModelConfig`` shape).
2. ``ServerCls = ray.remote(vLLMOmniHttpServerLocal)`` then
   ``ServerCls.options(...).remote(config=..., model_config=...,
   rollout_mode=RolloutMode.STANDALONE, workers=[], replica_rank=0,
   node_rank=0, gpus_per_node=1, nnodes=1, cuda_visible_devices="0")``.
3. ``ray.get(server.launch_server.remote())``.
4. ``server.generate.remote(prompt_ids=..., sampling_params=..., request_id=...)``
   and assert the returned ``DiffusionOutput``.

The inlined classes / helpers mirror their ``verl`` / ``verl_omni``
counterparts; bits not reached by this test (AR mode, LoRA,
quantization, run_uvicorn, headless, profiler, port wiring) are removed.

Usage:
    pytest tests/e2e/offline_inference/rlhf_test/\
test_ray_async_omni_qwen_image_generate.py -v -s
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from enum import Enum
from functools import lru_cache
from typing import Any
from uuid import uuid4

import pytest
import ray
import torch
import torchvision.transforms as T
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer
from vllm.entrypoints.openai.api_server import build_app
from vllm.utils.argparse_utils import FlexibleArgumentParser

import vllm_omni.entrypoints.cli.serve
from tests.helpers.mark import hardware_test
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

logger = logging.getLogger(__name__)

CUSTOM_PIPELINE_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline.qwen_image_pipeline_with_logprob.QwenImagePipelineWithLogProbForTest"
)
WORKER_EXTENSION_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline.worker_extension.vLLMOmniColocateWorkerExtensionForTest"
)

# Use your specified HF repo name/path directly here
MODEL = "tiny-random/Qwen-Image"

TOKENIZER_MODEL = "Qwen/Qwen2-1.5B-Instruct"


def _resolve_model_path(repo_id: str) -> str:
    """Resolve an HF repo ID to a local snapshot path, downloading if needed."""
    # Allow overriding with a pre-existing local path (skips download).
    if os.path.isdir(repo_id):
        return repo_id
    return snapshot_download(repo_id=repo_id)


_MIN_PROMPT_TOKENS = 35


# ---------------------------------------------------------------------
#       Inlined: verl.workers.rollout.replica.RolloutMode
# ---------------------------------------------------------------------


class RolloutMode(Enum):
    HYBRID = "hybrid"
    COLOCATED = "colocated"
    STANDALONE = "standalone"


# ---------------------------------------------------------------------
#       Inlined: verl_omni.workers.rollout.replica.DiffusionOutput
# ---------------------------------------------------------------------


class DiffusionOutput(BaseModel):
    """Pydantic mirror of ``verl_omni.workers.rollout.replica.DiffusionOutput``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    diffusion_output: Any
    log_probs: Any | None = None
    stop_reason: str | None = None
    num_preempted: int | None = None
    extra_fields: dict[str, Any] = {}


# ---------------------------------------------------------------------
#       Inlined: verl.utils.tokenizer.normalize_token_ids
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
#  Inlined: verl.workers.rollout.vllm_rollout.utils.build_cli_args_from_config
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
#                  Tokenizer helper (test-side)
# ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)


def _tokenize_prompt(text: str) -> list[int]:
    tokenizer = _get_tokenizer()
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    assert len(token_ids) > _MIN_PROMPT_TOKENS, (
        f"Prompt too short ({len(token_ids)} tokens, need >{_MIN_PROMPT_TOKENS}). "
        "The pipeline drops the first 34 chat-template prefix tokens; "
        "use a longer prompt so content tokens remain after the drop."
    )
    return token_ids


# ---------------------------------------------------------------------
#  Inlined: vLLMOmniHttpServer (verl-omni) + relevant vLLMHttpServer (verl)
#  bits, stripped to the diffusion-mode path the test exercises.
# ---------------------------------------------------------------------


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Uniform getter that works for both DictConfig and dataclass-like objects."""
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class vLLMOmniHttpServerLocal:
    """Self-contained simulation of ``vLLMOmniHttpServer`` for diffusion mode.

    Constructor signature matches ``vLLMHttpServer.__init__`` exactly so the
    Ray-actor call site in the test is identical to the verl-omni one.
    """

    # ------------------------------------------------------------------ init
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

        # Resolve configs. verl uses omega_conf_to_dataclass; here we keep
        # them as DictConfig — every read site below uses ``_cfg_get``.
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

        # Diffusion-mode parity with ``vLLMOmniHttpServer._post_init``.
        self._to_tensor = T.PILToTensor()

        logger.info(
            "%s, replica_rank: %d, node_rank: %d, CUDA_VISIBLE_DEVICES: %s",
            self.__class__.__name__,
            replica_rank,
            node_rank,
            cuda_visible_devices,
        )

    # ---------------------------------------------------------- launch_server
    async def launch_server(self) -> None:
        """Diffusion-only mirror of ``vLLMHttpServer.launch_server``."""

        # vLLMOmniHttpServer._get_engine_kwargs_key() returns "vllm_omni".
        engine_kwargs_root = _cfg_get(self.config, "engine_kwargs", {}) or {}
        if isinstance(engine_kwargs_root, (dict, DictConfig)):
            engine_kwargs = engine_kwargs_root.get("vllm_omni", {}) or {}
        else:
            engine_kwargs = {}
        engine_kwargs = {k: v for k, v in dict(engine_kwargs).items() if v is not None}

        # vLLMOmniHttpServer._preprocess_engine_kwargs (diffusion path).
        engine_kwargs.pop("output_mode", None)

        # ``compilation_config`` handling — verbatim from parent launch_server.
        compilation_config = engine_kwargs.pop("compilation_config", None) or {}
        if isinstance(compilation_config, str):
            compilation_config = json.loads(compilation_config)
        if isinstance(compilation_config, DictConfig):
            compilation_config = OmegaConf.to_container(compilation_config, resolve=True)
        compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")
        compilation_config = json.dumps(compilation_config)

        # Args dict (subset of parent's; the diffusion override drops AR-only
        # entries via ``_get_override_generation_config`` returning ``{}``).
        args = {
            "dtype": _cfg_get(self.config, "dtype", "bfloat16"),
            "load_format": _cfg_get(self.config, "load_format", "auto"),
            "skip_tokenizer_init": False,
            "distributed_executor_backend": "mp",
            "worker_extension_cls": WORKER_EXTENSION_CLASS,
            "trust_remote_code": _cfg_get(self.model_config, "trust_remote_code", True),
            "max_model_len": _cfg_get(self.config, "max_model_len", 1058),
            "max_num_seqs": _cfg_get(self.config, "max_num_seqs", 256),
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

        # Same parser pipeline as parent launch_server (with vllm-omni's
        # cmd_init injected, matching ``_get_cli_modules`` + ``_get_cli_description``).
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

        # node_rank == 0 → run_server (the test never sets node_rank != 0).
        await self.run_server(parsed)

    # ------------------------------------------------------------- run_server
    async def run_server(self, args: argparse.Namespace) -> None:
        """Diffusion branch of ``vLLMOmniHttpServer.run_server``."""

        engine_args = OmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)

        # Diffusion path: pin the in-repo test pipeline. The production code
        # calls ``VllmOmniPipelineBase.get_pipeline_path(architecture)``; we
        # bypass the registry and bind the test pipeline directly.
        engine_args["enable_dummy_pipeline"] = True
        engine_args["custom_pipeline_args"] = {"pipeline_class": CUSTOM_PIPELINE_CLASS}

        engine_client = AsyncOmni(**engine_args)
        app = build_app(args)
        await omni_init_app_state(engine_client, app.state, args)

        self.engine = engine_client
        # ``run_uvicorn`` intentionally skipped — the test drives the engine
        # via Ray RPC, not HTTP.

    # ---------------------------------------------------------------- generate
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
        # Verbatim copy of ``vLLMOmniHttpServer._generate_diffusion`` minus
        # the LoRA branch (the test does not enable ``lora_as_adapter``).
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


# ---------------------------------------------------------------------
#                          Fixtures & tests
# ---------------------------------------------------------------------


@pytest.fixture
def init_server():
    """Create and launch a vLLMOmniHttpServerLocal Ray actor with Qwen-Image."""
    model_path = _resolve_model_path(MODEL)

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
            }
        },
        ignore_reinit_error=True,
    )

    rollout_cfg = OmegaConf.create(
        {
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 256,
            "max_model_len": 1058,
            "dtype": "bfloat16",
            "load_format": "auto",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "logprobs_mode": "processed_logprobs",
            "scheduling_policy": "fcfs",
            "seed": 0,
            "n": 4,
            "pipeline": {
                "height": 512,
                "width": 512,
                "num_inference_steps": 10,
            },
        }
    )

    model_cfg = OmegaConf.create(
        {
            "path": str(model_path),
            "local_path": str(model_path),
            "tokenizer_path": TOKENIZER_MODEL,
            "trust_remote_code": True,
            "load_tokenizer": True,
        }
    )

    ServerCls = ray.remote(vLLMOmniHttpServerLocal)
    server = ServerCls.options(
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
            }
        },
        max_concurrency=10,
        num_gpus=1,
    ).remote(
        config=rollout_cfg,
        model_config=model_cfg,
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=1,
        nnodes=1,
        cuda_visible_devices="0",
    )

    ray.get(server.launch_server.remote())

    yield server

    ray.shutdown()


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_generate(init_server):
    """Concurrent generate() calls covering basic output, logprobs, and multi-request correctness."""
    server = init_server

    prompts = [
        "a beautiful sunset over the ocean with vibrant orange and purple clouds "
        "reflecting on the calm water surface near a rocky coastline",
        "a fluffy orange cat sitting on a wooden windowsill looking outside at "
        "a garden full of colorful flowers on a bright sunny afternoon",
        "a majestic mountain landscape covered with fresh white snow under a "
        "clear blue sky with pine trees in the foreground and a frozen lake",
        "a futuristic city at night with neon lights glowing on tall glass "
        "skyscrapers and flying vehicles soaring between the buildings",
    ]

    refs = []
    for i, prompt in enumerate(prompts):
        rid = f"test_{i}_{uuid4().hex[:8]}"
        ref = server.generate.remote(
            prompt_ids=_tokenize_prompt(prompt),
            sampling_params={
                "num_inference_steps": 10,
                "true_cfg_scale": 4.0,
                "height": 512,
                "width": 512,
                "logprobs": i == 0,  # first request includes logprobs
            },
            request_id=rid,
        )
        refs.append(ref)

    results = ray.get(refs, timeout=600)

    for i, output in enumerate(results):
        assert isinstance(output, DiffusionOutput), f"Request {i}: expected DiffusionOutput"
        img = output.diffusion_output
        assert isinstance(img, torch.Tensor) and img.ndim == 3, (
            f"Request {i}: expected CHW torch.Tensor, got {type(img).__name__} ndim={getattr(img, 'ndim', None)}"
        )
        c, h, w = img.shape
        assert c == 3, f"Request {i}: expected 3 channels (CHW), got {c}"
        assert h > 0 and w > 0, f"Request {i}: image dimensions must be positive"
        assert output.stop_reason in ("completed", "aborted", None), (
            f"Request {i}: unexpected stop_reason {output.stop_reason!r}"
        )
        assert 0.0 <= float(img[0, 0, 0]) <= 1.0, f"Request {i}: pixel values must be in [0, 1]"

    print(f"All {len(prompts)} concurrent requests returned valid DiffusionOutput")
