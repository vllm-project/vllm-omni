# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ray-serializable verl-omni AR server test double for RLHF E2E tests.

The autoregressive counterpart of ``verl_omni_server``: it follows the flow
``verl_omni`` drives for AR rollouts (``vLLMOmniHttpServer`` + ``ARStrategy``
on the worker-v2 runner) and inlines every ``verl`` / ``verl_omni`` symbol the
flow depends on, so the test needs neither package. The actor class must live
in an importable module (not a pytest test file) so Ray workers can unpickle
``vLLMOmniHttpServerARLocal``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
from dataclasses import asdict, fields
from typing import Any

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from vllm import SamplingParams

from tests.e2e.features.helpers.verl_omni_server import (
    RolloutMode,
    _cfg_get,
    build_cli_args_from_config,
    normalize_token_ids,
    parse_serve_args,
)
from vllm_omni.engine.arg_utils import OmniEngineArgs, OrchestratorArgs, orchestrator_field_names
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_THINKER_ONLY_PIPELINE
from vllm_omni.utils.tracking_parser import TrackingNamespace

logger = logging.getLogger(__name__)

# verl_omni.workers.rollout.vllm_rollout.process_hidden_states
TEACHER_HIDDEN_STATES_KEY = "teacher_hidden_states"
RETURN_HIDDEN_STATES_FLAG = "return_hidden_states"
# verl_omni.pipelines.qwen3_omni.omni_rollout_adapter (pipeline_mode="thinker_only")
THINKER_PIPELINE = QWEN3_OMNI_THINKER_ONLY_PIPELINE
THINKER_STAGE_ENGINE_EXTRAS = {"model_arch": "Qwen3OmniMoeThinkerForConditionalGeneration"}
THINKER_HF_OVERRIDES = {"enable_audio_output": False}
# verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy._SCORED_PROMPT_IDS_KEY
_SCORED_PROMPT_IDS_KEY = "_verl_omni_scored_prompt_ids"
WORKER_EXTENSION_CLASS = "tests.e2e.features.helpers.verl_omni_ar_worker.vLLMOmniColocateWorkerExtensionForARTest"


class TokenOutput(BaseModel):
    """Pydantic mirror of ``verl.workers.rollout.replica.TokenOutput``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    token_ids: list[int]
    log_probs: list[float] | None = None
    stop_reason: str | None = None
    num_preempted: int | None = None
    extra_fields: dict[str, Any] = {}


def extract_prompt_logprobs(output: Any, num_prompt_logprobs: int, result_dict: dict[str, list]) -> None:
    """Port of ``verl.workers.rollout.vllm_rollout.utils.extract_prompt_logprobs`` (``logprobs=0``)."""
    prompt_logprobs_ls, prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for logprobs_dict in output.prompt_logprobs[1:]:
        token_id_str = list(logprobs_dict.keys())[0]
        logprob = logprobs_dict[token_id_str].logprob
        prompt_logprobs_ls.append([logprob])
        prompt_ids_ls.append([int(token_id_str)])

    # NOTE: pad a dummy prompt logprob for last prompt token.
    prompt_logprobs_ls.append([0.0])
    prompt_ids_ls.append([0])

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls


def extract_teacher_hidden_states(req_output: Any) -> torch.Tensor | None:
    """Mirror of ``verl_omni`` ``process_hidden_states.extract_hidden_states``."""
    outputs = getattr(req_output, "outputs", None)
    if not outputs:
        return None
    multimodal_output = getattr(outputs[0], "multimodal_output", None)
    if multimodal_output is None:
        return None
    return multimodal_output.get(TEACHER_HIDDEN_STATES_KEY)


def _is_defaulted_value(key: str, value: Any, default: Any) -> bool:
    """Verbatim port of ``verl_omni`` ``vllm_omni_async_server._is_defaulted_value``."""
    if value == default:
        if isinstance(value, bool) != isinstance(default, bool):
            return False
        return True
    if default is None and value in ("", {}, []):
        return True
    if isinstance(default, str) and isinstance(value, dict | list):
        try:
            return json.loads(default) == value
        except (ValueError, TypeError):
            return False
    if isinstance(value, str) and isinstance(default, dict | list):
        try:
            return json.loads(value) == default
        except (ValueError, TypeError):
            return False
    return False


def _drop_defaulted_engine_args(engine_args: dict[str, Any]) -> dict[str, Any]:
    """Verbatim port of ``verl_omni`` ``vllm_omni_async_server._drop_defaulted_engine_args``."""
    defaults = asdict(OmniEngineArgs(model=engine_args.get("model", "")))
    defaults.update(asdict(OrchestratorArgs()))
    return {
        key: value
        for key, value in engine_args.items()
        if key == "model" or key not in defaults or not _is_defaulted_value(key, value, defaults[key])
    }


class ARStrategyForTest:
    """Mirror of ``verl_omni`` ``ARStrategy``: the hooks ``vLLMOmniHttpServer`` calls."""

    def __init__(self, server: vLLMOmniHttpServerARLocal):
        self.server = server

    def preprocess_engine_kwargs(self, engine_kwargs: dict[str, Any]) -> None:
        """Replace the rollout's pipeline selection with a generated deploy config."""
        for consumed in ("output_mode", "pipeline_name", "pipeline_mode", "custom_pipeline"):
            engine_kwargs.pop(consumed, None)
        # Stage devices are renumbered to local indices.
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        device_count = len([device for device in visible_devices.split(",") if device.strip()])
        deploy = {
            "pipeline": THINKER_PIPELINE.model_type,
            # verl-omni leaves this unset, which the AR strategy defaults to True.
            "async_chunk": engine_kwargs.get("async_chunk", engine_kwargs.get("async-chunk", True)),
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "devices": ",".join(str(index) for index in range(device_count)),
                    "tensor_parallel_size": int(_cfg_get(self.server.config, "tensor_model_parallel_size", 1)),
                    "engine_extras": dict(THINKER_STAGE_ENGINE_EXTRAS),
                }
                for stage in THINKER_PIPELINE.stages
            ],
        }
        self.server._deploy_config_dir = tempfile.mkdtemp(prefix="verl_omni_ar_deploy_")
        deploy_path = os.path.join(self.server._deploy_config_dir, "qwen3_omni_moe_thinker_only.yaml")
        with open(deploy_path, "w") as handle:
            yaml.safe_dump(deploy, handle)
        engine_kwargs["deploy_config"] = deploy_path
        hf_overrides = engine_kwargs.get("hf_overrides") or {}
        if isinstance(hf_overrides, str):
            hf_overrides = json.loads(hf_overrides)
        hf_overrides.update(THINKER_HF_OVERRIDES)
        engine_kwargs["hf_overrides"] = hf_overrides

    def preprocess_input(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: dict[str, Any],
        mm_processor_kwargs: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], SamplingParams]:
        """Mirror of ``ARStrategy.preprocess_input`` for a token prompt."""
        max_possible_tokens = self.server.config.max_model_len - len(prompt_ids)
        if max_possible_tokens <= 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) meets or exceeds the model's maximum context length "
                f"({self.server.config.max_model_len}), leaving no space for generation."
            )
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            max_tokens = min(
                self.server.config.response_length,
                self.server.config.prompt_length + self.server.config.response_length - len(prompt_ids),
            )
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        logprobs = sampling_params.pop("logprobs", None)
        if logprobs is True:
            sampling_params["logprobs"] = 0
        elif isinstance(logprobs, int) and not isinstance(logprobs, bool):
            sampling_params["logprobs"] = logprobs
        else:
            sampling_params["logprobs"] = None
        return_hidden = bool(sampling_params.pop(RETURN_HIDDEN_STATES_FLAG, False))
        policy_params = SamplingParams(max_tokens=max_tokens, **sampling_params)

        prompt: dict[str, Any] = {"prompt_token_ids": prompt_ids}
        if multi_modal_data:
            prompt["multi_modal_data"] = multi_modal_data
        if mm_processor_kwargs:
            prompt["mm_processor_kwargs"] = mm_processor_kwargs
        if policy_params.prompt_logprobs is not None:
            # A teacher scores the request's own ids; run_generation checks them.
            prompt[_SCORED_PROMPT_IDS_KEY] = list(prompt_ids)
        if return_hidden:
            policy_params.extra_args = {**(policy_params.extra_args or {}), RETURN_HIDDEN_STATES_FLAG: True}
        return prompt, policy_params

    async def run_generation(
        self,
        prompt: dict[str, Any],
        params: SamplingParams,
        request_id: str,
        lora_request: Any,
        priority: int,
    ) -> Any:
        scored_prompt_ids = prompt.pop(_SCORED_PROMPT_IDS_KEY, None)
        generator = self.server.live_engine().generate(
            prompt=prompt,
            sampling_params_list=[params],
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )
        final_res = None
        async for output in generator:
            final_res = output
        if scored_prompt_ids is not None and final_res is not None:
            req_output = getattr(final_res, "request_output", None) or final_res
            if list(getattr(req_output, "prompt_token_ids", None) or []) != scored_prompt_ids:
                raise RuntimeError(
                    "The omni teacher scored a different token sequence than the request carried; "
                    "its prompt log probabilities would not align with the student's tokens."
                )
        return final_res

    def process_output(self, final_res: Any, params: SamplingParams) -> TokenOutput:
        if final_res is None:
            raise RuntimeError("AR mode: vLLM-Omni engine yielded no output for the prompt.")
        req_output = getattr(final_res, "request_output", None) or final_res
        if not req_output.outputs:
            raise RuntimeError("AR mode expects outputs with token IDs, but got None or empty.")

        extra_fields: dict[str, Any] = {"global_steps": self.server.global_steps}
        hidden = extract_teacher_hidden_states(req_output)
        if hidden is not None:
            extra_fields[TEACHER_HIDDEN_STATES_KEY] = hidden
        token_ids = req_output.outputs[0].token_ids
        log_probs = None
        if params.prompt_logprobs is not None:
            if getattr(req_output, "prompt_logprobs", None) is None:
                raise RuntimeError("The omni teacher did not return requested prompt log probabilities.")
            extract_prompt_logprobs(req_output, params.prompt_logprobs, extra_fields)
            if not all(math.isfinite(logprob) for row in extra_fields["prompt_logprobs"] for logprob in row):
                raise RuntimeError("The omni teacher returned non-finite prompt log probabilities.")
        if params.logprobs is not None:
            log_probs = [
                logprobs[token_ids[index]].logprob for index, logprobs in enumerate(req_output.outputs[0].logprobs)
            ]

        finish_reason = req_output.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason
        return TokenOutput(
            token_ids=list(token_ids),
            log_probs=log_probs,
            stop_reason=stop_reason,
            extra_fields=extra_fields,
        )


class vLLMOmniHttpServerARLocal:
    """Self-contained simulation of ``vLLMOmniHttpServer`` for AR mode."""

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
        self._generate_strategy = ARStrategyForTest(self)
        self._deploy_config_dir: str | None = None

        logger.info(
            "%s, replica_rank: %d, node_rank: %d, CUDA_VISIBLE_DEVICES: %s",
            self.__class__.__name__,
            replica_rank,
            node_rank,
            cuda_visible_devices,
        )

    async def launch_server(self) -> None:
        engine_kwargs = _cfg_get(self.config, "engine_kwargs", {}) or {}
        engine_kwargs = dict(engine_kwargs.get("vllm_omni", {}) or {})
        engine_kwargs = {key: value for key, value in engine_kwargs.items() if value is not None}

        self._generate_strategy.preprocess_engine_kwargs(engine_kwargs)
        compilation_config = engine_kwargs.pop("compilation_config", None) or {}
        if isinstance(compilation_config, str):
            compilation_config = json.loads(compilation_config)
        compilation_config.setdefault("cudagraph_mode", "FULL_AND_PIECEWISE")

        args = {
            "dtype": _cfg_get(self.config, "dtype", "bfloat16"),
            "load_format": _cfg_get(self.config, "load_format", "auto"),
            "skip_tokenizer_init": False,
            "distributed_executor_backend": "mp",
            "worker_extension_cls": WORKER_EXTENSION_CLASS,
            "trust_remote_code": _cfg_get(self.model_config, "trust_remote_code", True),
            "max_model_len": _cfg_get(self.config, "max_model_len"),
            "max_num_seqs": _cfg_get(self.config, "max_num_seqs", 1),
            "enable_chunked_prefill": _cfg_get(self.config, "enable_chunked_prefill", False),
            "max_num_batched_tokens": _cfg_get(self.config, "max_num_batched_tokens", 8192),
            "enable_prefix_caching": _cfg_get(self.config, "enable_prefix_caching", False),
            "enable_sleep_mode": _cfg_get(self.config, "enable_sleep_mode", False),
            "logprobs_mode": _cfg_get(self.config, "logprobs_mode", "processed_logprobs"),
            "enforce_eager": _cfg_get(self.config, "enforce_eager", False),
            "gpu_memory_utilization": _cfg_get(self.config, "gpu_memory_utilization", 0.8),
            "disable_log_stats": _cfg_get(self.config, "disable_log_stats", True),
            "tensor_parallel_size": _cfg_get(self.config, "tensor_model_parallel_size", 1),
            "seed": self.replica_rank + int(_cfg_get(self.config, "seed", 0) or 0),
            "override_generation_config": json.dumps({}),
            "scheduling_policy": _cfg_get(self.config, "scheduling_policy", "fcfs"),
            "compilation_config": json.dumps(compilation_config),
            **engine_kwargs,
        }

        model_path = str(_cfg_get(self.model_config, "local_path", None) or _cfg_get(self.model_config, "path"))
        await self.run_server(parse_serve_args(["serve", model_path] + build_cli_args_from_config(args)))

    async def run_server(self, args: TrackingNamespace) -> None:
        # Match the production entrypoint: parser defaults are not user overrides.
        engine_args = args.get_explicit_kwargs_dict()
        engine_args["model"] = args.model
        # verl's ``OmniEngineArgs.from_cli_args`` keeps the OmniEngineArgs fields
        # and then restores the OrchestratorArgs fields it forwarded. The explicit
        # dict already holds the OmniEngineArgs that were passed, so only the
        # orchestrator-only fields are restored here.
        orchestrator_only = orchestrator_field_names() - {field.name for field in fields(OmniEngineArgs)}
        for key in orchestrator_only:
            value = getattr(args, key, None)
            if value is not None:
                engine_args[key] = value
        engine_args = _drop_defaulted_engine_args(engine_args)
        # verl sets this explicitly even when it equals the OmniEngineArgs default.
        engine_args["log_stats"] = not bool(_cfg_get(self.config, "disable_log_stats", True))
        self.engine = AsyncOmni(**engine_args)

    async def shutdown(self) -> None:
        """Stop the engine before Ray destroys this actor."""
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
        if self._deploy_config_dir is not None:
            shutil.rmtree(self._deploy_config_dir, ignore_errors=True)
            self._deploy_config_dir = None

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        audio_data: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        priority: int = 0,
    ) -> TokenOutput:
        """Mirror of ``OmniStrategyBase.generate``: preprocess, generate, postprocess.

        The rollout is adapter-free, so the LoRA request verl resolves here is
        always ``None``.
        """
        prompt_ids = normalize_token_ids(prompt_ids)
        multi_modal_data: dict[str, Any] = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        if audio_data is not None:
            multi_modal_data["audio"] = audio_data

        prompt, params = self._generate_strategy.preprocess_input(
            prompt_ids, sampling_params, multi_modal_data, mm_processor_kwargs
        )
        final_res = await self._generate_strategy.run_generation(prompt, params, request_id, None, priority)
        return self._generate_strategy.process_output(final_res, params)

    def live_engine(self) -> AsyncOmni:
        if self.engine is None:
            raise RuntimeError("Server must be launched before generating")
        return self.engine

    # Control-plane calls verl's trainer issues between rollout phases.

    async def sleep(self) -> None:
        self._validate_acks("sleep", await self.live_engine().sleep(level=1))

    async def wake_up(self, tags: list[str] | None = None) -> None:
        # An AR engine core keeps generation paused after wake until resumed.
        self._validate_acks("wake_up", await self.live_engine().wake_up(tags=tags))
        await self.live_engine().resume_generation()

    @staticmethod
    def _validate_acks(method: str, acks: Any) -> None:
        """Port of ``vLLMOmniHttpServer._validate_acks``: fail closed on a failed handshake."""
        for ack in acks or []:
            if isinstance(ack, dict):
                raise RuntimeError(f"{method} failed on a stage: {ack.get('error', ack)}")
            if ack.status != "SUCCESS":
                raise RuntimeError(f"{method} failed on a stage: {getattr(ack, 'error_msg', None) or ack!r}")

    async def collective_rpc(self, method: str, **kwargs) -> list:
        return await self.live_engine().collective_rpc(method=method, **kwargs)
