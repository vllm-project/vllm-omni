# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Serving layer for robot policy inference via `/v1/realtime/robot/openpi`.

Flow: raw obs → engine request → actions.
The loaded policy model owns dataset transforms inside its pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np
from omegaconf import OmegaConf
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType, OmniSamplingParams
    from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

ActionOutput: TypeAlias = np.ndarray | dict[str, np.ndarray]


def _to_builtin_container(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {key: _to_builtin_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin_container(item) for item in value]
    return value


@dataclass(frozen=True)
class PolicyServerConfig:
    """OpenPI policy server handshake config.

    Values are model-specific and must be provided by the loaded policy model.
    """

    values: dict[str, Any]

    @classmethod
    def from_model_config(cls, model_config: Any) -> PolicyServerConfig:
        if isinstance(model_config, Mapping):
            raw_config = model_config.get("policy_server_config")
        else:
            raw_config = getattr(model_config, "policy_server_config", None)

        if raw_config is None:
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
        if isinstance(raw_config, cls):
            return raw_config
        if not isinstance(raw_config, Mapping):
            raise ValueError("Robot OpenPI serving requires policy_server_config.")
        return cls(_to_builtin_container(raw_config))

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin_container(self.values)


class OpenPIPromptBuilder(Protocol):
    """Builds the first stage's prompt of a multi-stage policy from an observation.

    Named by ``openpi_prompt_builder`` next to ``policy_server_config`` and
    constructed with the engine's vLLM ``ModelConfig`` and the handshake values.
    """

    def build_prompt(self, obs: Mapping[str, Any]) -> OmniPromptType: ...


class ServingRealtimeRobotOpenPI:
    """Robot policy serving layer for OpenPI protocol.

    Model-specific transform/state lives in the diffusion pipeline.
    """

    def __init__(
        self,
        engine_client: Any,
        model_name: str | None = None,
    ) -> None:
        self.engine_client = engine_client
        self.model_name = model_name
        policy_model_config = self._get_policy_model_config(engine_client)
        self.policy_server_config = PolicyServerConfig.from_model_config(policy_model_config)
        self.prompt_builder = self._load_prompt_builder(engine_client, policy_model_config, self.policy_server_config)
        self._request_counter = count()

    @classmethod
    def create_policy_server(
        cls,
        engine_client: Any,
        model_name: str | None = None,
    ) -> ServingRealtimeRobotOpenPI | None:
        try:
            return cls(engine_client=engine_client, model_name=model_name)
        except ValueError as exc:
            if "policy_server_config" not in str(exc):
                raise
            logger.info("Robot OpenPI serving disabled for model %s", model_name)
            return None

    @staticmethod
    def _get_policy_model_config(engine_client: Any) -> Any:
        """The model config that declares ``policy_server_config``, normally the diffusion stage's."""
        model_config = None
        get_od_config = getattr(engine_client, "get_diffusion_od_config", None)
        if callable(get_od_config):
            od_config = get_od_config()
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            for stage_config in getattr(engine_client, "stage_configs", []) or []:
                if getattr(stage_config, "stage_type", None) != "diffusion":
                    continue
                # Typed diffusion stages keep model-owned OpenPI handshake
                # metadata in diffusion_config.model_config. The out-of-process
                # head only has this stage view because full od_config lives in
                # the worker.
                diffusion_config = getattr(stage_config, "diffusion_config", None)
                model_config = getattr(diffusion_config, "model_config", None)
                if model_config is not None:
                    break
                engine_args = getattr(stage_config, "engine_args", None)
                model_config = getattr(engine_args, "model_config", None)
                if model_config is not None:
                    break

        if model_config is None:
            od_config = getattr(engine_client, "od_config", None)
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            model_config = getattr(engine_client, "model_config", None)
        return model_config

    @staticmethod
    def _load_prompt_builder(
        engine_client: Any,
        policy_model_config: Any,
        policy_server_config: PolicyServerConfig,
    ) -> OpenPIPromptBuilder | None:
        """Instantiate the policy's ``openpi_prompt_builder`` (a qualified class name), if any.

        A policy whose first stage is autoregressive needs a prompt (token ids,
        images) built from the observation; the deploy config names the class
        next to ``policy_server_config``. It is not sent to clients.
        """
        if isinstance(policy_model_config, Mapping):
            path = policy_model_config.get("openpi_prompt_builder")
        else:
            path = getattr(policy_model_config, "openpi_prompt_builder", None)
        if path is None:
            return None
        try:
            builder_cls = resolve_obj_by_qualname(str(path))
        except (ImportError, AttributeError, ValueError) as exc:
            raise RuntimeError(f"Cannot load openpi_prompt_builder {path!r}: {exc}") from exc
        return builder_cls(getattr(engine_client, "model_config", None), policy_server_config.to_dict())

    def reset(self, obs: dict) -> None:
        """Compatibility hook; per-connection state lives in RobotRealtimeConnection."""

    def drop_session(self, session_id: str) -> None:
        """Best-effort release of model-side session state for a closed rollout."""
        drop = getattr(self.engine_client, "drop_session", None)
        if callable(drop):
            drop(session_id)
            return
        pipeline = self._pipeline()
        for name in ("close_ar_diffusion_session", "drop_session_state"):
            close = getattr(pipeline, name, None)
            if callable(close):
                close(session_id)
                return

    def _pipeline(self) -> Any:
        engine = self.engine_client
        for attr in ("model_runner", "runner", "diffusion_model_runner"):
            runner = getattr(engine, attr, None)
            pipeline = getattr(runner, "pipeline", None) if runner is not None else None
            if pipeline is not None:
                return pipeline
        return getattr(engine, "pipeline", None)

    async def infer(self, obs: dict, *, session_id: str, reset: bool) -> ActionOutput:
        """raw obs → engine → actions."""
        # Build request, run inference through AsyncOmni
        request = self.build_request(obs, session_id=session_id, reset=reset)
        result = None
        # OpenPI policy serving is one request -> one action reply. AsyncOmni
        # exposes an async iterator, so consume it to completion and use the
        # final output, matching other non-streaming OpenAI serving paths. A
        # multi-stage policy may also return earlier stages' outputs (for
        # example reasoning text); the reply comes from the one with actions.
        async for output in self.engine_client.generate(
            prompt=request.prompt,
            request_id=request.request_id,
            sampling_params_list=self._stage_sampling_params_list(request.sampling_params),
        ):
            if result is None or _has_actions(output) or not _has_actions(result):
                result = output
        if result is None:
            raise RuntimeError("Robot OpenPI request produced no output.")

        return self._extract_actions(result)

    def _stage_sampling_params_list(self, diffusion_params: OmniDiffusionSamplingParams) -> list[OmniSamplingParams]:
        """One sampling params entry per stage.

        Multi-stage policies get every other stage's deploy defaults; the
        observation's seed applies to all stages.
        """
        from vllm_omni.entrypoints.openai.stage_params import (
            build_stage_sampling_params_list,
            get_default_sampling_params_list,
        )
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        stage_configs = list(getattr(self.engine_client, "stage_configs", None) or [])
        if len(stage_configs) <= 1:
            return [diffusion_params]
        params_list = build_stage_sampling_params_list(
            stage_configs,
            get_default_sampling_params_list(self.engine_client),
            diffusion_params=diffusion_params,
            replace_diffusion_params=True,
        )
        if diffusion_params.seed is not None:
            for params in params_list:
                if not isinstance(params, OmniDiffusionSamplingParams):
                    params.seed = diffusion_params.seed
        return params_list

    def _next_request_id(self, session_id: str) -> str:
        return f"robot-{session_id}-{next(self._request_counter)}"

    def build_request(self, obs: dict, *, session_id: str, reset: bool) -> Any:
        """Build an engine request from raw robot obs."""
        return self._build_request(obs, session_id=session_id, reset=reset)

    def _build_request(self, obs: dict, *, session_id: str, reset: bool) -> Any:
        """Build engine request from raw robot obs.

        Returns an `OmniDiffusionRequest` payload consumed by
        `AsyncOmni.generate()` and routed to the diffusion stage.
        """
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.entrypoints.openai.stage_params import (
            clone_sampling_params,
            get_default_sampling_params_list,
        )
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        # The engine applies stage default_sampling_params only to requests
        # that carry no explicit params; this endpoint always passes explicit
        # params, so start from a clone of the diffusion stage's defaults
        # (e.g. a policy deploy yaml's ``extra_args``) and layer the OpenPI
        # protocol fields on top.
        seed = obs.pop("seed", None)
        # Nested so engine knobs cannot collide with robot-defined obs keys.
        sampling = obs.get("sampling_params") or {}
        robot_obs = {key: value for key, value in obs.items() if key != "sampling_params"}
        sampling_params = OmniDiffusionSamplingParams()
        for default_params in get_default_sampling_params_list(self.engine_client):
            if isinstance(default_params, OmniDiffusionSamplingParams):
                sampling_params = clone_sampling_params(default_params)
                break

        extra_args = sampling_params.extra_args or {}
        extra_args.update(
            {
                "reset": reset,
                "session_id": session_id,
                "robot_obs": robot_obs,
            }
        )

        prompt = obs.get("prompt", "") if self.prompt_builder is None else self.prompt_builder.build_prompt(robot_obs)
        if seed is not None:
            sampling_params.seed = int(seed)
        if "num_inference_steps" in sampling:
            sampling_params.num_inference_steps = sampling["num_inference_steps"]
        sampling_params.extra_args = extra_args
        return OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=self._next_request_id(session_id),
        )

    def _extract_actions(self, result: Any) -> ActionOutput:
        """Extract actions from engine result."""
        multimodal_output = getattr(result, "multimodal_output", None)
        if not isinstance(multimodal_output, Mapping):
            raise RuntimeError("Missing multimodal_output in robot policy result")

        actions = multimodal_output.get("actions")
        if actions is None:
            raise RuntimeError("Missing multimodal_output['actions'] in robot policy result")
        if isinstance(actions, Mapping):
            return {str(key): np.asarray(value, dtype=np.float32) for key, value in actions.items()}
        return np.asarray(actions, dtype=np.float32)


def _has_actions(output: OmniRequestOutput) -> bool:
    multimodal_output = getattr(output, "multimodal_output", None)
    return isinstance(multimodal_output, Mapping) and multimodal_output.get("actions") is not None
