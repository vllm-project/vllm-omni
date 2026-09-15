# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Serving layer for robot policy inference via `/v1/realtime/robot/openpi`.

Flow: raw obs → engine request → actions.
The loaded policy model owns dataset transforms inside its pipeline.

``generate()`` needs one sampling-params object per configured stage, so this
layer reads the initialized topology: every participant gets the same normalized
session identity, only the encode-side stages get the raw observation.
"""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from typing import Any, TypeAlias

import numpy as np
from omegaconf import OmegaConf
from vllm.logger import init_logger

logger = init_logger(__name__)

ActionOutput: TypeAlias = np.ndarray | dict[str, np.ndarray]

# Roles running the observation encoders; the rest read the stage payload.
_OBSERVATION_ROLES = frozenset({"full", "encode"})

# Kept in sync with vllm_omni.engine.orchestrator; named here so this module does
# not pull the orchestrator into the serving import graph.
CLOSE_COORDINATED_SESSION = "close_coordinated_session"

# Upper bound on one coordinated close; a disconnect must not hang a socket.
DEFAULT_SESSION_CLOSE_TIMEOUT_S = 30.0


def _collect_rpc_errors(results: Any) -> list[str]:
    """Read failures out of a control-RPC reply; ``True`` is the only success."""
    errors: list[str] = []
    acknowledged = False
    for result in results or ():
        if result is True:
            acknowledged = True
            continue
        if isinstance(result, Mapping):
            errors.append(str(result.get("error") or f"unsupported reply {sorted(result)!r}"))
            continue
        errors.append(f"unexpected reply {result!r}")
    if not errors and not acknowledged:
        errors.append("no stage acknowledged the operation")
    return errors


def _resolve_coordinated_lifecycle(engine_client: Any) -> bool:
    """Whether any configured stage declares topology-coordinated lifecycle."""
    for stage_config in getattr(engine_client, "stage_configs", None) or ():
        if bool(getattr(stage_config, "coordinated_session_lifecycle", False)):
            return True
    return False


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


def _stage_type_name(stage_config: Any) -> str:
    """Read a stage config's type as a plain string (``StageType`` is a str enum)."""
    stage_type = getattr(stage_config, "stage_type", None)
    if stage_type is None:
        return ""
    return str(getattr(stage_type, "value", stage_type))


def _resolve_stage_roles(engine_client: Any) -> tuple[str | None, ...]:
    """Read one diffusion role per configured stage from the live topology.

    ``None`` marks a non-diffusion stage, which keeps its own defaults and takes
    no policy controls. No stage configs resolves to one ``full`` stage, i.e.
    the historical single-stage behavior.
    """
    from vllm_omni.config.stage_config import resolve_diffusion_stage_role

    stage_configs = list(getattr(engine_client, "stage_configs", None) or [])
    num_stages = getattr(engine_client, "num_stages", None)
    try:
        stage_count = int(num_stages)
    except (TypeError, ValueError):
        stage_count = len(stage_configs)
    if stage_count <= 0:
        stage_count = len(stage_configs) or 1

    roles: list[str | None] = []
    for index in range(stage_count):
        stage_config = stage_configs[index] if index < len(stage_configs) else None
        if stage_config is None:
            # Assume the policy stage so a single-stage engine without
            # stage_configs still gets its controls.
            roles.append("full" if stage_count == 1 else None)
            continue
        if _stage_type_name(stage_config) != "diffusion":
            roles.append(None)
            continue
        role = resolve_diffusion_stage_role(
            getattr(stage_config, "stage_role", None),
            getattr(stage_config, "model_stage", None),
        )
        roles.append(str(role.value))
    return tuple(roles)


@dataclass(frozen=True)
class _PolicyRequest:
    """One OpenPI inference request plus its per-stage sampling parameters.

    ``sampling_params_list[0]`` is the object ``request`` carries, so the seed
    ``OmniDiffusionRequest`` assigns is the one mirrored onto the other stages.
    """

    request: Any
    sampling_params_list: list[Any]

    @property
    def prompt(self) -> Any:
        return self.request.prompt

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def sampling_params(self) -> Any:
        return self.request.sampling_params


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
        self.policy_server_config = self._get_policy_server_config(engine_client)
        self._request_counter = count()
        self.stage_roles = _resolve_stage_roles(engine_client)
        self.coordinated_session_lifecycle = _resolve_coordinated_lifecycle(engine_client)
        self.session_close_timeout_s = DEFAULT_SESSION_CLOSE_TIMEOUT_S
        # Connections holding each live session id, so one socket disconnecting
        # does not close a session another is still driving.
        self._session_refcounts: Counter[str] = Counter()
        self._closing_sessions: set[str] = set()
        self._failed_closes: set[str] = set()

    @property
    def num_stages(self) -> int:
        return len(self.stage_roles)

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
    def _get_policy_server_config(engine_client: Any) -> PolicyServerConfig:
        model_config = None
        get_od_config = getattr(engine_client, "get_diffusion_od_config", None)
        if callable(get_od_config):
            od_config = get_od_config()
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            for stage_config in getattr(engine_client, "stage_configs", []) or []:
                if getattr(stage_config, "stage_type", None) != "diffusion":
                    continue
                engine_args = getattr(stage_config, "engine_args", None)
                model_config = getattr(engine_args, "model_config", None)
                if model_config is not None:
                    break

        if model_config is None:
            od_config = getattr(engine_client, "od_config", None)
            model_config = getattr(od_config, "model_config", None)

        if model_config is None:
            model_config = getattr(engine_client, "model_config", None)
        return PolicyServerConfig.from_model_config(model_config)

    def reset(self, obs: dict) -> None:
        """Compatibility hook; per-connection state lives in RobotRealtimeConnection."""

    def acquire_session(self, session_id: str) -> None:
        """Record that one more connection is using ``session_id``.

        Refused while that id is being torn down or has an unresolved failed
        close: acquiring it would attach a new rollout to state that is about to
        be destroyed, or that nothing has confirmed is gone.
        """
        key = str(session_id)
        if key in self._closing_sessions:
            raise RuntimeError(
                f"Robot OpenPI session {key!r} is being closed; a new rollout cannot take the id until that finishes."
            )
        if key in self._failed_closes:
            raise RuntimeError(
                f"Robot OpenPI session {key!r} has an unresolved failed close, so its model state "
                "may still exist. The id cannot be reused until cleanup is confirmed."
            )
        self._session_refcounts[key] += 1

    @property
    def unresolved_closes(self) -> frozenset[str]:
        """Session ids whose close failed and whose model state is unaccounted for."""
        return frozenset(self._failed_closes)

    def session_refcount(self, session_id: str) -> int:
        return int(self._session_refcounts.get(str(session_id), 0))

    async def release_session(self, session_id: str) -> bool:
        """Drop one connection's hold on ``session_id``; close it when last out.

        True when this call actually released model-side state.
        """
        key = str(session_id)
        remaining = self._session_refcounts.get(key, 0) - 1
        if remaining > 0:
            self._session_refcounts[key] = remaining
            logger.debug(
                "Robot OpenPI session %s still held by %d connection(s); not closing",
                key,
                remaining,
            )
            return False
        self._session_refcounts.pop(key, None)
        # Marked while the close runs so a concurrent acquire cannot take the id,
        # and recorded on failure so the unaccounted-for state is not forgotten
        # just because the last connection's reference is gone.
        self._closing_sessions.add(key)
        try:
            await self.close_session(key)
        except BaseException:
            # Cancellation included: an aborted await does not prove the remote
            # close stopped, so the id stays unresolved rather than reusable.
            self._failed_closes.add(key)
            raise
        finally:
            self._closing_sessions.discard(key)
        self._failed_closes.discard(key)
        return True

    async def close_session(self, session_id: str) -> None:
        """Release model-side session state for ``session_id``.

        A coordinated topology has its state in worker processes the serving layer
        cannot touch, so the close goes through the engine's control plane and is
        only reported done once the coordinator has retired every participant. An
        in-process client keeps the direct pipeline hook.
        """
        if self.coordinated_session_lifecycle:
            await self._close_remote_session(session_id)
            return
        result = self.drop_session(session_id)
        if inspect.isawaitable(result):
            await result

    async def _close_remote_session(self, session_id: str) -> None:
        """Close through the orchestrator's lifecycle coordinator, or fail loudly."""
        rpc = getattr(self.engine_client, "collective_rpc", None)
        if not callable(rpc):
            raise RuntimeError(
                f"Robot OpenPI cannot close session {session_id!r}: this deployment declares a "
                "coordinated session lifecycle, but its engine client exposes no collective_rpc() "
                "to reach the orchestrator. Worker state would leak on every disconnect."
            )
        results = await rpc(
            method=CLOSE_COORDINATED_SESSION,
            args=(str(session_id),),
            timeout=self.session_close_timeout_s,
        )
        errors = _collect_rpc_errors(results)
        if errors:
            raise RuntimeError(f"Robot OpenPI close of session {session_id!r} was not confirmed: " + "; ".join(errors))

    def drop_session(self, session_id: str) -> Any:
        """Best-effort release of model-side session state for an in-process client.

        Returns the hook's result so ``close_session`` can await an async hook. A
        model with no per-session state legitimately has no hook; a coordinated
        deployment never reaches here.
        """
        drop = getattr(self.engine_client, "drop_session", None)
        if callable(drop):
            return drop(session_id)
        pipeline = self._pipeline()
        for name in ("close_ar_diffusion_session", "drop_session_state"):
            close = getattr(pipeline, name, None)
            if callable(close):
                return close(session_id)
        logger.debug(
            "Robot OpenPI found no session-release hook for %s; this policy keeps no session state",
            session_id,
        )
        return None

    def _pipeline(self) -> Any:
        engine = self.engine_client
        for attr in ("model_runner", "runner", "diffusion_model_runner"):
            runner = getattr(engine, attr, None)
            pipeline = getattr(runner, "pipeline", None) if runner is not None else None
            if pipeline is not None:
                return pipeline
        return getattr(engine, "pipeline", None)

    async def infer(
        self,
        obs: dict,
        *,
        session_id: str,
        reset: bool,
        close_session: bool = False,
    ) -> ActionOutput:
        """raw obs → engine → actions."""
        # Build request, run inference through AsyncOmni
        request = self.build_request(
            obs,
            session_id=session_id,
            reset=reset,
            close_session=close_session,
        )
        result = None
        # OpenPI policy serving is one request -> one action reply. AsyncOmni
        # exposes an async iterator, so consume it to completion and use the
        # final output, matching other non-streaming OpenAI serving paths.
        async for output in self.engine_client.generate(
            prompt=request.prompt,
            request_id=request.request_id,
            sampling_params_list=request.sampling_params_list,
        ):
            result = output
        if result is None:
            raise RuntimeError("Robot OpenPI request produced no output.")

        return self._extract_actions(result)

    def _next_request_id(self, session_id: str) -> str:
        return f"robot-{session_id}-{next(self._request_counter)}"

    def build_request(
        self,
        obs: dict,
        *,
        session_id: str,
        reset: bool,
        close_session: bool = False,
    ) -> _PolicyRequest:
        """Build an engine request from raw robot obs."""
        return self._build_request(
            obs,
            session_id=session_id,
            reset=reset,
            close_session=close_session,
        )

    def _stage_default_params(self) -> list[Any]:
        """Per-stage templates: the engine's initialized defaults, padded."""
        from vllm_omni.entrypoints.openai.stage_params import get_default_sampling_params_list

        defaults = get_default_sampling_params_list(self.engine_client)
        return [defaults[index] if index < len(defaults) else None for index in range(self.num_stages)]

    def _normalize_session_controls(
        self,
        extra_args: Mapping[str, Any],
        *,
        session_id: str,
        reset: bool,
        close_session: bool,
    ) -> tuple[str, bool, bool]:
        """Resolve session identity and lifecycle intent once for every stage.

        A typed tick is authoritative, matching the runner: its controls replace
        the flat ones rather than being OR-ed, so a typed ``False`` cannot be
        overridden by a stale ``True`` left in a stage default.
        """
        from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest

        tick = ARDiffusionTickRequest.from_extra_args(extra_args)
        if tick is not None:
            return tick.session_id, bool(tick.reset), bool(tick.close_session)
        return str(session_id), bool(reset), bool(close_session)

    def _build_request(
        self,
        obs: dict,
        *,
        session_id: str,
        reset: bool,
        close_session: bool = False,
    ) -> _PolicyRequest:
        """Build engine request and per-stage sampling params from raw robot obs.

        Each diffusion stage gets its own clone of that stage's initialized
        defaults, so a deploy yaml's per-stage ``extra_args`` survive. Only the
        encode-side roles receive the raw observation.
        """
        from vllm import SamplingParams

        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params
        from vllm_omni.experimental.ar_diffusion.tick_protocol import AR_DIFFUSION_TICK_KEY
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        seed = obs.pop("seed", None)
        # A direct caller may drive the rollout with a typed tick; keep it on
        # extra_args rather than in the observation.
        typed_tick = obs.pop(AR_DIFFUSION_TICK_KEY, None)

        # The engine applies stage defaults only to requests without explicit
        # params, and this endpoint always passes them, so start from a clone of
        # each stage's defaults and layer the protocol fields on top.
        templates = self._stage_default_params()
        sampling_params_list: list[Any] = []
        resolved_session_id = str(session_id)
        resolved_reset = bool(reset)
        resolved_close = bool(close_session)
        controls_resolved = False

        for index, role in enumerate(self.stage_roles):
            template = templates[index]
            if role is None:
                # Keeps its own parameter type and takes no policy controls.
                sampling_params_list.append(
                    clone_sampling_params(template) if template is not None else SamplingParams()
                )
                continue
            if isinstance(template, OmniDiffusionSamplingParams):
                # A deep copy, so no two stages share an extra_args mapping and
                # the engine defaults are never mutated.
                params = clone_sampling_params(template)
            else:
                params = OmniDiffusionSamplingParams()

            extra_args = dict(params.extra_args or {})
            if typed_tick is not None:
                extra_args[AR_DIFFUSION_TICK_KEY] = typed_tick
            if not controls_resolved:
                resolved_session_id, resolved_reset, resolved_close = self._normalize_session_controls(
                    extra_args,
                    session_id=session_id,
                    reset=reset,
                    close_session=close_session,
                )
                controls_resolved = True

            extra_args["session_id"] = resolved_session_id
            extra_args["reset"] = resolved_reset
            extra_args["close_session"] = resolved_close
            if role in _OBSERVATION_ROLES:
                extra_args["robot_obs"] = obs
            else:
                # These read the stage payload; a second copy of the observation
                # would ship for nothing and could drift from what encode used.
                extra_args.pop("robot_obs", None)
            params.extra_args = extra_args
            if seed is not None:
                params.seed = int(seed)
            sampling_params_list.append(params)

        self._validate_stage_sampling_params(sampling_params_list)

        prompt = obs.get("prompt", "")
        request = OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=sampling_params_list[0],
            request_id=self._next_request_id(resolved_session_id),
        )
        # OmniDiffusionRequest auto-seeds an unseeded request; mirror that seed
        # so every participant derives the same generator state.
        resolved_seed = getattr(request.sampling_params, "seed", None)
        for params in sampling_params_list[1:]:
            if getattr(params, "seed", None) is None:
                params.seed = resolved_seed
        return _PolicyRequest(request=request, sampling_params_list=sampling_params_list)

    def _validate_stage_sampling_params(self, sampling_params_list: list[Any]) -> None:
        """Fail before generate() when the topology and params disagree."""
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        if len(sampling_params_list) != self.num_stages:
            raise ValueError(
                f"Robot OpenPI built {len(sampling_params_list)} sampling params for a "
                f"{self.num_stages}-stage topology (roles={list(self.stage_roles)}); "
                "AsyncOmni.generate() requires exactly one per stage."
            )
        if not sampling_params_list:
            raise ValueError(
                "Robot OpenPI resolved an empty stage topology; the policy deployment "
                "exposes no stages to send the observation to."
            )
        for index, (role, params) in enumerate(zip(self.stage_roles, sampling_params_list, strict=True)):
            if role is None:
                continue
            if not isinstance(params, OmniDiffusionSamplingParams):
                raise ValueError(
                    f"Robot OpenPI stage {index} has role {role!r} but resolved "
                    f"{type(params).__name__} sampling params; a diffusion stage requires "
                    "OmniDiffusionSamplingParams."
                )
        if not any(role in _OBSERVATION_ROLES for role in self.stage_roles):
            raise ValueError(
                "Robot OpenPI found no encode-capable diffusion stage in the deployed "
                f"topology (roles={list(self.stage_roles)}); the raw observation would "
                "reach no encoder. Declare a stage with stage_role 'full' or 'encode'."
            )

    @staticmethod
    def _raise_engine_error(result: Any) -> None:
        """Surface an engine-reported failure before reading actions off it.

        A lifecycle cleanup failure arrives as an error on the final output; the
        actionable message must reach the client instead of being replaced by a
        "Missing actions" complaint about the empty payload it came with.
        """
        error = getattr(result, "error", None)
        if not error:
            return
        error_type = getattr(result, "error_type", None)
        detail = f"{error_type}: {error}" if error_type else str(error)
        raise RuntimeError(f"Robot OpenPI request failed: {detail}")

    def _extract_actions(self, result: Any) -> ActionOutput:
        """Extract actions from engine result."""
        self._raise_engine_error(result)
        multimodal_output = getattr(result, "multimodal_output", None)
        if not isinstance(multimodal_output, Mapping):
            raise RuntimeError("Missing multimodal_output in robot policy result")

        actions = multimodal_output.get("actions")
        if actions is None:
            raise RuntimeError("Missing multimodal_output['actions'] in robot policy result")
        if isinstance(actions, Mapping):
            return {str(key): np.asarray(value, dtype=np.float32) for key, value in actions.items()}
        return np.asarray(actions, dtype=np.float32)
