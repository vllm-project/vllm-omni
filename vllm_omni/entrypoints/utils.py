# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import types
from dataclasses import fields, is_dataclass
from typing import Any, get_args, get_origin

from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.stage_utils import _to_dict
from vllm_omni.inputs.data import OmniSamplingParams

logger = init_logger(__name__)


def inject_omni_kv_config(stage: Any, omni_conn_cfg: dict[str, Any], omni_from: str, omni_to: str) -> None:
    """Inject connector configuration into stage engine arguments."""
    # Prepare omni_kv_config dict
    omni_conf_dict = {}
    try:
        # Access engine_args safely (might be OmegaConf or dict)
        existing_args = stage.engine_args
        if hasattr(existing_args, "get"):
            _oc = existing_args.get("omni_kv_config", None)
            if _oc:
                if hasattr(_oc, "items"):  # dict-like
                    omni_conf_dict = dict(_oc)
                else:  # object?
                    omni_conf_dict = _to_dict(_oc)
    except Exception:
        omni_conf_dict = {}

    # Inject connector info
    omni_conf_dict["connector_config"] = omni_conn_cfg
    omni_conf_dict["omni_from_stage"] = omni_from
    omni_conf_dict["omni_to_stage"] = omni_to

    # Write back to engine_args
    try:
        if hasattr(stage.engine_args, "__setitem__"):
            stage.engine_args["omni_kv_config"] = omni_conf_dict
        else:
            setattr(stage.engine_args, "omni_kv_config", omni_conf_dict)
    except Exception as e:
        # Fallback for OmegaConf or similar if direct set fails?
        logger.error(f"Failed to inject omni connector config into stage: {e}")


def parse_stage_overrides(value: Any) -> dict[str, dict[str, Any]] | None:
    """Parse and validate the shape of per-stage JSON overrides."""
    if not value:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--stage-overrides is not valid JSON: {exc}. Got: {value!r}") from exc
    else:
        parsed = value

    if not isinstance(parsed, dict):
        raise ValueError(
            "--stage-overrides must be a JSON object mapping stage_id -> overrides, "
            f"got {type(parsed).__name__}: {parsed!r}"
        )
    if not parsed:
        return None

    for stage_id, overrides in parsed.items():
        if not isinstance(stage_id, str) or not stage_id.isascii() or not stage_id.isdigit():
            raise ValueError(
                f"--stage-overrides keys must be non-negative integer stage ids (as strings), got {stage_id!r}"
            )
        if not isinstance(overrides, dict):
            raise ValueError(
                f"--stage-overrides[{stage_id!r}] must be an object, got {type(overrides).__name__}: {overrides!r}"
            )

    return parsed


def get_final_stage_id_for_e2e(
    output_modalities: list[str] | None, default_modalities: list[str], stage_list: list
) -> int:
    """Get the final stage id for e2e.

    Args:
        stage_list: List of stage configurations

    Returns:
        Final stage id for e2e
    """
    last_stage_id = len(stage_list) - 1
    if output_modalities is not None:
        prompt_modalities = []
        for modality in output_modalities:
            if modality not in default_modalities:
                logger.warning(f"Invalid output modality: {modality}, ignoring it")
                # TODO: if user specifies unsupported modalities, invalid it and raise an error
                continue
            prompt_modalities.append(modality)
        output_modalities = prompt_modalities
    else:
        output_modalities = default_modalities

    try:
        final_stage_id_for_e2e = last_stage_id
        for _sid in range(last_stage_id, -1, -1):
            if (
                getattr(stage_list[_sid], "final_output", False)
                and stage_list[_sid].final_output_type in output_modalities
            ):
                final_stage_id_for_e2e = _sid
                break
    except Exception as e:
        logger.debug(
            "[Orchestrator] Failed to determine final stage for E2E; \
                falling back to last: %s",
            e,
            exc_info=True,
        )
        final_stage_id_for_e2e = last_stage_id

    return final_stage_id_for_e2e


def filter_dataclass_kwargs(cls: Any, kwargs: dict) -> dict:
    """Filter kwargs to only include fields defined in the dataclass.

    Args:
        cls: Dataclass type
        kwargs: Keyword arguments to filter

    Returns:
        Filtered keyword arguments containing only valid dataclass fields
    """
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs must be a dictionary")

    def _filter_value(value: Any, annotation: Any) -> Any:
        """Recursively filter nested dict/list values based on dataclass annotations."""
        if annotation is None:
            return value

        origin = get_origin(annotation)
        if origin is None:
            if isinstance(annotation, type) and is_dataclass(annotation) and isinstance(value, dict):
                return filter_dataclass_kwargs(annotation, value)
            return value

        if origin in (list, tuple, set):
            args = get_args(annotation)
            inner = args[0] if args else None
            if isinstance(value, list | tuple | set):
                return type(value)(_filter_value(v, inner) for v in value)
            return value

        if origin is dict:
            args = get_args(annotation)
            val_type = args[1] if len(args) > 1 else None
            if isinstance(value, dict):
                return {k: _filter_value(v, val_type) for k, v in value.items()}
            return value

        if origin is types.UnionType or origin is getattr(types, "UnionType", None):
            for arg in get_args(annotation):
                if isinstance(arg, type) and is_dataclass(arg) and isinstance(value, dict):
                    return filter_dataclass_kwargs(arg, value)
                # Try container-style filtering for union members
                filtered = _filter_value(value, arg)
                if filtered is not value:
                    return filtered
            return value

        return value

    valid_fields = {f.name: f for f in fields(cls) if f.init}
    filtered_kwargs = {}
    for k, v in kwargs.items():
        if k not in valid_fields:
            logger.warning(
                "Dropping unknown %s field %r (not declared on the dataclass)",
                cls.__name__,
                k,
            )
            continue
        field = valid_fields[k]
        filtered_kwargs[k] = _filter_value(v, field.type)

    return filtered_kwargs


# The following code detects if the process is running in a container and if
# PID host is available. If so, we can use process-scoped memory tracking;
# otherwise we need sequential init locks.


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def in_container() -> bool:
    # Common Docker signal
    if os.path.exists("/.dockerenv"):
        return True

    # cgroup markers (works for Docker/containerd/K8s/Podman in many setups)
    cg = _read_text("/proc/1/cgroup") or ""
    markers = ("docker", "containerd", "kubepods", "libpod", "podman")
    return any(m in cg for m in markers)


def has_pid_host() -> bool | None:
    """
    Returns:
      True  -> very likely running with --pid=host (host PID namespace)
      False -> very likely isolated PID namespace (default)
      None  -> cannot determine
    """
    # Strong signal: in host pid namespace, PID 2 is usually kthreadd
    comm2 = _read_text("/proc/2/comm")
    if comm2 is not None:
        comm2 = comm2.strip()
        if comm2 == "kthreadd":
            return True
        # If PID 2 exists and is NOT kthreadd, we're almost certainly not in host pid ns
        return False

    # Fallback: check for other low-numbered kernel threads (best-effort)
    for pid, name in [(3, "rcu_gp"), (4, "rcu_par_gp"), (10, "ksoftirqd/0")]:
        comm = _read_text(f"/proc/{pid}/comm")
        if comm is not None:
            if comm.strip() == name:
                return True
            else:
                return False

    return False


def detect_pid_host() -> bool:
    ic = in_container()
    if not ic:
        return True

    return has_pid_host() is True


### Helpers for handling delta messages
def coerce_param_message_types(params: list[OmniSamplingParams], is_streaming: bool):
    """Iterate over the sampling params and convert to the message types
    to DELTA messages, if streaming is enabled, or FINAL_ONLY if
    it's disabled, while respecting `.skip_clone` on the params.

    This is needed to avoid emitting redundant multimodal data.
    """
    # Coerce vLLM's default output kinds as needed to handle streaming
    # (i.e., DELTA output kind). Note that this is only applied to non
    # Diffusion sampling params.
    #
    # NOTE: Hidden states will still be passed between stages.
    for idx, sp in enumerate(params):
        # For OmniDiffusionParams don't set output kind
        if isinstance(sp, SamplingParams):
            params[idx] = maybe_coerce_to_message_type(sp, is_streaming)
    return params


def maybe_coerce_to_message_type(params: SamplingParams, is_streaming: bool):
    """If this is a CUMULATIVE message, coerce it to DELTA if streaming, otherwise FINAL_ONLY."""
    target_type = RequestOutputKind.DELTA if is_streaming else RequestOutputKind.FINAL_ONLY
    if params.output_kind == target_type:
        return params
    elif is_streaming and params.output_kind == RequestOutputKind.FINAL_ONLY:
        logger.warning("Request appears to be streaming, but got request type final only!")
    elif not is_streaming and params.output_kind == RequestOutputKind.DELTA:
        logger.warning("Request appears to not be streaming, but got request type delta!")

    if not params.skip_clone:
        params = params.clone()
        params.skip_clone = True
    params.output_kind = target_type
    return params


class PureDiffusionLauncherAdapter:
    """vLLM launcher compatibility shim for pure-diffusion mode.

    The upstream launcher's shutdown path reads
    ``app.state.engine_client.vllm_config.shutdown_timeout``
    (vllm/entrypoints/launcher.py), but ``AsyncOmni.vllm_config`` returns
    ``None`` when the pipeline has no comprehension stage (pure diffusion),
    which crashes ``handle_shutdown`` with AttributeError and hangs server
    teardown (workers force-killed, spurious resource_tracker noise).

    This adapter only overrides the ``vllm_config`` property with a minimal
    fallback carrying ``shutdown_timeout`` and forwards every other attribute
    to the wrapped engine client, so the pure-diffusion detection
    (``get_vllm_config()`` still returns ``None``) is unaffected.
    """

    def __init__(self, engine_client: Any, shutdown_timeout: float) -> None:
        object.__setattr__(self, "_wrapped", engine_client)
        object.__setattr__(self, "_shutdown_timeout", float(shutdown_timeout))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    @property
    def vllm_config(self) -> Any:
        return types.SimpleNamespace(shutdown_timeout=self._shutdown_timeout)
