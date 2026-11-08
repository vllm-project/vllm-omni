# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Resolve diffusion pipeline classes, validate startup contracts, and read capabilities."""

import inspect
from collections.abc import Mapping

from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.diffusion.data import OmniDiffusionConfig, uses_diffusers_adapter
from vllm_omni.diffusion.registry import DiffusionModelRegistry


class CustomPipelineConfigError(ValueError):
    """Raised when a custom pipeline cannot satisfy the startup contract."""


def _qualified_identity(value: object) -> tuple[str, str]:
    """Return safe class and module labels without rendering arbitrary values."""
    if isinstance(value, str):
        module, separator, name = value.rpartition(".")
        return (name if separator else value, module if separator else "<unknown>")
    if isinstance(value, type):
        return value.__qualname__, getattr(value, "__module__", "<unknown>")
    value_type = type(value)
    return value_type.__qualname__, getattr(value_type, "__module__", "<unknown>")


def _config_error(
    pipeline_ref: object,
    missing_fields: tuple[str, ...],
    *,
    extension_ref: object | None = None,
) -> CustomPipelineConfigError:
    pipeline_name, pipeline_module = _qualified_identity(pipeline_ref)
    details = [
        "Invalid custom diffusion pipeline configuration.",
        f"pipeline_class={pipeline_name!r}",
        f"module={pipeline_module!r}",
        f"missing_fields={', '.join(missing_fields)}",
    ]
    if extension_ref is not None:
        extension_name, extension_module = _qualified_identity(extension_ref)
        details.extend(
            [
                f"worker_extension_class={extension_name!r}",
                f"worker_extension_module={extension_module!r}",
            ]
        )
    return CustomPipelineConfigError("; ".join(details))


def resolve_custom_pipeline_cls(custom_pipeline_args: Mapping[str, object] | None) -> type | None:
    if not custom_pipeline_args or "pipeline_class" not in custom_pipeline_args:
        return None

    pipeline_cls = custom_pipeline_args["pipeline_class"]

    if isinstance(pipeline_cls, type):
        return pipeline_cls
    if isinstance(pipeline_cls, str):
        try:
            resolved = resolve_obj_by_qualname(pipeline_cls)
        except (AttributeError, ImportError, OSError, RuntimeError, SyntaxError, TypeError, ValueError):
            raise _config_error(pipeline_cls, ("import",)) from None
        if isinstance(resolved, type):
            return resolved
        raise _config_error(pipeline_cls, ("pipeline_class (class)",))
    raise _config_error(pipeline_cls, ("pipeline_class (qualified name or class)",))


def _constructor_missing_fields(pipeline_cls: type) -> tuple[str, ...]:
    """Find constructor fields that prevent ``pipeline_cls(od_config=...)``."""
    try:
        signature = inspect.signature(pipeline_cls)
    except (TypeError, ValueError):
        # Some extension types do not expose a signature. Preserve their existing
        # behavior and let the loader report a runtime constructor error if needed.
        return ()

    parameters = signature.parameters
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    missing: list[str] = []
    od_config = parameters.get("od_config")
    if od_config is None and not accepts_kwargs:
        missing.append("constructor.od_config")
    elif od_config is not None and od_config.kind is inspect.Parameter.POSITIONAL_ONLY:
        missing.append("constructor.od_config")

    for name, parameter in parameters.items():
        if name == "od_config":
            continue
        if (
            parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and parameter.default is inspect.Parameter.empty
        ):
            missing.append(f"constructor.{name}")
    return tuple(missing)


def _resolve_worker_extension_cls(extension_ref: str | type) -> type:
    if isinstance(extension_ref, str):
        try:
            extension_cls = resolve_obj_by_qualname(extension_ref)
        except (AttributeError, ImportError, OSError, RuntimeError, SyntaxError, TypeError, ValueError):
            raise _config_error(extension_ref, ("worker_extension_class (import)",)) from None
    else:
        extension_cls = extension_ref
    if not isinstance(extension_cls, type):
        raise _config_error(extension_ref, ("worker_extension_class (class)",))
    return extension_cls


def validate_custom_pipeline_config(
    custom_pipeline_args: Mapping[str, object] | None,
    worker_extension_cls: str | type | None = None,
) -> None:
    """Validate custom pipeline startup requirements without constructing it.

    The validator runs only for configurations containing ``pipeline_class``.
    It checks the loader's keyword-constructor contract and, when an explicit
    worker extension is provided, the hook used to install the pipeline.
    """
    if not custom_pipeline_args or "pipeline_class" not in custom_pipeline_args:
        return

    pipeline_ref = custom_pipeline_args["pipeline_class"]
    pipeline_cls = resolve_custom_pipeline_cls(custom_pipeline_args)
    if pipeline_cls is None:
        return

    missing_fields = _constructor_missing_fields(pipeline_cls)
    if missing_fields:
        raise _config_error(pipeline_ref, missing_fields)

    if worker_extension_cls is None:
        return
    extension_cls = _resolve_worker_extension_cls(worker_extension_cls)
    if not callable(getattr(extension_cls, "re_init_pipeline", None)):
        raise _config_error(
            pipeline_ref,
            ("worker_extension.re_init_pipeline",),
            extension_ref=extension_cls,
        )


def resolve_pipeline_cls(od_config: OmniDiffusionConfig) -> type | None:
    """Return the pipeline class the engine will execute, or ``None`` if unknown.

    A custom pipeline wins over the registry; the Diffusers adapter wins over
    the model's native pipeline when ``diffusion_load_format="diffusers"``.
    Capability flags (``supports_*`` class attributes) are read from this class.
    """
    model_cls = resolve_custom_pipeline_cls(getattr(od_config, "custom_pipeline_args", None))
    if model_cls is not None:
        return model_cls
    if uses_diffusers_adapter(od_config):
        return DiffusionModelRegistry._try_load_model_cls("DiffusersAdapterPipeline")
    return DiffusionModelRegistry._try_load_model_cls(getattr(od_config, "model_class_name", None))


def pipeline_supports(od_config: OmniDiffusionConfig, capability: str) -> bool:
    return bool(getattr(resolve_pipeline_cls(od_config), capability, False))


def supports_pre_tokenized_prompt_ids(od_config: OmniDiffusionConfig) -> bool | None:
    """Whether the pipeline honors ``OmniCustomPrompt.prompt_ids`` verbatim.

    In-tree pipelines tokenize text themselves and never read the ids, so a
    request that carries them would silently run unconditioned; they default
    to ``False`` unless the class declares ``supports_pre_tokenized_prompt_ids
    = True``. A user-supplied ``custom_pipeline_args["pipeline_class"]`` is
    outside the engine's knowledge (RL rollouts feed exactly these ids to such
    pipelines), so without a declaration it returns ``None`` and admission
    skips the check; it may still declare ``False`` to opt into rejection.
    """
    flag = "supports_pre_tokenized_prompt_ids"
    custom_cls = resolve_custom_pipeline_cls(getattr(od_config, "custom_pipeline_args", None))
    if custom_cls is not None:
        declared = getattr(custom_cls, flag, None)
        return None if declared is None else bool(declared)
    return pipeline_supports(od_config, flag)
