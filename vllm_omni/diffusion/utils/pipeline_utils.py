# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Resolve the pipeline class a diffusion engine will run and read its capability flags."""

from typing import Any

from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.diffusion.data import OmniDiffusionConfig, uses_diffusers_adapter
from vllm_omni.diffusion.registry import DiffusionModelRegistry


def resolve_custom_pipeline_cls(custom_pipeline_args: dict[str, Any] | None) -> type | None:
    if not custom_pipeline_args or "pipeline_class" not in custom_pipeline_args:
        return None

    pipeline_cls = custom_pipeline_args["pipeline_class"]

    if isinstance(pipeline_cls, type):
        return pipeline_cls
    if isinstance(pipeline_cls, str):
        try:
            return resolve_obj_by_qualname(pipeline_cls)
        except (AttributeError, ImportError, ValueError) as exc:
            raise ValueError(f"Failed to resolve custom diffusion pipeline class {pipeline_cls!r}.") from exc
    raise TypeError(
        f"custom_pipeline_args['pipeline_class'] must be a qualified name string or a class, "
        f"got {type(pipeline_cls).__name__}"
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
