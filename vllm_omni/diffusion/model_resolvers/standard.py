"""Shared resolvers for diffusion model families with straightforward rules."""

from __future__ import annotations

from vllm_omni.diffusion.model_resolvers.types import ModelConfigLike


def resolve_standard_model_class_name(
    model: str | None,
    cfg: ModelConfigLike | None,
) -> str | None:
    """Resolve straightforward model families by config markers.

    ``cfg`` is expected to be a model config view with optional keys such as
    ``model_type`` and ``architectures``.
    """
    del model  # Reserved for future path-based standard-family resolution.
    if cfg is None:
        return None

    model_type = cfg.get("model_type")
    architectures = cfg.get("architectures") or []

    if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
        return "BagelPipeline"
    if model_type == "neo_chat":
        return "SenseNovaU1Pipeline"
    if "BailingMM2NativeForConditionalGeneration" in architectures or model_type in (
        "bailingmm_moe_v2_lite",
        "ming_flash_omni",
        "ming_flash_omni_thinker",
    ):
        return "MingImagePipeline"
    if model_type == "nextstep":
        return "NextStep11Pipeline"
    if model_type == "s2v":
        return "WanS2VPipeline"
    if model_type == "Gr00tN1d7" or "Gr00tN1d7" in architectures:
        return "Gr00tN1d7Pipeline"

    return None
