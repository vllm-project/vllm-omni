# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.models.registry import _LazyRegisteredModel, _ModelRegistry

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig, get_sp_plan_from_model
from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel

logger = init_logger(__name__)

_DIFFUSION_MODELS = {
    # arch:(mod_folder, mod_relname, cls_name)
    "QwenImagePipeline": (
        "qwen_image",
        "pipeline_qwen_image",
        "QwenImagePipeline",
    ),
    "QwenImageEditPipeline": (
        "qwen_image",
        "pipeline_qwen_image_edit",
        "QwenImageEditPipeline",
    ),
    "QwenImageEditPlusPipeline": (
        "qwen_image",
        "pipeline_qwen_image_edit_plus",
        "QwenImageEditPlusPipeline",
    ),
    "QwenImageLayeredPipeline": (
        "qwen_image",
        "pipeline_qwen_image_layered",
        "QwenImageLayeredPipeline",
    ),
    "GlmImagePipeline": (
        "glm_image",
        "pipeline_glm_image",
        "GlmImagePipeline",
    ),
    "ZImagePipeline": (
        "z_image",
        "pipeline_z_image",
        "ZImagePipeline",
    ),
    "OvisImagePipeline": (
        "ovis_image",
        "pipeline_ovis_image",
        "OvisImagePipeline",
    ),
    "WanPipeline": (
        "wan2_2",
        "pipeline_wan2_2",
        "Wan22Pipeline",
    ),
    "StableAudioPipeline": (
        "stable_audio",
        "pipeline_stable_audio",
        "StableAudioPipeline",
    ),
    "WanImageToVideoPipeline": (
        "wan2_2",
        "pipeline_wan2_2_i2v",
        "Wan22I2VPipeline",
    ),
    "LongCatImagePipeline": (
        "longcat_image",
        "pipeline_longcat_image",
        "LongCatImagePipeline",
    ),
    "BagelPipeline": (
        "bagel",
        "pipeline_bagel",
        "BagelPipeline",
    ),
    "LongCatImageEditPipeline": (
        "longcat_image",
        "pipeline_longcat_image_edit",
        "LongCatImageEditPipeline",
    ),
    "StableDiffusion3Pipeline": (
        "sd3",
        "pipeline_sd3",
        "StableDiffusion3Pipeline",
    ),
    "Flux2KleinPipeline": (
        "flux2_klein",
        "pipeline_flux2_klein",
        "Flux2KleinPipeline",
    ),
}


DiffusionModelRegistry = _ModelRegistry(
    {
        model_arch: _LazyRegisteredModel(
            module_name=f"vllm_omni.diffusion.models.{mod_folder}.{mod_relname}",
            class_name=cls_name,
        )
        for model_arch, (mod_folder, mod_relname, cls_name) in _DIFFUSION_MODELS.items()
    }
)


def initialize_model(
    od_config: OmniDiffusionConfig,
) -> nn.Module:
    """Initialize a diffusion model from the registry.

    This function:
    1. Loads the model class from the registry
    2. Instantiates the model with the config
    3. Configures VAE optimization settings
    4. Applies sequence parallelism if enabled (similar to diffusers' enable_parallelism)

    Args:
        od_config: The OmniDiffusion configuration.

    Returns:
        The initialized pipeline model.

    Raises:
        ValueError: If the model class is not found in the registry.
    """
    model_class = DiffusionModelRegistry._try_load_model_cls(od_config.model_class_name)
    if model_class is not None:
        model = model_class(od_config=od_config)
        # Configure VAE memory optimization settings from config
        if hasattr(model.vae, "use_slicing"):
            model.vae.use_slicing = od_config.vae_use_slicing
        if hasattr(model.vae, "use_tiling"):
            model.vae.use_tiling = od_config.vae_use_tiling

        # Apply sequence parallelism if enabled
        # This follows diffusers' pattern where enable_parallelism() is called
        # at model loading time, not inside individual model files
        _apply_sequence_parallel_if_enabled(model, od_config)

        return model
    else:
        raise ValueError(f"Model class {od_config.model_class_name} not found in diffusion model registry.")


def _apply_sequence_parallel_if_enabled(model, od_config: OmniDiffusionConfig) -> None:
    """Apply sequence parallelism hooks if SP is enabled.

    This is the centralized location for enabling SP, similar to diffusers'
    ModelMixin.enable_parallelism() method. It applies _sp_plan hooks to
    transformer models that define them.

    Note: Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP) in diffusers.
    We use _sp_plan instead of diffusers' _cp_plan.

    Args:
        model: The pipeline model (e.g., ZImagePipeline).
        od_config: The OmniDiffusion configuration.
    """

    try:
        sp_size = od_config.parallel_config.sequence_parallel_size
        if sp_size <= 1:
            return

        # Find transformer model(s) in the pipeline that have _sp_plan
        # Include transformer_2 for two-stage models (e.g., Wan MoE)
        transformer_attrs = ["transformer", "transformer_2", "dit", "unet"]
        applied_count = 0

        for attr in transformer_attrs:
            if not hasattr(model, attr):
                continue

            transformer = getattr(model, attr)
            if transformer is None:
                continue

            plan = get_sp_plan_from_model(transformer)
            if plan is None:
                continue

            # Create SP config
            sp_config = SequenceParallelConfig(
                ulysses_degree=od_config.parallel_config.ulysses_degree,
                ring_degree=od_config.parallel_config.ring_degree,
            )

            # Apply hooks according to the plan
            mode = (
                "hybrid"
                if sp_config.ulysses_degree > 1 and sp_config.ring_degree > 1
                else ("ulysses" if sp_config.ulysses_degree > 1 else "ring")
            )
            logger.info(
                f"Applying sequence parallelism to {transformer.__class__.__name__} ({attr}) "
                f"(sp_size={sp_size}, mode={mode}, ulysses={sp_config.ulysses_degree}, ring={sp_config.ring_degree})"
            )
            apply_sequence_parallel(transformer, sp_config, plan)
            applied_count += 1

        if applied_count == 0:
            logger.warning(
                f"Sequence parallelism is enabled (sp_size={sp_size}) but no transformer with _sp_plan found. "
                "SP hooks not applied. Consider adding _sp_plan to your transformer model."
            )

    except Exception as e:
        logger.warning(f"Failed to apply sequence parallelism: {e}. Continuing without SP hooks.")


_DIFFUSION_POST_PROCESS_FUNCS = {
    # arch: post_process_func
    # `post_process_func` function must be placed in {mod_folder}/{mod_relname}.py,
    # where mod_folder and mod_relname are  defined and mapped using `_DIFFUSION_MODELS` via the `arch` key
    "QwenImagePipeline": "get_qwen_image_post_process_func",
    "QwenImageEditPipeline": "get_qwen_image_edit_post_process_func",
    "QwenImageEditPlusPipeline": "get_qwen_image_edit_plus_post_process_func",
    "GlmImagePipeline": "get_glm_image_post_process_func",
    "ZImagePipeline": "get_post_process_func",
    "OvisImagePipeline": "get_ovis_image_post_process_func",
    "WanPipeline": "get_wan22_post_process_func",
    "StableAudioPipeline": "get_stable_audio_post_process_func",
    "WanImageToVideoPipeline": "get_wan22_i2v_post_process_func",
    "LongCatImagePipeline": "get_longcat_image_post_process_func",
    "BagelPipeline": "get_bagel_post_process_func",
    "LongCatImageEditPipeline": "get_longcat_image_post_process_func",
    "StableDiffusion3Pipeline": "get_sd3_image_post_process_func",
    "Flux2KleinPipeline": "get_flux2_klein_post_process_func",
}

_DIFFUSION_PRE_PROCESS_FUNCS = {
    # arch: pre_process_func
    # `pre_process_func` function must be placed in {mod_folder}/{mod_relname}.py,
    # where mod_folder and mod_relname are  defined and mapped using `_DIFFUSION_MODELS` via the `arch` key
    "GlmImagePipeline": "get_glm_image_pre_process_func",
    "QwenImageEditPipeline": "get_qwen_image_edit_pre_process_func",
    "QwenImageEditPlusPipeline": "get_qwen_image_edit_plus_pre_process_func",
    "LongCatImageEditPipeline": "get_longcat_image_edit_pre_process_func",
    "QwenImageLayeredPipeline": "get_qwen_image_layered_pre_process_func",
    "WanPipeline": "get_wan22_pre_process_func",
    "WanImageToVideoPipeline": "get_wan22_i2v_pre_process_func",
}


def _load_process_func(od_config: OmniDiffusionConfig, func_name: str):
    """Load and return a process function from the appropriate module."""
    mod_folder, mod_relname, _ = _DIFFUSION_MODELS[od_config.model_class_name]
    module_name = f"vllm_omni.diffusion.models.{mod_folder}.{mod_relname}"
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func(od_config)


def get_diffusion_post_process_func(od_config: OmniDiffusionConfig):
    if od_config.model_class_name not in _DIFFUSION_POST_PROCESS_FUNCS:
        return None
    func_name = _DIFFUSION_POST_PROCESS_FUNCS[od_config.model_class_name]
    return _load_process_func(od_config, func_name)


def get_diffusion_pre_process_func(od_config: OmniDiffusionConfig):
    if od_config.model_class_name not in _DIFFUSION_PRE_PROCESS_FUNCS:
        return None  # Return None if no pre-processing function is registered (for backward compatibility)
    func_name = _DIFFUSION_PRE_PROCESS_FUNCS[od_config.model_class_name]
    return _load_process_func(od_config, func_name)
