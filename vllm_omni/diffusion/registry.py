# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

from vllm_omni.diffusion.data import OmniDiffusionConfig

_DIFFUSION_MODELS = {
    # arch:(mod_folder, mod_relname, cls_name)
    "QwenImagePipeline": (
        "qwen_image",
        "pipeline_qwen_image",
        "QwenImagePipeline",
    ),
    "ZImagePipeline": (
        "z_image",
        "pipeline_z_image",
        "ZImagePipeline",
    ),
}


def initialize_model(
    od_config: OmniDiffusionConfig,
):
    if od_config.model_class_name in _DIFFUSION_MODELS:
        mod_folder, mod_relname, cls_name = _DIFFUSION_MODELS[od_config.model_class_name]
        module_name = f"vllm_omni.diffusion.models.{mod_folder}.{mod_relname}"
        module = importlib.import_module(module_name)
        model_class = getattr(module, cls_name)
        model = model_class(od_config=od_config, prefix=mod_relname)

        # Configure VAE memory optimization settings from config
        if hasattr(model.vae, "use_slicing"):
            model.vae.use_slicing = od_config.vae_use_slicing
        if hasattr(model.vae, "use_tiling"):
            model.vae.use_tiling = od_config.vae_use_tiling

        return model
    else:
        raise ValueError(f"Model class {od_config.model_class_name} not found in diffusion model registry.")


_DIFFUSION_POST_PROCESS_FUNCS = {
    # arch: post_process_func
    # `post_process_func` function must be placed in {mod_folder}/{mod_relname}.py,
    # where mod_folder and mod_relname are  defined and mapped using `_DIFFUSION_MODELS` via the `arch` key
    "QwenImagePipeline": "get_qwen_image_post_process_func",
    "ZImagePipeline": "get_post_process_func",
}


def get_diffusion_post_process_func(od_config: OmniDiffusionConfig):
    if od_config.model_class_name in _DIFFUSION_POST_PROCESS_FUNCS:
        mod_folder, mod_relname, _ = _DIFFUSION_MODELS[od_config.model_class_name]
        func_name = _DIFFUSION_POST_PROCESS_FUNCS[od_config.model_class_name]
        module_name = f"vllm_omni.diffusion.models.{mod_folder}.{mod_relname}"
        module = importlib.import_module(module_name)
        post_process_func = getattr(module, func_name)
        return post_process_func(od_config)
    else:
        raise ValueError(
            f"Post process function for model class {od_config.model_class_name} not found in diffusion model registry."
        )
