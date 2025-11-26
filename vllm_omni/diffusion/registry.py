import importlib

from vllm_omni.diffusion.data import OmniDiffusionConfig

_DIFFUSION_MODELS = {
    # arch:(mod_folder, mod_relname, cls_name)
    "QwenImagePipeline": (
        "qwen_image",
        "qwen_image",
        "QwenImagePipeline",
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
        return model
    else:
        raise ValueError(f"Model class {od_config.model_class_name} not found in diffusion model registry.")
