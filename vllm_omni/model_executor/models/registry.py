from vllm.model_executor.models.registry import _VLLM_MODELS, _LazyRegisteredModel, _ModelRegistry

_OMNI_MODELS = {
    "Qwen2_5OmniForConditionalGeneration": (
        "qwen2_5_omni",
        "Qwen2_5OmniForConditionalGeneration",
    ),
    "Qwen2_5OmniThinkerModel": (
        "qwen2_5_omni_thinker",
        "Qwen2_5OmniThinkerForConditionalGeneration",
    ),  # noqa: E501
    "Qwen2_5OmniTalkerModel": (
        "qwen2_5_omni_talker",
        "Qwen2_5OmniTalkerForConditionalGeneration",
    ),  # noqa: E501
    "Qwen2_5OmniToken2WavModel": (
        "qwen2_5_omni_token2wav",
        "Qwen2_5OmniToken2WavForConditionalGenerationVLLM",
    ),
    "Qwen2_5OmniToken2WavDiTModel": (
        "qwen2_5_omni_token2wav",
        "Qwen2_5OmniToken2WavModel",
    ),
    "Qwen2ForCausalLM_old": ("qwen2_old", "Qwen2ForCausalLM"),  # need to discuss
}

_VLLM_OMNI_MODELS = {
    **_VLLM_MODELS,
    **_OMNI_MODELS,
}


OmniModelRegistry = _ModelRegistry(
    {
        **{
            model_arch: _LazyRegisteredModel(
                module_name=f"vllm.model_executor.models.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
        },
        **{
            model_arch: _LazyRegisteredModel(
                module_name=f"vllm_omni.model_executor.models.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_relname, cls_name) in _OMNI_MODELS.items()
        },
    }
)
