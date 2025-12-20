from vllm.model_executor.models.registry import (
    _VLLM_MODELS,
    _LazyRegisteredModel,
    _ModelRegistry,
    ModelRegistry as VLLMModelRegistry,
)

_OMNI_MODELS = {
    "Qwen2_5OmniForConditionalGeneration": (
        "qwen2_5_omni",
        "qwen2_5_omni",
        "Qwen2_5OmniForConditionalGeneration",
    ),
    "Qwen2_5OmniThinkerModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_thinker",
        "Qwen2_5OmniThinkerForConditionalGeneration",
    ),  # noqa: E501
    "Qwen2_5OmniTalkerModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_talker",
        "Qwen2_5OmniTalkerForConditionalGeneration",
    ),  # noqa: E501
    "Qwen2_5OmniToken2WavModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_token2wav",
        "Qwen2_5OmniToken2WavForConditionalGenerationVLLM",
    ),
    "Qwen2_5OmniToken2WavDiTModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_token2wav",
        "Qwen2_5OmniToken2WavModel",
    ),
    "Qwen2ForCausalLM_old": ("qwen2_5_omni", "qwen2_old", "Qwen2ForCausalLM"),  # need to discuss
    # Qwen3 Omni MoE models
    "Qwen3OmniMoeForConditionalGeneration": (
        "qwen3_omni",
        "qwen3_omni",
        "Qwen3OmniMoeForConditionalGeneration",
    ),
    "Qwen3OmniMoeThinkerForConditionalGeneration": (
        "qwen3_omni",
        "qwen3_omni_moe_thinker",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    ),
    "Qwen3OmniMoeTalkerForConditionalGeneration": (
        "qwen3_omni",
        "qwen3_omni_moe_talker",
        "Qwen3OmniMoeTalkerForConditionalGeneration",
    ),
    "Qwen3OmniMoeCode2Wav": (
        "qwen3_omni",
        "qwen3_omni_code2wav",
        "Qwen3OmniMoeCode2Wav",
    ),
    "MammothModa2ARForConditionalGeneration": (
        "mammoth_moda2",
        "mammoth_moda2_ar",
        "MammothModa2ARForConditionalGeneration",
    ),
    "MammothModa2DiTForConditionalGeneration": (
        "mammoth_moda2",
        "mammoth_moda2_dit",
        "MammothModa2DiTForConditionalGeneration",
    ),
    # 统一入口：仿照 Qwen2_5OmniForConditionalGeneration，让每个 stage 都使用同一个 model_arch，
    # 再通过 engine_args.model_stage（ar/dit/vae）在 __init__ 内选择实际子模块。
    "MammothModa2ForConditionalGeneration": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2ForConditionalGeneration",
    ),
    # 顶层入口，匹配 HF 配置里的 architectures= ["Mammothmoda2Model"]
    "Mammothmoda2Model": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2ForConditionalGeneration",
    ),
    # 文本骨干架构别名，便于按名称显式加载。
    "MammothModa2Qwen2ForCausalLM": (
        "mammoth_moda2",
        "mammoth_moda2_ar",
        "MammothModa2Qwen2ForCausalLM",
    ),
}

_VLLM_OMNI_MODELS = {
    **_VLLM_MODELS,
    **_OMNI_MODELS,
}

# 兼容基础 vLLM 的全局 ModelRegistry，确保 architectures=["Mammothmoda2Model"] 可被识别。
if hasattr(VLLMModelRegistry, "register_model"):
    VLLMModelRegistry.register_model(
        "Mammothmoda2Model",
        "vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2:MammothModa2ForConditionalGeneration",
    )

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
                module_name=f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items()
        },
    }
)
