from vllm.model_executor.models.registry import (
    _VLLM_MODELS,
    _LazyRegisteredModel,
    _ModelRegistry,
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
    ),
    "Qwen2_5OmniTalkerModel": (
        "qwen2_5_omni",
        "qwen2_5_omni_talker",
        "Qwen2_5OmniTalkerForConditionalGeneration",
    ),
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
    "CosyVoice3Model": (
        "cosyvoice3",
        "cosyvoice3",
        "CosyVoice3Model",
    ),
    "MammothModa2Qwen2ForCausalLM": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2Qwen2ForCausalLM",
    ),
    "MammothModa2ARForConditionalGeneration": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2ARForConditionalGeneration",
    ),
    "MammothModa2DiTPipeline": (
        "mammoth_moda2",
        "pipeline_mammothmoda2_dit",
        "MammothModa2DiTPipeline",
    ),
    "MammothModa2ForConditionalGeneration": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2ForConditionalGeneration",
    ),
    "Mammothmoda2Model": (
        "mammoth_moda2",
        "mammoth_moda2",
        "MammothModa2ForConditionalGeneration",
    ),
    "Qwen3TTSForConditionalGeneration": (
        "qwen3_tts",
        "qwen3_tts_talker",
        "Qwen3TTSTalkerForConditionalGeneration",
    ),
    "Qwen3TTSTalkerForConditionalGeneration": (
        "qwen3_tts",
        "qwen3_tts_talker",
        "Qwen3TTSTalkerForConditionalGeneration",
    ),
    "Qwen3TTSCode2Wav": (
        "qwen3_tts",
        "qwen3_tts_code2wav",
        "Qwen3TTSCode2Wav",
    ),
    ## mimo_audio
    "MiMoAudioModel": (
        "mimo_audio",
        "mimo_audio",
        "MiMoAudioForConditionalGeneration",
    ),
    "MiMoAudioLLMModel": (
        "mimo_audio",
        "mimo_audio_llm",
        "MiMoAudioLLMForConditionalGeneration",
    ),
    "MiMoAudioToken2WavModel": (
        "mimo_audio",
        "mimo_audio_code2wav",
        "MiMoAudioToken2WavForConditionalGenerationVLLM",
    ),
    ## glm_image
    "GlmImageForConditionalGeneration": (
        "glm_image",
        "glm_image_ar",
        "GlmImageForConditionalGeneration",
    ),
    "OmniBagelForConditionalGeneration": (
        "bagel",
        "bagel",
        "OmniBagelForConditionalGeneration",
    ),
    "HunyuanImage3ForCausalMM": (
        "hunyuan_image3",
        "hunyuan_image3",
        "HunyuanImage3ForConditionalGeneration",
    ),
    ## fish_speech (Fish Speech S2 Pro)
    "FishSpeechSlowARForConditionalGeneration": (
        "fish_speech",
        "fish_speech_slow_ar",
        "FishSpeechSlowARForConditionalGeneration",
    ),
    "FishSpeechDACDecoder": (
        "fish_speech",
        "fish_speech_dac_decoder",
        "FishSpeechDACDecoder",
    ),
    # MiniCPM-o 2.6 Omni models
    "MiniCPMO26OmniForConditionalGeneration": (
        "minicpmo_2_6",
        "minicpmo_2_6_omni",
        "MiniCPMO26OmniForConditionalGeneration",
    ),
    "MiniCPMO26OmniLLMModel": (
        "minicpmo_2_6",
        "minicpmo_2_6_omni_llm",
        "MiniCPMO26OmniLLMForConditionalGeneration",
    ),
    "MiniCPMO26OmniTTSModel": (
        "minicpmo_2_6",
        "minicpmo_2_6_omni_tts",
        "MiniCPMO26OmniTTSForConditionalGeneration",
    ),
    "MiniCPMO26OmniT2WModel": (
        "minicpmo_2_6",
        "minicpmo_2_6_omni_t2w",
        "MiniCPMO26OmniT2WForConditionalGeneration",
    ),
    # MiniCPM-o 4.5 Omni models
    "MiniCPMO45OmniForConditionalGeneration": (
        "minicpmo_4_5",
        "minicpmo_4_5_omni",
        "MiniCPMO45OmniForConditionalGeneration",
    ),
    "MiniCPMO45OmniLLMModel": (
        "minicpmo_4_5",
        "minicpmo_4_5_omni_llm",
        "MiniCPMO45OmniLLMForConditionalGeneration",
    ),
    "MiniCPMO45OmniTTSModel": (
        "minicpmo_4_5",
        "minicpmo_4_5_omni_tts",
        "MiniCPMO45OmniTTSForConditionalGeneration",
    ),
    "MiniCPMO45OmniT2WModel": (
        "minicpmo_4_5",
        "minicpmo_4_5_omni_t2w",
        "MiniCPMO45OmniT2WForConditionalGeneration",
    ),
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
                module_name=f"vllm_omni.model_executor.models.{mod_folder}.{mod_relname}",
                class_name=cls_name,
            )
            for model_arch, (mod_folder, mod_relname, cls_name) in _OMNI_MODELS.items()
        },
    }
)
