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
    "OmniVoiceModel": (
        "omnivoice",
        "omnivoice",
        "OmniVoiceModel",
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
    "MiMoV2ASRForCausalLM": (
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
    ## VoxCPM
    "VoxCPMForConditionalGeneration": (
        "voxcpm",
        "voxcpm",
        "VoxCPMForConditionalGeneration",
    ),
    ## VoxCPM2
    "VoxCPM2TalkerForConditionalGeneration": (
        "voxcpm2",
        "voxcpm2_talker",
        "VoxCPM2TalkerForConditionalGeneration",
    ),
    ## Voxtral TTS
    "VoxtralTTSForConditionalGeneration": (
        "voxtral_tts",
        "voxtral_tts",
        "VoxtralTTSForConditionalGeneration",
    ),
    "VoxtralTTSAudioGeneration": (
        "voxtral_tts",
        "voxtral_tts_audio_generation",
        "VoxtralTTSAudioGenerationForConditionalGeneration",
    ),
    "VoxtralTTSAudioTokenizer": ("voxtral_tts", "voxtral_tts_audio_tokenizer", "VoxtralTTSAudioTokenizer"),
    ## covo_audio
    "CovoAudioForCausalLM": (
        "covo_audio",
        "covo_audio",
        "CovoAudioForConditionalGeneration",
    ),
    "CovoAudioForConditionalGeneration": (
        "covo_audio",
        "covo_audio",
        "CovoAudioForConditionalGeneration",
    ),
    "CovoAudioModel": (
        "covo_audio",
        "covo_audio",
        "CovoAudioForConditionalGeneration",
    ),
    "CovoAudioLLMModel": (
        "covo_audio",
        "covo_audio_llm",
        "CovoAudioLLMForConditionalGeneration",
    ),
    "CovoAudioCode2WavModel": (
        "covo_audio",
        "covo_audio_code2wav",
        "CovoAudioCode2WavForConditionalGeneration",
    ),
    ## MOSS-TTS-Nano
    "MossTTSNanoForCausalLM": (
        "moss_tts_nano",
        "modeling_moss_tts_nano",
        "MossTTSNanoForGeneration",
    ),
    "DyninOmniForConditionalGeneration": (
        "dynin_omni",
        "dynin_omni",
        "DyninOmniForConditionalGeneration",
    ),
    ## Ming-flash-omni-2.0
    "MingFlashOmniForConditionalGeneration": (
        "ming_flash_omni",
        "ming_flash_omni",
        "MingFlashOmniForConditionalGeneration",
    ),
    "MingFlashOmniThinkerForConditionalGeneration": (
        "ming_flash_omni",
        "ming_flash_omni_thinker",
        "MingFlashOmniThinkerForConditionalGeneration",
    ),
    "MingFlashOmniTalkerForConditionalGeneration": (
        "ming_flash_omni",
        "ming_flash_omni_talker",
        "MingFlashOmniTalkerForConditionalGeneration",
    ),
    # Alias: HF repo currently ships this architecture name in config.json
    "BailingMM2NativeForConditionalGeneration": (
        "ming_flash_omni",
        "ming_flash_omni",
        "MingFlashOmniForConditionalGeneration",
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
