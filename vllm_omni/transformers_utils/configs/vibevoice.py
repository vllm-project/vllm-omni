"""VibeVoice config registration with transformers AutoConfig."""

from transformers import AutoConfig

from vllm_omni.model_executor.models.vibevoice_tts.configuration_vibevoice import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceConfig,
    VibeVoiceDiffusionHeadConfig,
    VibeVoiceSemanticTokenizerConfig,
)

AutoConfig.register("vibevoice", VibeVoiceConfig, exist_ok=True)
AutoConfig.register("vibevoice_acoustic_tokenizer", VibeVoiceAcousticTokenizerConfig, exist_ok=True)
AutoConfig.register("vibevoice_semantic_tokenizer", VibeVoiceSemanticTokenizerConfig, exist_ok=True)
AutoConfig.register("vibevoice_diffusion_head", VibeVoiceDiffusionHeadConfig, exist_ok=True)

__all__ = [
    "VibeVoiceConfig",
    "VibeVoiceAcousticTokenizerConfig",
    "VibeVoiceSemanticTokenizerConfig",
    "VibeVoiceDiffusionHeadConfig",
]
