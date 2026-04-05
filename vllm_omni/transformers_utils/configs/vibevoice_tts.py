"""VibeVoice TTS config registration with transformers AutoConfig.

Registers VibeVoiceTTSConfig (model_type="vibevoice") so that
``AutoConfig.from_pretrained("microsoft/VibeVoice-1.5B")`` returns
the correct config class in all processes (including child workers).
"""

from transformers import AutoConfig

from vllm_omni.model_executor.models.vibevoice_tts.configuration_vibevoice_tts import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceDiffusionHeadConfig,
    VibeVoiceSemanticTokenizerConfig,
    VibeVoiceTTSConfig,
)

AutoConfig.register("vibevoice", VibeVoiceTTSConfig)
AutoConfig.register("vibevoice_diffusion_head", VibeVoiceDiffusionHeadConfig)
AutoConfig.register("vibevoice_acoustic_tokenizer", VibeVoiceAcousticTokenizerConfig)
AutoConfig.register("vibevoice_semantic_tokenizer", VibeVoiceSemanticTokenizerConfig)

__all__ = [
    "VibeVoiceTTSConfig",
    "VibeVoiceDiffusionHeadConfig",
    "VibeVoiceAcousticTokenizerConfig",
    "VibeVoiceSemanticTokenizerConfig",
]
