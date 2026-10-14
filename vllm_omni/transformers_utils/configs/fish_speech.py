"""Fish Speech S2 Pro config registration with transformers AutoConfig.

Registers FishSpeechConfig (model_type="fish_qwen3_omni") and sub-configs
so that ``AutoConfig.from_pretrained("fishaudio/s2-pro")`` returns the
correct config class.
"""

from transformers import AutoConfig

from vllm_omni.model_executor.models.fish_speech.configuration_fish_speech import (
    FishSpeechConfig,
    FishSpeechFastARConfig,
    FishSpeechSlowARConfig,
)

AutoConfig.register("fish_qwen3_omni", FishSpeechConfig)
AutoConfig.register("fish_qwen3", FishSpeechSlowARConfig)
AutoConfig.register("fish_qwen3_audio_decoder", FishSpeechFastARConfig)

__all__ = [
    "FishSpeechConfig",
    "FishSpeechSlowARConfig",
    "FishSpeechFastARConfig",
]
