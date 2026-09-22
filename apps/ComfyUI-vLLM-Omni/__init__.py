# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N999

"""Top-level package for comfyui_vllm_omni."""  # This is not a Python library intended to be imported.

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """vLLM-Omni Team"""
__email__ = "vllm-omni@vllm.ai"
__version__ = "0.0.1"

from .comfyui_vllm_omni.node_lingbot import (
    VLLMOmniLingBotOfflineJSON,
    VLLMOmniLingBotRealtimeJSON,
    VLLMOmniLingBotWorld,
)
from .comfyui_vllm_omni.nodes import (
    VLLMOmniARSampling,
    VLLMOmniDiffusionSampling,
    VLLMOmniFastH3Deployment,
    VLLMOmniGenerateImage,
    VLLMOmniGenerateMusic,
    VLLMOmniGenerateVideo,
    VLLMOmniLatentMaskEditing,
    VLLMOmniMiniMaxH3Params,
    VLLMOmniMiniMaxH3TemporalMask,
    VLLMOmniQwenTTSParams,
    VLLMOmniRemoteLoRA,
    VLLMOmniSamplingParamsList,
    VLLMOmniTTS,
    VLLMOmniUnderstanding,
    VLLMOmniVideoReferences,
    VLLMOmniVoiceClone,
    VLLMOmniWanParams,
)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "VLLMOmniMiniMaxH3TemporalMask": VLLMOmniMiniMaxH3TemporalMask,
    "VLLMOmniLingBotWorld": VLLMOmniLingBotWorld,
    "VLLMOmniLingBotRealtimeJSON": VLLMOmniLingBotRealtimeJSON,
    "VLLMOmniLingBotOfflineJSON": VLLMOmniLingBotOfflineJSON,
    # === Generation ===
    "VLLMOmniGenerateImage": VLLMOmniGenerateImage,
    "VLLMOmniGenerateVideo": VLLMOmniGenerateVideo,
    "VLLMOmniUnderstanding": VLLMOmniUnderstanding,
    "VLLMOmniTTS": VLLMOmniTTS,
    "VLLMOmniGenerateMusic": VLLMOmniGenerateMusic,
    "VLLMOmniVoiceClone": VLLMOmniVoiceClone,
    "VLLMOmniVideoReferences": VLLMOmniVideoReferences,
    "VLLMOmniLatentMaskEditing": VLLMOmniLatentMaskEditing,
    # === Params ===
    "VLLMOmniARSampling": VLLMOmniARSampling,
    "VLLMOmniDiffusionSampling": VLLMOmniDiffusionSampling,
    "VLLMOmniSamplingParamsList": VLLMOmniSamplingParamsList,
    "VLLMOmniRemoteLoRA": VLLMOmniRemoteLoRA,
    "VLLMOmniFastH3Deployment": VLLMOmniFastH3Deployment,
    "VLLMOmniQwenTTSParams": VLLMOmniQwenTTSParams,
    "VLLMOmniWanParams": VLLMOmniWanParams,
    "VLLMOmniMiniMaxH3Params": VLLMOmniMiniMaxH3Params,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLLMOmniMiniMaxH3TemporalMask": "MiniMax-H3 Temporal Mask",
    "VLLMOmniLingBotWorld": "LingBot World (Realtime)",
    "VLLMOmniLingBotRealtimeJSON": "LingBot World (Realtime JSON)",
    "VLLMOmniLingBotOfflineJSON": "LingBot World (Offline JSON)",
    # === Generation ===
    "VLLMOmniGenerateImage": "Generate Image",
    "VLLMOmniGenerateVideo": "Generate Video",
    "VLLMOmniUnderstanding": "Multimodality Understanding",
    "VLLMOmniTTS": "TTS (Text to Speech)",
    "VLLMOmniGenerateMusic": "Generate Music",
    "VLLMOmniVoiceClone": "TTS Voice Cloning",
    "VLLMOmniVideoReferences": "Video References",
    "VLLMOmniLatentMaskEditing": "Latent Mask Editing",
    # === Params ===
    "VLLMOmniARSampling": "AR Sampling Params",
    "VLLMOmniDiffusionSampling": "Diffusion Sampling Params",
    "VLLMOmniSamplingParamsList": "Multi-Stage Sampling Params List",
    "VLLMOmniRemoteLoRA": "LoRA",
    "VLLMOmniFastH3Deployment": "FastH3 Deployment",
    "VLLMOmniQwenTTSParams": "Qwen TTS Params",
    "VLLMOmniWanParams": "Wan Video Params",
    "VLLMOmniMiniMaxH3Params": "MiniMax-H3 Video Params",
}

WEB_DIRECTORY = "./web"
