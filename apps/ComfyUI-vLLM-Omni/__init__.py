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

from .comfyui_vllm_omni.nodes import (
    VLLMOmniARSampling,
    VLLMOmniDiffusionSampling,
    VLLMOmniFastH3Deployment,
    VLLMOmniGenerateImage,
    VLLMOmniGenerateMusic,
    VLLMOmniGenerateVideo,
    VLLMOmniLatentMaskEditing,
    VLLMOmniMiniMaxH3Guide,
    VLLMOmniMiniMaxH3Params,
    VLLMOmniMiniMaxH3TemporalMask,
    VLLMOmniQwenTTSParams,
    VLLMOmniRemoteLoRA,
    VLLMOmniRestoreVideo,
    VLLMOmniSamplingParamsList,
    VLLMOmniTTS,
    VLLMOmniUnderstanding,
    VLLMOmniVideoReferences,
    VLLMOmniVoiceClone,
    VLLMOmniWanParams,
    VLLMOmniYuE2Params,
)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    # === Generation ===
    "VLLMOmniGenerateImage": VLLMOmniGenerateImage,
    "VLLMOmniGenerateVideo": VLLMOmniGenerateVideo,
    "VLLMOmniRestoreVideo": VLLMOmniRestoreVideo,
    "VLLMOmniUnderstanding": VLLMOmniUnderstanding,
    "VLLMOmniTTS": VLLMOmniTTS,
    "VLLMOmniGenerateMusic": VLLMOmniGenerateMusic,
    "VLLMOmniVoiceClone": VLLMOmniVoiceClone,
    "VLLMOmniVideoReferences": VLLMOmniVideoReferences,
    "VLLMOmniLatentMaskEditing": VLLMOmniLatentMaskEditing,
    "VLLMOmniMiniMaxH3TemporalMask": VLLMOmniMiniMaxH3TemporalMask,
    "VLLMOmniMiniMaxH3Guide": VLLMOmniMiniMaxH3Guide,
    # === Params ===
    "VLLMOmniARSampling": VLLMOmniARSampling,
    "VLLMOmniDiffusionSampling": VLLMOmniDiffusionSampling,
    "VLLMOmniSamplingParamsList": VLLMOmniSamplingParamsList,
    "VLLMOmniRemoteLoRA": VLLMOmniRemoteLoRA,
    "VLLMOmniFastH3Deployment": VLLMOmniFastH3Deployment,
    "VLLMOmniQwenTTSParams": VLLMOmniQwenTTSParams,
    "VLLMOmniWanParams": VLLMOmniWanParams,
    "VLLMOmniYuE2Params": VLLMOmniYuE2Params,
    "VLLMOmniMiniMaxH3Params": VLLMOmniMiniMaxH3Params,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # === Generation ===
    "VLLMOmniGenerateImage": "Generate Image",
    "VLLMOmniGenerateVideo": "Generate Video",
    "VLLMOmniRestoreVideo": "Restore Video (SeedVR2)",
    "VLLMOmniUnderstanding": "Multimodality Understanding",
    "VLLMOmniTTS": "TTS (Text to Speech)",
    "VLLMOmniGenerateMusic": "Generate Music",
    "VLLMOmniVoiceClone": "TTS Voice Cloning",
    "VLLMOmniVideoReferences": "Video References",
    "VLLMOmniLatentMaskEditing": "Latent Mask Editing",
    "VLLMOmniMiniMaxH3TemporalMask": "MiniMax-H3 Temporal Mask",
    "VLLMOmniMiniMaxH3Guide": "MiniMax H3 Add Guide",
    # === Params ===
    "VLLMOmniARSampling": "AR Sampling Params",
    "VLLMOmniDiffusionSampling": "Diffusion Sampling Params",
    "VLLMOmniSamplingParamsList": "Multi-Stage Sampling Params List",
    "VLLMOmniRemoteLoRA": "LoRA",
    "VLLMOmniFastH3Deployment": "FastH3 Deployment",
    "VLLMOmniQwenTTSParams": "Qwen TTS Params",
    "VLLMOmniWanParams": "Wan Video Params",
    "VLLMOmniYuE2Params": "YuE2 Params",
    "VLLMOmniMiniMaxH3Params": "MiniMax-H3 Video Params",
}

WEB_DIRECTORY = "./web"

# The `model` dropdown is fed by a same-origin proxy route on the ComfyUI
# server (see comfyui_vllm_omni/routes.py): a browser-side fetch would resolve
# `localhost` to the user's own machine for remote deployments.
try:
    from server import PromptServer

    from .comfyui_vllm_omni.routes import register_routes

    register_routes(PromptServer.instance)
except ImportError:
    # Outside a ComfyUI runtime (e.g. unit tests) there is no server module.
    pass
