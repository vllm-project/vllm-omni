"""Top-level package for comfyui_vllm_omni."""  # noqa: N999  # This is not a python library intended to be imported

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
    VLLMOmniGenerateImage,
    VLLMOmniGenerateVideo,
    VLLMOmniQwenTTSParams,
    VLLMOmniRemoteLoRA,
    VLLMOmniSamplingParamsList,
    VLLMOmniTTS,
    VLLMOmniUnderstanding,
    VLLMOmniVoiceClone,
    VLLMOmniWanParams,
)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    # === Generation ===
    "VLLMOmniGenerateImage": VLLMOmniGenerateImage,
    "VLLMOmniGenerateVideo": VLLMOmniGenerateVideo,
    "VLLMOmniUnderstanding": VLLMOmniUnderstanding,
    "VLLMOmniTTS": VLLMOmniTTS,
    "VLLMOmniVoiceClone": VLLMOmniVoiceClone,
    # === Params ===
    "VLLMOmniARSampling": VLLMOmniARSampling,
    "VLLMOmniDiffusionSampling": VLLMOmniDiffusionSampling,
    "VLLMOmniSamplingParamsList": VLLMOmniSamplingParamsList,
    "VLLMOmniRemoteLoRA": VLLMOmniRemoteLoRA,
    "VLLMOmniQwenTTSParams": VLLMOmniQwenTTSParams,
    "VLLMOmniWanParams": VLLMOmniWanParams,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # === Generation ===
    "VLLMOmniGenerateImage": "Generate Image",
    "VLLMOmniGenerateVideo": "Generate Video",
    "VLLMOmniUnderstanding": "Multimodality Understanding",
    "VLLMOmniTTS": "TTS (Text to Speech)",
    "VLLMOmniVoiceClone": "TTS Voice Cloning",
    # === Params ===
    "VLLMOmniARSampling": "AR Sampling Params",
    "VLLMOmniDiffusionSampling": "Diffusion Sampling Params",
    "VLLMOmniSamplingParamsList": "Multi-Stage Sampling Params List",
    "VLLMOmniRemoteLoRA": "LoRA",
    "VLLMOmniQwenTTSParams": "Qwen TTS Params",
    "VLLMOmniWanParams": "Wan Video Params",
}

WEB_DIRECTORY = "./web"
