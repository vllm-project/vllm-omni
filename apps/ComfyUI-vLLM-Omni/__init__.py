"""Top-level package for comfyui_vllm_omni."""  # noqa: N999  # This is not a python library intended to be imported

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """vLLM-Omni Team"""
__email__ = "collaboration@vllm.ai"
__version__ = "0.0.1"

from .comfyui_vllm_omni.nodes import (
    VLLMOmniARSampling,
    VLLMOmniComprehension,
    VLLMOmniDiffusionSampling,
    VLLMOmniGenerateImage,
    VLLMOmniQwenTTSParams,
    VLLMOmniSamplingParamsList,
    VLLMOmniTTS,
    VLLMOmniVoiceClone,
)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    # === Generation ===
    "VLLMOmniGenerateImage": VLLMOmniGenerateImage,
    "VLLMOmniComprehension": VLLMOmniComprehension,
    "VLLMOmniTTS": VLLMOmniTTS,
    "VLLMOmniVoiceClone": VLLMOmniVoiceClone,
    # === Params ===
    "VLLMOmniARSampling": VLLMOmniARSampling,
    "VLLMOmniDiffusionSampling": VLLMOmniDiffusionSampling,
    "VLLMOmniSamplingParamsList": VLLMOmniSamplingParamsList,
    "VLLMOmniQwenTTSParams": VLLMOmniQwenTTSParams,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # === Generation ===
    "VLLMOmniGenerateImage": "Generate Image",
    "VLLMOmniComprehension": "Multimodality Comprehension",
    "VLLMOmniTTS": "TTS (Text to Speech)",
    "VLLMOmniVoiceClone": "TTS Voice Cloning",
    # === Params ===
    "VLLMOmniARSampling": "AR Sampling Params",
    "VLLMOmniDiffusionSampling": "Diffusion Sampling Params",
    "VLLMOmniSamplingParamsList": "Multi-Stage Sampling Params List",
    "VLLMOmniQwenTTSParams": "Qwen TTS Params",
}

WEB_DIRECTORY = "./web"
