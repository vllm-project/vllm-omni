"""Top-level package for comfyui_vllm_omni."""

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
    "VLLMOmniTTS": "Text-to-Speech (TTS)",
    "VLLMOmniVoiceClone": "TTS Voice Cloning",
    # === Params ===
    "VLLMOmniARSampling": "AR Sampling Params",
    "VLLMOmniDiffusionSampling": "Diffusion Sampling Params",
    "VLLMOmniSamplingParamsList": "Multi-Stage Sampling Params List",
    "VLLMOmniRemoteLoRA": "LoRA",
    "VLLMOmniQwenTTSParams": "Qwen TTS Params",
    "VLLMOmniWanParams": "Wan Video Params",
}

# New model: VoXtream2
# https://huggingface.co/herimor/voxtream2
# A 0.5B parameter zero-shot full-stream Text-to-Speech model with dynamic speaking-rate control.

# Add VoXtream2 to the list of supported models
NODE_CLASS_MAPPINGS["VLLMOmniVoXtream2"] = "VLLMOmniVoXtream2"
NODE_DISPLAY_NAME_MAPPINGS["VLLMOmniVoXtream2"] = "VoXtream2 TTS"


# Define the VoXtream2 class
class VLLMOmniVoXtream2:
    """VoXtream2 Text-to-Speech model."""

    def __init__(self):
        # Initialize the model
        pass

    def generate(self, text, speaking_rate=1.0):
        # Generate audio from text with dynamic speaking rate
        pass


# Add the VoXtream2 model to the list of supported models
WEB_DIRECTORY = "./web"
