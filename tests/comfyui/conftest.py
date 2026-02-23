"""
Conftest for ComfyUI-vLLM-Omni tests.

This module sets up the test environment by:
1. Adding the ComfyUI plugin to Python path
2. Mocking comfy_api.input module (AudioInput, VideoInput) since comfyui is not installed
3. Mocking comfy_extras.nodes_audio module
"""

import os
import sys
from typing import BinaryIO, TypedDict
from unittest.mock import MagicMock


def pytest_configure(config):
    """
    Called after command line options have been parsed and before test collection.
    This is the right place to set up sys.path and mock modules.
    """
    _setup_comfyui_test_environment()


def _setup_comfyui_test_environment():
    """Set up the test environment for ComfyUI plugin testing."""
    # === Add ComfyUI plugin path to allow importing comfyui_vllm_omni ===
    _COMFYUI_PLUGIN_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "apps", "ComfyUI-vLLM-Omni")
    )
    if not os.path.isdir(_COMFYUI_PLUGIN_PATH):
        raise FileNotFoundError(
            f"ComfyUI plugin not found at {_COMFYUI_PLUGIN_PATH}. "
            "If it is moved elsewhere, please update the path in this conftest.py."
        )
    if _COMFYUI_PLUGIN_PATH not in sys.path:
        sys.path.insert(0, _COMFYUI_PLUGIN_PATH)

    # Import torch after changing import paths. (To be used later)
    import torch

    # === Mock ComfyUI internal modules (comfy_api & comfy_extras) and "import" them to sys.module ===
    class AudioInput(TypedDict):
        """Mock AudioInput TypedDict from comfy_api.input"""

        waveform: torch.Tensor  # Shape: (B, C, T)
        sample_rate: int

    class VideoInput:
        """Mock VideoInput class from comfy_api.input"""

        def __init__(self, data: bytes = b"mock_video_data"):
            self._data = data

        def save_to(self, file: str | BinaryIO):
            """Save video data to file or file-like object."""
            if isinstance(file, str):
                print("Called VideoInput.save_to with file path. Saving to a path is no-op in tests.")
            else:
                file.write(self._data)

    mock_comfy_api = MagicMock()
    mock_comfy_api_input = MagicMock()
    mock_comfy_api_input.AudioInput = AudioInput
    mock_comfy_api_input.VideoInput = VideoInput
    mock_comfy_api.input = mock_comfy_api_input

    def mock_load(_: str | BinaryIO):
        """Mock nodes_audio.load that returns a waveform tensor (channels, samples) and sample rate."""
        waveform = torch.zeros((1, 24000), dtype=torch.float32)
        sample_rate = 24000
        return waveform, sample_rate

    mock_comfy_extras = MagicMock()
    mock_nodes_audio = MagicMock()
    mock_nodes_audio.load = mock_load
    mock_comfy_extras.nodes_audio = mock_nodes_audio

    # Install mock modules BEFORE importing any comfyui_vllm_omni code
    sys.modules["comfy_api"] = mock_comfy_api
    sys.modules["comfy_api.input"] = mock_comfy_api_input
    sys.modules["comfy_extras"] = mock_comfy_extras
    sys.modules["comfy_extras.nodes_audio"] = mock_nodes_audio


# Run setup immediately when conftest is imported (before pytest_configure for early loading)
_setup_comfyui_test_environment()
