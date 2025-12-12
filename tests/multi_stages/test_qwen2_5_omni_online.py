# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end tests for Qwen2.5-Omni online serving API.

These tests require:
- GPU with sufficient memory (tested on H200-140G)
- Model weights downloaded from HuggingFace

Run with:
    # If model already cached:
    pytest tests/multi_stages/test_qwen2_5_omni_online.py -v -s

    # To allow model download:
    VLLM_OMNI_DOWNLOAD_MODEL=1 pytest tests/multi_stages/test_qwen2_5_omni_online.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ..utils import (
    _check_model_available,
    _create_local_128_image_path,
    get_image_url_from_path,
)

# Skip entire module if basic dependencies are missing
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

CI_STAGE_CONFIG_PATH = str(Path(__file__).parent / "stage_configs" / "qwen2_5_omni_ci.yaml")

# Check GPU availability
_HAS_GPU = torch.cuda.is_available()
_GPU_COUNT = torch.cuda.device_count() if _HAS_GPU else 0

# Model configuration
_MODEL_NAME = "Qwen/Qwen2.5-Omni-3B"
_SEED = 42
models = [_MODEL_NAME]


_MODEL_AVAILABLE = _check_model_available(_MODEL_NAME)

# Skip markers
requires_gpu = pytest.mark.skipif(not _HAS_GPU, reason="GPU not available")
requires_model = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason=f"Model {_MODEL_NAME} not available. Set VLLM_OMNI_DOWNLOAD_MODEL=1 to download.",
)
requires_multi_gpu = pytest.mark.skipif(_GPU_COUNT < 2, reason="Multiple GPUs required for full pipeline")

# System prompt for Qwen2.5-Omni
_DEFAULT_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)

# Sampling parameters for the multi-stage pipeline (thinker/talker/code2wav)
_SAMPLING_PARAMS_LIST = [
    {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 10,
        "seed": _SEED,
        "detokenize": True,
        "repetition_penalty": 1.1,
    },
    {
        "temperature": 0.9,
        "top_p": 0.8,
        "top_k": 40,
        "max_tokens": 10,
        "seed": _SEED,
        "detokenize": True,
        "repetition_penalty": 1.05,
        "stop_token_ids": [8294],
    },
    {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 10,
        "seed": _SEED,
        "detokenize": True,
        "repetition_penalty": 1.1,
    },
]


def _create_openai_client(base_url: str):
    from openai import OpenAI

    return OpenAI(
        api_key="EMPTY",
        base_url=f"{base_url}/v1",
    )


@requires_gpu
@requires_model
@pytest.mark.parametrize("omni_server", models, indirect=True)
class TestQwen25OmniOnlineServingWithOpenAIClient:
    """Tests using the official OpenAI Python client."""

    def test_openai_client_text(self, omni_server):
        """Test using OpenAI Python client for text chat."""
        client = _create_openai_client(omni_server["base_url"])

        response = client.chat.completions.create(
            model=omni_server["model"],
            messages=[
                {"role": "system", "content": _DEFAULT_SYSTEM},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            temperature=0.0,
            max_tokens=32,
            seed=_SEED,
            extra_body={"sampling_params_list": _SAMPLING_PARAMS_LIST},
        )

        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    def test_openai_client_image(self, omni_server):
        """Test using OpenAI Python client with image input."""
        client = _create_openai_client(omni_server["base_url"])

        # Create a local 128x128 PNG and encode to data URL to avoid external fetch
        image_path = _create_local_128_image_path()
        image_url = get_image_url_from_path(image_path)
        os.remove(image_path)

        response = client.chat.completions.create(
            model=omni_server["model"],
            messages=[
                {"role": "system", "content": _DEFAULT_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "What is this?"},
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=64,
            seed=_SEED,
            extra_body={"sampling_params_list": _SAMPLING_PARAMS_LIST},
        )

        assert response.choices is not None
        assert len(response.choices) > 0

        content = response.choices[0].message.content
        print(f"Image response: {content}")
        assert len(content) > 5

    def test_openai_client_list_models(self, omni_server):
        """Test listing models via OpenAI Python client."""
        client = _create_openai_client(omni_server["base_url"])

        models_response = client.models.list()
        model_ids = [m.id for m in models_response.data]

        assert len(model_ids) > 0
        print(f"Models via OpenAI client: {model_ids}")
