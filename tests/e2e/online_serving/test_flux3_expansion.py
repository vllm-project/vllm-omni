# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flux3 image expansion tests — serving feature matrix (full_model / nightly).

Tests one serving feature at a time against the base T2I smoke request, confirming
that each parallel/acceleration knob initialises and runs end-to-end without error.

Deferred features (H100):
  - cpu_offload    — CPU offload support is handled in a follow-up change
  - cache_dit      — currently coupled with deferred layerwise CPU offload
  - hsdp           — HSDP support is handled in the same follow-up change

Enabled two-GPU features (2× H100):
  - cfg_parallel   — CFG batch split across 2 GPUs (--cfg-parallel-size 2)
  - tp_2           — Tensor parallelism across 2 GPUs (--tensor-parallel-size 2)

Deferred (not yet supported):
  - usp / ring     — sequence parallelism (_sp_plan = {})
  - vae_patch      — VAE patch parallelism (2.3 deferred)

All tests skip when FLUX3_MODEL_PATH is unset.
"""

from __future__ import annotations

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

MODEL = os.getenv("FLUX3_MODEL_PATH", "")

PROMPT = "a scenic mountain landscape at golden hour"
NEGATIVE_PROMPT = "blurry, dark, low quality"

FORM_DATA = {
    "prompt": PROMPT,
    "negative_prompt": NEGATIVE_PROMPT,
    "size": "256x256",
    "n": 1,
    "response_format": "b64_json",
    "num_inference_steps": 2,
    "guidance_scale": 3.5,
    "flow_shift": 6.93,
    "seed": 42,
}

H100_1CARD = hardware_marks(res={"cuda": "H100"}, num_cards=1)
H100_2CARD = hardware_marks(res={"cuda": "H100"}, num_cards=2)

CACHE_DIT_ARGS = ["--cache-backend", "cache_dit", "--enable-layerwise-offload"]
HSDP_ARGS = ["--use-hsdp", "--hsdp-shard-size", "2"]
HSDP_OFFLOAD_DEFERRED = pytest.mark.skip(reason="Flux3 HSDP and offload support is deferred to a follow-up change.")


def _feature_cases():
    if not MODEL:
        return []
    return [
        # ── single-GPU ─────────────────────────────────────────────────────────
        pytest.param(
            OmniServerParams(model=MODEL, server_args=["--enable-cpu-offload"]),
            id="cpu_offload",
            marks=[*H100_1CARD, HSDP_OFFLOAD_DEFERRED],
        ),
        pytest.param(
            OmniServerParams(model=MODEL, server_args=CACHE_DIT_ARGS),
            id="cache_dit",
            marks=[*H100_1CARD, HSDP_OFFLOAD_DEFERRED],
        ),
        # ── two-GPU ────────────────────────────────────────────────────────────
        pytest.param(
            OmniServerParams(model=MODEL, server_args=["--cfg-parallel-size", "2"]),
            id="cfg_parallel_2",
            marks=H100_2CARD,
        ),
        pytest.param(
            OmniServerParams(model=MODEL, server_args=["--tensor-parallel-size", "2"]),
            id="tp_2",
            marks=H100_2CARD,
        ),
        pytest.param(
            OmniServerParams(model=MODEL, server_args=HSDP_ARGS),
            id="hsdp_2",
            marks=[*H100_2CARD, HSDP_OFFLOAD_DEFERRED],
        ),
    ]


pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.full_model,
    pytest.mark.skipif(not MODEL, reason="Set FLUX3_MODEL_PATH to run."),
]


@pytest.mark.parametrize("omni_server", _feature_cases(), indirect=True)
def test_flux3_expansion(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """Flux3 T2I smoke through each serving feature — asserts successful generation."""
    responses = openai_client.send_images_generations_http_request(
        {
            "json": {
                "model": omni_server.model,
                **FORM_DATA,
            }
        }
    )
    response = responses[0]
    assert response.success, response.error_message
