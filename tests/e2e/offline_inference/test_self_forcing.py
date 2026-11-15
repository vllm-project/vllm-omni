# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end test for the Self-Forcing causal DMD pipeline.

Request mode rolls the clip out against request-local K/V; step execution
streams one AR block per chunk from the runner's paged session. Both draw the
block noise and every re-noise from the same seeded generator in the same
order, so the streamed chunks must reproduce the request-mode latents.
"""

from __future__ import annotations

import asyncio
import os

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path

# A directory assembled by tools/prepare_self_forcing_checkpoint.py.
MODEL = os.environ.get("VLLM_OMNI_SELF_FORCING_MODEL_PATH")
_PROMPT = (
    "A red fox trots through fresh snow in a quiet birch forest at sunrise, soft golden light filtering "
    "between the trees, the camera tracking alongside it at a low angle."
)
_HEIGHT, _WIDTH = 480, 832
_FRAMES_PER_BLOCK = 3
_NUM_CHUNKS = 3
_NUM_FRAMES = (_NUM_CHUNKS * _FRAMES_PER_BLOCK - 1) * 4 + 1
_SEED = 0

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
    pytest.mark.skipif(MODEL is None, reason="set VLLM_OMNI_SELF_FORCING_MODEL_PATH to an assembled model"),
]


def _sampling():
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionSamplingParams(
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_NUM_FRAMES,
        num_inference_steps=4,
        seed=_SEED,
        output_type="latent",
    )


def _request_mode_latents() -> torch.Tensor:
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=MODEL, enforce_eager=True)
    try:
        return omni.generate({"prompt": _PROMPT}, _sampling())[0].images[0].detach().float().cpu()
    finally:
        omni.shutdown()


async def _streamed_chunks() -> list[tuple[torch.Tensor, dict, bool]]:
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    engine = AsyncOmni(model=MODEL, deploy_config=get_deploy_config_path("self_forcing_stepwise.yaml"))
    chunks = []
    try:
        async for output in engine.generate({"prompt": _PROMPT}, _sampling(), request_id="self-forcing-e2e"):
            if not output.images:
                continue
            metadata = output.multimodal_output["metadata"]["ar_diffusion"]
            chunks.append((output.images[0].detach().float().cpu(), dict(metadata), bool(output.finished)))
    finally:
        engine.shutdown()
    return chunks


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_streamed_chunks_reproduce_request_mode() -> None:
    expected = _request_mode_latents()
    assert tuple(expected.shape) == (1, 16, _NUM_CHUNKS * _FRAMES_PER_BLOCK, _HEIGHT // 8, _WIDTH // 8)

    chunks = asyncio.run(_streamed_chunks())
    assert len(chunks) == _NUM_CHUNKS
    for index, (latent, metadata, finished) in enumerate(chunks):
        assert metadata["chunk_index"] == index
        assert metadata["session_id"] == metadata["request_id"]
        assert finished is (index == _NUM_CHUNKS - 1)
        torch.testing.assert_close(latent, expected[:, :, index * _FRAMES_PER_BLOCK : (index + 1) * _FRAMES_PER_BLOCK])
