# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end stepwise smoke test for LingBot-World v2.

One ``AsyncOmni.generate()`` must stream one latent chunk per AR block while
keeping a single request identity, which is what separates this path from the
tick control plane exercised by ``test_lingbot_world_v2.py``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_test

MODEL = os.environ.get(
    "VLLM_OMNI_LINGBOT_WORLD_V2_CHECKPOINT_PATH",
    "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
)
_IMAGE_PATH = os.environ.get("VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH")

_HEIGHT = 480
_WIDTH = 832
_FRAMES_PER_BLOCK = 3
_TEMPORAL_COMPRESSION = 4
_REQUEST_ID = "lingbot-world-v2-stepwise-e2e"
# One three-frame action list per generated chunk.
_CAMERA_ACTION_SCRIPT = [
    [["w"], ["w"], ["w"]],
    [["a"], [], []],
    [[], [], []],
]
_NUM_CHUNKS = len(_CAMERA_ACTION_SCRIPT)
_NUM_FRAMES = (_NUM_CHUNKS * _FRAMES_PER_BLOCK - 1) * _TEMPORAL_COMPRESSION + 1

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
    pytest.mark.skipif(
        _IMAGE_PATH is None,
        reason="VLLM_OMNI_LINGBOT_WORLD_V2_IMAGE_PATH is required",
    ),
]


def _chunk_metadata(output: Any) -> dict[str, Any]:
    multimodal = getattr(output, "multimodal_output", None) or {}
    metadata = multimodal.get("metadata") if isinstance(multimodal, dict) else None
    assert isinstance(metadata, dict), "streamed chunk is missing its metadata envelope"
    ar_diffusion = metadata.get("ar_diffusion")
    assert isinstance(ar_diffusion, dict), "streamed chunk is missing ar_diffusion metadata"
    return dict(ar_diffusion)


async def _stream_chunks(image: Path) -> list[tuple[torch.Tensor, dict[str, Any], bool]]:
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine = AsyncOmni(
        model=MODEL,
        engine_backend="vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=1,
        step_execution=True,
        diffusion_streaming_output=True,
        model_config={
            "ar_diffusion_height": _HEIGHT,
            "ar_diffusion_width": _WIDTH,
            "ar_diffusion_kv_config": {"gpu_memory_fraction": 0.6, "warmup_cudagraph": True},
        },
    )
    sampling = OmniDiffusionSamplingParams(
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_NUM_FRAMES,
        num_inference_steps=4,
        max_sequence_length=512,
        seed=42,
        output_type="latent",
        extra_args={"flow_shift": 5.0, "camera_action_script": _CAMERA_ACTION_SCRIPT},
    )
    prompt = {
        "prompt": "The camera moves slowly forward through the scene.",
        "multi_modal_data": {"image": str(image)},
    }

    chunks: list[tuple[torch.Tensor, dict[str, Any], bool]] = []
    try:
        async for output in engine.generate(prompt, sampling, request_id=_REQUEST_ID):
            images = getattr(output, "images", None)
            if not images:
                continue
            assert len(images) == 1, "each AR block yields exactly one latent tensor"
            latent = images[0].detach().float().cpu()
            chunks.append((latent, _chunk_metadata(output), bool(getattr(output, "finished", False))))
    finally:
        engine.shutdown()
    return chunks


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_lingbot_world_v2_stepwise_streams_one_chunk_per_block() -> None:
    """Load the real checkpoint and stream N chunks from a single request."""

    assert _IMAGE_PATH is not None
    chunks = asyncio.run(_stream_chunks(Path(_IMAGE_PATH).expanduser().resolve()))

    assert len(chunks) == _NUM_CHUNKS
    expected_shape = (
        1,
        16,
        _FRAMES_PER_BLOCK,
        _HEIGHT // 8,
        _WIDTH // 8,
    )
    for chunk_index, (latent, metadata, finished) in enumerate(chunks):
        assert tuple(latent.shape) == expected_shape
        assert torch.isfinite(latent).all()
        # The request is the session, so both ids must agree on every chunk,
        # and chunk indices must be contiguous from zero.
        assert metadata["chunk_index"] == chunk_index
        assert metadata["session_id"] == metadata["request_id"]
        assert metadata["applied_event_ids"] == []
        assert finished is (chunk_index == _NUM_CHUNKS - 1)

    # The engine suffixes the caller's id to keep it unique; what this path
    # guarantees is that one rollout keeps one identity from start to finish.
    request_ids = {metadata["request_id"] for _, metadata, _ in chunks}
    assert len(request_ids) == 1
    assert request_ids.pop().startswith(_REQUEST_ID)

    concatenated = torch.cat([latent for latent, _, _ in chunks], dim=2)
    assert concatenated.shape[2] == _NUM_CHUNKS * _FRAMES_PER_BLOCK
