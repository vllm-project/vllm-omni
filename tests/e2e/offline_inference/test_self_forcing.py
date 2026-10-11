# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end tests for the Self-Forcing-family causal DMD pipelines.

Request mode rolls the clip out against request-local K/V; step execution
streams one AR block per chunk from the runner's paged session. Both draw the
block noise and every re-noise from the same seeded generator in the same
order, so the streamed chunks must reproduce the request-mode latents. The
minWM case also drives its PRoPE camera branch from a trajectory.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path

_HEIGHT, _WIDTH = 480, 832
_NUM_CHUNKS = 3
_SEED = 0


@dataclass(frozen=True)
class _Case:
    # Environment variable naming a directory assembled by tools/prepare_self_forcing_checkpoint.py.
    model_env: str
    deploy_config: str
    frames_per_block: int
    prompt: str
    extra_args: dict[str, str] = field(default_factory=dict)

    @property
    def model(self) -> str | None:
        return os.environ.get(self.model_env)

    @property
    def num_frames(self) -> int:
        return (_NUM_CHUNKS * self.frames_per_block - 1) * 4 + 1


_CASES = [
    pytest.param(
        _Case(
            model_env="VLLM_OMNI_SELF_FORCING_MODEL_PATH",
            deploy_config="self_forcing_stepwise.yaml",
            frames_per_block=3,
            prompt="A red fox trots through fresh snow in a quiet birch forest at sunrise, soft golden light "
            "filtering between the trees, the camera tracking alongside it at a low angle.",
        ),
        id="self_forcing",
    ),
    pytest.param(
        _Case(
            model_env="VLLM_OMNI_MINWM_ACTION2V_MODEL_PATH",
            deploy_config="minwm_action2v_stepwise.yaml",
            frames_per_block=4,
            prompt="A brick pathway winds through manicured green bushes beside a dark building with framed "
            "butterfly displays on a quiet, overcast day after rain.",
            # Three 4-frame blocks: a strafe, a dolly and a turn across the block boundaries.
            extra_args={"trajectory": "a*4,w*4,j*3"},
        ),
        id="minwm_action2v",
    ),
]

pytestmark = [pytest.mark.slow, pytest.mark.diffusion]


def _sampling(case: _Case):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionSamplingParams(
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=case.num_frames,
        num_inference_steps=4,
        seed=_SEED,
        output_type="latent",
        extra_args=dict(case.extra_args),
    )


def _request_mode_latents(case: _Case) -> torch.Tensor:
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=case.model, enforce_eager=True)
    try:
        return omni.generate({"prompt": case.prompt}, _sampling(case))[0].images[0].detach().float().cpu()
    finally:
        omni.shutdown()


async def _streamed_chunks(case: _Case) -> list[tuple[torch.Tensor, dict, bool]]:
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    engine = AsyncOmni(model=case.model, deploy_config=get_deploy_config_path(case.deploy_config))
    chunks = []
    try:
        async for output in engine.generate({"prompt": case.prompt}, _sampling(case), request_id="causal-dmd-e2e"):
            if not output.images:
                continue
            metadata = output.multimodal_output["metadata"]["ar_diffusion"]
            chunks.append((output.images[0].detach().float().cpu(), dict(metadata), bool(output.finished)))
    finally:
        engine.shutdown()
    return chunks


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("case", _CASES)
def test_streamed_chunks_reproduce_request_mode(case: _Case) -> None:
    if case.model is None:
        pytest.skip(f"set {case.model_env} to an assembled model")
    block = case.frames_per_block
    expected = _request_mode_latents(case)
    assert tuple(expected.shape) == (1, 16, _NUM_CHUNKS * block, _HEIGHT // 8, _WIDTH // 8)

    chunks = asyncio.run(_streamed_chunks(case))
    assert len(chunks) == _NUM_CHUNKS
    for index, (latent, metadata, finished) in enumerate(chunks):
        assert metadata["chunk_index"] == index
        assert metadata["session_id"] == metadata["request_id"]
        assert finished is (index == _NUM_CHUNKS - 1)
        torch.testing.assert_close(latent, expected[:, :, index * block : (index + 1) * block])
