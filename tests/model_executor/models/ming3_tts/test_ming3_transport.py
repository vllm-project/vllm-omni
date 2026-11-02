# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.ming3_tts import (
    llm2audio_vae,
    llm2audio_vae_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


class _Request:
    def __init__(self, request_id: str = "req-1"):
        self.external_req_id = request_id
        self.finished = False

    def is_finished(self):
        return self.finished


def _manager(**extra):
    return SimpleNamespace(
        request_payload={},
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "initial_latent_chunk_size": 2,
                    "latent_chunk_size": 3,
                    "latent_left_context": 0,
                    **extra,
                }
            }
        ),
    )


def _step(value: float):
    return {
        "ming3_latent_patch": torch.full((1, 1, 64), value),
        "ming3_has_patch": torch.tensor([True]),
    }


def test_async_transport_emits_initial_regular_and_final_chunks_once():
    manager = _manager()
    request = _Request()

    assert llm2audio_vae_async_chunk(manager, _step(1), request) is None
    initial = llm2audio_vae_async_chunk(manager, _step(2), request)
    assert initial.latent.shape == (2, 1, 64)
    assert not bool(initial.meta.finished)

    assert llm2audio_vae_async_chunk(manager, _step(3), request) is None
    assert llm2audio_vae_async_chunk(manager, _step(4), request) is None
    regular = llm2audio_vae_async_chunk(manager, _step(5), request)
    assert regular.latent.shape == (3, 1, 64)
    assert not bool(regular.meta.finished)

    request.finished = True
    final = llm2audio_vae_async_chunk(manager, _step(6), request)
    assert final.latent.shape == (1, 1, 64)
    assert bool(final.meta.finished)
    assert llm2audio_vae_async_chunk(manager, None, request) is None


def test_async_transport_ramps_chunks_up_to_the_regular_size():
    manager = _manager(initial_latent_chunk_size=1, latent_chunk_size=5, latent_chunk_ramp=True)
    request = _Request()

    sizes = []
    for step in range(20):
        chunk = llm2audio_vae_async_chunk(manager, _step(step), request)
        if chunk is not None:
            sizes.append(chunk.latent.shape[0])
    request.finished = True
    final = llm2audio_vae_async_chunk(manager, None, request)

    # 1 -> 2 -> 4 -> capped at 5, then the remainder on finish.
    assert sizes == [1, 2, 4, 5, 5]
    assert final.latent.shape[0] == 3
    assert bool(final.meta.finished)


def test_async_transport_keeps_interleaved_requests_isolated():
    manager = _manager()
    request_a = _Request("req-a")
    request_b = _Request("req-b")

    assert llm2audio_vae_async_chunk(manager, _step(1), request_a) is None
    assert llm2audio_vae_async_chunk(manager, _step(10), request_b) is None

    chunk_a = llm2audio_vae_async_chunk(manager, _step(2), request_a)
    chunk_b = llm2audio_vae_async_chunk(manager, _step(20), request_b)

    assert torch.equal(chunk_a.latent[:, 0, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(chunk_b.latent[:, 0, 0], torch.tensor([10.0, 20.0]))
    assert chunk_a.kv_metadata["ming_request_id"] == "req-a"
    assert chunk_b.kv_metadata["ming_request_id"] == "req-b"


def test_sync_transport_preserves_all_accumulated_latents():
    latents = torch.randn(7, 1, 64)
    stage_output = SimpleNamespace(
        finished=True,
        request_id="req-sync",
        outputs=[SimpleNamespace(multimodal_output={"ming3_latent_patch": latents})],
    )
    prompts = llm2audio_vae([stage_output])
    assert len(prompts) == 1
    assert torch.equal(prompts[0]["additional_information"]["latent"], latents)
