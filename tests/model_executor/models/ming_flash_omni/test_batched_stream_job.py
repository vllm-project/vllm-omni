# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage of the consumer half of the batched Ming talker."""

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.ming_flash_omni.streaming import MingBatchedStreamJob

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def vae():
    value = Mock()
    value.decode.side_effect = lambda latent, **kwargs: (latent, (1, 2, 3), "cache")
    return value


def test_urgent_wakes_for_first_audio_and_final_flush_only(vae):
    job = MingBatchedStreamJob(vae, torch.device("cpu"))
    assert not job.urgent()
    job.publish(torch.ones(1, 1, 1), False)
    assert job.urgent()
    job.next_audio()
    job.publish(torch.ones(1, 1, 1), False)
    assert not job.urgent(), "later chunks coalesce within the polling budget"
    job.publish(torch.ones(1, 1, 1), True)
    assert job.urgent()


def test_next_audio_drains_every_ready_latent_and_keeps_vae_state(vae):
    job = MingBatchedStreamJob(vae, torch.device("cpu"))
    job.publish(torch.ones(1, 1, 1), False)
    job.publish(torch.full((1, 1, 1), 2.0), False)
    audio, last = job.next_audio()
    assert not last
    torch.testing.assert_close(audio, torch.tensor([[[1.0], [2.0]]]))
    empty, last = job.next_audio()
    assert empty.numel() == 0 and not last
    job.publish(torch.full((1, 1, 1), 3.0), True)
    audio, last = job.next_audio()
    assert last and job.finished
    assert vae.decode.call_count == 2
    assert vae.decode.call_args.kwargs == {
        "past_key_values": "cache",
        "stream_state": (1, 2, 3),
        "use_cache": True,
        "last_chunk": True,
    }


def test_request_without_segments_finishes_once_without_decoding(vae):
    job = MingBatchedStreamJob(vae, torch.device("cpu"))
    job.final_published = True
    audio, last = job.next_audio()
    assert last and audio.numel() == 0
    vae.decode.assert_not_called()


def test_unstreamed_request_decodes_all_latents_at_the_end(vae):
    decode_full = Mock(return_value=torch.zeros(1, 1, 4))
    job = MingBatchedStreamJob(vae, torch.device("cpu"), decode_full=decode_full)
    job.publish(torch.ones(1, 1, 1), False)
    assert not job.urgent()
    audio, last = job.next_audio()
    assert not last and audio.numel() == 0
    job.publish(torch.ones(1, 1, 1), True)
    assert job.urgent()
    audio, last = job.next_audio()
    assert last and audio.shape == (1, 1, 4)
    assert len(decode_full.call_args.args[0]) == 2
    vae.decode.assert_not_called()


def test_failed_request_raises_and_cancels_once(vae):
    job = MingBatchedStreamJob(vae, torch.device("cpu"))
    job.request = object()
    job.fail(ValueError("bad prompt"))
    assert job.urgent()
    with pytest.raises(ValueError, match="bad prompt"):
        job.next_audio()
    engine = Mock()
    job.close(engine)
    engine.cancel.assert_called_once_with(job.request)
