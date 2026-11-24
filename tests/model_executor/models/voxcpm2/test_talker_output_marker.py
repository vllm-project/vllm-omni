# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VoxCPM2 MRV2 audio output alignment and request identity."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

torch = pytest.importorskip("torch")


@functools.lru_cache(maxsize=1)
def _talker_cls():
    """Defer talker import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        VoxCPM2TalkerForConditionalGeneration,
    )

    return VoxCPM2TalkerForConditionalGeneration


def _make_dense_talker():
    cls = _talker_cls()
    talker = cls.__new__(cls)
    # Dense mode: emit every step, no delayed copy, no batched VAE decode.
    talker._audio_emit_every = 1
    talker._vae_decode_every = 1
    talker._coalesce_audio_d2h = False
    talker._sample_rate = 16000
    talker._audio_queue = []
    return talker

    # (The coalesce-D2H sibling branch carries the same marker; it needs
    # device-resident chunks, so its marker contract is pinned here via the
    # shared dense-subset shape rather than a GPU-only test.)


def _make_mrv2_talker(monkeypatch):
    from vllm_omni.model_executor.models.voxcpm2.model_state import VoxCPM2ModelState
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    talker = _make_dense_talker()
    talker.config = SimpleNamespace(hidden_size=4)
    talker._side_dtype = torch.float32
    talker._patch_size = 4
    talker._feat_dim = 2
    talker._n_decode_pad_frames = 4
    talker._tts = SimpleNamespace(audio_vae=SimpleNamespace(decode_chunk_size=1))
    talker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100))

    def init_base(owner, _config, model, _cache, _device):
        owner.model = model
        owner.scheduler_config = SimpleNamespace(max_num_seqs=2)

    monkeypatch.setattr(OmniModelState, "__init__", init_base)
    state = VoxCPM2ModelState(None, talker, None, torch.device("cpu"))
    talker._active_states = {}
    for slot, req_id in enumerate(("r0", "r1")):
        state.slots[slot].request_id = req_id
        state.slots[slot].slot_index = slot
        talker._active_states[req_id] = state.slots[slot]
    return talker


def _materialize_mrv2_audio(talker, request_ids):
    state = talker._mrv2_model_state
    batch = SimpleNamespace(
        num_reqs=len(request_ids),
        idx_mapping_np=np.array([talker._active_states[req_id].slot_index for req_id in request_ids]),
        num_computed_tokens_np=np.zeros(len(request_ids), dtype=np.int32),
        num_scheduled_tokens=[1] * len(request_ids),
    )
    # This is each request's last allowed step, so partial PCM also emits.
    req_states = SimpleNamespace(max_seq_len=np.full(2, 2))
    return state.prepare_streaming_audio_output(batch, req_states, {}).get_output()


def test_mrv2_sparse_audio_materializes_by_request_id(monkeypatch):
    talker = _make_mrv2_talker(monkeypatch)
    talker._last_audio_output_req_ids = ["r0", "r1"]
    audio = torch.arange(4, dtype=torch.float32)
    talker._audio_queue = [("r0", None), ("r1", audio)]

    mm = talker.make_omni_output(torch.zeros(2)).multimodal_outputs
    assert mm == {"voxcpm2_audio_pending": True}
    output = _materialize_mrv2_audio(talker, ["r0", "r1"])

    assert output[0] is None
    assert torch.equal(output[1]["model_outputs"], audio)

    talker._audio_queue = [("r0", None), ("r1", None)]
    talker.make_omni_output(torch.zeros(2))
    assert _materialize_mrv2_audio(talker, ["r0", "r1"]) == [None, None]


def test_mrv2_sparse_audio_preserves_request_order(monkeypatch):
    talker = _make_mrv2_talker(monkeypatch)
    talker._last_audio_output_req_ids = ["r1", "r0"]
    talker._audio_queue = [("r1", torch.tensor([2.0])), ("r0", torch.tensor([1.0]))]

    # Output follows the runner's slot mapping, even when preprocess visited
    # the requests in a different order.
    talker.make_omni_output(torch.zeros(2))
    output = _materialize_mrv2_audio(talker, ["r0", "r1"])

    assert output[0]["model_outputs"].item() == 1.0
    assert output[1]["model_outputs"].item() == 2.0


def test_mrv2_preprocess_uses_scheduler_req_id(monkeypatch):
    talker = _make_mrv2_talker(monkeypatch)
    talker.config = SimpleNamespace(hidden_size=4)
    talker._side_dtype = torch.float32
    talker._pending_requests = []
    talker._pending_batch_rows = []

    for slot, req_id in enumerate(("r0", "r1")):
        talker.preprocess(
            torch.tensor([0]),
            torch.zeros(1, 4),
            req_id=req_id,
            slot_index=slot,
            batch_row=slot,
            _omni_is_prefill=False,
        )

    assert [entry[0] for entry in talker._pending_requests] == ["r0", "r1"]
