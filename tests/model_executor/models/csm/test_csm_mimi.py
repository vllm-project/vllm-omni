# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU correctness tests for the CSM-1B Stage-1 Mimi vocoder (code2wav).

Locks the framework-glue invariants that do not need real codec weights:
  * Reserved-id clamp ([0, 2047] on a COPY) -- ids 2048/2049/2050 must never
    reach Mimi, and the caller's tensor must not be mutated.
  * Codebook-major reshape ``[32*F] -> (1, 32, F)`` into ``decode``.
  * I1 delta streaming -- the leading ``left_context_size`` frames are trimmed
    so each chunk emits only its new audio.
  * Per-request split + "all return paths emit model_outputs" -- empty and
    malformed-length requests still produce a length-matched output list.
  * I5 -- per-request streaming state is freed on finish.

Bit-parity of ``_mimi_decode`` against the real ``transformers.MimiModel`` lives
in ``test_csm_gpu_parity.py``.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.csm.csm_mimi import _MIMI_CODEBOOK_SIZE, CsmMimiVocoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SAMPLES_PER_FRAME = 4  # tiny stand-in for the real 1920 so trims are checkable


class _FakeCodec(nn.Module):
    """Records every ``decode`` call and returns ``arange`` audio so trimming is
    bit-checkable. Emits ``F * _SAMPLES_PER_FRAME`` samples per request row."""

    def __init__(self):
        super().__init__()
        self.received: list[torch.Tensor] = []

    def decode(self, audio_codes: torch.Tensor):
        self.received.append(audio_codes.clone())
        b = int(audio_codes.shape[0])
        f = int(audio_codes.shape[-1])
        samples = f * _SAMPLES_PER_FRAME
        wav = torch.arange(samples, dtype=torch.float32).view(1, 1, -1).expand(b, 1, samples).clone()
        return SimpleNamespace(audio_values=wav)


def _make_vocoder() -> CsmMimiVocoder:
    v = object.__new__(CsmMimiVocoder)
    nn.Module.__init__(v)
    v.num_codebooks = 32
    v.sample_rate = 24000
    v._device = torch.device("cpu")
    v.config = SimpleNamespace(codec_samples_per_frame=_SAMPLES_PER_FRAME)
    v._stream_state_by_req = {}
    v._mimi_codec = _FakeCodec()
    return v


def test_mimi_decode_clamps_reserved_ids_on_a_copy():
    v = _make_vocoder()
    # A single frame whose codes include all three reserved ids.
    frame = torch.full((32, 1), 2050, dtype=torch.long)
    frame[0, 0] = 2049
    frame[1, 0] = 2048
    original = frame.clone()

    v._mimi_decode(frame)

    received = v._mimi_codec.received[-1]
    assert received.shape == (1, 32, 1)
    # Nothing >= 2048 reached the codec.
    assert int(received.max()) <= _MIMI_CODEBOOK_SIZE - 1
    # The caller's tensor was NOT mutated (clamp ran on a copy).
    assert torch.equal(frame, original)


def test_forward_reshapes_codebook_major_into_codec():
    v = _make_vocoder()
    # F = 2 frames, codebook-major flat [32*2]; values stay in range.
    ids = torch.arange(64, dtype=torch.long)
    v.forward(input_ids=ids, runtime_additional_information=[{"meta": {"left_context_size": 0}}])

    received = v._mimi_codec.received[-1]
    assert received.shape == (1, 32, 2)
    # [Q*F] codebook-major reshapes to [Q, F] row-major.
    assert torch.equal(received[0], ids.reshape(32, 2))


def test_forward_delta_trims_left_context():
    v = _make_vocoder()
    # 3 frames -> fake emits arange(12); 2 context frames -> trim 2*4 = 8 samples.
    ids = torch.arange(96, dtype=torch.long)  # 32 * 3
    out = v.forward(input_ids=ids, runtime_additional_information=[{"meta": {"left_context_size": 2}}])
    audio = out.multimodal_outputs["model_outputs"][0]
    torch.testing.assert_close(audio, torch.arange(8, 12, dtype=torch.float32))


def test_forward_no_context_emits_full_chunk():
    v = _make_vocoder()
    ids = torch.arange(96, dtype=torch.long)
    out = v.forward(input_ids=ids, runtime_additional_information=[{"meta": {"left_context_size": 0}}])
    audio = out.multimodal_outputs["model_outputs"][0]
    torch.testing.assert_close(audio, torch.arange(12, dtype=torch.float32))


def test_forward_empty_input_emits_one_empty_output():
    v = _make_vocoder()
    out = v.forward(input_ids=torch.empty((0,), dtype=torch.long))
    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 1
    assert audios[0].numel() == 0
    assert "sr" in out.multimodal_outputs


def test_forward_skips_malformed_length_but_keeps_output_slot():
    v = _make_vocoder()
    # 33 ids is not divisible by 32 -> skip, but the output list still has 1 slot.
    out = v.forward(
        input_ids=torch.arange(33, dtype=torch.long),
        runtime_additional_information=[{"meta": {"left_context_size": 0}}],
    )
    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 1
    assert audios[0].numel() == 0
    assert v._mimi_codec.received == []  # codec never called for a bad length


def test_forward_per_request_split_emits_one_output_each():
    v = _make_vocoder()
    ids = torch.arange(64, dtype=torch.long)  # two 32-code frames, one per request
    out = v.forward(
        input_ids=ids,
        seq_token_counts=[32, 32],
        runtime_additional_information=[
            {"meta": {"left_context_size": 0}},
            {"meta": {"left_context_size": 0}},
        ],
    )
    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert all(a.numel() == _SAMPLES_PER_FRAME for a in audios)  # 1 frame each


def test_forward_mixed_valid_and_malformed_keeps_indices_aligned():
    v = _make_vocoder()
    # req0: 32 valid ids (1 frame); req1: 5 ids (malformed) -> empty, not dropped.
    ids = torch.arange(37, dtype=torch.long)
    out = v.forward(
        input_ids=ids,
        seq_token_counts=[32, 5],
        runtime_additional_information=[{"meta": {}}, {"meta": {}}],
    )
    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert audios[0].numel() == _SAMPLES_PER_FRAME
    assert audios[1].numel() == 0


def test_on_requests_finished_frees_stream_state():
    v = _make_vocoder()
    v._stream_state_by_req = {"a": object(), "b": object()}
    v.on_requests_finished(["a"])
    assert "a" not in v._stream_state_by_req
    assert "b" in v._stream_state_by_req


def test_make_omni_output_passthrough_and_rejects_bad_type():
    v = _make_vocoder()
    out = v.forward(input_ids=torch.empty((0,), dtype=torch.long))
    assert v.make_omni_output(out) is out
    with pytest.raises(TypeError):
        v.make_omni_output(torch.zeros(3))


def test_compute_logits_is_none_for_non_ar_stage():
    v = _make_vocoder()
    assert v.compute_logits(torch.zeros(2, 4)) is None
