from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.fish_speech import (
    slow_ar_to_dac_decoder_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_transfer_manager(
    *,
    tensor_payload: bool = False,
    chunk_frames: int = 2,
    left_context: int = 1,
    initial_chunk_frames: int = 0,
    single_initial_chunk: bool = False,
    backlog_chunk_frames: int = 0,
    backlog_load_threshold: float = 0.75,
    max_num_seqs: int = 1,
):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        scheduler_max_num_seqs=max_num_seqs,
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                    "initial_codec_chunk_frames": initial_chunk_frames,
                    "fish_speech_tensor_codes": tensor_payload,
                    "fish_speech_single_initial_chunk": single_initial_chunk,
                    "fish_speech_backlog_codec_chunk_frames": backlog_chunk_frames,
                    "fish_speech_backlog_load_threshold": backlog_load_threshold,
                }
            }
        ),
    )


def _make_request(req_id="req", *, finished=False):
    return SimpleNamespace(
        external_req_id=req_id,
        additional_information=None,
        is_finished=lambda: finished,
    )


def _seed_frames(transfer_manager, req_id: str, n_frames: int, n_codebooks: int = 3):
    frame = torch.arange(1, n_codebooks + 1, dtype=torch.long)
    transfer_manager.code_prompt_token_ids[req_id] = [frame + i * n_codebooks for i in range(n_frames)]


def _call_seeded(transfer_manager, req_id: str = "req", *, n_frames: int, finished: bool = False):
    _seed_frames(transfer_manager, req_id, n_frames)
    return slow_ar_to_dac_decoder_async_chunk(
        transfer_manager,
        {},
        _make_request(req_id, finished=finished),
        is_finished=finished,
    )


def test_slow_ar_to_dac_async_chunk_can_emit_tensor_codes():
    transfer_manager = _make_transfer_manager(tensor_payload=True)
    request = _make_request()

    assert (
        slow_ar_to_dac_decoder_async_chunk(
            transfer_manager,
            {"audio_codes": torch.tensor([[1, 2, 3]])},
            request,
        )
        is None
    )
    payload = slow_ar_to_dac_decoder_async_chunk(
        transfer_manager,
        {"audio_codes": torch.tensor([[4, 5, 6]])},
        request,
    )

    assert payload is not None
    audio_codes = payload.codes.audio
    assert isinstance(audio_codes, torch.Tensor)
    assert audio_codes.dtype == torch.long
    assert torch.equal(audio_codes, torch.tensor([[1, 4], [2, 5], [3, 6]], dtype=torch.long))
    assert payload.meta.left_context_size == 0


def test_slow_ar_to_dac_async_chunk_uses_valid_mask_for_zero_codes():
    transfer_manager = _make_transfer_manager(tensor_payload=True, chunk_frames=1, left_context=0)
    request = _make_request()

    assert (
        slow_ar_to_dac_decoder_async_chunk(
            transfer_manager,
            {
                "audio_codes": torch.tensor([[0, 0, 0]]),
                "audio_code_valid": torch.tensor([False]),
            },
            request,
        )
        is None
    )
    payload = slow_ar_to_dac_decoder_async_chunk(
        transfer_manager,
        {
            "audio_codes": torch.tensor([[4, 5, 6]]),
            "audio_code_valid": torch.tensor([True]),
        },
        request,
    )

    assert payload is not None
    assert torch.equal(payload.codes.audio, torch.tensor([[4], [5], [6]], dtype=torch.long))


def test_slow_ar_to_dac_async_chunk_keeps_flat_codes_by_default():
    transfer_manager = _make_transfer_manager(tensor_payload=False)
    request = _make_request()

    slow_ar_to_dac_decoder_async_chunk(
        transfer_manager,
        {"audio_codes": torch.tensor([[1, 2]])},
        request,
    )
    payload = slow_ar_to_dac_decoder_async_chunk(
        transfer_manager,
        {"audio_codes": torch.tensor([[3, 4]])},
        request,
    )

    assert payload is not None
    assert torch.equal(payload.codes.audio, torch.tensor([1, 3, 2, 4], dtype=torch.long))


def test_qwen3_style_single_initial_chunk_only_emits_first_chunk_early():
    transfer_manager = _make_transfer_manager(
        chunk_frames=5,
        left_context=5,
        initial_chunk_frames=2,
        single_initial_chunk=True,
    )

    first = _call_seeded(transfer_manager, n_frames=2)
    assert first is not None
    assert first.meta.left_context_size == 0
    assert first.codes.audio.numel() == 3 * 2

    # After the first small chunk, return to normal chunk boundaries:
    # initial_coverage=2, so the next emit is at 2 + 5 = 7 frames.
    assert _call_seeded(transfer_manager, n_frames=5) is None

    second = _call_seeded(transfer_manager, n_frames=7)
    assert second is not None
    assert second.meta.left_context_size == 2
    assert second.codes.audio.numel() == 3 * 7


def test_backlog_chunk_size_uses_larger_post_initial_boundary():
    transfer_manager = _make_transfer_manager(
        chunk_frames=5,
        left_context=5,
        initial_chunk_frames=2,
        single_initial_chunk=True,
        backlog_chunk_frames=8,
        backlog_load_threshold=0.5,
        max_num_seqs=4,
    )
    for idx in range(2):
        _seed_frames(transfer_manager, f"other-{idx}", 10)

    first = _call_seeded(transfer_manager, "req", n_frames=2)
    assert first is not None

    # Under backlog, steady chunk is 8 instead of 5. The next emit is
    # initial_coverage=2 plus 8 frames, so n=7 must still be held.
    assert _call_seeded(transfer_manager, "req", n_frames=7) is None
    second = _call_seeded(transfer_manager, "req", n_frames=10)
    assert second is not None
    assert second.meta.left_context_size == 2
    assert second.codes.audio.numel() == 3 * 10


def test_backlog_chunk_size_keeps_multi_initial_boundary_aligned():
    transfer_manager = _make_transfer_manager(
        chunk_frames=25,
        left_context=25,
        initial_chunk_frames=10,
        single_initial_chunk=False,
        backlog_chunk_frames=50,
        backlog_load_threshold=0.5,
        max_num_seqs=4,
    )
    for idx in range(2):
        _seed_frames(transfer_manager, f"other-{idx}", 50)

    first = _call_seeded(transfer_manager, "req", n_frames=10)
    assert first is not None
    transfer_manager.put_req_chunk["req"] += 1
    second = _call_seeded(transfer_manager, "req", n_frames=20)
    assert second is not None
    transfer_manager.put_req_chunk["req"] += 1

    assert _call_seeded(transfer_manager, "req", n_frames=50) is None
    third = _call_seeded(transfer_manager, "req", n_frames=70)
    assert third is not None
    assert third.meta.left_context_size == 20
    assert third.codes.audio.numel() == 3 * 70


def test_invalid_backlog_config_fails_loudly():
    transfer_manager = _make_transfer_manager(
        chunk_frames=5,
        left_context=5,
        backlog_chunk_frames="invalid",
    )
    request = _make_request()

    with pytest.raises(ValueError, match="fish_speech_backlog_codec_chunk_frames"):
        slow_ar_to_dac_decoder_async_chunk(
            transfer_manager,
            {"audio_codes": torch.tensor([[1, 2, 3]])},
            request,
        )
