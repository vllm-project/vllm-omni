# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.funaudiochat.pipeline import FUN_AUDIO_CHAT_PIPELINE
from vllm_omni.model_executor.stage_input_processors.funaudiochat import (
    funaudiochat2code2wav_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_transfer_manager(chunk_frames: int = 3, lookahead_frames: int = 1):
    return SimpleNamespace(
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_pre_lookahead_frames": lookahead_frames,
                    "codec_vocab_size": 16,
                }
            }
        ),
        request_payload={},
        code_prompt_token_ids=defaultdict(list),
    )


def _make_request(request_id: str = "request-1", finished: bool = False):
    return SimpleNamespace(
        external_req_id=request_id,
        is_finished=lambda: finished,
    )


def test_async_chunk_waits_for_chunk_and_lookahead_then_emits_prefix():
    transfer_manager = _make_transfer_manager()
    request = _make_request()

    assert (
        funaudiochat2code2wav_async_chunk(
            transfer_manager,
            {"audio_token_ids": torch.tensor([[1, 2]])},
            request,
        )
        is None
    )

    payload = funaudiochat2code2wav_async_chunk(
        transfer_manager,
        {"audio_token_ids": torch.tensor([[3, 4]])},
        request,
    )

    assert payload is not None
    assert payload.codes.audio.tolist() == [1, 2, 3, 4]
    assert payload.meta.left_context_size == 0
    assert payload.meta.codec_chunk_frames == 4
    assert payload.meta.finished.item() is False


def test_async_chunk_accepts_cumulative_tensor_snapshots_without_duplicate_tokens():
    transfer_manager = _make_transfer_manager()
    request = _make_request()
    funaudiochat2code2wav_async_chunk(
        transfer_manager,
        {"audio_token_ids": [torch.tensor([[1, 2]])]},
        request,
    )

    payload = funaudiochat2code2wav_async_chunk(
        transfer_manager,
        {"audio_token_ids": [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]},
        request,
    )

    assert payload is not None
    assert payload.codes.audio.tolist() == [1, 2, 3, 4]


def test_async_chunk_reuses_lookahead_and_flushes_final_tail():
    transfer_manager = _make_transfer_manager()
    request = _make_request()
    funaudiochat2code2wav_async_chunk(
        transfer_manager,
        {"audio_token_ids": torch.tensor([[1, 2, 3, 4]])},
        request,
    )

    payload = funaudiochat2code2wav_async_chunk(
        transfer_manager,
        {"audio_token_ids": torch.tensor([[5, 6]])},
        request,
        is_finished=True,
    )

    assert payload is not None
    assert payload.codes.audio.tolist() == [1, 2, 3, 4, 5, 6]
    assert payload.meta.left_context_size == 3
    assert payload.meta.stream_finished.item() is True
    assert payload.meta.finished.item() is True


def test_async_chunk_emits_exact_chunk_size_with_reused_lookahead():
    transfer_manager = _make_transfer_manager(chunk_frames=3, lookahead_frames=1)
    request = _make_request()

    first = funaudiochat2code2wav_async_chunk(
        transfer_manager,
        {"audio_token_ids": torch.tensor([[1, 2, 3, 4]])},
        request,
    )
    second = funaudiochat2code2wav_async_chunk(
        transfer_manager,
        {"audio_token_ids": torch.tensor([[5, 6, 7]])},
        request,
    )

    assert first is not None
    assert first.codes.audio.tolist() == [1, 2, 3, 4]
    assert first.meta.codec_chunk_frames == 4
    assert first.meta.left_context_size == 0
    assert second is not None
    assert second.codes.audio.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert second.meta.codec_chunk_frames == 4
    assert second.meta.left_context_size == 3
    assert second.meta.chunk_seq == 1


def test_async_chunk_emits_terminal_marker_for_empty_audio():
    transfer_manager = _make_transfer_manager()

    payload = funaudiochat2code2wav_async_chunk(
        transfer_manager,
        {"audio_token_ids": torch.full((1, 5), -1)},
        _make_request(),
        is_finished=True,
    )

    assert payload is not None
    assert payload.codes.audio.numel() == 0
    assert payload.meta.stream_finished.item() is True


def test_async_chunk_state_is_isolated_per_request():
    transfer_manager = _make_transfer_manager()

    for request_id, token_id in (("request-a", 1), ("request-b", 2)):
        payload = funaudiochat2code2wav_async_chunk(
            transfer_manager,
            {"audio_token_ids": torch.tensor([[token_id, token_id, token_id, token_id]])},
            _make_request(request_id),
        )
        assert payload is not None
        assert payload.codes.audio.tolist() == [token_id] * 4

    assert transfer_manager.request_payload["request-a"]["_funaudiochat_async_chunk_state"]["tokens"] == [1] * 4
    assert transfer_manager.request_payload["request-b"]["_funaudiochat_async_chunk_state"]["tokens"] == [2] * 4


def test_async_chunk_rejects_invalid_chunk_configuration():
    transfer_manager = _make_transfer_manager(chunk_frames=0)

    with pytest.raises(ValueError, match="Invalid FunAudioChat codec chunk configuration"):
        funaudiochat2code2wav_async_chunk(
            transfer_manager,
            None,
            _make_request(),
        )


def test_funaudiochat_deploy_resolves_async_decoder_edge():
    deploy_path = Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy" / "funaudiochat.yaml"

    stages = merge_pipeline_deploy(FUN_AUDIO_CHAT_PIPELINE, load_deploy_config(deploy_path))

    assert (
        stages[0]
        .yaml_engine_args["custom_process_next_stage_input_func"]
        .endswith(".funaudiochat2code2wav_async_chunk")
    )
    assert stages[0].yaml_engine_args["hf_overrides"]["audio_config"]["max_source_positions"] == 100
    assert stages[1].yaml_engine_args["model_arch"] == "FunAudioChatCosyVoice3Code2Wav"
    assert stages[1].yaml_engine_args["model"] == "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
