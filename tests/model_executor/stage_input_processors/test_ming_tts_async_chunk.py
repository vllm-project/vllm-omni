# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import pytest
import torch

from vllm_omni.model_executor.models.ming_tts.constants import (
    KEY_CHUNK_ID,
    KEY_REQUEST_ID,
    LATENT_DIM,
    PATCH_SIZE,
)
from vllm_omni.model_executor.models.ming_tts.pipeline import MING_TTS_PIPELINE
from vllm_omni.model_executor.stage_input_processors.ming_tts import (
    MING_EMIT_PATCH_COUNT_KEY,
    MING_ESTIMATED_BYTES_KEY,
    MING_FINAL_DECODE_STEP_KEY,
    MING_FINAL_FLUSH_KEY,
    MING_LATENT_SHAPE_KEY,
    MING_STOP_REASON_KEY,
    _extract_ming_output_snapshot,
    llm2audio_vae,
    llm2audio_vae_async_chunk,
)
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_transfer_manager(*, chunk_size: int = 5, initial_chunk_size: int = 2):
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        put_req_chunk=defaultdict(int),
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "latent_chunk_size": chunk_size,
                    "initial_latent_chunk_size": initial_chunk_size,
                    "latent_left_context": 0,
                }
            }
        ),
    )


def _make_request(req_id: str = "req", *, finished: bool = False):
    return SimpleNamespace(
        external_req_id=req_id,
        is_finished=lambda: finished,
    )


def _make_output(value: float):
    return {
        "ming_has_patch": torch.tensor([1], dtype=torch.bool),
        "ming_latent_patch": torch.full((1, PATCH_SIZE, LATENT_DIM), value),
    }


def _append_patch(tm, req_id: str, idx: int, *, finished: bool = False):
    return llm2audio_vae_async_chunk(
        tm,
        _make_output(float(idx)),
        _make_request(req_id, finished=finished),
        is_finished=finished,
    )


def test_ming_output_snapshot_selects_last_active_row():
    patches = torch.arange(3 * PATCH_SIZE * LATENT_DIM, dtype=torch.float32).reshape(
        3,
        PATCH_SIZE,
        LATENT_DIM,
    )
    pooling_output = {
        "ming_has_patch": torch.tensor([1, 0, 1], dtype=torch.bool),
        "ming_latent_patch": patches,
        "ming_decode_step": torch.tensor([11, 22, 33]),
        MING_STOP_REASON_KEY: torch.tensor([0, 1, 2]),
    }

    patch, decode_step, stop_reason = _extract_ming_output_snapshot(pooling_output)

    torch.testing.assert_close(patch, patches[2])
    assert decode_step == 33
    assert stop_reason == "max_decode_steps"


def test_ming_output_snapshot_selects_all_active_rows_for_full_payload():
    patches = torch.arange(3 * PATCH_SIZE * LATENT_DIM, dtype=torch.float32).reshape(
        3,
        PATCH_SIZE,
        LATENT_DIM,
    )
    pooling_output = {
        "ming_has_patch": torch.tensor([1, 0, 1], dtype=torch.bool),
        "ming_latent_patch": patches,
        "ming_decode_step": torch.tensor([11, 22, 33]),
        MING_STOP_REASON_KEY: torch.tensor([0, 1, 2]),
    }

    selected, decode_step, stop_reason = _extract_ming_output_snapshot(
        pooling_output,
        all_patches=True,
    )

    torch.testing.assert_close(selected, patches[[0, 2]])
    assert decode_step == 33
    assert stop_reason == "max_decode_steps"


def test_ming_output_snapshot_returns_empty_when_no_row_is_active():
    pooling_output = {
        "ming_has_patch": torch.tensor([0, 0], dtype=torch.bool),
        "ming_latent_patch": torch.ones(2, PATCH_SIZE, LATENT_DIM),
        "ming_decode_step": torch.tensor([11, 22]),
        MING_STOP_REASON_KEY: torch.tensor([0, 1]),
    }

    assert _extract_ming_output_snapshot(pooling_output) == (None, None, None)


def test_ming_output_snapshot_preserves_list_metadata_fallback():
    patches = torch.arange(2 * PATCH_SIZE * LATENT_DIM, dtype=torch.float32).reshape(
        2,
        PATCH_SIZE,
        LATENT_DIM,
    )
    pooling_output = {
        "ming_has_patch": torch.tensor([0, 1], dtype=torch.bool),
        "ming_latent_patch": patches,
        "ming_decode_step": [11, 22],
        MING_STOP_REASON_KEY: [0, 1],
    }

    patch, decode_step, stop_reason = _extract_ming_output_snapshot(pooling_output)

    torch.testing.assert_close(patch, patches[1])
    assert decode_step == 22
    assert stop_reason == "stop_head"


@pytest.mark.parametrize("all_patches", [False, True])
@pytest.mark.parametrize("metadata_type", [list, tuple])
def test_ming_output_snapshot_tensor_mask_selects_sequence_metadata_once(monkeypatch, all_patches, metadata_type):
    patches = torch.arange(3 * PATCH_SIZE * LATENT_DIM, dtype=torch.float32).reshape(3, PATCH_SIZE, LATENT_DIM)
    pooling_output = {
        "ming_has_patch": torch.tensor([1, 1, 0], dtype=torch.bool),
        "ming_latent_patch": patches,
        "ming_decode_step": metadata_type([11, 22, 33]),
        MING_STOP_REASON_KEY: metadata_type([0, 1, 2]),
    }
    original_item = torch.Tensor.item
    item_calls = []

    def counted_item(tensor, *args, **kwargs):
        item_calls.append(tensor)
        return original_item(tensor, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch.Tensor, "item", counted_item)
        selected, decode_step, stop_reason = _extract_ming_output_snapshot(pooling_output, all_patches=all_patches)

    assert len(item_calls) == int(all_patches)
    torch.testing.assert_close(selected, patches[:2] if all_patches else patches[1])
    assert decode_step == 22
    assert stop_reason == "stop_head"


@pytest.mark.parametrize("full_payload", [False, True])
def test_ming_two_dimensional_patch_preserves_consumer_payload(full_payload):
    patch = torch.arange(PATCH_SIZE * LATENT_DIM, dtype=torch.float16).reshape(PATCH_SIZE, LATENT_DIM)
    multimodal_output = {
        "ming_has_patch": torch.tensor([1], dtype=torch.bool),
        "ming_latent_patch": patch,
        "ming_decode_step": torch.tensor([17]),
        MING_STOP_REASON_KEY: torch.tensor([2]),
    }

    if full_payload:
        stage_output = SimpleNamespace(
            finished=True,
            request_id="req",
            outputs=[SimpleNamespace(multimodal_output=multimodal_output)],
        )
        outputs = llm2audio_vae([stage_output])
        assert len(outputs) == 1
        metadata = outputs[0]["additional_information"]
        latents = metadata["ming_latent_patches"]
    else:
        payload = llm2audio_vae_async_chunk(_make_transfer_manager(), multimodal_output, _make_request(finished=True))
        assert payload is not None
        metadata = payload.kv_metadata
        latents = payload.latent

    torch.testing.assert_close(latents, patch.float().unsqueeze(0))
    assert latents.device.type == "cpu"
    assert metadata[KEY_REQUEST_ID] == "req"
    assert metadata[MING_FINAL_DECODE_STEP_KEY] == 17
    assert metadata[MING_STOP_REASON_KEY] == "max_decode_steps"


def test_ming_full_payload_reuses_snapshot_for_patches_and_metadata():
    patches = torch.arange(3 * PATCH_SIZE * LATENT_DIM, dtype=torch.float32).reshape(
        3,
        PATCH_SIZE,
        LATENT_DIM,
    )
    multimodal_output = {
        "ming_has_patch": torch.tensor([1, 0, 1], dtype=torch.bool),
        "ming_latent_patch": patches,
        "ming_decode_step": torch.tensor([11, 22, 33]),
        MING_STOP_REASON_KEY: torch.tensor([0, 1, 2]),
    }
    stage_output = SimpleNamespace(
        finished=True,
        request_id="req",
        outputs=[SimpleNamespace(multimodal_output=multimodal_output)],
    )

    outputs = llm2audio_vae([stage_output])

    assert len(outputs) == 1
    additional_information = outputs[0]["additional_information"]
    torch.testing.assert_close(
        additional_information["ming_latent_patches"],
        patches[[0, 2]],
    )
    assert additional_information[MING_FINAL_DECODE_STEP_KEY] == 33
    assert additional_information[MING_STOP_REASON_KEY] == "max_decode_steps"


def test_ming_async_chunk_emits_small_initial_chunk_then_steady_chunks():
    tm = _make_transfer_manager(chunk_size=5, initial_chunk_size=2)

    assert _append_patch(tm, "req", 0) is None

    first = _append_patch(tm, "req", 1)
    assert first is not None
    assert first.latent.shape == (2, PATCH_SIZE, LATENT_DIM)
    assert first.kv_metadata[MING_EMIT_PATCH_COUNT_KEY] == 2
    assert first.meta.finished.item() is False

    for idx in range(2, 6):
        assert _append_patch(tm, "req", idx) is None

    second = _append_patch(tm, "req", 6)
    assert second is not None
    assert second.latent.shape == (5, PATCH_SIZE, LATENT_DIM)
    assert second.kv_metadata[MING_EMIT_PATCH_COUNT_KEY] == 5


def test_ming_async_chunk_flushes_short_final_chunk():
    tm = _make_transfer_manager(chunk_size=5, initial_chunk_size=2)

    payload = _append_patch(tm, "req", 0, finished=True)

    assert payload is not None
    assert payload.latent.shape == (1, PATCH_SIZE, LATENT_DIM)
    assert payload.kv_metadata[MING_EMIT_PATCH_COUNT_KEY] == 1
    assert payload.meta.finished.item() is True


def test_ming_async_chunk_preserves_selected_final_metadata():
    transfer_manager = _make_transfer_manager()
    transfer_manager.put_req_chunk["req"] = 3
    patches = torch.arange(3 * PATCH_SIZE * LATENT_DIM, dtype=torch.float32).reshape(3, PATCH_SIZE, LATENT_DIM)
    multimodal_output = {
        "ming_has_patch": torch.tensor([1, 1, 0], dtype=torch.bool),
        "ming_latent_patch": patches,
        "ming_decode_step": torch.tensor([11, 22, 33]),
        MING_STOP_REASON_KEY: torch.tensor([0, 1, 2]),
    }

    payload = llm2audio_vae_async_chunk(transfer_manager, multimodal_output, _make_request(), is_finished=True)

    assert payload is not None
    torch.testing.assert_close(payload.latent, patches[1:2])
    assert payload.kv_metadata[KEY_REQUEST_ID] == "req"
    assert payload.kv_metadata[KEY_CHUNK_ID] == 3
    assert payload.kv_metadata[MING_FINAL_DECODE_STEP_KEY] == 22
    assert payload.kv_metadata[MING_STOP_REASON_KEY] == "stop_head"
    assert payload.kv_metadata[MING_EMIT_PATCH_COUNT_KEY] == 1
    assert payload.kv_metadata[MING_LATENT_SHAPE_KEY] == (1, PATCH_SIZE, LATENT_DIM)
    assert payload.kv_metadata[MING_ESTIMATED_BYTES_KEY] == PATCH_SIZE * LATENT_DIM * 4
    assert payload.kv_metadata[MING_FINAL_FLUSH_KEY] is True
    assert payload.meta.finished.item() is True
    assert payload.meta.stream_finished.item() is True


def test_ming_async_chunk_zero_patch_terminal_flush_preserves_metadata():
    transfer_manager = _make_transfer_manager(chunk_size=1, initial_chunk_size=1)
    assert _append_patch(transfer_manager, "req", 0) is not None
    transfer_manager.put_req_chunk["req"] = 1
    multimodal_output = {
        "ming_decode_step": torch.tensor([23]),
        MING_STOP_REASON_KEY: torch.tensor([2]),
    }

    payload = llm2audio_vae_async_chunk(transfer_manager, multimodal_output, _make_request(finished=True))

    assert payload is not None
    assert payload.latent is None
    assert payload.codes.audio.numel() == 0
    assert payload.kv_metadata[KEY_REQUEST_ID] == "req"
    assert payload.kv_metadata[KEY_CHUNK_ID] == 1
    assert payload.kv_metadata[MING_FINAL_DECODE_STEP_KEY] == 23
    assert payload.kv_metadata[MING_STOP_REASON_KEY] == "max_decode_steps"
    assert payload.kv_metadata[MING_EMIT_PATCH_COUNT_KEY] == 0
    assert payload.kv_metadata[MING_LATENT_SHAPE_KEY] is None
    assert payload.kv_metadata[MING_ESTIMATED_BYTES_KEY] == 0
    assert payload.kv_metadata[MING_FINAL_FLUSH_KEY] is True
    assert payload.meta.finished.item() is True
    assert payload.meta.stream_finished.item() is True
    assert transfer_manager.request_payload["req"]["_ming_async_state"]["terminal_sent"] is True
    assert llm2audio_vae_async_chunk(transfer_manager, multimodal_output, _make_request(finished=True)) is None


def test_ming_async_chunk_does_not_extract_snapshot_after_terminal_sent():
    transfer_manager = _make_transfer_manager()
    terminal = _append_patch(transfer_manager, "req", 0, finished=True)
    assert terminal is not None
    state = transfer_manager.request_payload["req"]["_ming_async_state"]
    assert state["terminal_sent"] is True

    with mock_patch(
        "vllm_omni.model_executor.stage_input_processors.ming_tts._extract_ming_output_snapshot",
        wraps=_extract_ming_output_snapshot,
    ) as extract_snapshot:
        assert _append_patch(transfer_manager, "req", 1, finished=True) is None
        extract_snapshot.assert_not_called()

    assert len(transfer_manager.code_prompt_token_ids["req"]) == 1
    assert state["seen_patch_len"] == 1


@pytest.mark.parametrize("metadata_only", [False, True])
def test_ming_worker_loads_hook_and_builds_terminal_payload(metadata_only):
    worker = OmniConnectorModelRunnerMixin()
    worker.init_omni_connectors(
        model_config=SimpleNamespace(
            stage_connector_config=None,
            stage_id=0,
            async_chunk=True,
            worker_type="ar",
            custom_process_next_stage_input_func=MING_TTS_PIPELINE.stages[0].async_chunk_process_next_stage_input_func,
        ),
    )
    try:
        assert worker._custom_process_func is llm2audio_vae_async_chunk
        worker.put_req_chunk["external-req"] = 3
        request = _make_request("external-req", finished=True)
        patches = torch.arange(3 * PATCH_SIZE * LATENT_DIM, dtype=torch.float32).reshape(3, PATCH_SIZE, LATENT_DIM)
        if metadata_only:
            multimodal_output = {
                "ming_decode_step": torch.tensor([22]),
                MING_STOP_REASON_KEY: torch.tensor([1]),
            }
        else:
            multimodal_output = {
                "ming_has_patch": torch.tensor([1, 1, 0], dtype=torch.bool),
                "ming_latent_patch": patches,
                "ming_decode_step": torch.tensor([11, 22, 33]),
                MING_STOP_REASON_KEY: torch.tensor([0, 1, 2]),
            }

        payload = worker._build_custom_process_payload("internal-req", request, multimodal_output)

        assert payload is not None
        assert payload.kv_metadata[KEY_REQUEST_ID] == "external-req"
        assert payload.kv_metadata[KEY_CHUNK_ID] == 3
        assert payload.kv_metadata[MING_FINAL_DECODE_STEP_KEY] == 22
        assert payload.kv_metadata[MING_STOP_REASON_KEY] == "stop_head"
        assert payload.meta.finished.item() is True
        assert payload.meta.stream_finished.item() is True
        assert payload.kv_metadata[MING_FINAL_FLUSH_KEY] is True
        assert payload.kv_metadata[MING_EMIT_PATCH_COUNT_KEY] == (0 if metadata_only else 1)
        if metadata_only:
            assert payload.latent is None
            assert payload.codes.audio.numel() == 0
        else:
            torch.testing.assert_close(payload.latent, patches[1:2])
        state = worker.request_payload["external-req"]["_ming_async_state"]
        assert state["terminal_sent"] is True
        assert state["seen_patch_len"] == (0 if metadata_only else 1)

        with mock_patch(
            "vllm_omni.model_executor.stage_input_processors.ming_tts._extract_ming_output_snapshot",
            wraps=_extract_ming_output_snapshot,
        ) as extract_snapshot:
            assert worker._build_custom_process_payload("internal-req", request, multimodal_output) is None
            extract_snapshot.assert_not_called()
    finally:
        worker.shutdown_omni_connectors()


def test_ming_async_chunk_flushes_leftover_after_initial_chunk():
    tm = _make_transfer_manager(chunk_size=5, initial_chunk_size=2)

    assert _append_patch(tm, "req", 0) is None
    assert _append_patch(tm, "req", 1) is not None
    assert _append_patch(tm, "req", 2) is None

    final = _append_patch(tm, "req", 3, finished=True)

    assert final is not None
    assert final.latent.shape == (2, PATCH_SIZE, LATENT_DIM)
    assert final.kv_metadata[MING_EMIT_PATCH_COUNT_KEY] == 2
    assert final.meta.finished.item() is True
