# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.data_entry_keys import SKIP_TRANSFER
from vllm_omni.model_executor.models.personaplex.personaplex_talker import (
    PersonaPlexTalkerForConditionalGeneration,
)
from vllm_omni.model_executor.stage_input_processors.personaplex import (
    talker2code2wav_async_chunk,
    talker2code2wav_async_chunk_batch,
    talker2code2wav_full_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _raw_agent_codes() -> torch.Tensor:
    return torch.arange(16, dtype=torch.long).reshape(2, 8)


@pytest.mark.parametrize("source", ["pooling_output", "additional_information_cpu"])
def test_full_payload_accepts_worker_and_cached_payload_sources(source: str) -> None:
    audio = _raw_agent_codes()
    payload = talker2code2wav_full_payload(
        pooling_output={"codes": {"audio": audio}} if source == "pooling_output" else None,
        request=(
            SimpleNamespace()
            if source == "pooling_output"
            else SimpleNamespace(additional_information_cpu={"codes": {"audio": audio}})
        ),
    )

    expected = torch.cat([audio[:-1, :1], audio[1:, 1:]], dim=1).reshape(-1)
    assert torch.equal(payload.codes.audio, expected)


def test_async_chunk_keeps_delay_tail_across_resumable_segments() -> None:
    manager = SimpleNamespace(
        connector=SimpleNamespace(
            config={
                "extra": {
                    "initial_codec_chunk_frames": 1,
                    "codec_chunk_frames": 5,
                }
            }
        )
    )
    request = SimpleNamespace(
        request_id="req",
        external_req_id="req",
        resumable=True,
        is_finished=lambda: True,
        additional_information=None,
    )
    first_frame = torch.arange(8, dtype=torch.long).reshape(1, 8)
    second_frame = torch.arange(8, 16, dtype=torch.long).reshape(1, 8)

    request.additional_information = {"codes": {"audio": first_frame}}
    first = talker2code2wav_async_chunk(
        manager,
        multimodal_output=None,
        request=request,
        is_finished=True,
    )
    request.additional_information = {"codes": {"audio": second_frame}}
    second = talker2code2wav_async_chunk(
        manager,
        multimodal_output=None,
        request=request,
        is_finished=True,
    )

    assert first is SKIP_TRANSFER
    expected = torch.cat([first_frame[:, :1], second_frame[:, 1:]], dim=1).reshape(-1)
    assert torch.equal(second.codes.audio, expected)
    assert manager.request_payload["req"]["personaplex_frames"][0].equal(second_frame.reshape(-1))


def test_async_chunk_emits_initial_then_fixed_chunks_and_skips_the_rest() -> None:
    manager = SimpleNamespace(
        connector=SimpleNamespace(config={"extra": {"initial_codec_chunk_frames": 1, "codec_chunk_frames": 5}})
    )
    request = SimpleNamespace(
        request_id="req",
        external_req_id="req",
        resumable=True,
        is_finished=lambda: True,
        additional_information=None,
    )
    frames = [torch.arange(8, dtype=torch.long).reshape(1, 8) + 100 * i for i in range(17)]
    chunks = []
    for frame in frames:
        request.additional_information = {"codes": {"audio": frame}}
        payload = talker2code2wav_async_chunk(manager, multimodal_output=None, request=request, is_finished=True)
        if payload is SKIP_TRANSFER:
            continue
        assert payload.meta.finished.item() is False
        chunks.append(payload.codes.audio.reshape(8, -1))

    assert [codes.shape[1] for codes in chunks] == [1, 5, 5, 5]
    raw = torch.cat(frames, dim=0)
    dedelayed = torch.cat([raw[:-1, :1], raw[1:, 1:]], dim=1).transpose(0, 1)
    assert torch.equal(torch.cat(chunks, dim=1), dedelayed[:, :16])


def test_post_sample_talker_mtp_uses_current_temporal_state() -> None:
    received: dict[str, torch.Tensor] = {}
    recorded: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def depformer(
        text_token: torch.Tensor,
        hidden: torch.Tensor,
        *,
        audio_tokens: torch.Tensor | None = None,
        audio_provided: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        received["text_token"] = text_token
        received["hidden"] = hidden
        received["audio_tokens"] = audio_tokens
        received["audio_provided"] = audio_provided
        received["num_steps"] = num_steps
        return torch.arange(num_steps or 16, dtype=torch.long).reshape(1, -1)

    model = SimpleNamespace(
        _dtype=torch.float32,
        num_active_codebooks=8,
        depformer=depformer,
        _duplex_stage0_runtime=lambda: SimpleNamespace(
            record_sample=lambda *, request_id, text_token, agent_codes: recorded.append(
                (request_id, text_token.clone(), agent_codes.clone())
            )
        ),
    )
    method = getattr(PersonaPlexTalkerForConditionalGeneration, "post_sample_talker_mtp", None)
    assert callable(method)

    codes = method(
        model,
        input_ids=torch.tensor([101]),
        hidden_states=torch.arange(4, dtype=torch.float32).reshape(1, 4),
        req_ids=["r1"],
        req_infos=[
            {
                "duplex": {"data_plane": True},
                "pplex_depformer_audio_tokens": torch.arange(16),
                "pplex_depformer_audio_provided": torch.arange(16) > 0,
            }
        ],
    )

    # Only the vocoded agent codebooks are drawn on the duplex path.
    assert received["num_steps"] == 8
    assert codes.shape == (1, 8)
    assert received["text_token"].tolist() == [101]
    assert received["hidden"].shape == (1, 1, 4)
    assert torch.equal(received["audio_tokens"], torch.arange(16).reshape(1, 16))
    assert torch.equal(received["audio_provided"], (torch.arange(16) > 0).reshape(1, 16))
    assert [(row[0], row[1].item()) for row in recorded] == [("r1", 101)]
    assert torch.equal(recorded[0][2], codes[0])


def _chunk_manager(initial: int = 1, chunk: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        connector=SimpleNamespace(
            config={"extra": {"initial_codec_chunk_frames": initial, "codec_chunk_frames": chunk}},
        )
    )


def _frame_row(request_id: str, frame: torch.Tensor | None, *, resumable: bool = True) -> dict:
    return {
        "multimodal_output": None if frame is None else {"codes.audio": frame},
        "request": SimpleNamespace(
            request_id=request_id,
            external_req_id=request_id,
            resumable=resumable,
            additional_information=None,
        ),
        # Every duplex frame ends a resumable segment.
        "is_finished": True,
    }


def test_async_chunk_batch_matches_the_one_row_processor_row_by_row() -> None:
    generator = torch.Generator().manual_seed(0)
    batched, one_row = _chunk_manager(), _chunk_manager()
    sessions = [f"s{index}" for index in range(6)]
    for step in range(13):
        rows = []
        for index, session in enumerate(sessions):
            frame = torch.randint(0, 2048, (1, 16), generator=generator)
            if index == 2 and step == 4:
                frame[0, 3] = -1  # a frame the de-delay drops
            if index == 1 and step < 5:
                frame = None  # joins late: its first chunk is due with the others' second
            # Session 4 ends at the last step and flushes its tail.
            rows.append(_frame_row(session, frame, resumable=not (index == 4 and step == 12)))

        expected = [talker2code2wav_async_chunk(one_row, **row) for row in rows]
        results = talker2code2wav_async_chunk_batch(batched, rows)

        assert len(results) == len(rows)
        for got, want in zip(results, expected, strict=True):
            if want is SKIP_TRANSFER:
                assert got is SKIP_TRANSFER
            else:
                assert torch.equal(got.codes.audio, want.codes.audio)
                assert bool(got.meta.finished) == bool(want.meta.finished)
    assert batched.request_payload.keys() == one_row.request_payload.keys()
    for session, state in one_row.request_payload.items():
        frames = batched.request_payload[session]["personaplex_frames"]
        assert len(frames) == len(state["personaplex_frames"])
        assert all(torch.equal(a, b) for a, b in zip(frames, state["personaplex_frames"], strict=True))
