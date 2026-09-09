# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.personaplex.personaplex_talker import (
    PersonaPlexTalkerForConditionalGeneration,
)
from vllm_omni.model_executor.stage_input_processors.personaplex import (
    talker2code2wav_async_chunk,
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

    assert first is not None
    assert first.codes is None
    assert first.meta is not None
    assert first.meta.finished.item() is False
    assert first.meta.is_segment_finished.item() is False
    expected = torch.cat([first_frame[:, :1], second_frame[:, 1:]], dim=1).reshape(-1)
    assert torch.equal(second.codes.audio, expected)
    assert manager.request_payload["req"]["personaplex_frames"][0].equal(second_frame.reshape(-1))


def test_post_sample_talker_mtp_uses_current_temporal_state() -> None:
    received: dict[str, torch.Tensor] = {}
    recorded: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def depformer(
        text_token: torch.Tensor,
        hidden: torch.Tensor,
        *,
        audio_tokens: torch.Tensor | None = None,
        audio_provided: torch.Tensor | None = None,
    ) -> torch.Tensor:
        received["text_token"] = text_token
        received["hidden"] = hidden
        received["audio_tokens"] = audio_tokens
        received["audio_provided"] = audio_provided
        return torch.arange(16, dtype=torch.long).reshape(1, 16)

    model = SimpleNamespace(
        _dtype=torch.float32,
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

    assert codes.shape == (1, 16)
    assert received["text_token"].tolist() == [101]
    assert received["hidden"].shape == (1, 1, 4)
    assert torch.equal(received["audio_tokens"], torch.arange(16).reshape(1, 16))
    assert torch.equal(received["audio_provided"], (torch.arange(16) > 0).reshape(1, 16))
    assert [(row[0], row[1].item()) for row in recorded] == [("r1", 101)]
    assert torch.equal(recorded[0][2], codes[0])
