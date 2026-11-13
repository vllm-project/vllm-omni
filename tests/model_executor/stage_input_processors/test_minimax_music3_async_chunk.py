# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax Music 3 terminal span forwarding on both model runners.

The talker flags its last span itself. On V1 the forwarder sees it with the
request already finished; on the Model Runner V2 data plane it sees it on the
step that samples ``<|audio_end|>``, before the scheduler acknowledges the
finish. Neither is a talker running ahead of the engine.
"""

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.data_entry_keys import OmniPayloadStruct
from vllm_omni.model_executor.models.minimax_music3.constants import AR_HIDDEN_SIZE, NUM_CODEBOOKS
from vllm_omni.model_executor.models.minimax_music3.prompt import AUDIO_CODE_OFFSET, SPECIAL_TOKEN_IDS
from vllm_omni.model_executor.stage_input_processors import minimax_music3 as music3_proc
from vllm_omni.outputs.mm_outputs import MultimodalPayload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AUDIO_END = SPECIAL_TOKEN_IDS["<|audio_end|>"]


@dataclass
class _TransferManager:
    request_payload: dict = field(default_factory=dict)


@dataclass
class _V1Request:
    """A V1 engine request: finished once the engine has retired it."""

    finished: bool
    external_req_id: str = "r"
    sampling_params: SamplingParams | None = None

    def is_finished(self) -> bool:
        return self.finished


@dataclass
class _V2Snapshot(_V1Request):
    """The MRv2 data plane's request snapshot, which also carries the step's tokens."""

    sampled_token_ids: list[int] = field(default_factory=list)


def _terminal_span(frames: int = 150, start: int = 600) -> dict:
    return {
        "latent": torch.zeros(frames, AR_HIDDEN_SIZE, dtype=torch.bfloat16),
        "codes": {"audio": torch.zeros(frames, NUM_CODEBOOKS, dtype=torch.long)},
        "meta": {"num_processed_tokens": start, "last_chunk": True, "audio_seed": 1},
    }


def _forward(request) -> tuple[OmniPayloadStruct, list[str]]:
    manager = _TransferManager()
    with patch.object(music3_proc, "logger") as logger:
        payload = music3_proc.ar2acoustic_async_chunk(manager, _terminal_span(), request)
    assert payload is not None
    return payload, [call.args[0] for call in logger.warning.call_args_list]


def _flagged(warnings: list[str]) -> bool:
    return any("flagged a terminal span" in message for message in warnings)


def test_v2_terminal_span_on_the_stop_step_is_not_reported() -> None:
    request = _V2Snapshot(finished=False, sampling_params=SamplingParams(), sampled_token_ids=[AUDIO_END])

    payload, warnings = _forward(request)

    assert payload.meta.last_chunk is True
    assert payload.meta.num_processed_tokens == 600
    assert not _flagged(warnings)


def test_v1_terminal_span_of_a_finished_request_is_not_reported() -> None:
    request = _V1Request(finished=True)

    payload, warnings = _forward(request)

    assert payload.meta.last_chunk is True
    assert not _flagged(warnings)


def test_terminal_span_while_generation_continues_is_still_reported() -> None:
    # An audio code: the talker keeps decoding.
    request = _V2Snapshot(finished=False, sampled_token_ids=[AUDIO_CODE_OFFSET + 5])

    payload, warnings = _forward(request)

    assert payload.meta.last_chunk is True
    assert _flagged(warnings)


@dataclass
class _Completion:
    multimodal_output: MultimodalPayload


@dataclass
class _ARRequestOutput:
    outputs: list
    request_id: str = "song"
    finished: bool = True


@pytest.mark.parametrize(
    "published_seed",
    [torch.tensor(7), 7],
    ids=["model_runner_v1_tensor", "model_runner_v2_int"],
)
def test_whole_song_handoff_reads_the_published_seed(published_seed) -> None:
    """Without async_chunk the engine hands over the whole song as a
    ``MultimodalPayload``, with the seed as a 0-d tensor on Model Runner V1."""
    song = MultimodalPayload.from_dict(
        {
            "latent": torch.zeros(250, AR_HIDDEN_SIZE, dtype=torch.bfloat16),
            "codes": {"audio": torch.zeros(250, NUM_CODEBOOKS, dtype=torch.long)},
            "meta": {"audio_seed": published_seed, "last_chunk": True},
        }
    )

    (prompt,) = music3_proc.ar2acoustic([_ARRequestOutput([_Completion(song)])])

    info = prompt["additional_information"]
    assert info["meta"]["audio_seed"] == 7
    assert info["meta"]["codec_chunk_frames"] == 250
    assert info["latent"].shape == (250, AR_HIDDEN_SIZE)
    # One placeholder token per 100-frame hop.
    assert prompt["prompt_token_ids"] == [0, 0, 0]
