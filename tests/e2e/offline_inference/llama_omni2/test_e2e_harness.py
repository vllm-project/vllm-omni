# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.logprobs import Logprob

from tests.e2e.offline_inference.llama_omni2.run_llama_omni2_e2e import (
    RequestResult,
    _build_parser,
    _engine_kwargs,
    _requested_text_prompts,
    _require_multiple_audio_chunks,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_request_result_reads_audio_from_omni_output_property():
    completion = SimpleNamespace(text="", token_ids=[])
    item = SimpleNamespace(
        final_output_type="audio",
        request_output=SimpleNamespace(outputs=[completion]),
        multimodal_output={
            "audio": torch.tensor([0.25, -0.25]),
            "sr": torch.tensor(24000),
            "codec_units": torch.tensor([12, 34]),
            "sequence_index": torch.tensor(0),
            "consumed_units": torch.tensor(1),
            "finished": torch.tensor(True),
        },
    )
    result = RequestResult(request_id="request-a")

    result.add_output(item)

    assert result.sample_rate == 24000
    assert result.codec_token_ids == [12, 34]
    assert result.sequence_indices == [0]
    assert result.consumed_units == [1]
    assert result.terminal_audio_chunks == 1
    assert torch.equal(result.audio_chunks[0], torch.tensor([0.25, -0.25]))


def test_request_result_accepts_generation_model_outputs_audio_key():
    item = SimpleNamespace(
        final_output_type="audio",
        request_output=SimpleNamespace(outputs=[SimpleNamespace(text="", token_ids=[])]),
        multimodal_output={
            "model_outputs": torch.tensor([0.5, -0.5]),
            "sr": torch.tensor(24000),
        },
    )
    result = RequestResult(request_id="request-b")

    result.add_output(item)

    assert torch.equal(result.audio_chunks[0], torch.tensor([0.5, -0.5]))


def test_request_result_does_not_count_buffered_codec_event_as_audio_chunk():
    item = SimpleNamespace(
        final_output_type="audio",
        request_output=SimpleNamespace(outputs=[SimpleNamespace(text="", token_ids=[])]),
        multimodal_output={
            "audio": None,
            "sr": None,
            "codec_units": torch.tensor([12]),
            "sequence_index": None,
            "consumed_units": None,
            "finished": None,
        },
    )
    result = RequestResult(request_id="request-buffered")

    result.add_output(item)

    assert result.codec_token_ids == [12]
    assert result.audio_chunks == []
    assert result.sequence_indices == []
    assert result.consumed_units == []
    assert result.terminal_audio_chunks == 0


def test_request_result_validation_requires_one_terminal_audio_chunk():
    result = RequestResult(
        request_id="request-no-terminal",
        text_token_ids=[1],
        audio_chunks=[torch.tensor([0.5])],
        sample_rate=24000,
    )

    with pytest.raises(AssertionError, match="terminal audio chunk"):
        result.validate(require_multiple_audio_chunks=False)


def test_request_result_serializes_text_top_logprobs():
    item = SimpleNamespace(
        final_output_type="text",
        request_output=SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    text="Hello",
                    token_ids=[9707],
                    logprobs=[
                        {
                            9707: Logprob(-0.2, rank=1, decoded_token="Hello"),
                            13347: Logprob(-0.3, rank=2, decoded_token="Hi"),
                        }
                    ],
                )
            ]
        ),
    )
    result = RequestResult(request_id="request-c")

    result.add_output(item)

    assert result.text_logprobs == [
        [
            {
                "token_id": 9707,
                "logprob": -0.2,
                "rank": 1,
                "decoded_token": "Hello",
            },
            {
                "token_id": 13347,
                "logprob": -0.3,
                "rank": 2,
                "decoded_token": "Hi",
            },
        ]
    ]


def test_e2e_parser_accepts_multiple_custom_text_prompts():
    args = _build_parser().parse_args(
        [
            "--model",
            "model",
            "--deploy-config",
            "deploy.yaml",
            "--output-dir",
            "output",
            "--label",
            "parity",
            "--mode",
            "text",
            "--text-prompt",
            "Reply with exactly: Hello.",
            "--text-prompt",
            "Reply with exactly: Yes.",
        ]
    )

    assert args.text_prompts == [
        "Reply with exactly: Hello.",
        "Reply with exactly: Yes.",
    ]


def test_e2e_uses_high_margin_default_parity_prompt():
    assert _requested_text_prompts(SimpleNamespace(text_prompts=None)) == ["Answer with exactly one word: OK"]


def test_sync_entrypoint_forces_full_payload_mode():
    args = _build_parser().parse_args(
        [
            "--model",
            "model",
            "--deploy-config",
            "deploy.yaml",
            "--output-dir",
            "output",
            "--label",
            "sync",
            "--mode",
            "speech",
            "--entrypoint",
            "sync",
        ]
    )

    assert args.entrypoint == "sync"
    assert _engine_kwargs(args)["async_chunk"] is False
    assert _require_multiple_audio_chunks(args) is False


def test_async_entrypoint_requires_multiple_audio_chunks():
    args = _build_parser().parse_args(
        [
            "--model",
            "model",
            "--deploy-config",
            "deploy.yaml",
            "--output-dir",
            "output",
            "--label",
            "stream",
            "--mode",
            "text",
        ]
    )

    assert _require_multiple_audio_chunks(args) is True
