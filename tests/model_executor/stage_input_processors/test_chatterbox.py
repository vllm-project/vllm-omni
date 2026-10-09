# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.data_entry_keys import to_dict, to_struct
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import SAMPLES_PER_FRAME, S3GenDecoder
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioning, build_prompt
from vllm_omni.model_executor.stage_input_processors.chatterbox import t3_to_s3gen
from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import talker2code2wav_async_chunk
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def request_prompt(prompt_tokens: int) -> dict:
    conditioning = VoiceConditioning(
        cond_tokens=torch.randint(0, 6561, (1, 375)),
        speaker_emb=torch.randn(1, 256),
        prompt_token=torch.randint(0, 6561, (1, prompt_tokens)),
        prompt_feat=torch.randn(1, 2 * prompt_tokens, 80),
        embedding=torch.randn(1, 192),
    )
    return build_prompt([5, 6, 7], conditioning, ChatterboxConfig())


def talker_output(token_ids: list[int], finished: bool = True) -> SimpleNamespace:
    return SimpleNamespace(finished=finished, request_id="r", outputs=[SimpleNamespace(cumulative_token_ids=token_ids)])


def test_codes_are_the_valid_speech_tokens_001() -> None:
    """The stop token, and any placeholder echoed from the prompt, never reach the decoder."""
    (stage_input,) = t3_to_s3gen([talker_output([6561, 6561, 10, 20, 6560, 6562])], request_prompt(250))
    assert stage_input["prompt_token_ids"] == [10, 20, 6560]


def test_payload_is_one_final_chunk_in_the_requests_own_voice_001() -> None:
    prompt = request_prompt(202)
    (stage_input,) = t3_to_s3gen([talker_output([10, 20, 6562])], prompt)
    payload = to_struct(stage_input["additional_information"])

    assert bool(payload.meta.stream_finished) is True
    assert payload.meta.left_context_size == 0
    reference = prompt["additional_information"]["embed"]
    assert payload.embed.speech_token is reference["speech_token"]
    assert payload.embed.speech_feat is reference["speech_feat"]
    assert payload.embed.embedding is reference["embedding"]
    # Stage 0's own conditioning stays behind.
    assert payload.embed.voice is None and payload.ids is None


def test_unfinished_outputs_are_skipped_001() -> None:
    assert t3_to_s3gen([talker_output([10, 20], finished=False)], request_prompt(250)) == []


def test_an_utterance_with_no_speech_tokens_is_an_error_001() -> None:
    with pytest.raises(RuntimeError, match="no speech tokens"):
        t3_to_s3gen([talker_output([6562])], request_prompt(250))


def test_streamed_chunks_from_the_shared_processor_decode_to_the_whole_utterance_001() -> None:
    """Streaming reuses CosyVoice3's chunk processor.

    It must find the reference under the names build_prompt writes, chunk as
    the deploy file's connector settings say, and emit payloads stage 1
    accepts, one token of talker output at a time.
    """
    deploy = yaml.safe_load(Path(get_deploy_config_path("chatterbox_turbo.yaml")).read_text())
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(config={"extra": deploy["connectors"]["connector_of_shared_memory"]["extra"]}),
    )
    prompt = request_prompt(250)
    tokens = torch.randint(0, 6561, (60,)).tolist()
    decoder = S3GenDecoder(ChatterboxConfig()).eval()

    merged: dict = {}
    plan, samples = [], 0
    for produced in range(1, 62):
        finished = produced == 61  # the 61st sampled token is the stop token
        request = SimpleNamespace(
            external_req_id="external-id",
            output_token_ids=tokens[:produced] + ([6562] if finished else []),
            additional_information=prompt["additional_information"],
            is_finished=lambda finished=finished: finished,
        )
        payload = talker2code2wav_async_chunk(
            transfer_manager=transfer_manager, multimodal_output=None, request=request, is_finished=finished
        )
        if payload is None:
            continue
        # The runner merges each update into the request's payload; the
        # reference arrives once, on the first chunk.
        merged.update(to_dict(payload))
        codes = payload.codes.audio
        plan.append((codes.numel(), payload.meta.left_context_size, bool(payload.meta.stream_finished)))
        (audio,) = decoder.decode_step(codes, [codes.numel()], [merged], ["scheduler-id"])
        samples += audio.numel()

    # First hop 15 plus 5 to align the 250-token reference, then 30; every
    # chunk but the last waits for the 3-token lookahead.
    assert plan == [(23, 0, False), (53, 20, False), (60, 50, True)]
    assert samples == 2 * (60 + 3) * SAMPLES_PER_FRAME
    assert decoder.streams == {}
