# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Same-request regular and tail PCM must be consumed in wire order."""

import pytest
import torch
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest, FinishReason

from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("kind", list(RequestOutputKind))
def test_batched_regular_tail_and_terminal_preserve_audio(kind):
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False, engine_core_output_type="audio")
    processor.add_request(
        EngineCoreRequest(
            request_id="r",
            external_req_id="r",
            prompt_token_ids=[1],
            mm_features=None,
            sampling_params=SamplingParams(detokenize=False, output_kind=kind),
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        ),
        prompt=None,
    )
    chunks = [torch.full((2, frames * 4), float(i)) for i, frames in enumerate((1, 15, 1))]

    def output(audio=None, finish=None):
        return OmniEngineCoreOutput(
            request_id="r",
            new_token_ids=[],
            finish_reason=finish,
            multimodal_output={"model_outputs": audio, "sr": torch.tensor(48000)} if audio is not None else None,
        )

    results = processor.process_outputs([output(chunks[0])]).request_outputs
    results += processor.process_outputs(
        [output(chunks[1]), output(chunks[2]), output(finish=FinishReason.LENGTH)]
    ).request_outputs
    if kind == RequestOutputKind.DELTA:
        audio = [
            result.outputs[0].multimodal_output["audio"]
            for result in results
            if getattr(result.outputs[0], "multimodal_output", None) and "audio" in result.outputs[0].multimodal_output
        ]
        assert len(audio) == 3 and all(isinstance(chunk, torch.Tensor) for chunk in audio)
        got = torch.cat(audio, dim=-1)
    else:
        got = results[-1].outputs[0].multimodal_output["audio"]
    torch.testing.assert_close(got, torch.cat(chunks, dim=-1), rtol=0, atol=0)
    assert results[-1].finished and "r" not in processor.request_states
