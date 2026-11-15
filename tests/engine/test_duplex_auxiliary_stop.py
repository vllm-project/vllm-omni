# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Private token prompts must retain native end-of-generation semantics."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GenerationConfig, LlamaConfig, PreTrainedTokenizerFast
from vllm.config import ModelConfig, VllmConfig
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request, RequestStatus

from tests.engine.test_duplex_orchestrator import _build
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    duplex_resource_request_id,
)
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.engine.stage_init_utils import build_stage0_input_processor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def native_processor(tmp_path_factory):
    # No weights or network: the actual tokenizer defines EOS even though the
    # model config does not. Generation config contributes another stop ID.
    path = tmp_path_factory.mktemp("auxiliary-stop-model")
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        eos_token_id=None,
    ).save_pretrained(path)
    backend = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<bos>": 1, "<eos>": 2, "<extra_eos>": 3, "<stop>": 4, "hello": 5, "tail": 6},
            unk_token="<unk>",
        )
    )
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
    ).save_pretrained(path)
    GenerationConfig(eos_token_id=[2, 3]).save_pretrained(path)
    config = VllmConfig(
        model_config=ModelConfig(model=str(path), tokenizer=str(path), max_model_len=4096, dtype="float32")
    )
    return build_stage0_input_processor(config)


async def _submit_private(native_processor, params, *, before_accept=None):
    orchestrator, clients, _, _ = _build(stages=1)
    orchestrator._stage_input_processors[0] = native_processor
    fence = DuplexFence("auxiliary-stop")
    request_id = duplex_resource_request_id(fence, "auxiliary-memory-stop")
    context = DuplexStageRequestContext(
        request_id=request_id,
        session_id=fence.session_id,
        fence=fence,
        stage_id=0,
        final_stage_id=0,
        config_generation=0,
        sampling_params=(params,),
    )
    orchestrator.ensure_request(context)
    try:
        await orchestrator.submit(
            DuplexStageSubmission(
                context=context,
                prompt={"prompt_token_ids": [5], "additional_information": {"speaker": ["Ryan"]}},
                already_submitted=False,
                resumable=False,
            ),
            before_accept=before_accept,
        )
        return clients[0].add_request_calls[0][0]
    finally:
        await orchestrator.cleanup([request_id], abort=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_id", [2, 3, 4], ids=["tokenizer_eos", "generation_eos", "explicit_stop"])
async def test_private_tokens_stop_at_native_boundaries(native_processor, stop_id):
    params = SamplingParams(max_tokens=512, temperature=0, stop_token_ids=[4], output_kind=RequestOutputKind.FINAL_ONLY)
    wire = await _submit_private(native_processor, params)
    request = Request.from_engine_core_request(wire, block_hasher=None)
    request.append_output_token_ids([5, stop_id])

    # The real scheduler must stop at EOS, before unrelated tokens can follow.
    assert check_stop(request, 4096)
    assert request.status is RequestStatus.FINISHED_STOPPED
    assert wire.sampling_params.eos_token_id == 2
    assert wire.sampling_params.all_stop_token_ids == {2, 3, 4}
    assert set(wire.sampling_params.stop_token_ids) == {3, 4}
    assert wire.sampling_params.max_tokens == 512
    assert wire.sampling_params.output_kind is RequestOutputKind.FINAL_ONLY
    assert wire.prompt_token_ids == [5]
    assert wire.external_req_id == wire.request_id
    assert deserialize_additional_information(wire.additional_information)["speaker"] == ["Ryan"]
    # Native processing owns a copy; the caller's fresh policy stays reusable.
    assert params.eos_token_id is None
    assert params.all_stop_token_ids == {4}
    assert params.stop_token_ids == [4]


@pytest.mark.asyncio
async def test_private_tokens_respect_ignore_eos_and_explicit_stop(native_processor):
    params = SamplingParams(max_tokens=16, ignore_eos=True, stop_token_ids=[4])
    wire = await _submit_private(native_processor, params)
    request = Request.from_engine_core_request(wire, block_hasher=None)
    for token in (2, 3):
        request.append_output_token_ids(token)
        assert not check_stop(request, 4096)
    request.append_output_token_ids(4)
    assert check_stop(request, 4096)
    assert request.status is RequestStatus.FINISHED_STOPPED
    assert wire.sampling_params.eos_token_id is None
    assert wire.sampling_params.stop_token_ids == [4]
    assert wire.sampling_params.all_stop_token_ids == {2, 3, 4}


@pytest.mark.asyncio
async def test_private_tokens_keep_stop_suppression_metadata_and_length_limit(native_processor):
    params = SamplingParams(max_tokens=4, min_tokens=2)
    wire = await _submit_private(native_processor, params)
    # The sampler needs all stop IDs to suppress early EOS until min_tokens.
    assert wire.sampling_params.min_tokens == 2
    assert wire.sampling_params.all_stop_token_ids == {2, 3}
    request = Request.from_engine_core_request(wire, block_hasher=None)
    request.append_output_token_ids([5, 5, 5])
    assert not check_stop(request, 4096)
    request.append_output_token_ids(5)
    assert check_stop(request, 4096)
    assert request.status is RequestStatus.FINISHED_LENGTH_CAPPED


@pytest.mark.asyncio
async def test_private_admission_receives_initialized_sampling_params(native_processor):
    accepted = []

    def before_accept(request):
        assert request.sampling_params.eos_token_id == 2
        assert request.sampling_params.all_stop_token_ids == {2, 3}
        accepted.append(request)

    wire = await _submit_private(native_processor, SamplingParams(max_tokens=16), before_accept=before_accept)
    assert len(accepted) == 1
    assert accepted[0] is wire
