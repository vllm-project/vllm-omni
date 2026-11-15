# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A real Qwen private prompt must stop at its native tokenizer boundary."""

import os

import pytest
from vllm.config import VllmConfig
from vllm.sampling_params import RequestOutputKind
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request, RequestStatus

from tests.engine.test_duplex_orchestrator import _build
from tests.model_executor.models.qwen3_omni.test_duplex_context_view_processor import (
    PROCESSOR_REVISION,
)
from tests.model_executor.models.qwen3_omni.test_duplex_context_view_processor import (
    qwen_processor as _processor_fixture,
)
from tests.model_executor.models.qwen3_omni.test_memory_card_extraction import context as _context_fixture
from vllm_omni.config.model import OmniModelConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexFence,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    duplex_resource_request_id,
)
from vllm_omni.engine.stage_init_utils import build_stage0_input_processor
from vllm_omni.model_executor.models.qwen3_omni.duplex.memory_request import build_memory_request
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

context = _context_fixture
qwen_processor = _processor_fixture

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def native_qwen_input_processor(qwen_processor):
    # Reuse the pinned fixture's cached config/tokenizer assets. Building this
    # processor does not instantiate model weights or launch an engine worker.
    source = os.environ.get("QWEN3_OMNI_PROCESSOR_PATH", qwen_processor.tokenizer.name_or_path)
    model_config = OmniModelConfig(
        model=source,
        tokenizer=source,
        revision=PROCESSOR_REVISION,
        tokenizer_revision=PROCESSOR_REVISION,
        trust_remote_code=False,
        max_model_len=4096,
        dtype="float32",
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        model_stage="thinker",
        hf_config_name="thinker_config",
        session_mode="duplex",
        engine_output_type="latent",
    )
    return build_stage0_input_processor(VllmConfig(model_config=model_config))


@pytest.mark.asyncio
async def test_private_memory_prompt_stops_at_qwen_eos(context, qwen_processor, native_qwen_input_processor):
    auxiliary = build_memory_request(
        context,
        processor=qwen_processor,
        input_token_limit=2048,
        output_token_limit=512,
        max_model_len=4096,
    )
    assert auxiliary is not None
    original_ids = list(auxiliary.prompt["prompt_token_ids"])
    original_sampling = auxiliary.sampling_params
    assert qwen_processor.tokenizer.eos_token_id == 151645
    assert original_sampling.eos_token_id is None
    assert original_sampling.all_stop_token_ids == set()

    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = qwen_processor
    orchestrator, clients, _, _ = _build(stages=1, plugin=plugin)
    orchestrator._stage_input_processors[0] = native_qwen_input_processor
    fence = DuplexFence("qwen-memory-stop")
    request_id = duplex_resource_request_id(fence, "auxiliary-memory-stop")
    stage_context = DuplexStageRequestContext(
        request_id=request_id,
        session_id=fence.session_id,
        fence=fence,
        stage_id=0,
        final_stage_id=0,
        config_generation=0,
        sampling_params=(original_sampling,),
    )
    orchestrator.ensure_request(stage_context)
    try:
        await orchestrator.submit(
            DuplexStageSubmission(
                context=stage_context,
                prompt=auxiliary.prompt,
                already_submitted=False,
                resumable=False,
            )
        )
        wire = clients[0].add_request_calls[0][0]
        request = Request.from_engine_core_request(wire, block_hasher=None)
        # Feed a synthetic completion into the real scheduler. This proves the
        # request boundary, not that a model forward will produce valid memory.
        json_ids = qwen_processor.tokenizer.encode(
            '{"operations":[],"resolutions":[],"reconciled":false}', add_special_tokens=False
        )
        assert 151645 not in json_ids
        request.append_output_token_ids(json_ids)
        assert not check_stop(request, 4096)
        request.append_output_token_ids(151645)
        assert check_stop(request, 4096)
        assert request.status is RequestStatus.FINISHED_STOPPED

        assert wire.prompt_token_ids == original_ids
        assert auxiliary.prompt["prompt_token_ids"] == original_ids
        assert wire.sampling_params.eos_token_id == 151645
        assert 151645 in wire.sampling_params.all_stop_token_ids
        assert wire.sampling_params.max_tokens == 512
        assert wire.sampling_params.temperature == 0
        assert wire.sampling_params.output_kind is RequestOutputKind.FINAL_ONLY
        assert wire.sampling_params.detokenize
        assert original_sampling.eos_token_id is None
        assert original_sampling.all_stop_token_ids == set()
        assert original_sampling.max_tokens == 512
        assert original_sampling.output_kind is RequestOutputKind.FINAL_ONLY
    finally:
        await orchestrator.cleanup([request_id], abort=True)
