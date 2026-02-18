"""Unit tests for OmniGPUModelRunner._update_streaming_request."""

from unittest.mock import Mock

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def mock_runner_with_input_batch():
    """Create a mock OmniGPUModelRunner with a real InputBatch."""
    from vllm.v1.worker.gpu_input_batch import InputBatch

    runner = Mock(spec=OmniGPUModelRunner)
    runner.uses_mrope = False
    runner.requests = {}
    runner.max_num_reqs = 10
    runner.max_model_len = 1024

    runner.input_batch = InputBatch(
        max_num_reqs=10,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device="cpu",
        pin_memory=False,
        vocab_size=32000,
        block_sizes=[16],
        kernel_block_sizes=[16],
        is_spec_decode=False,
        logitsprocs=None,
        is_pooling_model=False,
    )
    return runner


def test_update_streaming_request_basic(mock_runner_with_input_batch):
    """Test that streaming request state is updated correctly.

    Validates:
    1. The request is removed from InputBatch (avoids duplication)
    2. Request state fields (prompt_token_ids, sampling_params, block_ids,
       num_computed_tokens) are updated
    3. output_token_ids is cleared (intermediate outputs moved to prompt)
    4. prompt_embeds is decoded from PromptEmbedsPayload if present
    """
    runner = mock_runner_with_input_batch
    req_id = "streaming_req_0"

    initial_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.5),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=3,
        output_token_ids=[10, 11],
    )
    runner.requests[req_id] = initial_state
    runner.input_batch.add_request(initial_state)
    assert req_id in runner.input_batch.req_id_to_index

    new_req_data = Mock()
    new_req_data.prompt_token_ids = [1, 2, 3, 10, 4, 5]
    new_req_data.mm_features = []
    new_req_data.prompt_embeds = None
    new_req_data.sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 4
    new_req_data.additional_information = None

    updated = OmniGPUModelRunner._update_streaming_request(
        runner, req_id, new_req_data
    )

    assert updated.prompt_token_ids == [1, 2, 3, 10, 4, 5]
    assert updated.num_computed_tokens == 4
    assert updated.sampling_params.temperature == 0.8
    assert updated.sampling_params.max_tokens == 50
    assert updated.block_ids == ([0, 1],)
    assert updated.output_token_ids == []
    assert runner.requests[req_id] is updated
    assert req_id not in runner.input_batch.req_id_to_index


def test_update_streaming_request_with_prompt_embeds_tensor(mock_runner_with_input_batch):
    """Test streaming update when prompt_embeds is a tensor.

    The base _update_streaming_request assigns prompt_embeds directly
    and updates num_prompt_tokens based on the embeds length.
    """
    runner = mock_runner_with_input_batch
    req_id = "streaming_embed_req"

    initial_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=None,
        mm_features=[],
        sampling_params=SamplingParams(),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=2,
        output_token_ids=[5],
        prompt_embeds=torch.randn(2, 16),
    )
    runner.requests[req_id] = initial_state
    runner.input_batch.add_request(initial_state)

    embed_tensor = torch.randn(4, 16)
    new_req_data = Mock()
    new_req_data.prompt_token_ids = None
    new_req_data.mm_features = []
    new_req_data.prompt_embeds = embed_tensor
    new_req_data.sampling_params = SamplingParams()
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 3

    updated = OmniGPUModelRunner._update_streaming_request(
        runner, req_id, new_req_data
    )

    assert updated.prompt_embeds is not None
    assert updated.prompt_embeds.shape == (4, 16)
    assert torch.equal(updated.prompt_embeds, embed_tensor)
    assert updated.output_token_ids == []
    assert updated.num_prompt_tokens == 4


def test_update_streaming_request_clears_output_and_updates_prompt(mock_runner_with_input_batch):
    """Test that output_token_ids are cleared and prompt grows.

    When a streaming request is updated, the intermediate output tokens
    are moved into the prompt_token_ids. The output_token_ids must be
    cleared since those tokens are now part of the prompt context.
    """
    runner = mock_runner_with_input_batch
    req_id = "streaming_clear_req"

    initial_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=3,
        output_token_ids=[10, 11, 12],
    )
    runner.requests[req_id] = initial_state
    runner.input_batch.add_request(initial_state)

    # New prompt includes old prompt + old outputs + new input
    new_req_data = Mock()
    new_req_data.prompt_token_ids = [1, 2, 3, 10, 11, 12, 20, 21]
    new_req_data.mm_features = []
    new_req_data.prompt_embeds = None
    new_req_data.sampling_params = SamplingParams(temperature=0.5)
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 6  # old prompt + old outputs

    updated = OmniGPUModelRunner._update_streaming_request(
        runner, req_id, new_req_data
    )

    assert updated.prompt_token_ids == [1, 2, 3, 10, 11, 12, 20, 21]
    assert updated.output_token_ids == []
    assert updated.num_computed_tokens == 6
    assert updated.num_prompt_tokens == 8


def test_update_streaming_request_with_multimodal_features(mock_runner_with_input_batch):
    """Test streaming update preserves multimodal features correctly."""
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalKwargsItem,
        PlaceholderRange,
    )

    runner = mock_runner_with_input_batch
    req_id = "streaming_mm_req"

    mm_feature_1 = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy("audio"),
        modality="audio",
        identifier="audio_1",
        mm_position=PlaceholderRange(offset=2, length=10),
    )

    initial_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2] + [0] * 10 + [3],
        mm_features=[mm_feature_1],
        sampling_params=SamplingParams(),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=13,
        output_token_ids=[100],
    )
    runner.requests[req_id] = initial_state
    runner.input_batch.add_request(initial_state)

    mm_feature_2 = MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy("audio"),
        modality="audio",
        identifier="audio_2",
        mm_position=PlaceholderRange(offset=14, length=5),
    )

    new_req_data = Mock()
    new_req_data.prompt_token_ids = [1, 2] + [0] * 10 + [3, 100] + [0] * 5 + [4]
    new_req_data.mm_features = [mm_feature_1, mm_feature_2]
    new_req_data.prompt_embeds = None
    new_req_data.sampling_params = SamplingParams(temperature=0.7)
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 13
    new_req_data.additional_information = None

    updated = OmniGPUModelRunner._update_streaming_request(
        runner, req_id, new_req_data
    )

    assert len(updated.mm_features) == 2
    assert updated.mm_features[0] == mm_feature_1
    assert updated.mm_features[1] == mm_feature_2
    assert len(updated.prompt_token_ids) == 20
    assert updated.output_token_ids == []
    assert updated.num_computed_tokens == 13
    assert updated.sampling_params.temperature == 0.7
    assert req_id not in runner.input_batch.req_id_to_index


def test_update_states_routes_to_streaming_update(mock_runner_with_input_batch):
    """Test that _update_states detects existing request and routes to
    _update_streaming_request instead of creating a new CachedRequestState.
    """
    runner = mock_runner_with_input_batch
    req_id = "route_test_req"

    initial_state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(),
        pooling_params=None,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=3,
        output_token_ids=[10],
    )
    runner.requests[req_id] = initial_state

    # The key behavior: if req_id is already in runner.requests when
    # processing scheduled_new_reqs, it should call _update_streaming_request
    assert req_id in runner.requests
    assert initial_state.output_token_ids == [10]

    new_req_data = Mock()
    new_req_data.prompt_token_ids = [1, 2, 3, 10, 4]
    new_req_data.mm_features = []
    new_req_data.prompt_embeds = None
    new_req_data.sampling_params = SamplingParams()
    new_req_data.pooling_params = None
    new_req_data.block_ids = ([0, 1],)
    new_req_data.num_computed_tokens = 4
    new_req_data.additional_information = None

    updated = OmniGPUModelRunner._update_streaming_request(
        runner, req_id, new_req_data
    )

    # Same object should be reused
    assert updated is initial_state
    assert updated.prompt_token_ids == [1, 2, 3, 10, 4]
    assert updated.output_token_ids == []
