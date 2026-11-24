# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned codec draws must survive batching and CUDA graph replay."""

import pytest
import torch
from transformers import Qwen2Config

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.mimo_audio.mimo_audio import MiMoAudioForConditionalGeneration
from vllm_omni.model_executor.models.mimo_audio.mimo_audio_llm import (
    MiMoAudioLLMForConditionalGeneration,
    MiMoLocalDecodeBuffer,
    MiMoLocalDecodeCudaGraph,
    MiMoLocalSampler,
    MiMoLocalSamplerTensor,
)

pytestmark = [pytest.mark.core_model]


def _samplers(batch_size, device):
    eager = MiMoLocalSampler(do_sample=True, temperature=0.9, top_p=0.95)
    captured = MiMoLocalSamplerTensor(
        temperature=torch.full((batch_size,), 0.9, device=device),
        top_k=torch.full((batch_size,), -1, dtype=torch.long, device=device),
        top_p=torch.full((batch_size,), 0.95, device=device),
    )
    return eager, captured


def _bare_model():
    model = MiMoAudioLLMForConditionalGeneration.__new__(MiMoAudioLLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.group_size = 4
    model.audio_channels = 8
    model._local_generators = {}
    model._cached_new_audio_emb_by_req = {}
    return model


def _draw(model, ids, seeds, valid=None):
    return model._draw_local_sampling_uniforms(ids, seeds, valid or [True] * len(ids), torch.device("cpu"))


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_uniform_sampling_parity_and_exclusion(dtype):
    scores = torch.zeros(8, 4, dtype=dtype)
    scores[:, 1] = 30  # Excluded maximum must not consume probability mass.
    original = scores.clone()
    uniforms = torch.tensor([0.0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0])
    uniforms[-1] = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0))
    expected = torch.tensor([0, 0, 0, 2, 2, 3, 3, 3])
    eager, captured = _samplers(8, "cpu")
    for sampler in (eager, captured):
        torch.testing.assert_close(sampler.sample(scores, [1], uniform=uniforms), expected)
    torch.testing.assert_close(scores, original)
    # The categorical draw is per row, including when a scheduler reorders it.
    order = torch.tensor([7, 3, 1, 5, 0, 2, 6, 4])
    torch.testing.assert_close(eager.sample(scores[order], [1], uniform=uniforms[order]), expected[order])


@pytest.mark.cpu
def test_request_draws_survive_reordering_and_inactive_rows():
    batched, serial = _bare_model(), _bare_model()
    first = _draw(batched, ["a", "b"], [42, 7])
    torch.testing.assert_close(first[0], _draw(serial, ["a"], [42])[0])
    torch.testing.assert_close(first[1], _draw(serial, ["b"], [7])[0])
    middle = _draw(batched, ["b", "text", "a"], [7, 999, 42], [True, False, False])
    torch.testing.assert_close(middle[0], _draw(serial, ["b"], [7])[0])
    assert torch.all(middle[1:] == -1)
    assert "text" not in batched._local_generators
    last = _draw(batched, ["a", "b"], [42, 7])
    torch.testing.assert_close(last[0], _draw(serial, ["a"], [42])[0])
    torch.testing.assert_close(last[1], _draw(serial, ["b"], [7])[0])
    assert not torch.equal(first[0], last[0])


@pytest.mark.cpu
def test_seeded_draws_ignore_global_rng_and_default_dtype():
    before = torch.random.get_rng_state()
    model = _bare_model()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        draws = _draw(model, ["a"], [0])
    finally:
        torch.set_default_dtype(previous_dtype)
    assert draws.dtype == torch.float32
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    torch.testing.assert_close(draws, _draw(_bare_model(), ["new-id"], [0]))


@pytest.mark.cpu
def test_finished_request_releases_audio_and_rng_state():
    model = _bare_model()
    first = _draw(model, ["a", "b"], [42, 7])
    model._cached_new_audio_emb_by_req = {"a": torch.ones(1), "b": torch.ones(1)}
    wrapper = MiMoAudioForConditionalGeneration.__new__(MiMoAudioForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.fused_thinker_talker = model
    wrapper.on_requests_finished(["a", "unknown"])
    assert set(model._local_generators) == {"b"}
    assert set(model._cached_new_audio_emb_by_req) == {"b"}
    torch.testing.assert_close(first[0], _draw(model, ["a"], [42])[0])


@pytest.mark.cpu
def test_smaller_graph_batch_clears_previous_draws():
    model = _bare_model()
    model.local_config = Qwen2Config(hidden_size=4)
    model.hidden_states_downcast = torch.nn.Linear(4, 4, bias=False)
    buffer = MiMoLocalDecodeBuffer(model, max_batch_size=8)
    sampler = MiMoLocalSampler(do_sample=True, temperature=0.9, top_p=0.95)
    buffer.prepare(torch.ones(8, 1, 4), sampler, torch.full((8, 32), 0.8))
    next_draws = torch.full((3, 32), 0.2)
    buffer.prepare(torch.ones(3, 1, 4), sampler, next_draws)
    torch.testing.assert_close(buffer.sampling_uniforms[:3], next_draws)
    assert torch.all(buffer.sampling_uniforms[3:] == -1)
    assert torch.all(buffer.input_tensor[3:] == 0)
    buffer.prepare(torch.ones(3, 1, 4), MiMoLocalSampler(do_sample=False))
    assert torch.all(buffer.sampling_uniforms == -1)


@pytest.mark.cpu
def test_seeded_local_decode_keeps_graph_path(mocker):
    model = _bare_model()
    graph = mocker.Mock(spec=MiMoLocalDecodeCudaGraph)
    graph.forward.return_value = torch.zeros(3, 4, 8, dtype=torch.long)
    model.local_forward_cg_by_bs = {4: graph}
    sampler = MiMoLocalSampler(do_sample=True, temperature=0.9, top_p=0.95)
    hidden = torch.ones(3, 1, 4)
    uniforms = torch.full((3, 32), 0.5)
    result = model.local_forward(hidden, local_sampler=sampler, sampling_uniforms=uniforms)
    assert result is graph.forward.return_value
    graph.forward.assert_called_once_with(hidden, sampler, uniforms)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("batch_size", [1, 8])
def test_cuda_replay_consumes_fresh_uniforms(batch_size):
    scores = torch.tensor([[0.0, 30.0, 0.0, 0.0]], device="cuda", dtype=torch.bfloat16).repeat(batch_size, 1)
    uniforms = torch.zeros(batch_size, device="cuda")
    eager, captured = _samplers(batch_size, "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            captured.sample(scores, [1], uniform=uniforms)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = captured.sample(scores, [1], uniform=uniforms)
    for uniform, expected in [(0.0, 0), (0.4, 2), (0.9, 3), (-1.0, 0)]:
        uniforms.fill_(uniform)
        graph.replay()
        assert torch.all(result == expected)
        torch.testing.assert_close(result, eager.sample(scores, [1], uniform=uniforms))
