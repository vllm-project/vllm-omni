# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 Talker codec ids match V1's seeded sampler on the same logits."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.sample.states import SamplingStates

from vllm_omni.model_executor.models.minicpmo_4_5.talker_mrv2_sampling import draw_v1_codec_ids
from vllm_omni.worker.v1_compat.logits_processor.state import LogitsProcessors
from vllm_omni.worker.v1_compat.metadata import SamplingMetadata
from vllm_omni.worker.v1_compat.sample_ops import topk_topp_sampler
from vllm_omni.worker.v1_compat.sampler import Sampler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_VOCAB = 64


@pytest.fixture(autouse=True)
def _cpu_sampling_backend(monkeypatch):
    # These fixtures use CPU tensors even on CUDA hosts. Keep upstream's
    # backend discovery from dispatching them to FlashInfer or Triton.
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    monkeypatch.setattr(topk_topp_sampler, "HAS_TRITON", False)


def _generators(n: int, seed: int) -> dict[int, torch.Generator]:
    generators = {}
    for row in range(n):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + row)
        generators[row] = generator
    return generators


def _sampling_states(temperature, top_k, top_p):
    # Use the real upstream API with CPU-backed buffers. Constructing UVA
    # buffers normally requires CUDA; only that allocation is substituted.
    states = SamplingStates.__new__(SamplingStates)
    states.vocab_size = _VOCAB
    for name, values in (("temperature", temperature), ("top_k", top_k), ("top_p", top_p)):
        tensor = torch.tensor(values)
        setattr(states, name, SimpleNamespace(gpu=tensor, np=tensor.numpy()))
    states.min_p = SimpleNamespace(np=np.zeros(len(temperature), dtype=np.float32))
    states.num_logprobs = np.full(len(temperature), -1, dtype=np.int32)
    return states


def _v1_sample(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """The V1 sampler tail: temperature, top-k/top-p, seeded exponential draw."""
    metadata = SamplingMetadata(
        temperature=temperature,
        all_greedy=bool(torch.all(temperature < 1e-5).item()),
        all_random=bool(torch.all(temperature >= 1e-5).item()),
        top_p=top_p,
        top_k=top_k,
        generators=generators,
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(logits.shape[0]),
        presence_penalties=torch.zeros(logits.shape[0]),
        repetition_penalties=torch.ones(logits.shape[0]),
        output_token_ids=[[] for _ in range(logits.shape[0])],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    sampled, _ = Sampler().sample(logits.clone(), metadata)
    return sampled


def _mrv2_sample(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Scale like the MRv2 logit pipeline, then the shared V1 draw."""
    scale = torch.where(temperature < 1e-5, torch.ones_like(temperature), temperature)
    processed = logits / scale.unsqueeze(1)
    return draw_v1_codec_ids(
        processed,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        host_temperature=temperature.numpy(),
        generators=generators,
    )


def test_seeded_codec_ids_match_v1_across_steps():
    temperature = torch.tensor([0.8, 0.8])
    top_k = torch.tensor([8, 8])
    top_p = torch.tensor([0.85, 0.85])
    v1_generators = _generators(2, 42)
    mrv2_generators = _generators(2, 42)
    for step in range(6):
        logits = torch.arange(_VOCAB, dtype=torch.float32).repeat(2, 1)
        logits = logits + torch.randn(2, _VOCAB).mul_(step + 1)
        v1_ids = _v1_sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, generators=v1_generators)
        mrv2_ids = _mrv2_sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, generators=mrv2_generators)
        assert torch.equal(v1_ids, mrv2_ids), step


def test_mixed_greedy_rows_match_v1_and_keep_the_generator():
    temperature = torch.tensor([0.0, 0.8])
    top_k = torch.tensor([5, 5])
    top_p = torch.tensor([1.0, 0.9])
    logits = torch.randn(2, _VOCAB)
    v1_ids = _v1_sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, generators=_generators(2, 7))
    mrv2_ids = _mrv2_sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, generators=_generators(2, 7))
    assert torch.equal(v1_ids, mrv2_ids)
    assert int(v1_ids[0]) == int(logits[0].argmax())

    # An all-greedy step must not consume the generator: the next seeded draw
    # matches a generator that never saw that step.
    greedy_logits = torch.randn(2, _VOCAB)
    held = _generators(2, 7)
    fresh = _generators(2, 7)
    _mrv2_sample(
        greedy_logits,
        temperature=torch.zeros(2),
        top_k=None,
        top_p=None,
        generators=held,
    )
    follow = torch.randn(2, _VOCAB)
    warm = torch.tensor([0.8, 0.8])
    assert torch.equal(
        _mrv2_sample(follow, temperature=warm, top_k=top_k, top_p=top_p, generators=held),
        _mrv2_sample(follow, temperature=warm, top_k=top_k, top_p=top_p, generators=fresh),
    )


@pytest.mark.parametrize(
    ("num_logprobs", "stock_feature"),
    [(-1, None), (0, None), (-1, "trace_replay_state"), (-1, "return_sampling_mask")],
)
def test_talker_sampler_uses_upstream_logprobs_contract(mocker, num_logprobs, stock_feature):
    """Exercise the adapter entry point used by MRv2 startup profiling."""
    from types import SimpleNamespace

    import numpy as np
    from vllm.v1.worker.gpu.sample.sampler import Sampler as Mrv2Sampler

    from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts as tts

    # Spec rejects removed upstream methods rather than silently creating them.
    base = mocker.Mock(spec=object.__new__(Mrv2Sampler))
    base.trace_replay_state = object() if stock_feature == "trace_replay_state" else None
    base.return_sampling_mask = stock_feature == "return_sampling_mask"
    base.use_fp64_gumbel = False
    base.get_logprobs_dims.side_effect = lambda rows: Mrv2Sampler.get_logprobs_dims(base, rows)
    base.sampling_states = _sampling_states([0.0, 0.0], [_VOCAB, _VOCAB], [1.0, 1.0])
    base.sampling_states.num_logprobs.fill(num_logprobs)
    base.logprob_token_ids_state = SimpleNamespace(max_num_token_ids=lambda rows: 0)
    base.compute_nans = False
    base.needs_logits_processing = np.zeros(2, dtype=bool)
    base.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=torch.ones(2)), index_to_req_id={0: "a", 1: "b"})
    batch = SimpleNamespace(
        expanded_idx_mapping=torch.arange(2),
        idx_mapping_np=np.arange(2),
        idx_mapping=torch.arange(2),
        seq_lens=torch.ones(2, dtype=torch.int32),
        cu_num_logits=torch.arange(3),
        num_reqs=2,
    )
    mocker.patch.object(
        tts,
        "get_num_sampled_and_rejected",
        return_value=(torch.ones(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)),
    )
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    sampler = tts.MiniCPMO45TalkerSampler(base, SimpleNamespace())
    output = sampler._sample_like_v1(logits, batch)
    if num_logprobs == -1 and stock_feature is None:
        base.assert_not_called()
        assert output.sampled_token_ids.tolist() == [[2], [0]]
        assert output.num_sampled.tolist() == [1, 1]
        assert output.logprobs_tensors is None
    else:
        # logprobs=0 still asks for the sampled token's logprob; trace replay
        # and sampling masks are stock-sampler features too.
        base.assert_called_once_with(logits, batch)
        assert output is base.return_value


@pytest.mark.parametrize("top_k", [[_VOCAB, _VOCAB], [8, 8], [8, _VOCAB]])
@pytest.mark.parametrize("top_p", [[1.0, 1.0], [0.85, 0.9]])
def test_partial_prefill_preserves_seeded_streams_with_upstream_sampling_states(mocker, top_k, top_p):
    from vllm.v1.worker.gpu.sample.sampler import Sampler as Mrv2Sampler

    from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts as tts
    from vllm_omni.model_executor.models.minicpmo_4_5 import talker_mrv2_sampling as draw

    base = Mrv2Sampler.__new__(Mrv2Sampler)
    base.sampling_states = _sampling_states([1.0, 1.0], top_k, top_p)
    base.trace_replay_state = None
    base.return_sampling_mask = False
    base.compute_nans = False
    base.use_fp64_gumbel = False
    base.needs_logits_processing = np.zeros(2, dtype=bool)
    base.logprob_token_ids_state = SimpleNamespace(max_num_token_ids=lambda _: 0)
    base.req_states = SimpleNamespace(
        index_to_req_id={0: "prefill", 1: "decode"}, prefill_len=SimpleNamespace(gpu=torch.tensor([6, 1]))
    )
    talker = SimpleNamespace(take_mrv2_forced_eos=lambda *args: None)
    wrapped = tts.MiniCPMO45TalkerSampler(base, talker)
    actual = _generators(2, 42)
    reference = _generators(2, 42)
    wrapped._v1_generators = {"prefill": actual[0], "decode": actual[1]}
    # Reorder slots so the prefill mask must follow batch rows, not slot ids.
    batch = SimpleNamespace(
        num_reqs=2,
        expanded_idx_mapping=torch.tensor([1, 0]),
        idx_mapping=torch.tensor([1, 0]),
        idx_mapping_np=np.array([1, 0]),
        cu_num_logits=torch.arange(3),
        has_prefill=True,
        prefill_len_np=np.array([1, 6]),
        num_scheduled_tokens=np.array([1, 2]),
    )

    def counts(ones, seq_lens, cu_num_logits, mapping, prefill):
        # CPU equivalent of the upstream Triton bookkeeping kernel.
        return (seq_lens >= prefill[mapping]).to(torch.int32), torch.zeros_like(ones)

    mocker.patch.object(tts, "get_num_sampled_and_rejected", side_effect=counts)
    filtering = mocker.spy(draw, "apply_top_k_top_p")
    expected_topp_only = any(k >= _VOCAB and p < 1.0 for k, p in zip(top_k, top_p))
    logits = torch.linspace(-1, 1, _VOCAB).repeat(2, 1)
    # Two discarded chunks, final prefill, then decode: only the decode row
    # advances during the discarded chunks; both streams advance afterwards.
    for computed in (0, 2, 4, 6, 6, 6):
        batch.num_computed_prefill_tokens_np = np.array([1, computed])
        batch.seq_lens = torch.tensor([8, computed + 2])
        batch.has_prefill = computed < 6
        before = reference[0].get_state()
        expected = _v1_sample(
            logits,
            temperature=torch.ones(2),
            top_k=torch.tensor(top_k).flip(0),
            top_p=torch.tensor(top_p).flip(0),
            generators={0: reference[1], 1: reference[0]},
        )
        if computed < 4:
            reference[0].set_state(before)
        output = wrapped(logits.clone(), batch)
        assert output.num_sampled.tolist() == [1, int(computed >= 4)]
        assert torch.equal(output.sampled_token_ids.flatten(), expected)
        assert filtering.call_args.kwargs["has_topp_only_rows"] == expected_topp_only
        for slot in actual:
            assert torch.equal(actual[slot].get_state(), reference[slot].get_state())


def test_talker_sampler_keeps_seeded_generator_until_the_request_finishes(mocker):
    """MRv2 re-adds a request after preemption and on streaming-input updates;
    V1 keeps the request's generator across both and drops it on finish."""
    from types import SimpleNamespace

    from vllm.sampling_params import SamplingParams

    from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts as tts

    base = mocker.Mock()
    base.req_states = SimpleNamespace(device="cpu", index_to_req_id={0: "seeded", 1: "plain"})
    sampler = tts.MiniCPMO45TalkerSampler(base, SimpleNamespace())
    sampler.add_request(0, SamplingParams(seed=5))
    sampler.add_request(1, SamplingParams())
    generator = sampler._v1_generators["seeded"]
    assert sampler._v1_generators["plain"] is None
    first = torch.empty(4).exponential_(generator=generator)

    # Resume in another slot: the stream continues instead of restarting.
    base.req_states.index_to_req_id = {2: "seeded"}
    sampler.add_request(2, SamplingParams(seed=5))
    assert sampler._v1_generators["seeded"] is generator
    reference = torch.Generator().manual_seed(5)
    torch.empty(4).exponential_(generator=reference)
    assert torch.equal(
        torch.empty(4).exponential_(generator=generator), torch.empty(4).exponential_(generator=reference)
    )
    assert base.add_request.call_count == 3

    sampler.forget_requests({"seeded", "plain"})
    assert sampler._v1_generators == {}
    # A resubmitted id starts a fresh stream.
    sampler.add_request(2, SamplingParams(seed=5))
    assert torch.equal(torch.empty(4).exponential_(generator=sampler._v1_generators["seeded"]), first)


def test_talker_sampler_warmup_does_not_create_request_generators(mocker):
    from vllm.sampling_params import SamplingParams

    from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts as tts

    base = mocker.Mock()
    base.req_states = SimpleNamespace(device="cpu", index_to_req_id={})
    sampler = tts.MiniCPMO45TalkerSampler(base, SimpleNamespace())
    params = SamplingParams.for_sampler_warmup()
    for slot in range(3):
        sampler.add_request(slot, params)
    assert base.add_request.call_args_list == [mocker.call(slot, params) for slot in range(3)]
    assert sampler._v1_generators == {}
    base.req_states.index_to_req_id[0] = "real"
    sampler.add_request(0, SamplingParams(seed=42))
    assert set(sampler._v1_generators) == {"real"}
    assert torch.equal(sampler._v1_generators["real"].get_state(), torch.Generator().manual_seed(42).get_state())


def test_seeded_talker_sampler_requires_the_request_slot_first(mocker):
    from types import SimpleNamespace

    from vllm.sampling_params import SamplingParams

    from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts as tts

    base = mocker.Mock()
    base.req_states = SimpleNamespace(device="cpu", index_to_req_id={})
    sampler = tts.MiniCPMO45TalkerSampler(base, SimpleNamespace())
    with pytest.raises(RuntimeError, match="RequestState.add_request must assign the slot"):
        sampler.add_request(3, SamplingParams(seed=1))
    base.add_request.assert_not_called()


def test_runner_finish_path_drops_finished_generators_and_keeps_preempted(mocker, monkeypatch):
    """Drive the MRv2 runner's finish/preempt hook, not the sampler directly."""
    from types import SimpleNamespace

    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts as tts
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    talker = object.__new__(tts.MiniCPMO45OmniTTSForConditionalGeneration)
    talker._deferred_cleanup_ids = set()
    base = mocker.Mock()
    base.req_states = SimpleNamespace(device="cpu", index_to_req_id={0: "done", 1: "paused"})
    sampler = tts.MiniCPMO45TalkerSampler(base, talker)
    talker._mrv2_sampler = sampler
    sampler.add_request(0, SamplingParams(seed=1))
    sampler.add_request(1, SamplingParams(seed=2))
    paused = sampler._v1_generators["paused"]

    # The stage model is the Omni wrapper, which forwards to the Talker.
    wrapper = SimpleNamespace(model=talker)
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(
        on_requests_finished=lambda ids: MiniCPMO45OmniForConditionalGeneration.on_requests_finished(wrapper, ids)
    )
    runner.model_state = mocker.MagicMock(spec=["remove_request"])
    runner.req_states = SimpleNamespace(req_id_to_index={"done": 0, "paused": 1})
    # The upstream base is stubbed: this covers only the Omni override's
    # finished/preempted split, not hooks the base itself might invoke.
    monkeypatch.setattr(GPUModelRunner, "finish_requests", lambda *args: None)
    runner.finish_requests(SimpleNamespace(finished_req_ids={"done"}, preempted_req_ids={"paused"}))

    assert sampler._v1_generators == {"paused": paused}
    assert talker._deferred_cleanup_ids == {"done"}
