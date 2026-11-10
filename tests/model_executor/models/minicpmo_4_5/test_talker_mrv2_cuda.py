# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model Runner V2 sampler of MiniCPM-o 4.5's Talker on CUDA. The CPU cases are
in ``test_talker_mrv2.py``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.minicpmo_4_5.test_talker_mrv2 import _talker

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_talker_sampler_supports_upstream_dummy_sampler_profiling():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from types import SimpleNamespace

    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _install_mrv2_talker_sampler

    device, vocab = torch.device("cuda"), 6562
    reqs = RequestState(4, 64, 64, 0, vocab, device)
    wrapped = _install_mrv2_talker_sampler(Sampler(VllmConfig(), 4, vocab, device, reqs), _talker(max_reqs=4))
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.sampler = wrapped
    runner.model = SimpleNamespace(compute_logits=lambda hidden: hidden)
    runner.input_buffers = InputBuffers(4, 64, device)
    # Exercise both real upstream profiling passes, including add_request
    # with for_sampler_warmup() before any RequestState admission.
    runner._dummy_sampler_run(torch.randn(3, vocab, device=device))
    assert reqs.index_to_req_id == {}
    assert wrapped._v1_generators == {}


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_mrv2_sampler_applies_codec_window_penalty_instead_of_stock_penalty():
    """The real MRv2 sampler pipeline scores the V1 16-frame codec penalty.

    The stock penalty would tax every code in the prompt (the scheduler's
    placeholder id 0) and in the whole output once; the Talker's penalty taxes
    ``penalty ** count`` over the last 16 sampled codes only.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _CODEC_PENALTY_WINDOW,
        _apply_batched_repetition_penalty,
        _install_mrv2_talker_sampler,
    )

    device, vocab, penalty = torch.device("cuda"), 32, 1.05
    reqs = RequestState(4, 64, 64, 0, vocab, device)
    sampler = Sampler(VllmConfig(), 4, vocab, device, reqs)
    talker = _talker(max_reqs=4)
    wrapped = _install_mrv2_talker_sampler(sampler, talker)

    prompt = [0, 0, 0]
    # Code 5 only before the window, code 7 three times and code 9 once inside it.
    output = [5, 5, 5, 5] + [7, 1, 2, 7, 3, 4, 7, 6, 8, 10, 11, 12, 13, 14, 15, 9]
    assert len(output) - 4 == _CODEC_PENALTY_WINDOW
    reqs.add_request("talker", len(prompt), prompt + output, len(prompt) + len(output), 64)
    slot = reqs.req_id_to_index["talker"]
    wrapped.add_request(slot, SamplingParams(repetition_penalty=penalty, temperature=1.0))
    reqs.apply_staged_writes()
    wrapped.apply_staged_writes()

    logits = torch.linspace(-3.0, 3.0, vocab, device=device).reshape(1, vocab)
    idx = torch.tensor([slot], dtype=torch.int32, device=device)
    processed = wrapped.apply_sampling_params(
        logits.clone(),
        idx,
        idx,
        np.array([slot]),
        torch.tensor([len(prompt) + len(output)], device=device),
        torch.tensor([output[-1]], dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        np.array([len(prompt) + len(output) + 1]),
        skip_top_k_top_p=True,
    )
    expected = _apply_batched_repetition_penalty(
        logits.cpu(), [torch.tensor(output)], penalty=penalty, window_size=_CODEC_PENALTY_WINDOW
    )
    torch.testing.assert_close(processed.cpu(), expected)
    # Neither the placeholder prompt id nor codes outside the window are taxed.
    assert processed[0, 0].item() == pytest.approx(logits[0, 0].item())
    assert processed[0, 5].item() == pytest.approx(logits[0, 5].item())
    assert processed[0, 7].item() == pytest.approx(logits[0, 7].item() * penalty**3)
    # The runner still finds the output bin counts on ``penalties_state``.
    assert sampler.penalties_state.output_bin_counts is not None


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_mrv2_talker_sampler_follows_v1_streams_across_preemption_without_host_sync():
    """The wrapper entry point applies masked and forced EOS around the V1
    seeded draw, keeps each request's generator across preemption and slot
    changes, restarts it only after the request finishes, and never blocks
    the host on the device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from types import SimpleNamespace

    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.ops.topk_topp_sampler import random_sample
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _install_mrv2_talker_sampler

    device, vocab = torch.device("cuda"), 64
    reqs = RequestState(4, 64, 64, 0, vocab, device)
    sampler = Sampler(VllmConfig(), 4, vocab, device, reqs)
    talker = _talker(max_reqs=4)
    talker._codec_eos_id = eos = vocab - 1
    talker._deferred_cleanup_ids = set()
    wrapped = _install_mrv2_talker_sampler(sampler, talker)
    seeds = {"a": 11, "b": 23}
    params = {"a": SamplingParams(seed=11), "b": SamplingParams(seed=23), "greedy": SamplingParams(temperature=0.0)}
    reference = {}

    def admit(req_id: str) -> None:
        reqs.add_request(req_id, 3, [0, 0, 0], 3, 64)
        wrapped.add_request(reqs.req_id_to_index[req_id], params[req_id])
        reqs.apply_staged_writes()
        wrapped.apply_staged_writes()
        if req_id in seeds and req_id not in reference:
            reference[req_id] = torch.Generator(device=device).manual_seed(seeds[req_id])

    def step(req_ids: list[str], *, masked: list[bool], forced: list[bool], unfinished: str | None = None) -> None:
        slots = [reqs.req_id_to_index[req_id] for req_id in req_ids]
        idx = torch.tensor(slots, dtype=torch.int32, device=device)
        computed = np.array([0 if req_id == unfinished else 3 for req_id in req_ids])
        batch = SimpleNamespace(
            num_reqs=len(slots),
            expanded_idx_mapping=idx,
            idx_mapping=idx,
            idx_mapping_np=np.array(slots),
            seq_lens=torch.tensor(computed + 1, dtype=torch.int32, device=device),
            cu_num_logits=torch.arange(len(slots) + 1, dtype=torch.int32, device=device),
            has_prefill=unfinished is not None,
            num_computed_prefill_tokens_np=computed,
            num_scheduled_tokens=np.ones(len(slots), dtype=np.int32),
            prefill_len_np=np.full(len(slots), 3, dtype=np.int32),
        )
        logits = torch.randn(len(slots), vocab, device=device)
        expected = logits.clone()
        expected[:, eos].masked_fill_(torch.tensor(masked, device=device), float("-inf"))
        generators = {row: reference[req_id] for row, req_id in enumerate(req_ids) if req_id in reference}
        held = reference[unfinished].get_state() if unfinished is not None else None
        expected_ids = random_sample(expected.softmax(dim=-1), generators, False).tolist()
        if unfinished is not None:
            reference[unfinished].set_state(held)
        for row, req_id in enumerate(req_ids):
            if req_id == "greedy":
                expected_ids[row] = int(expected[row].argmax())
            if forced[row]:
                expected_ids[row] = eos
        talker._mrv2_masked_eos = torch.tensor(masked, device=device)
        talker._mrv2_forced_eos = torch.tensor(forced, device=device)
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            output = wrapped(logits, batch)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        assert output.sampled_token_ids.view(-1).tolist() == expected_ids
        assert output.num_sampled.tolist() == [int(req_id != unfinished) for req_id in req_ids]
        for req_id in req_ids:
            if req_id in reference:
                assert torch.equal(wrapped._v1_generators[req_id].get_state(), reference[req_id].get_state())

    for req_id in ("a", "b", "greedy"):
        admit(req_id)
    step(["b", "a"], masked=[False, False], forced=[False, False], unfinished="a")
    step(["a", "b", "greedy"], masked=[True, False, False], forced=[False, False, True])
    step(["a", "b", "greedy"], masked=[False, True, False], forced=[False, False, False])

    # Preemption frees the slot without finishing the request; the resumed
    # request lands in another slot and continues its V1 stream.
    old_slot = reqs.req_id_to_index["a"]
    reqs.remove_request("a")
    reqs.add_request("hold", 3, [0, 0, 0], 3, 64)
    admit("a")
    assert reqs.req_id_to_index["a"] != old_slot
    step(["b", "a"], masked=[False, False], forced=[False, False])

    # A finished request id that is resubmitted starts again from its seed.
    talker.on_requests_finished({"a"})
    reqs.remove_request("a")
    reference.pop("a")
    admit("a")
    step(["a", "b"], masked=[False, False], forced=[False, False])
