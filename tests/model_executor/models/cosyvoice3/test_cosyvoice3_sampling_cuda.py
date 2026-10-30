# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise mixed RAS trajectories and generator ownership on CUDA."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

from tests.helpers.mark import hardware_test
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_model_helpers import (
    _make_sampling_metadata,
    _make_talker_model,
)
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_model_helpers import (
    test_mixed_ras_preserves_seeded_trajectories_and_greedy_rng as _check_mixed_ras,
)
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model
from vllm_omni.model_executor.models.cosyvoice3.v1_sampling import commit_rejection_rng, conditional_sample
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.gpu_ar_model_runner import _SamplingFeedbackOutput

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("async_output,history", [(True, False), (False, True)])
def test_invalid_logits_are_request_local_at_token_consumption(async_output, history):
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[0] * 10 if history else [], []])
    for name in ("temperature", "top_p", "top_k", "frequency_penalties", "presence_penalties", "repetition_penalties"):
        setattr(metadata, name, getattr(metadata, name).cuda())
    metadata.generators = {row: torch.Generator(device="cuda").manual_seed(42 + row) for row in range(2)}
    good_state = metadata.generators[1].get_state()
    logits = torch.tensor([[float("nan")] * 6761, [0.0] * 6761], device="cuda")
    with pytest.raises(ValueError, match="no finite logits"):
        model.sample(logits.clone(), metadata)
    output = model.sample(logits.clone(), metadata, defer_rng=True)
    if async_output:
        stream = torch.cuda.Stream()
        wrapped = AsyncGPUModelRunnerOutput(
            model_runner_output=SimpleNamespace(req_ids=["bad", "good"]),
            sampled_token_ids=output.sampled_token_ids,
            logprobs_tensors=None,
            invalid_req_indices=[],
            async_output_copy_stream=stream,
            vocab_size=6761,
        )
        feedback = model.stage_deferred_sampling(stream, wrapped.async_copy_ready_event)
        result = _SamplingFeedbackOutput(wrapped, feedback).get_output()
        assert result.invalid_logits_req_ids == {"bad"}
    else:
        feedback = model.finalize_deferred_sampling(output.sampled_token_ids)
    assert feedback.invalid_rows() == [0]
    actual_good = output.sampled_token_ids[1].item()
    actual_state = metadata.generators[1].get_state()
    model.commit_deferred_sampling()

    # The healthy row still follows the original sampling and RNG behavior.
    metadata.generators[1].set_state(good_state)
    reference_metadata = _make_sampling_metadata(output_token_ids=[[]])
    for name in ("temperature", "top_p", "top_k", "frequency_penalties", "presence_penalties", "repetition_penalties"):
        setattr(reference_metadata, name, getattr(reference_metadata, name).cuda())
    reference_metadata.generators = {0: metadata.generators[1]}
    expected = model.sample(logits[1:].clone(), reference_metadata)
    assert actual_good == expected.sampled_token_ids.item()
    assert torch.equal(actual_state, metadata.generators[1].get_state())
    # No device assert: subsequent CUDA work remains usable.
    assert (torch.ones(1, device="cuda") + 1).item() == 2


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("layout", ["default", "mixed_alias"])
def test_conditional_replacement_matches_native_rng(layout):
    device = torch.device("cuda", torch.accelerator.current_device_index())
    default = torch.cuda.default_generators[device.index]
    original = default.get_state()
    shared = torch.Generator(device=device).manual_seed(71)
    layouts = {
        "default": {},
        "mixed_alias": {3: shared, 2: default, 0: shared},
    }
    generators = layouts[layout]
    unique = {id(g): g for g in [default, *generators.values()]}
    scores = torch.sin(torch.arange(4 * 6761, device=device).reshape(4, 6761).float())
    try:
        for flags in ([False, True, False, True], [True, False, True, True], [False] * 4):
            before = {key: g.get_state().clone() for key, g in unique.items()}
            picked = torch.tensor([0, 1, 2, 3], device=device)
            rejected = torch.tensor(flags, device=device)
            actual = conditional_sample(scores, picked.clone(), rejected, generators)
            # Scheduling the GPU draw must not speculatively advance any generator.
            assert all(torch.equal(g.get_state(), before[key]) for key, g in unique.items())
            commit_rejection_rng(flags, generators, scores.shape[1], device)
            actual_states = {key: g.get_state().clone() for key, g in unique.items()}
            for key, g in unique.items():
                g.set_state(before[key])
            rows = [i for i, active in enumerate(flags) if active]
            expected = picked.clone()
            if rows:
                weights = scores[rows].clone()
                weights[torch.arange(len(rows), device=device), picked[rows]] = -float("inf")
                probs = weights.softmax(1)
                noise = torch.empty_like(probs)
                if any(row not in generators for row in rows):
                    noise.exponential_()
                for compact, row in enumerate(rows):
                    if row in generators:
                        noise[compact].exponential_(generator=generators[row])
                expected[rows] = (probs / noise).argmax(1)
            assert torch.equal(actual, expected)
            assert all(torch.equal(g.get_state(), actual_states[key]) for key, g in unique.items())
    finally:
        default.set_state(original)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("seeded_rows", [(0, 2)])
def test_deferred_ras_trajectory_and_token_feedback(monkeypatch, seeded_rows):
    sample = CosyVoice3Model.sample

    def deferred(model, *args, **kwargs):
        output = sample(model, *args, **kwargs, defer_rng=True)
        model.finalize_deferred_sampling(output.sampled_token_ids)
        cached = model.consume_sampled_token_ids_cpu_override(output.sampled_token_ids)
        if cached is not None:
            assert cached == output.sampled_token_ids.tolist()
        return output

    monkeypatch.setattr(CosyVoice3Model, "sample", deferred)
    _check_mixed_ras("mixed", seeded_rows, use_host_params=True, device="cuda")


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("consumer", ["history", "output"])
def test_async_feedback_commits_rng_once_at_token_consumption(consumer):
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[0] * 10, [0] * 10])
    for name in ("temperature", "top_p", "top_k", "frequency_penalties", "presence_penalties", "repetition_penalties"):
        setattr(metadata, name, getattr(metadata, name).cuda())
    metadata.top_k.fill_(1)
    generator = torch.Generator(device="cuda").manual_seed(71)
    metadata.generators = {0: generator}
    default = torch.cuda.default_generators[torch.accelerator.current_device_index()]
    original = default.get_state()
    before = (generator.get_state(), default.get_state())
    logits = torch.full((2, 6761), -float("inf"), device="cuda")
    logits[:, :3] = torch.tensor([3.0, 1.0, 0.0], device="cuda")
    try:
        reference = model.sample(logits.clone(), metadata).sampled_token_ids.cpu()
        # Match the runner's existing discarded-request rollback ordering.
        generator.set_offset(generator.get_offset() - 4)
        expected = (generator.get_state(), default.get_state())
        generator.set_state(before[0])
        default.set_state(before[1])
        output = model.sample(logits.clone(), metadata, defer_rng=True)
        generator.set_offset(generator.get_offset() - 4)
        # Pending feedback must retain ownership if the live batch is compacted.
        metadata.generators.clear()
        stream = torch.cuda.Stream()
        wrapped = AsyncGPUModelRunnerOutput(
            model_runner_output=SimpleNamespace(),
            sampled_token_ids=output.sampled_token_ids,
            logprobs_tensors=None,
            invalid_req_indices=[],
            async_output_copy_stream=stream,
            vocab_size=6761,
        )
        feedback = model.stage_deferred_sampling(stream, wrapped.async_copy_ready_event)
        result = _SamplingFeedbackOutput(wrapped, feedback)
        metadata.output_token_ids = [[-1], [-1]]
        batch = SimpleNamespace(
            sampling_metadata=metadata,
            sampled_token_ids_cpu=wrapped.sampled_token_ids_cpu,
            async_copy_ready_event=wrapped.async_copy_ready_event,
            prev_req_id_to_index={"a": 0, "b": 1},
            req_ids=["a", "b"],
        )
        if consumer == "history":
            InputBatch.update_async_output_token_ids(batch)
            model.commit_deferred_sampling()
            actual = result.get_output().sampled_token_ids
        else:
            actual = result.get_output().sampled_token_ids
            InputBatch.update_async_output_token_ids(batch)
            model.commit_deferred_sampling()
        feedback.commit()
        assert actual == reference.tolist() == metadata.output_token_ids
        assert torch.equal(generator.get_state(), expected[0])
        assert torch.equal(default.get_state(), expected[1])
    finally:
        default.set_state(original)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("vocab", [6761])
def test_fused_replacement_matches_native_tokens_and_rng(vocab):
    scores = torch.sin(torch.arange(5 * vocab, device="cuda").reshape(5, vocab).float())
    scores[1].zero_()
    scores[2] = torch.linspace(-1000, 0, vocab, device="cuda")
    scores[3].fill_(-float("inf"))
    scores[3, 3] = 0
    picked = torch.arange(5, device="cuda")
    flags = [True, True, True, True, False]
    generators = {row: torch.Generator(device=scores.device).manual_seed(42 + row) for row in range(5)}
    before = {row: g.get_state().clone() for row, g in generators.items()}
    actual = conditional_sample(scores, picked.clone(), torch.tensor(flags, device="cuda"), generators)
    assert all(torch.equal(g.get_state(), before[row]) for row, g in generators.items())
    commit_rejection_rng(flags, generators, vocab, scores.device)
    actual_states = {row: g.get_state().clone() for row, g in generators.items()}
    for row, g in generators.items():
        g.set_state(before[row])

    # Independent ATen oracle, including the sole-finite-candidate fallback.
    logits = scores[:4].clone()
    selected = logits.gather(1, picked[:4, None])
    logits.scatter_(1, picked[:4, None], -float("inf"))
    logits.scatter_(
        1, picked[:4, None], torch.where(torch.isfinite(logits).any(1, keepdim=True), -float("inf"), selected)
    )
    probs = logits.softmax(1)
    noise = torch.empty_like(probs)
    for row in range(4):
        noise[row].exponential_(generator=generators[row])
    expected = picked.clone()
    expected[:4] = (probs / noise).argmax(1)
    assert torch.equal(actual, expected)  # Accepted row keeps its original token.
    assert all(torch.equal(g.get_state(), actual_states[row]) for row, g in generators.items())
