# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise mixed RAS trajectories and generator ownership on CUDA."""

import pytest

pytest.importorskip(
    "vllm.v1.worker.gpu_input_batch", reason="CosyVoice V1-only sampling contract is retired; V2 sampling tests remain"
)
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.cosyvoice3.test_cosyvoice3_model_helpers import (
    test_mixed_ras_preserves_seeded_trajectories_and_greedy_rng as _check_mixed_ras,
)
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model
from vllm_omni.model_executor.models.cosyvoice3.v1_sampling import commit_rejection_rng, conditional_sample
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


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
