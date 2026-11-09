# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small-vocabulary contracts for the upstream sampler used by the AR runner.

These exercise real verifier kernels, not a second rejection implementation.
They do not establish model-level numerical or codec-frame equivalence.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.accelerator.is_available() or torch.version.hip is not None, reason="requires CUDA"),
]


def _sample(drafts, logits, bonus, *, greedy, draft_probs=None):
    from vllm.v1.sample.rejection_sampler import rejection_sample

    batch, width = drafts.shape
    metadata = SimpleNamespace(
        all_greedy=greedy,
        all_random=not greedy,
        temperature=None if greedy else torch.ones(batch, device=drafts.device),
        generators={},
    )
    return rejection_sample(
        draft_token_ids=drafts.flatten().contiguous(),
        num_draft_tokens=[width] * batch,
        max_spec_len=width,
        cu_num_draft_tokens=torch.arange(1, batch + 1, device=drafts.device, dtype=torch.int32) * width,
        draft_probs=draft_probs,
        target_logits=logits,
        bonus_token_ids=bonus.reshape(batch, 1).contiguous(),
        sampling_metadata=metadata,
    )


def test_greedy_accept_replacement_bonus_and_padding():
    # Same verifier argmax rows; each request rejects at a different position.
    drafts = torch.tensor([[0, 1, 2], [0, 2, 2], [2, 1, 2]], device="cuda", dtype=torch.int32)
    logits = torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]], device="cuda").repeat(3, 1)
    bonus = torch.ones(3, device="cuda", dtype=torch.int32)
    output = _sample(drafts, logits, bonus, greedy=True)
    assert output.tolist() == [[0, 1, 2, 1], [0, 1, -1, -1], [0, -1, -1, -1]]


@pytest.mark.parametrize("seed", [17, 83, 251])
@pytest.mark.parametrize("proposal", ["point_mass", "categorical", "equal_target"])
def test_stochastic_verifier_preserves_target_distribution(seed, proposal):
    # N independent one-draft requests. The first output must follow p even
    # when the proposal differs strongly. No same-seed sequence comparison:
    # categorical draws, accept uniforms and recovered draws consume RNG apart.
    torch.manual_seed(seed)
    n = 32768
    p = torch.tensor([0.1, 0.3, 0.6], device="cuda")
    q = p if proposal == "equal_target" else torch.tensor([0.7, 0.2, 0.1], device="cuda")
    if proposal == "point_mass":
        drafts = torch.zeros((n, 1), device="cuda", dtype=torch.int32)
        draft_probs = None  # DFlash's deterministic-proposal interface.
        expected_acceptance = p[0]
    else:
        drafts = torch.multinomial(q, n, replacement=True).to(torch.int32).reshape(n, 1)
        draft_probs = q.expand(n, -1).contiguous()
        expected_acceptance = torch.minimum(p, q).sum()
    logits = p.log().expand(n, -1).contiguous()
    bonus = torch.multinomial(p, n, replacement=True).to(torch.int32)
    output = _sample(drafts, logits, bonus, greedy=False, draft_probs=draft_probs)
    assert bool(((output[:, 0] >= 0) & (output[:, 0] < 3)).all())
    frequencies = torch.bincount(output[:, 0].long(), minlength=3) / n
    # Six marginal standard errors + 0.002 slack; all seeds are fixed. Also
    # check acceptance analytically so a naive target-only sampler cannot pass.
    tolerance = 6 * torch.sqrt(p * (1 - p) / n) + 0.002
    assert bool(((frequencies - p).abs() < tolerance).all()), frequencies.tolist()
    accepted = output[:, 1] >= 0
    rate = accepted.float().mean()
    acceptance_tolerance = 6 * torch.sqrt(expected_acceptance * (1 - expected_acceptance) / n) + 0.002
    assert bool((rate - expected_acceptance).abs() < acceptance_tolerance), float(rate)
    assert torch.equal(output[accepted, 0], drafts[accepted, 0])
    assert torch.equal(output[accepted, 1], bonus[accepted])
