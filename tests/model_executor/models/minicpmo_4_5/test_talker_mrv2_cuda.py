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
