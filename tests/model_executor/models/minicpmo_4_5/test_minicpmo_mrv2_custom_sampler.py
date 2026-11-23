# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dispatch tests for MiniCPM-o Talker ``mrv2_custom_sampler``.

The stage-1 ``speculative_config`` carries two unrelated meanings on the
MRv2 Talker: a duplex session uses it to arm the K-step frame loop (the
drafts are booked by the runner's ``_propose_draft_tokens``, no speculator
installed), while a turn session uses it to verify DSpark draft codes
(``MiniCPMO45TalkerRejectionSampler``). Merging an upstream turn-draft
feature must not turn the duplex K-step deployment into a hard error, and
vice versa.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5 import (
    minicpmo_4_5_omni_tts as tts_mod,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_duplex_kstep_returns_no_speculator(monkeypatch):
    # Duplex + K>0 is how the K-step count reaches the Talker; it must not
    # raise and must not install a draft speculator.
    fake_sampler = object()
    monkeypatch.setattr(tts_mod, "_install_mrv2_talker_sampler", lambda s, talker: s)
    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    talker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(session_mode="duplex"),
        speculative_config=SimpleNamespace(num_speculative_tokens=7),
    )
    talker._k_step_frames = 8
    assert tts_mod.MiniCPMO45OmniTTSForConditionalGeneration.mrv2_custom_sampler(talker, fake_sampler) == (
        fake_sampler,
        None,
    )


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
@pytest.mark.parametrize("spec_method", [None, "ngram"])
def test_thinker_ngram_verifies_with_underlying_sampler(monkeypatch, device_type, spec_method):
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2_sampling import MiniCPMO45DuplexSampler
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    spec = SimpleNamespace(method=spec_method) if spec_method else None
    model = SimpleNamespace(
        model_stage="llm",
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                model_stage="llm",
                session_mode="duplex",
                architecture="MiniCPMO45OmniForConditionalGeneration",
                use_v2_model_runner=True,
            ),
            device_config=SimpleNamespace(device_type=device_type),
            speculative_config=spec,
            scheduler_config=SimpleNamespace(async_scheduling=False),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
            ),
            kv_transfer_config=None,
        ),
    )
    device = object()
    base_sampler = SimpleNamespace(req_states=SimpleNamespace(draft_tokens=SimpleNamespace(device=device)))
    monkeypatch.setattr(rejection_sampler, "RejectionSampler", lambda *args: args)

    sampler, verifier = MiniCPMO45OmniForConditionalGeneration.mrv2_custom_sampler(model, base_sampler)

    assert isinstance(sampler, MiniCPMO45DuplexSampler)
    assert verifier == ((base_sampler, spec, device) if spec is not None else None)
