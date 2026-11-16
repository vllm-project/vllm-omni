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


def _talker(session_mode, k_step_frames):
    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    talker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(session_mode=session_mode),
        speculative_config=SimpleNamespace(num_speculative_tokens=7),
    )
    talker._k_step_frames = k_step_frames
    talker._use_v2_model_runner = True
    return talker


@pytest.fixture()
def fake_sampler(monkeypatch):
    sampler = object()
    monkeypatch.setattr(tts_mod, "_install_mrv2_talker_sampler", lambda s, talker: s)
    return sampler


def test_duplex_kstep_returns_no_speculator(fake_sampler):
    # Duplex + K>0 is how the K-step count reaches the Talker; it must not
    # raise and must not install a draft speculator.
    talker = _talker("duplex", 8)
    assert tts_mod.MiniCPMO45OmniTTSForConditionalGeneration.mrv2_custom_sampler(talker, fake_sampler) == (
        fake_sampler,
        None,
    )


def test_no_spec_config_returns_no_speculator(fake_sampler):
    talker = _talker("duplex", 8)
    talker.vllm_config.speculative_config = None
    assert MiniCPMO45OmniTTSForConditionalGeneration.mrv2_custom_sampler(talker, fake_sampler) == (fake_sampler, None)


def test_turn_session_installs_rejection_sampler(fake_sampler, monkeypatch):
    # Turn + K=0 is the DSpark draft-verification path from upstream.
    sentinel = object()
    monkeypatch.setattr(
        tts_mod,
        "MiniCPMO45TalkerRejectionSampler",
        lambda sampler, spec, talker: sentinel,
    )
    talker = _talker("turn", 0)
    out = MiniCPMO45OmniTTSForConditionalGeneration.mrv2_custom_sampler(talker, fake_sampler)
    assert out == (fake_sampler, sentinel)


def test_duplex_k0_draft_method_raises(fake_sampler):
    # Duplex + K=0 with a stage-1 speculative_config is a draft method the
    # v2 runner cannot verify: its draft rows would be checked without the
    # Talker's codec-window penalty and forced EOS. Fail closed.
    talker = _talker("duplex", 0)
    with pytest.raises(NotImplementedError):
        MiniCPMO45OmniTTSForConditionalGeneration.mrv2_custom_sampler(talker, fake_sampler)


def test_thinker_ngram_verifies_with_underlying_sampler(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2_sampling import MiniCPMO45DuplexSampler
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

    spec = SimpleNamespace(method="ngram")
    model = SimpleNamespace(
        model_stage="llm",
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                model_stage="llm",
                session_mode="duplex",
                architecture="MiniCPMO45OmniForConditionalGeneration",
                use_v2_model_runner=True,
            ),
            device_config=SimpleNamespace(device_type="npu"),
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
    assert verifier == (base_sampler, spec, device)
