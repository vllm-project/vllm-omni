# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-layer gate tests for the shared Talker K-step configuration contract.

The generic AR scheduler and the model-side gate used to parse the engine's
``speculative_config`` independently and had already drifted: a missing
``method`` armed the scheduler while the model stayed disabled, and only one
side enforced the frame ceiling. Both now resolve the config through
``talker_kstep_contract``, so these tests pin the contract once and assert
that both consumers agree -- including the scheduler reading
``vllm_config.speculative_config`` the way the real engine presents it.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.model_executor.models.minicpmo_4_5.talker_kstep_contract import (
    MINICPMO_TALKER_FRAMES_MAX,
    talker_kstep_frames,
)


def _scheduler(spec, stage: str = "tts") -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._omni_talker_kstep_cache = None
    sched.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(model_stage=stage),
        speculative_config=spec,
    )
    return sched


def _assert_agree(spec, stage: str = "tts") -> int:
    """The scheduler's armed flag must mirror the shared contract's verdict."""
    frames = talker_kstep_frames(stage, spec)
    assert _scheduler(spec, stage)._talker_kstep_armed() is (frames > 0)
    return frames


def test_cap_is_pinned_to_the_model_ceiling():
    assert MINICPMO_TALKER_FRAMES_MAX == 16


def test_missing_method_is_disabled_on_both_sides():
    # The engine injects method explicitly; a config without it is not a
    # K-step config. This used to be the scheduler/model disagreement point:
    # the scheduler defaulted to ngram (armed) while the model read disabled.
    for spec in ({"num_speculative_tokens": 7}, SimpleNamespace(num_speculative_tokens=7)):
        assert _assert_agree(spec) == 0


def test_non_ngram_method_is_disabled_on_both_sides():
    for spec in (
        {"method": "eagle", "num_speculative_tokens": 7},
        SimpleNamespace(method="eagle", num_speculative_tokens=7),
    ):
        assert _assert_agree(spec) == 0


def test_single_frame_request_stays_disabled():
    # K = num_speculative_tokens + 1; K == 1 degenerates to the ordinary
    # one-frame path, so the contract reads it as off on both sides.
    for spec in ({"method": "ngram"}, {"method": "ngram", "num_speculative_tokens": 0}):
        assert _assert_agree(spec) == 0


def test_valid_ngram_config_arms_both_sides():
    assert _assert_agree({"method": "ngram", "num_speculative_tokens": 3}) == 4
    assert _assert_agree(SimpleNamespace(method="ngram", num_speculative_tokens=3)) == 4
    # The deploy YAML pins K=8 (num_speculative_tokens=7).
    assert _assert_agree({"method": "ngram", "num_speculative_tokens": 7}) == 8


def test_non_talker_stage_stays_disabled():
    spec = {"method": "ngram", "num_speculative_tokens": 7}
    assert _assert_agree(spec, stage="llm") == 0
    assert _assert_agree(spec, stage=None) == 0


def test_out_of_range_k_raises_on_both_sides():
    spec = {"method": "ngram", "num_speculative_tokens": MINICPMO_TALKER_FRAMES_MAX}
    with pytest.raises(ValueError, match="capped at"):
        talker_kstep_frames("tts", spec)
    with pytest.raises(ValueError, match="capped at"):
        _scheduler(spec)._talker_kstep_armed()


def test_scheduler_reads_vllm_config_when_own_attr_is_absent():
    # Real vLLM Scheduler instances expose no .speculative_config attribute,
    # so the armed check falls back to vllm_config.speculative_config.
    sched = _scheduler({"method": "ngram", "num_speculative_tokens": 7})
    sched.speculative_config = None
    assert sched._talker_kstep_armed() is True
