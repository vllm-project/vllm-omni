# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Engine-policy half of the MiniCPM-o 4.5 duplex plugin, as the session runner calls it.

Re-homed from ``tests/engine/duplex/test_duplex_runtime.py`` of the
pre-framework runtime extension: the per-stage sampling overrides a session's
runtime config applies, the listen decision on a finished Stage0 segment, and
the scheduler slots one PCM append reserves.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.contracts import DuplexOutputAction, DuplexOutputDecision
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import (
    MiniCPMO45DuplexPlugin,
    duplex_scheduler_token_budget,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LISTEN_TOKEN_ID = 151705


def _plugin() -> MiniCPMO45DuplexPlugin:
    return MiniCPMO45DuplexPlugin(lambda audio, sample_rate_hz, response_format, speed: None)


def _decide(
    output: object,
    *,
    segment_token_ids: tuple[int, ...] = (),
    segment_output_metadata: dict[str, object] | None = None,
) -> DuplexOutputDecision | None:
    return _plugin().decide_output(
        stage_id=0,
        final_stage_id=1,
        segment_finished=True,
        segment_token_ids=segment_token_ids,
        segment_output_metadata=segment_output_metadata or {},
        output=output,
    )


def test_plugin_owns_stage_sampling_overrides_without_mutating_defaults() -> None:
    defaults = (SamplingParams(max_tokens=4), SamplingParams(max_tokens=8))

    configured = _plugin().configure_sampling_params(
        runtime_config={
            "duplex_stage_max_tokens": {"0": 20},
            "duplex_stage_sampling_params": {"1": {"stop_token_ids": [151645]}},
        },
        defaults=defaults,
    )

    assert configured[0].max_tokens == 20
    assert configured[1].stop_token_ids == [151645]
    assert defaults[0].max_tokens == 4
    assert 151645 not in (defaults[1].stop_token_ids or [])


def test_listen_decision_uses_the_raw_streaming_token_snapshot() -> None:
    decision = _decide(
        SimpleNamespace(outputs=[SimpleNamespace()]),
        segment_token_ids=(LISTEN_TOKEN_ID,),
        segment_output_metadata={"special_token_ids": {"listen_token_id": LISTEN_TOKEN_ID}},
    )

    assert decision is not None
    assert decision.action is DuplexOutputAction.DIRECT_RESPONSE
    assert decision.metadata["duplex_native_decision"] == "listen"
    assert decision.metadata["model_listen"] is True


@pytest.mark.parametrize("attr", ["token_ids", "cumulative_token_ids"])
def test_listen_decision_ignores_output_level_token_history(attr: str) -> None:
    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": LISTEN_TOKEN_ID}},
        outputs=[SimpleNamespace()],
        **{attr: [42, LISTEN_TOKEN_ID]},
    )

    assert _decide(output) is None


@pytest.mark.parametrize("attr", ["token_ids", "cumulative_token_ids"])
def test_listen_decision_uses_completion_token_ids(attr: str) -> None:
    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": LISTEN_TOKEN_ID}},
        outputs=[SimpleNamespace(**{attr: [42, LISTEN_TOKEN_ID]})],
    )

    assert _decide(output) is not None


def test_listen_decision_uses_completion_stop_reason() -> None:
    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": LISTEN_TOKEN_ID}},
        outputs=[SimpleNamespace(stop_reason=LISTEN_TOKEN_ID)],
    )

    assert _decide(output) is not None


def test_a_speak_segment_is_not_a_listen_decision() -> None:
    output = SimpleNamespace(
        multimodal_output={"special_token_ids": {"listen_token_id": LISTEN_TOKEN_ID}},
        outputs=[SimpleNamespace(token_ids=[42, 43], stop_reason=None)],
    )

    assert _decide(output) is None


def test_scheduler_token_budget_estimates_pcm_slots() -> None:
    assert duplex_scheduler_token_budget({"audio": "AAAAAA==", "format": "pcm_f32le", "sample_rate_hz": 16000}) == 16


def test_scheduler_token_budget_ignores_client_budget_fields() -> None:
    assert (
        duplex_scheduler_token_budget(
            {"audio": "AAAAAA==", "format": "pcm_f32le", "duplex_num_input_tokens": 999, "num_input_tokens": 999}
        )
        == 16
    )
