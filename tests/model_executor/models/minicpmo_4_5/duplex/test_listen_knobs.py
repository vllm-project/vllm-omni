# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The listen decision knobs: validation, runtime-config plumbing, and the payload they reach.

``listen_prob_scale`` / ``listen_top_k`` mirror the released
``streaming_generate`` arguments of the same names: the listen logit scales
after the repetition penalty, and ``<|listen|>`` ranking within ``listen_top_k``
is returned outright. A session asks for them through ``extra_body``; the
sampler reads them off the append payload, next to ``force_listen``.
"""

from __future__ import annotations

import math

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import (
    MiniCPMO45ClientRuntimeConfigError,
    MiniCPMO45DuplexPlugin,
    _duplex_listen_knobs,
    build_duplex_data_plane_prompt,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("extra_body", "expected"),
    [
        (None, (1.0, None)),
        ({}, (1.0, None)),
        ({"listen_prob_scale": 0.5}, (0.5, None)),
        ({"listen_prob_scale": 2}, (2.0, None)),
        ({"listen_top_k": 5}, (1.0, 5)),
        ({"listen_prob_scale": 0.25, "listen_top_k": 2}, (0.25, 2)),
        ({"force_listen_count": 3}, (1.0, None)),
    ],
)
def test_listen_knobs_parse_the_documented_values(extra_body, expected):
    assert _duplex_listen_knobs(extra_body) == expected


@pytest.mark.parametrize(
    "extra_body",
    [
        {"listen_prob_scale": True},
        {"listen_prob_scale": "0.5"},
        {"listen_prob_scale": 0},
        {"listen_prob_scale": -1.0},
        {"listen_prob_scale": math.nan},
        {"listen_prob_scale": math.inf},
        {"listen_top_k": True},
        {"listen_top_k": 2.0},
        {"listen_top_k": "2"},
        {"listen_top_k": 0},
        {"listen_top_k": -3},
    ],
)
def test_listen_knobs_reject_invalid_values_at_session_creation(extra_body):
    with pytest.raises(MiniCPMO45ClientRuntimeConfigError):
        _duplex_listen_knobs(extra_body)


def _prompt(runtime_config: dict[str, object], payload: dict[str, object]) -> dict[str, object]:
    return build_duplex_data_plane_prompt(
        request_id="listen-knobs",
        fence=DuplexFence("sid", turn_id=1),
        session_config={},
        runtime_config=runtime_config,
        seq=2,
        turn_seq=2,
        payload=payload,
        final=False,
    )


def _prompt_payload(prompt: dict[str, object]) -> dict[str, object]:
    return prompt["model_intermediate_buffer"]["duplex"]["payload"]


def test_listen_knobs_ride_the_append_payload():
    payload = {"audio": "", "format": "pcm_f32le"}
    prompt = _prompt({"duplex_listen_prob_scale": 0.5, "duplex_listen_top_k": 3}, dict(payload))
    assert _prompt_payload(prompt)["listen_prob_scale"] == 0.5
    assert _prompt_payload(prompt)["listen_top_k"] == 3


@pytest.mark.parametrize(
    "runtime_config",
    [
        {},
        {"duplex_listen_prob_scale": 1.0, "duplex_listen_top_k": 0},
        {"duplex_listen_prob_scale": "broken", "duplex_listen_top_k": None},
    ],
)
def test_neutral_or_unreadable_listen_knobs_leave_the_append_payload_untouched(runtime_config):
    # A session that did not ask for the knobs samples exactly as before, and a
    # runtime config rebuilt without validation must not crash the append.
    payload = {"audio": "", "format": "pcm_f32le"}
    assert _prompt_payload(_prompt(runtime_config, dict(payload))) == payload


def test_listen_knobs_cannot_change_after_the_session_is_created():
    plugin = MiniCPMO45DuplexPlugin(lambda *args, **kwargs: b"")
    current = {
        "instructions": "be brief",
        "duplex_listen_prob_scale": 0.5,
        "duplex_listen_top_k": 3,
    }
    kept = plugin.runtime_config_for_update(
        DuplexSessionConfig(
            model="openbmb/MiniCPM-o-4_5",
            instructions="be brief",
            extra_body={"listen_prob_scale": 0.5, "listen_top_k": 3},
        ),
        current,
    )
    assert kept["duplex_listen_prob_scale"] == 0.5
    assert kept["duplex_listen_top_k"] == 3
    with pytest.raises(MiniCPMO45ClientRuntimeConfigError):
        plugin.runtime_config_for_update(
            DuplexSessionConfig(
                model="openbmb/MiniCPM-o-4_5",
                instructions="be brief",
                extra_body={"listen_prob_scale": 0.25},
            ),
            current,
        )
    with pytest.raises(MiniCPMO45ClientRuntimeConfigError):
        plugin.runtime_config_for_update(
            DuplexSessionConfig(
                model="openbmb/MiniCPM-o-4_5",
                instructions="be brief",
                extra_body={"listen_top_k": 8},
            ),
            current,
        )
