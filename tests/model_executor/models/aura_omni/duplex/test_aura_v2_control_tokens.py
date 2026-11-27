# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AURA_v2 two-token control markers (local day0; silent / respond)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.contracts import DuplexOutputAction
from vllm_omni.model_executor.models.aura_omni.control_tokens import AURA_V1, AURA_V2
from vllm_omni.model_executor.models.aura_omni.duplex.data_plane import AuraDataPlaneSession
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    RESPOND_TEXT,
    SILENT_TEXT,
    _speech_text,
    _strip_assistant_text,
    _trim_aura_response_token_ids,
    is_control_text_prefix,
    is_effectively_silent,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode_audio(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str | None:
    del audio, sample_rate, fmt, speed
    return "ZmFrZQ=="


def test_strip_drops_object_ref_tags_but_keeps_the_name() -> None:
    text = "<|object_ref_start|>右侧沙发上的毯子<|object_ref_end|>在哪里？"
    assert _strip_assistant_text(text) == "右侧沙发上的毯子在哪里？"
    # The box span is not object_ref; the client still receives it to draw.
    boxed = "<|object_ref_start|>毯子<|object_ref_end|><|box_start|>(1,2),(3,4)<|box_end|>"
    assert _strip_assistant_text(boxed) == "毯子<|box_start|>(1,2),(3,4)<|box_end|>"
    # Talker must not hear the tags or the coordinates.
    assert _speech_text(boxed) == "毯子"


def test_strip_drops_leading_respond_but_keeps_silent() -> None:
    assert _strip_assistant_text(f"{RESPOND_TEXT}The screen is red.") == "The screen is red."
    assert _strip_assistant_text(RESPOND_TEXT) == ""
    assert _strip_assistant_text(SILENT_TEXT) == SILENT_TEXT
    assert is_effectively_silent(SILENT_TEXT)
    assert not is_effectively_silent(RESPOND_TEXT)
    assert not is_effectively_silent(f"{RESPOND_TEXT}hi")
    assert is_control_text_prefix("<|res")
    assert is_control_text_prefix(RESPOND_TEXT)
    assert not is_control_text_prefix(f"{RESPOND_TEXT}hi")


def _four_stage_defaults(stage1_stops: list[int] | None = None) -> tuple[SamplingParams, ...]:
    return (
        SamplingParams(max_tokens=16),
        SamplingParams(max_tokens=16, stop_token_ids=stage1_stops),
        SamplingParams(max_tokens=64),
        SamplingParams(max_tokens=16),
    )


def test_stage_stop_ids_select_one_family() -> None:
    import vllm_omni.model_executor.models.aura_omni.duplex.plugin as plugin_mod

    v1 = plugin_mod.AuraDuplexPlugin(_encode_audio)
    v1_configured = v1.configure_sampling_params(
        runtime_config={},
        defaults=_four_stage_defaults([AURA_V1.silent_id, AURA_V1.im_end_id]),
    )
    v1_stops = set(v1_configured[1].stop_token_ids or [])
    assert {AURA_V1.silent_id, AURA_V1.im_end_id} <= v1_stops
    assert AURA_V2.silent_id not in v1_stops
    assert AURA_V2.im_end_id not in v1_stops

    v2 = plugin_mod.AuraDuplexPlugin(_encode_audio)
    v2_configured = v2.configure_sampling_params(
        runtime_config={},
        defaults=_four_stage_defaults([AURA_V2.silent_id, AURA_V2.im_end_id, AURA_V1.im_end_id]),
    )
    v2_stops = set(v2_configured[1].stop_token_ids or [])
    assert {AURA_V2.silent_id, AURA_V2.im_end_id} <= v2_stops
    assert AURA_V1.silent_id not in v2_stops
    assert AURA_V1.im_end_id not in v2_stops
    assert AURA_V2.respond_id not in v2_stops


def test_decide_output_v2_silent_and_respond() -> None:
    import vllm_omni.model_executor.models.aura_omni.duplex.plugin as plugin_mod

    plugin = plugin_mod.AuraDuplexPlugin(_encode_audio)
    configured = plugin.configure_sampling_params(
        runtime_config={},
        defaults=_four_stage_defaults([AURA_V2.silent_id, AURA_V2.im_end_id]),
    )
    stops = set(configured[1].stop_token_ids or [])
    assert {AURA_V2.silent_id, AURA_V2.im_end_id} <= stops
    assert AURA_V2.respond_id not in stops

    # v1 silent id inside a v2 body is an ordinary piece, not a listen turn.
    ordinary = (100, 101, 102, AURA_V1.silent_id)
    ordinary_out = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=list(ordinary), text="hello", cumulative_text="hello")]
    )
    assert (
        plugin.decide_output(
            stage_id=1,
            final_stage_id=3,
            segment_finished=True,
            segment_token_ids=ordinary,
            segment_output_metadata={},
            output=ordinary_out,
        )
        is None
    )

    silent_out = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[AURA_V2.silent_id],
                text=SILENT_TEXT,
                cumulative_text=SILENT_TEXT,
            )
        ]
    )
    decision = plugin.decide_output(
        stage_id=1,
        final_stage_id=3,
        segment_finished=True,
        segment_token_ids=(AURA_V2.silent_id,),
        segment_output_metadata={},
        output=silent_out,
    )
    assert decision is not None
    assert decision.action == DuplexOutputAction.DIRECT_RESPONSE

    respond_out = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                token_ids=[AURA_V2.respond_id, 100, AURA_V2.im_end_id],
                text=f"{RESPOND_TEXT}hello",
                cumulative_text=f"{RESPOND_TEXT}hello",
            )
        ]
    )
    assert (
        plugin.decide_output(
            stage_id=1,
            final_stage_id=3,
            segment_finished=True,
            segment_token_ids=(AURA_V2.respond_id, 100, AURA_V2.im_end_id),
            segment_output_metadata={},
            output=respond_out,
        )
        is None
    )


def test_trim_keeps_v2_body_when_v1_im_end_is_an_ordinary_piece() -> None:
    body = [11, AURA_V1.im_end_id, 22]
    trimmed = _trim_aura_response_token_ids([AURA_V2.respond_id, *body, AURA_V2.im_end_id])
    assert trimmed == body
    v1_trimmed = _trim_aura_response_token_ids(
        [AURA_V1.im_start_id, AURA_V1.assistant_id, 198, 11, 22, AURA_V1.im_end_id]
    )
    assert v1_trimmed == [11, 22]


def test_data_plane_strips_respond_from_client_text() -> None:
    session = AuraDataPlaneSession(_encode_audio)
    request_id = "duplex-s.abc.e.0.r.stage1-v2"
    session.begin_request(request_id)
    # Partial respond marker: hold.
    held = list(
        session.project(
            {
                "data_plane_outputs": [
                    SimpleNamespace(
                        request_id=request_id,
                        finished=False,
                        stage_id=1,
                        outputs=[SimpleNamespace(text="<|res", cumulative_text="<|res", finished=False)],
                    )
                ]
            }
        )
    )
    assert held == []
    # Full respond + body: client sees body only.
    events = list(
        session.project(
            {
                "data_plane_outputs": [
                    SimpleNamespace(
                        request_id=request_id,
                        finished=False,
                        stage_id=1,
                        outputs=[
                            SimpleNamespace(
                                text=f"{RESPOND_TEXT}Hello there",
                                cumulative_text=f"{RESPOND_TEXT}Hello there",
                                finished=False,
                            )
                        ],
                    )
                ]
            }
        )
    )
    texts = [e.get("text") for e in events if e.get("text")]
    assert texts == ["Hello there"]
    assert all(RESPOND_TEXT not in (t or "") for t in texts)


def test_v2_sampling_opts_into_two_token_logits() -> None:
    import vllm_omni.model_executor.models.aura_omni.duplex.plugin as plugin_mod

    plugin = plugin_mod.AuraDuplexPlugin(_encode_audio)
    defaults = list(_four_stage_defaults([AURA_V2.silent_id, AURA_V2.im_end_id]))
    defaults[1] = SamplingParams(max_tokens=16, temperature=0.2, stop_token_ids=[AURA_V2.silent_id, AURA_V2.im_end_id])
    configured = plugin.configure_sampling_params(runtime_config={}, defaults=tuple(defaults))
    stage1 = configured[1]
    assert stage1.temperature == 0.2
    assert stage1.min_tokens == 0
    assert (stage1.extra_args or {}).get("aura_mode") == "streaming"


def test_v2_default_system_prompt_matches_native() -> None:
    import asyncio

    import vllm_omni.model_executor.models.aura_omni.duplex.plugin as plugin_mod
    from vllm_omni.engine.duplex.config import DuplexSessionConfig
    from vllm_omni.model_executor.stage_input_processors.aura_omni import (
        DEFAULT_AURA_V2_SYSTEM_PROMPT,
    )

    plugin = plugin_mod.AuraDuplexPlugin(_encode_audio)
    plugin.configure_sampling_params(
        runtime_config={},
        defaults=_four_stage_defaults([AURA_V2.silent_id, AURA_V2.im_end_id]),
    )
    runtime = asyncio.run(plugin.prepare_runtime_config(DuplexSessionConfig(), model_config=None))
    assert runtime["aura_two_token"] is True
    assert runtime["aura_system_prompt"] == DEFAULT_AURA_V2_SYSTEM_PROMPT
    assert "output `<|respond|>` followed immediately" in str(runtime["aura_system_prompt"])
    assert "output exactly `<|silent|>`" in str(runtime["aura_system_prompt"])


def test_asr2aura_two_token_appends_empty_think_shell() -> None:
    from vllm_omni.model_executor.stage_input_processors.aura_omni import (
        AURA_V2_EMPTY_THINK_PREFIX,
        asr2aura,
    )

    prompt = {
        "additional_information": {
            "aura_two_token": True,
            "aura_system_prompt": "sys",
            "deferred_multi_modal_data": {"image": ["frame"]},
        },
        "multi_modal_data": {},
    }
    source = SimpleNamespace(
        request_id="r0",
        outputs=[SimpleNamespace(text="hi", token_ids=[1])],
    )
    [next_input] = asr2aura([source], prompt=[prompt])
    assert next_input["prompt"].endswith(f"<|im_start|>assistant\n{AURA_V2_EMPTY_THINK_PREFIX}")


def test_history_keeps_respond_like_v1_text() -> None:
    from vllm_omni.model_executor.models.aura_omni.duplex.history import (
        drop_session_history,
        get_or_create_session_history,
    )

    drop_session_history("hist-respond")
    history = get_or_create_session_history("hist-respond")
    history.messages.append({"role": "user", "content": "old", "video": None})
    history.messages.append({"role": "assistant", "content": f"{RESPOND_TEXT}old answer"})
    history.messages.append({"role": "user", "content": "now", "video": ("clip", {})})
    history.messages.append({"role": "assistant", "content": f"{RESPOND_TEXT}new answer"})
    prefix = history.render_prefix()
    assert f"{RESPOND_TEXT}old answer" in prefix
    assert f"{RESPOND_TEXT}new answer" in prefix
    drop_session_history("hist-respond")


def test_v2_history_renders_empty_think_before_every_assistant_turn() -> None:
    """Training and the native chat_template put the empty think before every assistant turn."""
    from vllm_omni.model_executor.models.aura_omni.duplex.history import (
        drop_session_history,
        get_or_create_session_history,
    )
    from vllm_omni.model_executor.stage_input_processors.aura_omni import AURA_V2_EMPTY_THINK_PREFIX

    drop_session_history("hist-think")
    history = get_or_create_session_history("hist-think")
    history.begin_user_turn("", video=("clip-a", {"fps": 2.0}))
    history.commit_turn("<|silent|>")
    history.begin_user_turn("這是什麼？", video=("clip-b", {"fps": 2.0}))
    history.commit_turn(f"{RESPOND_TEXT}這是一隻杯子。")
    think = AURA_V2_EMPTY_THINK_PREFIX
    assert history.render_prefix(think_prefix=True) == (
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|><|im_end|>\n"
        f"<|im_start|>assistant\n{think}<|silent|><|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>這是什麼？<|im_end|>\n"
        f"<|im_start|>assistant\n{think}{RESPOND_TEXT}這是一隻杯子。<|im_end|>\n"
    )
    assert think not in history.render_prefix()
    drop_session_history("hist-think")


@pytest.mark.parametrize("two_token", [True, False])
def test_asr2aura_history_think_prefix_follows_two_token(two_token: bool) -> None:
    from vllm_omni.model_executor.models.aura_omni.duplex.history import (
        drop_session_history,
        get_or_create_session_history,
    )
    from vllm_omni.model_executor.stage_input_processors.aura_omni import (
        AURA_V2_EMPTY_THINK_PREFIX,
        asr2aura,
    )

    session = f"hist-think-asr2aura-{two_token}"
    drop_session_history(session)
    history = get_or_create_session_history(session)
    history.messages.append({"role": "user", "content": "old", "video": None})
    history.messages.append({"role": "assistant", "content": f"{RESPOND_TEXT}old answer"})
    prompt = {
        "additional_information": {
            "aura_duplex": True,
            "aura_session_id": session,
            "aura_two_token": two_token,
            "aura_system_prompt": "sys",
        },
        "multi_modal_data": {},
    }
    source = SimpleNamespace(request_id="r0", outputs=[SimpleNamespace(text="now", token_ids=[1])])
    [next_input] = asr2aura([source], prompt=[prompt])
    stored = f"<|im_start|>assistant\n{AURA_V2_EMPTY_THINK_PREFIX}{RESPOND_TEXT}old answer<|im_end|>"
    assert (stored in next_input["prompt"]) is two_token
    drop_session_history(session)
