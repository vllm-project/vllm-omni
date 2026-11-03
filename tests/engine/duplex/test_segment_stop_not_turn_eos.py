# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A codec segment stop must not become duplex turn EOS on the projector output."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.contracts import DuplexFence, DuplexOutputContext, DuplexRequestIdentity
from vllm_omni.engine.duplex.session.context import StageOutput
from vllm_omni.engine.duplex.session.model_channel import ModelChannel
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _channel(*, projects_intermediate: bool = False) -> ModelChannel:
    async def _unused(*args, **kwargs):
        del args, kwargs

    return ModelChannel(
        SimpleNamespace(plugin=SimpleNamespace(projects_intermediate_outputs=projects_intermediate)),
        SimpleNamespace(),
        close_from_runtime=_unused,
        schedule_silence_continuation=_unused,
        abort_request=_unused,
    )


def _item(output: object, *, segment_finished: bool, stage_id: int = 3, final_stage_id: int = 3) -> StageOutput:
    return StageOutput(
        stage_id=stage_id,
        output=output,
        metrics=None,
        request_id="duplex-s.seg.e.0.r.stage0-turn0",
        context=DuplexOutputContext(
            identity=DuplexRequestIdentity(session_id="seg", fence=DuplexFence(session_id="seg")),
            final_stage_id=final_stage_id,
            segment_finished=segment_finished,
        ),
        decision=None,
    )


def test_segment_stop_does_not_restore_finished_on_omni_output() -> None:
    output = OmniRequestOutput(request_id="duplex-s.seg.e.0.r.stage0-turn0", finished=False)
    built = _channel()._build_stage_output(_item(output, segment_finished=True))
    assert built.finished is False


def test_segment_stop_keeps_raw_output_unfinished() -> None:
    output = SimpleNamespace(request_id="duplex-s.seg.e.0.r.stage0-turn0", finished=False, outputs=[])
    built = _channel()._build_stage_output(_item(output, segment_finished=True))
    assert built.finished is False


def test_real_stop_stays_finished() -> None:
    output = SimpleNamespace(request_id="duplex-s.seg.e.0.r.stage0-turn0", finished=True, outputs=[])
    built = _channel()._build_stage_output(_item(output, segment_finished=False))
    assert built.finished is True


def test_intermediate_stage_stays_unfinished() -> None:
    output = SimpleNamespace(request_id="duplex-s.seg.e.0.r.stage0-turn0", finished=True, outputs=[])
    built = _channel(projects_intermediate=True)._build_stage_output(
        _item(output, segment_finished=False, stage_id=1, final_stage_id=3)
    )
    assert built.finished is False
