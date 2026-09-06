from types import SimpleNamespace

import pytest

from vllm_omni.config.stage_config import StageConfig
from vllm_omni.core.sched.omni_ar_scheduler import _should_emit_engine_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_final_output_is_propagated_to_engine_model_config() -> None:
    config = StageConfig(
        stage_id=0,
        model_stage="thinker",
        final_output=True,
    )

    assert config.to_omegaconf().engine_args.final_output is True


@pytest.mark.parametrize(
    ("async_chunk", "final_output", "use_v2", "stopped", "has_control", "expected"),
    [
        (True, False, True, False, False, False),
        (True, False, False, False, False, True),
        (True, False, True, True, False, True),
        (True, False, True, False, True, True),
        (True, True, True, False, False, True),
        (False, False, True, False, False, True),
    ],
)
def test_async_chunk_intermediate_stage_emits_only_control_outputs(
    async_chunk: bool,
    final_output: bool,
    use_v2: bool,
    stopped: bool,
    has_control: bool,
    expected: bool,
) -> None:
    model_config = SimpleNamespace(
        async_chunk=async_chunk,
        final_output=final_output,
        use_v2_model_runner=use_v2,
    )

    assert (
        _should_emit_engine_output(
            model_config,
            stopped=stopped,
            has_control=has_control,
        )
        is expected
    )
