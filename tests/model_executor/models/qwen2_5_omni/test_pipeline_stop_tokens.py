import pytest

from vllm_omni.model_executor.models.qwen2_5_omni.pipeline import (
    QWEN2_5_OMNI_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_talker_stops_only_on_codec_eos():
    talker = QWEN2_5_OMNI_PIPELINE.get_stage(1)

    assert talker.sampling_constraints["stop_token_ids"] == [8294]
