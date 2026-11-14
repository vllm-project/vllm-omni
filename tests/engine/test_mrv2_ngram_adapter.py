# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.worker_v2.ngram_adapter import scheduler_ngram_factory, use_scheduler_ngram

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def thinker_config():
    return SimpleNamespace(
        device_config=SimpleNamespace(device_type="npu"),
        model_config=SimpleNamespace(
            use_v2_model_runner=True,
            model_stage="llm",
            architecture="MiniCPMO45OmniForConditionalGeneration",
        ),
        speculative_config=SimpleNamespace(method="ngram"),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        kv_transfer_config=None,
    )


def test_ascend_mrv2_thinker_uses_scheduler_drafts(thinker_config):
    fallback = object()
    factory = scheduler_ngram_factory(lambda *_: fallback)
    speculator = factory(thinker_config, object())

    assert speculator is not fallback
    assert speculator.draft_logits is None
    assert speculator.propose(dummy_run=True) is None

    thinker_config.model_config.model_stage = "talker"
    assert factory(thinker_config, object()) is fallback


def test_ascend_mrv2_thinker_rejects_async_ngram(thinker_config):
    thinker_config.scheduler_config.async_scheduling = True
    with pytest.raises(ValueError, match="async_scheduling: false"):
        use_scheduler_ngram(thinker_config)
