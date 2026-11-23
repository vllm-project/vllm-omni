# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.worker_v2.ngram_adapter import is_thinker_ngram, scheduler_ngram_factory, use_scheduler_ngram

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


@pytest.mark.parametrize("method", ["ngram", "ngram_gpu"])
def test_ascend_mrv1_thinker_keeps_worker_drafts_with_duplex_guard(thinker_config, method):
    thinker_config.model_config.use_v2_model_runner = False
    thinker_config.speculative_config.method = method

    assert is_thinker_ngram(thinker_config)
    assert not use_scheduler_ngram(thinker_config)
    worker_speculator = object()
    assert scheduler_ngram_factory(lambda *_: worker_speculator)(thinker_config) is worker_speculator


def test_ascend_mrv2_thinker_rejects_ngram_gpu(thinker_config):
    thinker_config.speculative_config.method = "ngram_gpu"

    with pytest.raises(ValueError, match="ngram_gpu is unsupported on Ascend MRv2"):
        use_scheduler_ngram(thinker_config)


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
@pytest.mark.parametrize("model_stage", ["llm", "tts"])
def test_ngram_duplex_guard_is_thinker_only_on_both_platforms(thinker_config, device_type, model_stage):
    thinker_config.device_config.device_type = device_type
    thinker_config.model_config.model_stage = model_stage

    assert is_thinker_ngram(thinker_config) is (model_stage == "llm")
    assert use_scheduler_ngram(thinker_config) is (model_stage == "llm" and device_type == "npu")


def test_cuda_ngram_keeps_worker_owned_proposals(thinker_config):
    thinker_config.device_config.device_type = "cuda"
    worker_speculator = object()
    assert scheduler_ngram_factory(lambda *_: worker_speculator)(thinker_config) is worker_speculator
