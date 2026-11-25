# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Rendered media must reach every stage-0 replica, including late joiners."""

import contextlib
import multiprocessing
import uuid

import pytest
import torch
from vllm.config import ModelConfig, ParallelConfig, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.cache import (
    LruKeyReplicatedReceiverCache,
    MultiModalProcessorOnlyCache,
    ShmObjectStoreReceiverCache,
    processor_cache_from_config,
    processor_only_cache_from_config,
)
from vllm.multimodal.cache.base import MultiModalProcessorCacheInItem
from vllm.multimodal.inputs import MultiModalBatchedField, MultiModalFieldElem, MultiModalKwargsItem
from vllm.renderers import BaseRenderer
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.engine.stage_runtime import StageRuntime
from vllm_omni.inputs.preprocess import omni_renderer_cls

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CacheRenderer(BaseRenderer):
    """Skip model loading, retaining upstream cache selection and its property."""

    def __init__(self, config):
        self._mm_processor_cache = processor_cache_from_config(config)
        self._mm_processor_only_cache = processor_only_cache_from_config(config)

    def render_messages(self, messages, params):
        raise NotImplementedError


@pytest.fixture
def frontend(mocker, monkeypatch):
    model = mocker.Mock(spec=ModelConfig, skip_tokenizer_init=False, supports_multimodal_inputs=True)
    model.get_multimodal_config.return_value = MultiModalConfig(mm_processor_cache_gb=0.001)
    model.try_get_generation_config.return_value = {}
    config = mocker.Mock(
        spec=VllmConfig,
        model_config=model,
        parallel_config=mocker.Mock(spec=ParallelConfig, world_size=1, _api_process_count=1, data_parallel_size=1),
    )
    monkeypatch.setenv("VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME", f"omni-stage0-{uuid.uuid4().hex}")
    processor = mocker.Mock(spec=InputProcessor)
    with contextlib.ExitStack() as cleanup:

        def build_renderer(cfg):
            renderer = omni_renderer_cls(_CacheRenderer)(cfg)
            for cache in (renderer._mm_processor_cache, renderer._mm_processor_only_cache):
                if cache is not None:
                    cleanup.callback(cache.close)
            processor.renderer = renderer
            return renderer

        def build_processor(*, vllm_config, renderer):
            assert vllm_config is config
            assert renderer is processor.renderer
            return processor

        monkeypatch.setattr(stage_init_utils, "build_omni_renderer", build_renderer)
        monkeypatch.setattr(stage_init_utils, "_build_token_only_renderer", build_renderer)
        monkeypatch.setattr(stage_init_utils, "InputProcessor", build_processor)
        yield config, processor, cleanup


@pytest.mark.parametrize("cache_type", ["lru", "shm"])
@pytest.mark.parametrize("num_replicas,distributed", [(1, False), (2, False), (3, False), (0, True), (1, True)])
def test_rendered_audio_reaches_each_replica(frontend, mocker, cache_type, num_replicas, distributed):
    config, processor, cleanup = frontend
    config.model_config.get_multimodal_config().mm_processor_cache_type = cache_type
    if cache_type == "shm":
        config.model_config.get_multimodal_config().mm_shm_cache_max_object_size_mb = 1
    pool = mocker.Mock(
        spec=StagePool,
        stage_client=None,
        stage_vllm_config=config,
        output_processor=None,
        live_num_replicas=num_replicas,
    )
    engine = object.__new__(AsyncOmniEngine)
    engine.stage_configs = []
    engine.single_stage_mode = distributed
    engine._runtime = mocker.Mock(spec=StageRuntime, stage_pools=[pool])
    engine._assign_stage_runtime_fields()
    assert engine.input_processor is processor
    cache = processor.renderer.mm_processor_cache

    def new_receiver():
        if cache_type == "shm":
            receiver = ShmObjectStoreReceiverCache(config, multiprocessing.Lock())
            cleanup.callback(receiver._shm_cache.close)
        else:
            receiver = LruKeyReplicatedReceiverCache(config.model_config)
        return receiver

    receivers = [new_receiver() for _ in range(max(num_replicas, 1))]
    item = MultiModalKwargsItem({"audio": MultiModalFieldElem(data=torch.arange(16), field=MultiModalBatchedField())})
    preprocessing_calls = 0
    for request in range(12):
        if distributed and request == 1:
            receivers.append(new_receiver())  # Join after the frontend cached the audio.
        receiver = receivers[request % len(receivers)]
        if request == 6 and len(receivers) > 1:
            receiver.clear_cache()
        # A rendered cache hit skips HF processing, before a replica is selected.
        processed: MultiModalProcessorCacheInItem = None if cache.is_cached_item("audio-A") else (item, [])
        preprocessing_calls += processed is not None
        data, updates = cache.get_and_update_item(processed, "audio-A")
        received = receiver.get_and_update_item(data, "audio-A")
        receiver.touch_receiver_cache_item("audio-A", data)
        torch.testing.assert_close(received["audio"].data, item["audio"].data)
        assert updates == []
    assert preprocessing_calls == 1


@pytest.mark.parametrize("cache_policy", ["disabled", "processor_only", "text_only"])
@pytest.mark.parametrize("skip_tokenizer_init", [False, True])
def test_existing_cache_policy_preserved(frontend, cache_policy, skip_tokenizer_init):
    config, _, _ = frontend
    config.model_config.skip_tokenizer_init = skip_tokenizer_init
    if cache_policy == "disabled":
        config.model_config.get_multimodal_config().mm_processor_cache_gb = 0
    elif cache_policy == "text_only":
        config.model_config.supports_multimodal_inputs = False
    elif cache_policy == "processor_only":
        config.parallel_config._api_process_count = 2
    processor = stage_init_utils.build_stage0_input_processor(config, replicated=True)
    cache = processor.renderer.mm_processor_cache
    if cache_policy == "processor_only":
        assert isinstance(cache, MultiModalProcessorOnlyCache)
        assert cache is not processor.renderer._mm_processor_only_cache
    else:
        assert cache is None
