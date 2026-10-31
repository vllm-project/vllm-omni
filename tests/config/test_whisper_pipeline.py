# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for Whisper ASR pipeline, deploy config, and engine task wiring."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from transformers import WhisperConfig
from vllm.config import CUDAGraphMode, ModelConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.cudagraph_dispatcher import BatchDescriptor

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import (
    _DEPLOY_DIR,
    StageExecutionType,
    _apply_platform_overrides,
    load_deploy_config,
)
from vllm_omni.engine.async_engine_utils import scope_stage_replica_mm_uuids
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.model_executor.models.whisper import WHISPER_PIPELINE
from vllm_omni.worker import gpu_ar_model_runner
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _StubStageClient:
    default_sampling_params: object = None
    final_output: bool = True
    final_output_type: str = "text"
    stage_type: str = "llm"
    model_stage: str = "whisper"


@dataclass
class _StubVllmConfig:
    model_config: ModelConfig


@dataclass
class _StubStagePool:
    stage_client: _StubStageClient
    stage_vllm_config: _StubVllmConfig
    output_processor: object = None


@dataclass
class _StubStageRuntime:
    stage_pools: list[_StubStagePool]

    def initialize(self) -> None:
        return None


@dataclass
class _StubStageConfig:
    is_comprehension: bool = True
    bypass_without_modalities: list[str] = field(default_factory=list)


@dataclass
class _StubCacheConfig:
    block_size: int = 16
    cache_dtype: str = "auto"
    kv_sharing_fast_prefill: bool = False


@dataclass
class _StubKvTransferManager:
    handle_finished_requests_kv_transfer: Callable[..., None] = lambda **_kw: None


@dataclass
class _StubParallelConfig:
    distributed_executor_backend: str = "uni"
    data_parallel_size: int = 1
    num_ubatches: int = 1


@dataclass
class _StubInputBatch:
    num_reqs: int = 1
    req_ids: list[str] = field(default_factory=lambda: ["r1"])
    num_computed_tokens_cpu: np.ndarray = field(default_factory=lambda: np.array([0], dtype=np.int32))
    sampling_metadata: object = None


@dataclass
class _StubKvCacheConfig:
    kv_cache_groups: list[object] = field(default_factory=list)


@dataclass
class _StubModelConfig:
    is_encoder_decoder: bool = False


def test_whisper_pipeline_registered_and_valid() -> None:
    assert "whisper" in OMNI_PIPELINES
    pipeline = resolve_pipeline_config("whisper", WhisperConfig())
    assert pipeline is WHISPER_PIPELINE
    assert pipeline.model_type == "whisper"
    assert pipeline.model_arch == "WhisperForConditionalGeneration"
    assert pipeline.default_deploy_config_name == "whisper.yaml"
    assert len(pipeline.stages) == 1

    stage0 = pipeline.stages[0]
    assert stage0.stage_id == 0
    assert stage0.model_stage == "whisper"
    assert stage0.execution_type is StageExecutionType.LLM_AR
    assert stage0.final_output is True
    assert stage0.final_output_type == "text"
    assert stage0.owns_tokenizer is True
    assert stage0.requires_multimodal_data is True


def test_whisper_deploy_config_loads_for_cuda_and_tpu() -> None:
    deploy_cuda = load_deploy_config(_DEPLOY_DIR / "whisper.yaml")
    _apply_platform_overrides(deploy_cuda, platform="cuda")
    assert deploy_cuda.pipeline == "whisper"
    assert deploy_cuda.model_runner == "v1"
    assert deploy_cuda.async_chunk is False
    assert deploy_cuda.enable_prefix_caching is False
    assert len(deploy_cuda.stages) == 1
    assert deploy_cuda.stages[0].max_model_len == 448
    assert deploy_cuda.stages[0].max_num_batched_tokens == 4096

    deploy_tpu = load_deploy_config(_DEPLOY_DIR / "whisper.yaml")
    _apply_platform_overrides(deploy_tpu, platform="tpu")
    assert deploy_tpu.model_runner == "v1"
    assert deploy_tpu.stages[0].devices == "0"
    assert deploy_tpu.stages[0].gpu_memory_utilization == 0.75


def test_stage_config_factory_resolves_whisper_from_hf_config(monkeypatch: pytest.MonkeyPatch) -> None:
    hf_cfg = WhisperConfig()
    monkeypatch.setattr(StageConfigFactory, "try_infer_model_type", classmethod(lambda cls, **kw: "whisper"))
    monkeypatch.setattr(StageConfigFactory, "get_hf_config", classmethod(lambda cls, **kw: hf_cfg))

    omni_cfg = StageConfigFactory.create_from_model(
        "openai/whisper-large-v3-turbo",
        trust_remote_code=False,
        cli_overrides={},
        deploy_config_path=None,
    )
    assert omni_cfg is not None
    assert omni_cfg.pipeline_config.model_type == "whisper"
    assert len(omni_cfg.stage_configs) == 1


def test_whisper_supported_tasks_transcription_only(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeWhisperModelCls:
        supports_transcription = True
        supports_transcription_only = True

    stub_model_cfg = object.__new__(ModelConfig)
    fake_pool = _StubStagePool(
        stage_client=_StubStageClient(),
        stage_vllm_config=_StubVllmConfig(model_config=stub_model_cfg),
    )
    fake_runtime = _StubStageRuntime(stage_pools=[fake_pool])

    monkeypatch.setattr(
        "vllm_omni.engine.omni_engine_base.create_stage_runtime",
        lambda **_kw: fake_runtime,
    )
    monkeypatch.setattr(
        "vllm_omni.engine.omni_engine_base.build_stage0_input_processor",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "vllm_omni.engine.omni_engine_base.get_model_cls",
        lambda _mc: _FakeWhisperModelCls,
    )

    engine = object.__new__(AsyncOmniEngine)
    engine.model = "openai/whisper-large-v3-turbo"
    engine.config_path = None
    engine.single_stage_mode = False
    engine.async_chunk = False
    engine.tokenizer = None
    engine._parallel_stage_init = False
    engine._single_stage_id_filter = None
    engine.stage_configs = [_StubStageConfig()]
    engine._omni_master_address = None
    engine._omni_master_port = None
    engine._omni_dp_size = 1
    engine._omni_dp_size_local = 1
    engine._omni_heartbeat_timeout = 30.0
    engine._omni_lb_policy = None
    engine.request_queue = None
    engine._log_stats = False
    engine._client_config = None
    engine._entry_stage_renderers = {}

    engine._initialize_stages(stage_init_timeout=30)
    assert engine.supported_tasks == ("transcription",)


def test_scope_stage_replica_mm_uuids_handles_encoder_decoder_prompt() -> None:
    audio = (np.zeros(1600, dtype=np.float32), 16000)
    enc_prompt: dict[str, object] = {
        "prompt": "",
        "multi_modal_data": {"audio": audio},
    }
    enc_dec_prompt: dict[str, dict[str, object]] = {
        "encoder_prompt": enc_prompt,
        "decoder_prompt": {
            "prompt": "<|startoftranscript|>",
        },
    }
    scoped = scope_stage_replica_mm_uuids(
        enc_dec_prompt,
        stage_id=0,
        replica_id=1,
        model_id="openai/whisper-large-v3-turbo",
        mm_hasher_algorithm="blake3",
    )
    assert scoped is not enc_dec_prompt
    assert "multi_modal_uuids" not in enc_prompt
    uuids = scoped["encoder_prompt"]["multi_modal_uuids"]["audio"]
    assert len(uuids) == 1
    assert uuids[0].startswith("stage0:rep1:")


@pytest.mark.parametrize(
    ("is_encoder_decoder", "scheduled_encoder_inputs", "expected_skip_compiled"),
    [
        (True, {"r1": [0]}, True),
        (True, {}, False),
        (False, {"r1": [0]}, False),
    ],
)
def test_gpu_ar_model_runner_sets_skip_compiled_for_encoder_decoder_prefill(
    monkeypatch: pytest.MonkeyPatch,
    is_encoder_decoder: bool,
    scheduled_encoder_inputs: dict[str, list[int]],
    expected_skip_compiled: bool,
) -> None:
    class _StopAfterForwardContextError(RuntimeError):
        pass

    captured_kwargs: dict[str, object] = {}

    def _fake_set_forward_context(*_args: object, **fk_kwargs: object):
        captured_kwargs.update(fk_kwargs)
        return nullcontext()

    monkeypatch.setattr(gpu_ar_model_runner, "set_forward_context", _fake_set_forward_context)
    monkeypatch.setattr(gpu_ar_model_runner, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(gpu_ar_model_runner, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(
        gpu_ar_model_runner,
        "maybe_create_ubatch_slices",
        lambda *_a, **_kw: (None, None),
    )

    runner = object.__new__(GPUARModelRunner)
    runner.execute_model_state = None
    runner.kv_extracted_req_ids = None
    runner.kv_caches = []
    runner.cache_config = _StubCacheConfig()
    runner._resolve_global_request_id = lambda rid: rid
    runner.kv_transfer_manager = _StubKvTransferManager()
    runner.speculative_config = None
    runner.synchronize_input_prep = nullcontext
    runner._update_states = lambda _sched: None
    runner.model = object()
    runner.parallel_config = _StubParallelConfig()
    runner.input_batch = _StubInputBatch()
    runner._prepare_inputs = lambda *_a, **_kw: (torch.tensor([0]), None, 1)
    runner._determine_batch_execution_and_padding = lambda **_kw: (
        CUDAGraphMode.NONE,
        BatchDescriptor(num_tokens=4, num_reqs=1),
        False,
        None,
        None,
    )
    runner._get_runner_assisted_full_attention_metadata_request = lambda **_kw: None
    runner.kv_cache_config = _StubKvCacheConfig()
    runner.attn_groups = []
    runner._get_slot_mappings = lambda **_kw: ({}, {})
    runner._build_attention_metadata = lambda **_kw: ({}, None)
    runner._maybe_attach_attention_metadata_extensions = lambda **_kw: None
    runner._preprocess = lambda *_a, **_kw: (
        torch.zeros(4, dtype=torch.int32),
        None,
        torch.zeros(4, dtype=torch.int32),
        None,
        {},
        None,
    )
    runner.model_config = _StubModelConfig(is_encoder_decoder=is_encoder_decoder)
    runner._prepare_padding_mask = lambda *_a, **_kw: None
    runner.vllm_config = object()
    runner.maybe_get_kv_connector_output = lambda *_a, **_kw: nullcontext(None)
    runner.sampler = None

    def _stop_forward(**_kw: object) -> None:
        raise _StopAfterForwardContextError

    runner._model_forward = _stop_forward

    sched = SchedulerOutput.make_empty()
    sched.num_scheduled_tokens = {"r1": 4}
    sched.total_num_scheduled_tokens = 4
    sched.scheduled_encoder_inputs = scheduled_encoder_inputs

    with pytest.raises(_StopAfterForwardContextError):
        runner.execute_model(sched)

    assert captured_kwargs["skip_compiled"] is expected_skip_compiled
    assert "is_padding" in captured_kwargs
