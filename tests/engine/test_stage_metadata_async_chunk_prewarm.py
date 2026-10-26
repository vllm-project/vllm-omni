# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolution of ``async_chunk_prewarm_payload_func`` into stage metadata and clients.

The legacy (``to_omegaconf``) and structured (``VllmOmniConfig``) paths must
resolve the same callable. MiniCPM-o 4.5 declares it on Code2Wav only.
"""

import copy
import operator
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.v1.engine.core_client import AsyncMPClient, DPLBAsyncMPClient

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.engine.stage_engine_core_client import DPLBStageEngineCoreClient, StageEngineCoreClient
from vllm_omni.engine.stage_init_utils import (
    StageMetadata,
    extract_legacy_stage_metadata,
    extract_stage_metadata_from_omni_stage_config,
)
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import code2wav_prewarm_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _minicpmo_inputs() -> tuple[PipelineConfig, DeployConfig]:
    return MINICPMO_4_5_PIPELINE, load_deploy_config(get_deploy_config_path("minicpmo_4_5.yaml"))


def _generic_inputs() -> tuple[PipelineConfig, DeployConfig]:
    pipeline = PipelineConfig(
        model_type="prewarm-metadata-test",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="thinker",
                execution_type=StageExecutionType.LLM_AR,
                owns_tokenizer=True,
                engine_output_type="latent",
            ),
            StagePipelineConfig(
                stage_id=1,
                model_stage="decoder",
                execution_type=StageExecutionType.LLM_GENERATION,
                input_sources=(0,),
                final_output=True,
                final_output_type="audio",
                engine_output_type="audio",
                async_chunk_prewarm_payload_func="operator.neg",
            ),
        ),
    )
    deploy = DeployConfig(
        async_chunk=False,
        stages=[StageDeployConfig(stage_id=0, devices="0"), StageDeployConfig(stage_id=1, devices="0")],
    )
    return pipeline, deploy


def test_minicpmo_declares_prewarm_payload_on_code2wav_only() -> None:
    assert [(s.model_stage, s.async_chunk_prewarm_payload_func) for s in MINICPMO_4_5_PIPELINE.stages] == [
        ("llm", None),
        ("tts", None),
        ("code2wav", "vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni.code2wav_prewarm_payload"),
    ]


@pytest.mark.parametrize(
    ("make_inputs", "expected"),
    [
        pytest.param(_minicpmo_inputs, [None, None, code2wav_prewarm_payload], id="minicpmo_4_5"),
        pytest.param(_generic_inputs, [None, operator.neg], id="generic"),
    ],
)
def test_legacy_and_structured_metadata_resolve_prewarm_payload(make_inputs, expected) -> None:
    pipeline, deploy = make_inputs()

    legacy = [
        extract_legacy_stage_metadata(stage.to_omegaconf())
        for stage in merge_pipeline_deploy(pipeline, copy.deepcopy(deploy))
    ]
    omni_config = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=copy.deepcopy(deploy))
    structured = [
        extract_stage_metadata_from_omni_stage_config(omni_config.stage_by_id(stage.stage_id))
        for stage in pipeline.stages
    ]

    assert [m.async_chunk_prewarm_payload_func for m in legacy] == expected
    assert [m.async_chunk_prewarm_payload_func for m in structured] == expected


_STAGE2: dict[str, Any] = dict(
    stage_id=2,
    stage_type="llm",
    engine_output_type="audio",
    is_comprehension=False,
    requires_multimodal_data=False,
    engine_input_source=[1],
    final_output=True,
    final_output_type="audio",
    default_sampling_params=None,
    custom_process_input_func=None,
    model_stage="code2wav",
    runtime_cfg=None,
)


@pytest.mark.parametrize(
    ("client_class", "base_client_class"),
    [(StageEngineCoreClient, AsyncMPClient), (DPLBStageEngineCoreClient, DPLBAsyncMPClient)],
)
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        pytest.param(
            StageMetadata(**_STAGE2, async_chunk_prewarm_payload_func=code2wav_prewarm_payload),
            code2wav_prewarm_payload,
            id="declared",
        ),
        pytest.param(StageMetadata(**_STAGE2), None, id="default"),
        # Older metadata doubles lack the field; the client reads it with getattr.
        pytest.param(
            SimpleNamespace(**_STAGE2, prompt_transform_func=None, prompt_expand_func=None),
            None,
            id="field-missing",
        ),
    ],
)
def test_stage_engine_core_client_exposes_prewarm_payload_func(
    monkeypatch, client_class, base_client_class, metadata, expected
) -> None:
    def fake_base_init(self, config, *_args, **_kwargs):
        self.vllm_config = config
        self.resources = SimpleNamespace(engine_dead=False)

    monkeypatch.setattr(base_client_class, "__init__", fake_base_init)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(omni_kv_config=None, hf_config=None, stage_connector_config=None)
    )

    client = client_class(vllm_config, object, metadata=metadata)

    assert client.async_chunk_prewarm_payload_func is expected
