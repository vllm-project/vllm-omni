# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Production Qwen3-Omni deploy literals, resolved without loading weights.

The Merge offline test used to boot ``qwen3_omni_moe.yaml`` on the 30B
Instruct checkpoint just to read these values back from the live engine.
That was a separate cold start from the CI Instruct runner shared with
``test_video_to_audio``. The literals stay here; the e2e test checks that
whatever the CI deploy resolved still reaches runtime.
"""

import copy

import pytest
from transformers import Qwen3OmniMoeConfig

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.omni_config import (
    VllmOmniARStageConfig,
    VllmOmniConfig,
    VllmOmniGenerationStageConfig,
)
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Mirrors vllm_omni/deploy/qwen3_omni_moe.yaml. Pipeline constraints
# (``detokenize``, ``stop_token_ids``) may add keys; these must remain.
_EXPECTED_SAMPLING = (
    {"temperature": 0.0, "max_tokens": 2048},
    {"temperature": 0.9, "top_k": 50, "max_tokens": 4096, "repetition_penalty": 1.05},
    {
        "temperature": 0.0,
        "top_p": 1.0,
        # Deploy YAML literal. SamplingParams later rewrites greedy top_k -1 to 0;
        # the live-engine normalization check stays on the CI runner e2e.
        "top_k": -1,
        "max_tokens": 65536,
        "repetition_penalty": 1.1,
    },
)


def _production_stages(monkeypatch: pytest.MonkeyPatch) -> tuple:
    """Resolve the packaged production deploy as CUDA would, with no checkpoint."""
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    pipeline = resolve_pipeline_config("qwen3_omni_moe", Qwen3OmniMoeConfig(enable_audio_output=True))
    assert pipeline is not None
    deploy = load_deploy_config(get_deploy_config_path("qwen3_omni_moe.yaml"))
    config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=copy.deepcopy(deploy),
    )
    return config.stage_configs


def test_production_yaml_literals_reach_resolved_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """``max_num_seqs``, utilization, and ``enforce_eager`` survive resolution."""
    resolved_stages = _production_stages(monkeypatch)
    assert len(resolved_stages) == 3
    thinker, talker, code2wav = resolved_stages
    assert isinstance(thinker, VllmOmniARStageConfig)
    assert isinstance(talker, VllmOmniARStageConfig)
    assert isinstance(code2wav, VllmOmniGenerationStageConfig)
    assert [stage.stage_id for stage in resolved_stages] == [0, 1, 2]
    assert [stage.model_stage for stage in resolved_stages] == ["thinker", "talker", "code2wav"]
    assert code2wav.final_output_type == "audio"
    assert thinker.runtime_config.devices == "0"
    assert talker.runtime_config.devices == code2wav.runtime_config.devices == "1"
    assert talker.connector_config.input_connectors == {"from_stage_0": "connector_of_shared_memory"}
    assert code2wav.connector_config.input_connectors == {"from_stage_1": "connector_of_shared_memory"}

    assert thinker.scheduler_config.max_num_seqs == 64
    assert talker.scheduler_config.max_num_seqs == 64
    assert code2wav.scheduler_config.max_num_seqs == 64
    assert thinker.scheduler_config.max_num_batched_tokens == 32768
    assert talker.scheduler_config.max_num_batched_tokens == 32768
    assert code2wav.scheduler_config.max_num_batched_tokens == 65536
    assert thinker.cache_config.gpu_memory_utilization == 0.9
    assert talker.cache_config.gpu_memory_utilization == 0.6
    assert code2wav.cache_config.gpu_memory_utilization == 0.1
    assert thinker.cache_config.enable_prefix_caching is True
    assert talker.cache_config.enable_prefix_caching is False
    assert code2wav.cache_config.enable_prefix_caching is False
    assert thinker.model_config.trust_remote_code is True
    assert talker.model_config.trust_remote_code is True
    assert code2wav.model_config.trust_remote_code is True
    assert code2wav.scheduler_config.enable_chunked_prefill is False
    assert code2wav.scheduler_config.async_scheduling is False
    # CUDA production runs thinker and talker with cudagraphs (eager left unset,
    # so the model-config default is false). Code2Wav spells enforce_eager: false.
    assert thinker.model_config.enforce_eager is False
    assert talker.model_config.enforce_eager is False
    assert code2wav.model_config.enforce_eager is False

    assert len(resolved_stages) == len(_EXPECTED_SAMPLING)
    for stage, expected in zip(resolved_stages, _EXPECTED_SAMPLING, strict=True):
        assert expected.items() <= (stage.model_config.default_sampling_params or {}).items()
