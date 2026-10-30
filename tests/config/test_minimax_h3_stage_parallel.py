# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
import yaml

from vllm_omni.config.stage_config import load_deploy_config, validate_stage_parallel_layout
from vllm_omni.model_executor.models.minimax_h3.pipeline import (
    MINIMAX_H3_DECODE_PIPELINE,
    MINIMAX_H3_PARALLEL_DECODE_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _config():
    path = Path(__file__).parents[2] / "vllm_omni/deploy/minimax_h3_disaggregated_parallel_decode.yaml"
    return yaml.safe_load(path.read_text())


def _load(tmp_path, raw):
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_deploy_config(path)


@pytest.mark.parametrize("video,audio", [(4, 4), (7, 1), (2, 2), (3, 1), (1, 7)])
def test_parallel_partitions_allow_frontend_device_reuse(tmp_path, video, audio):
    raw = _config()
    config = raw["stage_parallel"]
    config.update(
        total_devices=video + audio, video_devices=list(range(video)), audio_devices=list(range(video, video + audio))
    )
    raw["stages"][2]["devices"] = ",".join(map(str, config["video_devices"]))
    raw["stages"][3]["devices"] = ",".join(map(str, config["audio_devices"]))
    for stage_id, count in ((2, video), (3, audio)):
        raw["stages"][stage_id]["parallel_config"].update(
            ulysses_degree=count,
            vae_patch_parallel_size=count if stage_id == 2 else 1,
        )
    deploy = _load(tmp_path, raw)
    validate_stage_parallel_layout(deploy)


def test_overlap_requires_explicit_opt_in(tmp_path):
    raw = _config()
    raw["stage_parallel"].update(audio_devices=[2], require_full_device_partition=False)
    raw["stages"][3]["devices"] = "2"
    deploy = _load(tmp_path, raw)
    with pytest.raises(ValueError, match="overlap"):
        validate_stage_parallel_layout(deploy)
    deploy.stage_parallel.allow_device_overlap = True
    validate_stage_parallel_layout(deploy)


def test_serial_mode_restores_legacy_pipeline(tmp_path):
    raw = _config()
    raw.pop("stage_parallel")
    deploy = _load(tmp_path, raw)
    assert deploy.pipeline == "minimax_h3_disaggregated_decode"
    assert [s.stage_id for s in deploy.stages] == [0, 1, 2]
    assert not any(s.fanout_targets for s in MINIMAX_H3_DECODE_PIPELINE.stages)
    assert MINIMAX_H3_DECODE_PIPELINE.stages[2].inline_diffusion
    assert all(not s.inline_diffusion for s in MINIMAX_H3_PARALLEL_DECODE_PIPELINE.stages[2:])


def test_layout_revalidated_after_device_override(tmp_path):
    deploy = _load(tmp_path, _config())
    validate_stage_parallel_layout(deploy)
    deploy.stages[3].devices = "0"
    with pytest.raises(ValueError, match="disagree"):
        validate_stage_parallel_layout(deploy)


def test_parallel_dimensions_match_device_group(tmp_path):
    raw = _config()
    raw["stages"][2]["parallel_config"]["ulysses_degree"] = 2
    with pytest.raises(ValueError, match="world_size"):
        validate_stage_parallel_layout(_load(tmp_path, raw))


def test_audio_patch_parallel_is_not_claimed(tmp_path):
    raw = _config()
    raw["stages"][3]["parallel_config"]["vae_patch_parallel_size"] = 2
    with pytest.raises(ValueError, match="vae_patch_parallel_size"):
        validate_stage_parallel_layout(_load(tmp_path, raw))


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "20"])
def test_parallel_decode_timeout_requires_finite_positive_number(value):
    from vllm_omni.config.stage_config import StageParallelConfig

    with pytest.raises(ValueError, match="decode_timeout_s"):
        StageParallelConfig(decode_timeout_s=value)


def test_timeout_projects_only_into_parallel_runtime(tmp_path):
    from vllm_omni.config.stage_config import merge_pipeline_deploy

    raw = _config()
    raw["stage_parallel"]["decode_timeout_s"] = 45
    parallel = merge_pipeline_deploy(MINIMAX_H3_PARALLEL_DECODE_PIPELINE, _load(tmp_path, raw))
    assert all(stage.to_omegaconf().runtime.parallel_decode_timeout_s == 45 for stage in parallel)
    raw.pop("stage_parallel")
    serial = merge_pipeline_deploy(MINIMAX_H3_DECODE_PIPELINE, _load(tmp_path, raw))
    assert all("parallel_decode_timeout_s" not in stage.to_omegaconf().runtime for stage in serial)


def test_parallel_timeout_stays_out_of_backend_engine_args(tmp_path):
    from vllm_omni.config.omni_config import VllmOmniConfig
    from vllm_omni.engine.stage_init_utils import (
        _project_omni_stage_engine_args,
        extract_stage_metadata_from_omni_stage_config,
    )

    raw = _config()
    raw["stage_parallel"]["decode_timeout_s"] = 45
    config = VllmOmniConfig.from_pipeline_config(
        MINIMAX_H3_PARALLEL_DECODE_PIPELINE, user_deploy_config=_load(tmp_path, raw)
    )
    for stage_id in range(4):
        stage = config.stage_by_id(stage_id)
        assert stage.runtime_config.parallel_decode_timeout_s == 45
        assert extract_stage_metadata_from_omni_stage_config(stage).runtime_cfg.parallel_decode_timeout_s == 45
        assert "parallel_decode_timeout_s" not in _project_omni_stage_engine_args(stage)


@pytest.mark.parametrize("layout", ["2plus2", "3plus1", "4plus4", "7plus1"])
def test_delivered_partition_examples_validate(layout):
    path = Path(__file__).parents[2] / f"vllm_omni/deploy/minimax_h3_disaggregated_parallel_decode_{layout}.yaml"
    validate_stage_parallel_layout(load_deploy_config(path))


@pytest.mark.parametrize("dlo", [False, True])
def test_decoder_dlo_switch_is_independent_of_dit(tmp_path, dlo):
    from vllm_omni.config.stage_config import merge_pipeline_deploy

    raw = _config()
    for stage in raw["stages"][2:]:
        assert stage["enable_distributed_layerwise_offload"]
        stage["enable_distributed_layerwise_offload"] = dlo
    merged = merge_pipeline_deploy(MINIMAX_H3_PARALLEL_DECODE_PIPELINE, _load(tmp_path, raw))
    args = [stage.to_omegaconf().engine_args for stage in merged]
    assert args[1].enable_distributed_layerwise_offload
    assert args[2].enable_distributed_layerwise_offload is dlo
    assert args[3].enable_distributed_layerwise_offload is dlo
