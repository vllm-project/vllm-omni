# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
import yaml

from tests.helpers.stage_config import modify_stage_config, stage_config_path_for_run_level
from vllm_omni.config.stage_config import resolve_deploy_yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def inherited_deploy(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "stages": [
                    {"stage_id": 0, "max_num_seqs": 4, "load_format": "dummy"},
                    {"stage_id": 1, "max_num_seqs": 2, "load_format": "dummy"},
                ],
                "duplex_session": {"max_sessions": 4},
            }
        )
    )
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump({"base_config": "base.yaml", "stages": [{"stage_id": 0}]}))
    return overlay


def test_modified_deploy_preserves_inheritance_in_temporary_directory(inherited_deploy, tmp_path, monkeypatch):
    from tests.helpers import stage_config

    output_dir = tmp_path / "generated"
    output_dir.mkdir()
    monkeypatch.setattr(stage_config.tempfile, "tempdir", str(output_dir))
    modified = modify_stage_config(str(inherited_deploy), updates={"stages": {1: {"max_num_seqs": 3}}})
    assert Path(modified).parent == output_dir
    resolved = resolve_deploy_yaml(modified)
    assert resolved["duplex_session"] == {"max_sessions": 4}
    assert resolved["stages"] == [
        {"stage_id": 0, "max_num_seqs": 4, "load_format": "dummy"},
        {"stage_id": 1, "max_num_seqs": 3, "load_format": "dummy"},
    ]


def test_core_model_loads_all_inherited_stages_as_dummy(inherited_deploy):
    base = inherited_deploy.parent / "base.yaml"
    config = yaml.safe_load(base.read_text())
    for stage in config["stages"]:
        stage["load_format"] = "auto"
    base.write_text(yaml.safe_dump(config))
    modified = stage_config_path_for_run_level(str(inherited_deploy), "core_model")
    resolved = resolve_deploy_yaml(modified)
    assert len(resolved["stages"]) == 2
    assert all(stage["load_format"] == "dummy" for stage in resolved["stages"])


@pytest.mark.parametrize("run_level", ["advanced_model", "full_model"])
def test_real_weight_levels_remove_inherited_dummy_loading(inherited_deploy, run_level):
    modified = stage_config_path_for_run_level(str(inherited_deploy), run_level)
    resolved = resolve_deploy_yaml(modified)
    assert len(resolved["stages"]) == 2
    assert all("load_format" not in stage for stage in resolved["stages"])
    assert [stage["max_num_seqs"] for stage in resolved["stages"]] == [4, 2]
