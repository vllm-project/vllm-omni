# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run-level weight selection works for inherited deploy profiles."""

import pytest

from tests.helpers.stage_config import stage_config_path_for_run_level
from vllm_omni.config.stage_config import resolve_deploy_yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("run_level", ["core_model", "advanced_model", "full_model"])
def test_run_level_selects_weights_for_all_inherited_stages(tmp_path, run_level):
    base = tmp_path / "base.yaml"
    base.write_text(
        "stages:\n"
        "  - stage_id: 0\n"
        "    load_format: dummy\n"
        "  - stage_id: 1\n"
        "    load_format: dummy\n"
        "    enforce_eager: false\n"
        "    compilation_config:\n"
        "      cudagraph_capture_sizes: [1, 2, 4]\n"
    )
    overlay = tmp_path / "profile.yaml"
    overlay.write_text("base_config: base.yaml\nplatforms:\n  npu:\n    model_runner: v2\n")
    generated = stage_config_path_for_run_level(str(overlay), run_level)
    assert generated is not None
    resolved = resolve_deploy_yaml(generated)
    assert [stage["stage_id"] for stage in resolved["stages"]] == [0, 1]
    assert resolved["platforms"]["npu"]["model_runner"] == "v2"
    assert resolved["stages"][1]["enforce_eager"] is False
    assert resolved["stages"][1]["compilation_config"]["cudagraph_capture_sizes"] == [1, 2, 4]
    if run_level == "core_model":
        assert all(stage["load_format"] == "dummy" for stage in resolved["stages"])
    else:
        assert all("load_format" not in stage for stage in resolved["stages"])
