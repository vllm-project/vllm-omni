# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path

import pytest

from vllm_omni.diffusion.models.hyperflow_h3_flash.checkpoint import (
    HYPERFLOW_H3_FLASH_MODEL_ID,
    HyperFlowH3FlashLayout,
    looks_like_hyperflow_h3_flash,
    resolve_hyperflow_partition,
    resolve_hyperflow_subfolder,
    write_omni_model_index,
)
from vllm_omni.diffusion.utils.hf_utils import resolve_native_diffusion_model_class

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_hub_id_resolves_to_minimax_pipeline():
    assert looks_like_hyperflow_h3_flash(HYPERFLOW_H3_FLASH_MODEL_ID)
    assert resolve_native_diffusion_model_class(HYPERFLOW_H3_FLASH_MODEL_ID) == "MiniMaxH3Pipeline"
    assert not looks_like_hyperflow_h3_flash("MiniMaxAI/MiniMax-H3")


def test_local_partition_detection(tmp_path: Path):
    (tmp_path / "fl2va_int8_convrot").mkdir()
    (tmp_path / "fl2va_int8_convrot" / "config.json").write_text("{}", encoding="utf-8")
    assert looks_like_hyperflow_h3_flash(str(tmp_path))


def test_partition_and_subfolder_mapping():
    assert resolve_hyperflow_partition("t2va") == "fl2va"
    assert resolve_hyperflow_partition("ref2va") == "ref2va"
    assert resolve_hyperflow_subfolder("fl2va") == "fl2va_int8_convrot"


def test_write_omni_model_index_reuses_minimax_class(tmp_path: Path):
    layout = HyperFlowH3FlashLayout(
        weights_root=tmp_path,
        transformer_subfolder="fl2va_int8_convrot",
        base_partition_root=tmp_path / "FL2VA",
        partition="fl2va",
        transformer_config={"dropped_blocks": [6, 21, 22], "hyperflow_gate": 0.25},
    )
    path = write_omni_model_index(layout, tmp_path / "staged" / "FL2VA")
    index = json.loads(path.read_text(encoding="utf-8"))
    assert index["_class_name"] == "MiniMaxH3Pipeline"
    assert set(index["_minimax_h3"]["tasks"]) == {"t2va", "fl2va"}
