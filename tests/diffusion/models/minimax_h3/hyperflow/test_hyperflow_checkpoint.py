# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.minimax_h3.hyperflow.checkpoint import (
    HYPERFLOW_H3_FLASH_MODEL_ID,
    HyperFlowH3FlashLayout,
    looks_like_hyperflow_h3_flash,
    resolve_hyperflow_partition,
    resolve_hyperflow_subfolder,
    write_omni_model_index,
)
from vllm_omni.diffusion.models.minimax_h3.hyperflow.prepare import prepare_hyperflow_on_od_config
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


@pytest.mark.parametrize(("partition", "native"), [("fl2va", "FL2VA"), ("ref2va", "Ref2VA")])
def test_prepare_reuses_staged_layout(tmp_path, monkeypatch, partition, native):
    source = tmp_path / "source"
    transformer = source / resolve_hyperflow_subfolder(partition)
    transformer.mkdir(parents=True)
    config = {"time_embed_dim": 16, "dropped_blocks": [9], "hyperflow_gate": 0.5}
    (transformer / "config.json").write_text(json.dumps(config), encoding="utf-8")
    base = tmp_path / "base"
    (base / native / "video_vae").mkdir(parents=True)
    od_config = SimpleNamespace(model=str(source), task_type=partition, extras={"base_model": str(base)})
    staged = tmp_path / "staged"
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.hyperflow.prepare.tempfile.mkdtemp",
        lambda **kwargs: str(staged),
    )

    first = prepare_hyperflow_on_od_config(od_config)
    od_config.tf_model_config = None
    retry = prepare_hyperflow_on_od_config(od_config)

    assert first is not None and retry is not None
    assert Path(od_config.model) == staged
    assert retry.partition == first.partition == partition
    assert retry.transformer_root.resolve() == first.transformer_root
    assert (retry.base_partition_root / "video_vae").resolve() == base / native / "video_vae"
    assert retry.transformer_config == config
    assert od_config.tf_model_config.to_dict()["time_embed_dim"] == 16
    assert od_config.extras["hyperflow_dropped_blocks"] == [9]


def test_prepare_rejects_teacache_before_resolving_weights(monkeypatch):
    def fail_to_resolve(*args, **kwargs):
        pytest.fail("TeaCache must be rejected before resolving weights")

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.hyperflow.prepare.resolve_hyperflow_layout", fail_to_resolve
    )
    od_config = SimpleNamespace(model=HYPERFLOW_H3_FLASH_MODEL_ID, task_type="fl2va", cache_backend="tea_cache")
    with pytest.raises(ValueError, match="HyperFlow H3 Flash does not support --cache-backend tea_cache"):
        prepare_hyperflow_on_od_config(od_config)

    od_config.model = "MiniMaxAI/MiniMax-H3"
    assert prepare_hyperflow_on_od_config(od_config) is None
