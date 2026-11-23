# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint-owned Ming TN resources, without network or model weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
    MingFlashOmniTalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.ming_flash_omni.native_text_frontend import normalize_native_fragment
from vllm_omni.model_executor.models.ming_flash_omni.talker_tn.talker_tn import TalkerTN
from vllm_omni.model_executor.models.ming_flash_omni.text_processing import segment_and_normalize
from vllm_omni.transformers_utils.configs.ming_talker import MingFlashOmniTalkerConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_upstream_tn_reports_incomplete_model_assets(tmp_path):
    (tmp_path / "en_tn_tagger.fst").touch()
    with pytest.raises(FileNotFoundError, match="zh_tn_tagger.fst") as error:
        TalkerTN(tmp_path)
    assert str(tmp_path) in str(error.value)
    assert "en_tn_verbalizer.fst" in str(error.value)


def test_upstream_tn_requires_checkpoint_re_rules(tmp_path):
    for lang in ("zh", "en"):
        for kind in ("tagger", "verbalizer"):
            (tmp_path / f"{lang}_tn_{kind}.fst").touch()
    with pytest.raises(FileNotFoundError, match=r"talker_re\.json"):
        TalkerTN(tmp_path)


def test_talker_validates_upstream_tn_before_constructing_models(tmp_path, mocker):
    qwen = mocker.patch("vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker.Qwen2Model")
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=MingFlashOmniTalkerConfig(normalize_text=True, tn_mode="upstream"), model=str(tmp_path)
        )
    )
    with pytest.raises(FileNotFoundError, match="talker_tn"):
        MingFlashOmniTalkerForConditionalGeneration(vllm_config=config)
    qwen.assert_not_called()


def test_normalizer_is_cached_per_talker_not_globally(tmp_path, mocker):
    mocker.patch("vllm_omni.model_executor.models.ming_flash_omni.text_processing._get_inflect", return_value=object())
    load = mocker.patch(
        "vllm_omni.model_executor.models.ming_flash_omni.talker_tn.talker_tn.TalkerTN",
        side_effect=lambda resource_dir: SimpleNamespace(resource_dir=resource_dir),
    )
    models = []
    for name in ("first", "second"):
        model = object.__new__(MingFlashOmniTalkerForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.talker_dir = str(tmp_path / name)
        models.append(model)
    first, second = models
    assert first._upstream_tn is first._upstream_tn
    assert second._upstream_tn is second._upstream_tn
    assert first._upstream_tn is not second._upstream_tn
    assert first._upstream_tn.resource_dir == tmp_path / "first/talker_tn"
    assert second._upstream_tn.resource_dir == tmp_path / "second/talker_tn"
    assert load.call_count == 2


def test_upstream_talker_requires_english_preprocessing_dependency(tmp_path, mocker):
    mocker.patch("vllm_omni.model_executor.models.ming_flash_omni.talker_tn.talker_tn.TalkerTN")
    mocker.patch("vllm_omni.model_executor.models.ming_flash_omni.text_processing._get_inflect", return_value=None)
    model = object.__new__(MingFlashOmniTalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.talker_dir = str(tmp_path)
    with pytest.raises(ModuleNotFoundError, match="missing inflect"):
        _ = model._upstream_tn


def test_checkpoint_graphs_are_read_by_native_code(tmp_path: Path):
    pynini = pytest.importorskip("pynini")
    pytest.importorskip("inflect")
    # Tiny real FSTs keep path/format coverage independent of network and weights.
    for lang, source, spoken in (("zh", "甲", "乙"), ("en", "word", "spoken")):
        tagged = f'name {{ value: "{spoken}" }}'
        pynini.cross(source, tagged).write(str(tmp_path / f"{lang}_tn_tagger.fst"))
        pynini.cross(tagged, spoken).write(str(tmp_path / f"{lang}_tn_verbalizer.fst"))
    (tmp_path / "__init__.py").write_text("raise RuntimeError('Remote model code must not execute')")
    (tmp_path / "talker_re.json").write_text(
        json.dumps(
            {
                "money": {},
                "measure_before": {},
                "measure_after": {},
                "serial_before": {},
                "serial_after": {},
                "char_map0": {},
                "proper_noun": {"原文": "甲"},
            }
        )
    )
    tn = TalkerTN(tmp_path)
    assert tn("原文") == "乙"
    assert tn("甲") == "乙"
    assert tn("word") == "spoken"
    assert segment_and_normalize("word", tn_mode="upstream", normalizer=tn) == ["spoken"]
    assert normalize_native_fragment("甲", tn_mode="upstream", normalizer=tn) == "乙"


def test_missing_pynini_has_install_guidance(tmp_path, monkeypatch):
    import builtins

    for lang in ("zh", "en"):
        for kind in ("tagger", "verbalizer"):
            (tmp_path / f"{lang}_tn_{kind}.fst").touch()
    (tmp_path / "talker_re.json").write_text("{}")
    original_import = builtins.__import__

    def without_pynini(name, *args, **kwargs):
        if name == "normalizer":
            raise ModuleNotFoundError("No module named 'pynini'", name="pynini")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pynini)
    with pytest.raises(ModuleNotFoundError, match=r"vllm-omni\[ming-tn\]"):
        TalkerTN(tmp_path)
