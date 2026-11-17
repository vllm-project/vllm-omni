# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming talker checkpoint resolution without preset voice data."""

import json
from pathlib import Path

import pytest
from vllm import envs

from vllm_omni.model_executor.models.ming_flash_omni.checkpoint import (
    prepare_talker_checkpoint,
    resolve_ming_talker_checkpoint,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("with_tn", [False, True])
@pytest.mark.parametrize("use_resolver", [False, True], ids=["explicit-cache", "stage-resolver"])
def test_prepare_talker_checkpoint_does_not_require_or_link_data(tmp_path: Path, with_tn, use_resolver, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path / "runtime-cache"))
    root = tmp_path / "deployment"
    talker = root / "talker"
    (talker / "llm").mkdir(parents=True)
    (talker / "vae").mkdir()
    (root / "data").mkdir()
    (talker / "config.json").write_text(json.dumps({"architectures": ["BailingTalker2"]}))
    (talker / "llm/config.json").write_text("{}")
    (talker / "vae/config.json").write_text("{}")
    for name in ("model.safetensors", "vae/model.safetensors", "campplus.onnx"):
        (talker / name).touch()
    (root / "data/voice_name.json").write_text("{}")
    if with_tn:
        (talker / "talker_tn").mkdir()
        (talker / "talker_tn/talker_re.json").write_text("{}")
        for lang in ("zh", "en"):
            for kind in ("tagger", "verbalizer"):
                (talker / "talker_tn" / f"{lang}_tn_{kind}.fst").touch()

    if use_resolver:
        from vllm_omni.engine.stage_init_utils import _resolve_model_path

        cache = Path(envs.VLLM_CACHE_ROOT) / "ming_talker"
        view = Path(
            _resolve_model_path(
                str(root),
                {
                    "model_path_resolver": (
                        "vllm_omni.model_executor.models.ming_flash_omni.checkpoint.resolve_ming_talker_checkpoint"
                    )
                },
            )
        )
    else:
        cache = tmp_path / "cache"
        view = Path(prepare_talker_checkpoint(str(root), cache_dir=str(cache)))

    assert view.parent == cache
    assert not (view / "data").exists()
    assert (view / "model.safetensors").resolve() == talker / "model.safetensors"
    if with_tn:
        assert (view / "talker_tn").resolve() == talker / "talker_tn"
        assert len(list((view / "talker_tn").glob("*.fst"))) == 4
        assert (view / "talker_tn/talker_re.json").read_text() == "{}"

    (root / "data/voice_name.json").unlink()
    assert prepare_talker_checkpoint(str(root), cache_dir=str(cache)) == str(view)
    assert prepare_talker_checkpoint(str(view), cache_dir=str(cache)) == str(view)


def test_resolve_talker_checkpoint_preserves_hub_revision(mocker, monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "VLLM_CACHE_ROOT", str(tmp_path / "runtime-cache"))
    download = mocker.patch(
        "vllm_omni.model_executor.model_loader.weight_utils.download_weights_from_hf_specific",
        return_value="/cached/snapshot",
    )
    prepare = mocker.patch(
        "vllm_omni.model_executor.models.ming_flash_omni.checkpoint.prepare_talker_checkpoint",
        return_value="/normalized/talker",
    )
    assert resolve_ming_talker_checkpoint("example/Ming3", "revision-sha", None) == "/normalized/talker"
    download.assert_called_once()
    assert download.call_args.kwargs["model_name_or_path"] == "example/Ming3"
    assert download.call_args.kwargs["revision"] == "revision-sha"
    assert download.call_args.kwargs["allow_patterns"] == ["config.json", "talker/**"]
    assert download.call_args.kwargs["cache_dir"] is None  # Preserve the standard HF weight cache.
    prepare.assert_called_once_with("/cached/snapshot", cache_dir=str(tmp_path / "runtime-cache/ming_talker"))


def test_resolve_talker_checkpoint_rejects_unsupported_uri():
    with pytest.raises(ValueError, match="local directory or a Hugging Face Hub model ID"):
        resolve_ming_talker_checkpoint("https://example.com/model", None, None)
