# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from pathlib import Path
from shutil import rmtree
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.model_tests.diffusion import anima_builder, test_common_offline, test_common_online
from tests.model_tests.diffusion.config_types import DiffusionTasks

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize("model_name", ["AnimaPipeline", "FluxPipeline"])
def test_common_runners_resolve_native_checkpoint(model_name, tmp_path, monkeypatch):
    """Both entry points receive the file and class; directory models keep their existing arguments."""
    paths = {model_name: str(tmp_path)}
    subtests = SimpleNamespace(test=lambda **kwargs: nullcontext())
    build_omni = MagicMock()
    monkeypatch.setattr(test_common_offline, "build_omni_from_diff_accelerations", build_omni)
    monkeypatch.setattr(test_common_offline, "run_and_validate_text_to_image_request", MagicMock())
    test_common_offline.test_pipeline_on_supported_tasks(
        model_name, None, [DiffusionTasks.TEXT_TO_IMAGE], False, False, False, paths, subtests
    )
    server = MagicMock()
    monkeypatch.setattr(test_common_online, "OmniServer", server)
    monkeypatch.setattr(test_common_online, "OnlineOmniClient", MagicMock())
    monkeypatch.setattr(test_common_online, "run_and_validate_online_text_to_image_request", MagicMock())
    test_common_online.test_online_on_supported_tasks(
        model_name, None, [DiffusionTasks.TEXT_TO_IMAGE], False, False, paths, "core_model", subtests
    )
    if model_name == "AnimaPipeline":
        checkpoint = str(tmp_path / anima_builder.CHECKPOINT_FILENAME)
        build_omni.assert_called_once_with(
            accelerations=None, model=checkpoint, enforce_eager=True, model_class_name=model_name
        )
        server.assert_called_once_with(checkpoint, ["--enforce-eager", "--model-class-name", model_name])
    else:
        build_omni.assert_called_once_with(accelerations=None, model=str(tmp_path), enforce_eager=True)
        server.assert_called_once_with(str(tmp_path), ["--enforce-eager"])


def test_real_anima_assets_leave_cache_intact(tmp_path, monkeypatch):
    """The fixture may delete the assembled directory without deleting cached model assets."""
    components = tmp_path / "components"
    for name in ("text_encoder", "vae", "tokenizer", "t5_tokenizer"):
        (components / name).mkdir(parents=True)
        (components / name / "asset").write_text(name)
    checkpoint = tmp_path / "real.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(anima_builder, "snapshot_download", lambda *args, **kwargs: str(components))
    monkeypatch.setattr(anima_builder, "hf_hub_download", lambda *args, **kwargs: str(checkpoint))
    directory = Path(anima_builder.real_anima_model())
    try:
        assert (directory / anima_builder.CHECKPOINT_FILENAME).read_bytes() == b"checkpoint"
        assert (directory / "vae" / "asset").read_text() == "vae"
        # Missing scheduler is supported by Anima's default scheduler fallback.
        assert not (directory / "scheduler").exists()
    finally:
        rmtree(directory)
    assert checkpoint.read_bytes() == b"checkpoint"
    assert (components / "vae" / "asset").read_text() == "vae"
