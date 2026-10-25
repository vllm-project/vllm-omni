# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import REQUIRED_FILES
from vllm_omni.entrypoints import omni_base

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_local_hy_nomas_checkpoint_skips_shared_download(tmp_path, mocker):
    for relative in REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    download = mocker.patch.object(omni_base, "download_weights_from_hf_specific")
    probe = mocker.patch.object(omni_base, "file_or_path_exists")
    assert omni_base.omni_snapshot_download(str(tmp_path)) == str(tmp_path)
    download.assert_not_called()
    probe.assert_not_called()


def test_hy_nomas_hub_id_keeps_standard_snapshot_download(monkeypatch, mocker):
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    model_id = "tencent/Hy-Nomas"
    download = mocker.patch.object(omni_base, "download_weights_from_hf_specific")
    mocker.patch.object(omni_base, "file_or_path_exists", return_value=False)
    assert omni_base.omni_snapshot_download(model_id) == model_id
    download.assert_called_once_with(
        model_name_or_path=model_id, cache_dir=None, allow_patterns=["*"], require_all=True
    )
