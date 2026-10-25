# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.entrypoints import omni_base

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_hy_nomas_defers_download_to_pipeline(mocker):
    model_id = "tencent/Hy-Nomas"
    download = mocker.patch.object(omni_base, "download_weights_from_hf_specific")
    probe = mocker.patch.object(omni_base, "file_or_path_exists")
    assert omni_base.omni_snapshot_download(model_id) == model_id
    download.assert_not_called()
    probe.assert_not_called()


def test_native_model_without_deferral_keeps_shared_download(monkeypatch, mocker):
    monkeypatch.delenv("VLLM_USE_MODELSCOPE", raising=False)
    model_id = "sand-ai/MAGI-2-preview"
    download = mocker.patch.object(omni_base, "download_weights_from_hf_specific")
    mocker.patch.object(omni_base, "file_or_path_exists", return_value=False)
    assert omni_base.omni_snapshot_download(model_id) == model_id
    download.assert_called_once_with(
        model_name_or_path=model_id, cache_dir=None, allow_patterns=["*"], require_all=True
    )
