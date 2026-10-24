# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.entrypoints import omni_base

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_hy_nomas_defers_download_to_pipeline(mocker):
    download = mocker.patch.object(omni_base, "download_weights_from_hf_specific")
    probe = mocker.patch.object(omni_base, "file_or_path_exists")
    assert omni_base.omni_snapshot_download("tencent/Hy-Nomas") == "tencent/Hy-Nomas"
    download.assert_not_called()
    probe.assert_not_called()
