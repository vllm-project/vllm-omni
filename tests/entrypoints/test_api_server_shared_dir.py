# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The multi-API server scratch directory and the MOSS reference-code default."""

import os
import tempfile

import pytest

from vllm_omni.entrypoints.api_server_shared_dir import create_api_server_shared_dir, remove_api_server_shared_dir

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_directory_has_a_short_path_and_is_removed(monkeypatch, tmp_path):
    from vllm_omni.model_executor.models.moss_tts.shared_reference_encoder import socket_path_fits

    long_tmp = tmp_path / ("t" * 120)
    long_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(long_tmp))
    path = create_api_server_shared_dir()
    try:
        assert os.path.isdir(path) and not path.startswith(str(long_tmp))
        assert socket_path_fits(os.path.join(path, "moss-ref-codes"))
        with open(os.path.join(path, "entry"), "w") as handle:
            handle.write("x")
    finally:
        remove_api_server_shared_dir(path)
    assert not os.path.exists(path)


def test_moss_reference_codes_follow_the_server_directory(monkeypatch, tmp_path):
    from vllm_omni.model_executor.models.moss_tts.reference_encoder import shared_codes_dir

    monkeypatch.delenv("VLLM_OMNI_MOSS_REF_CODES_SHARED_DIR", raising=False)
    assert shared_codes_dir(None) is None  # one API process: nothing to share
    assert shared_codes_dir(str(tmp_path)) == os.path.join(str(tmp_path), "moss-ref-codes")

    monkeypatch.setenv("VLLM_OMNI_MOSS_REF_CODES_SHARED_DIR", "/dev/shm/explicit")
    assert shared_codes_dir(str(tmp_path)) == "/dev/shm/explicit"

    monkeypatch.setenv("VLLM_OMNI_MOSS_REF_CODES_SHARED_DIR", "")
    assert shared_codes_dir(str(tmp_path)) is None
