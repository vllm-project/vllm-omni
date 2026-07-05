# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for version compatibility warnings."""

import warnings
from unittest import mock

import pytest

from vllm_omni.version import warn_if_misaligned_vllm_version


@mock.patch("vllm_omni.version.__version_tuple__", (0, 19, 0))
@mock.patch("vllm_omni.version.__version__", "0.19.0")
@mock.patch("vllm.__version_tuple__", (0, 18, 0))
@mock.patch("vllm.__version__", "0.18.0")
def test_version_mismatch_warning():
    """Ensure that we warn when vLLM and vLLM-Omni major/minor versions differ."""
    with pytest.warns(RuntimeWarning, match="mismatched major/minor versions"):
        warn_if_misaligned_vllm_version()


@pytest.mark.parametrize(
    "vllm_ver,vllm_tuple,omni_ver,omni_tuple",
    [
        ("0.19.0", (0, 19, 0), "0.19.5", (0, 19, 5)),  # Patch differs
        ("0.18.0", (0, 18, 0), "dev", (0, 0, "dev")),  # Omni dev
        ("dev", (0, 0, "dev"), "0.19.0", (0, 19, 0)),  # vLLM dev
        # Ensure local identifies don't matter for the warning
        ("0.19.0+foo", (0, 19, 0, "foo"), "0.19.5", (0, 19, 0)),
        ("0.19.0", (0, 19, 0), "0.19.5+bar", (0, 19, 0, "bar")),
        ("0.19.0+foo", (0, 19, 0, "foo"), "0.19.5+bar", (0, 19, 0, "bar")),
    ],
)
def test_no_warning_cases(vllm_ver, vllm_tuple, omni_ver, omni_tuple):
    """Ensure we don't warn when minor versions match or either is a dev build."""
    with (
        mock.patch.multiple("vllm", __version__=vllm_ver, __version_tuple__=vllm_tuple),
        mock.patch.multiple("vllm_omni.version", __version__=omni_ver, __version_tuple__=omni_tuple),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_if_misaligned_vllm_version()


@mock.patch("vllm_omni.version.__version_tuple__", (0, 19, 0))
@mock.patch("vllm_omni.version.__version__", "0.19.0rc2.dev21")
@mock.patch("vllm.__version_tuple__", (0, 18, 0))
@mock.patch("vllm.__version__", "0.18.0")
def test_warning_contains_version_strings():
    """Ensure that the warning contains the full version strings."""
    with pytest.warns(RuntimeWarning) as record:
        warn_if_misaligned_vllm_version()

    assert len(record) == 1
    msg = str(record[0].message)
    assert "0.19.0rc2.dev21" in msg
    assert "0.18.0" in msg
