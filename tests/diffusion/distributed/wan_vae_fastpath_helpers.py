# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolate the upstream numerical reference from Wan's global norm patch."""

from __future__ import annotations

import importlib.util

import pytest
from diffusers.models.autoencoders import autoencoder_kl_wan
from torch import nn


@pytest.fixture(scope="session")
def original_wan_rms_norm() -> type[nn.Module]:
    # Wan's package import replaces the public class, including aliases held
    # by already collected tests. Load the installed upstream source under a
    # private name instead of reloading (and changing) the live module/classes.
    spec = importlib.util.spec_from_file_location(
        "diffusers.models.autoencoders._wan_fastpath_reference", autoencoder_kl_wan.__file__
    )
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    return reference.WanRMS_norm


@pytest.fixture
def unpatched_wan_rms_norm(monkeypatch, request, original_wan_rms_norm):
    # Import the patching package before temporarily restoring the reference,
    # so importing RMSNormVAE in a test cannot change it halfway through.
    from vllm_omni.diffusion.models.wan2_2.norm import RMSNormVAE

    assert original_wan_rms_norm is not RMSNormVAE
    monkeypatch.setattr(autoencoder_kl_wan, "WanRMS_norm", original_wan_rms_norm)
    if hasattr(request.module, "WanRMS_norm"):
        monkeypatch.setattr(request.module, "WanRMS_norm", original_wan_rms_norm)
