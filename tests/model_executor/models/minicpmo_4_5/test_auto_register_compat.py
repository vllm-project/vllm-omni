# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for the transformers 5.x auto-registration compatibility shim in
vllm_omni.patch.

Remote model files written against the pre-5.3 signature
``AutoImageProcessor.register(str_name, cls)`` crash on transformers >= 5.3
with ``AttributeError: 'str' object has no attribute '__module__'``. vLLM
imports remote model modules during model-architecture inspection (in a
subprocess), so this crash breaks ``vllm serve`` startup. The shim in
vllm_omni.patch forwards every call unchanged and drops a legacy str
registration only when the installed transformers rejects it; new-style
config-class registrations pass through untouched.
"""

import pytest
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoProcessor, AutoTokenizer
from transformers.image_processing_utils import BaseImageProcessor

from vllm_omni import patch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _apply_compat_shim():
    patch._patch_auto_register_compat()


class _StubImageProcessor(BaseImageProcessor):
    """Stand-in for a remote image processor class."""


def _register_is_wrapped(auto_cls):
    return getattr(auto_cls.__dict__["register"].__func__, "_omni_str_register_compat", False)


@pytest.mark.parametrize(
    "auto_cls",
    [AutoProcessor, AutoImageProcessor, AutoFeatureExtractor, AutoTokenizer],
)
def test_legacy_str_registration_is_noop(auto_cls):
    """Old-style (str, cls) registration must not raise on transformers >= 5.3."""
    assert _register_is_wrapped(auto_cls)
    # Old-style registration by name: either accepted by the installed
    # transformers (older releases) or dropped by the shim — never raised.
    auto_cls.register("SomeLegacyProcessorName", _StubImageProcessor)


@pytest.mark.parametrize(
    "auto_cls",
    [AutoProcessor, AutoImageProcessor, AutoFeatureExtractor, AutoTokenizer],
)
def test_new_style_config_class_registration_forwarded(auto_cls):
    """New-style (config_class, cls) registration must reach the original register."""
    from transformers import PretrainedConfig

    assert _register_is_wrapped(auto_cls)

    class _StubConfig(PretrainedConfig):
        model_type = "omni-test-stub"

    # Should not raise (and must not hit the str branch).
    auto_cls.register(_StubConfig, _StubImageProcessor, exist_ok=True)
