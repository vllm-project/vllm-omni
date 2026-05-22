# SPDX-License-Identifier: Apache-2.0
"""Unit tests for step-execution interface (no model weights required).

These tests verify the structural correctness of the step-execution
implementation without loading the full model. They can run quickly
in CI environments without GPU access to model weights.

Usage:
    python -m pytest tests/diffusion/models/sensenova_u1/test_7_step_execution_unit.py -v
"""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
)


def test_class_declares_step_execution():
    """SenseNovaU1Pipeline declares supports_step_execution = True."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    assert hasattr(SenseNovaU1Pipeline, "supports_step_execution")
    assert SenseNovaU1Pipeline.supports_step_execution is True


def test_class_has_step_methods():
    """All four SupportsStepExecution methods are defined."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    for name in ("prepare_encode", "denoise_step", "step_scheduler", "post_decode"):
        method = getattr(SenseNovaU1Pipeline, name, None)
        assert method is not None, f"Missing method: {name}"
        assert callable(method), f"{name} is not callable"


def test_class_has_step_states_dict():
    """Pipeline __init__ creates _step_states dict."""
    import inspect

    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    source = inspect.getsource(SenseNovaU1Pipeline.__init__)
    assert "_step_states" in source


def test_protocol_isinstance_check():
    """Pipeline class satisfies SupportsStepExecution protocol structurally."""
    from vllm_omni.diffusion.models.interface import SupportsStepExecution
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    # Protocols with non-method members (ClassVar) don't support issubclass
    # in Python 3.12+. Check structural conformance manually.
    assert SenseNovaU1Pipeline.supports_step_execution is True
    for method_name in ("prepare_encode", "denoise_step", "step_scheduler", "post_decode"):
        assert callable(getattr(SenseNovaU1Pipeline, method_name, None))


def test_helper_methods_exist():
    """Helper methods for step execution are defined."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    helpers = [
        "_parse_request_from_state",
        "_build_t2i_caches",
        "_build_it2i_caches",
        "_step_denoise_single",
    ]
    for name in helpers:
        assert hasattr(SenseNovaU1Pipeline, name), f"Missing helper: {name}"


if __name__ == "__main__":
    test_class_declares_step_execution()
    print("[1/5] Class declaration: PASS")

    test_class_has_step_methods()
    print("[2/5] Step methods exist: PASS")

    test_class_has_step_states_dict()
    print("[3/5] _step_states in __init__: PASS")

    test_protocol_isinstance_check()
    print("[4/5] Protocol conformance: PASS")

    test_helper_methods_exist()
    print("[5/5] Helper methods exist: PASS")

    print("\nAll unit tests passed.")
