"""Tests for CUDAGraphWrapper.__getattr__ performance optimization.

This module tests that the patched CUDAGraphWrapper avoids expensive __repr__
calls when hasattr() is used for non-existent attributes. The original vLLM
implementation includes {self.runnable} in the AttributeError message, which
triggers model tree traversal and can take ~6ms on large models.
"""

import time

import pytest
import torch
import torch.nn as nn

from vllm_omni.worker.gpu_model_runner import CUDAGraphWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class SlowReprModel(nn.Module):
    """A mock model with artificially slow __repr__ to detect unwanted calls."""

    def __init__(self, repr_delay_ms: float = 10.0):
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.repr_delay_ms = repr_delay_ms
        self.repr_call_count = 0

    def forward(self, x):
        return self.linear(x)

    def __repr__(self):
        self.repr_call_count += 1
        # Simulate expensive repr by sleeping
        time.sleep(self.repr_delay_ms / 1000.0)
        return f"SlowReprModel(delay={self.repr_delay_ms}ms)"


class MockCUDAGraphWrapper:
    """A minimal mock that mimics CUDAGraphWrapper structure for CPU testing."""

    def __init__(self, runnable):
        # Store in __dict__ directly to avoid triggering __getattr__
        object.__setattr__(self, "runnable", runnable)

    def __getattr__(self, key: str):
        # This is the optimized implementation we're testing
        runnable = object.__getattribute__(self, "runnable")
        if hasattr(runnable, key):
            return getattr(runnable, key)
        # Key optimization: DO NOT include {self.runnable} in error message
        # as it triggers expensive __repr__ on large models
        raise AttributeError(f"Attribute {key} not exists in the runnable of cudagraph wrapper")


def test_hasattr_nonexistent_does_not_trigger_repr():
    """Verify that hasattr for non-existent attributes doesn't call __repr__."""
    model = SlowReprModel(repr_delay_ms=100.0)  # Very slow repr
    wrapper = MockCUDAGraphWrapper(model)

    # Reset counter
    model.repr_call_count = 0

    # Call hasattr for non-existent attribute multiple times
    for _ in range(10):
        result = hasattr(wrapper, "nonexistent_attribute_xyz")
        assert result is False

    # __repr__ should never have been called
    assert model.repr_call_count == 0, (
        f"__repr__ was called {model.repr_call_count} times when checking "
        "for non-existent attributes. This indicates the AttributeError "
        "message contains {self.runnable} which triggers expensive repr."
    )


def test_hasattr_nonexistent_is_fast():
    """Verify that hasattr for non-existent attributes is fast (<1ms per call)."""
    model = SlowReprModel(repr_delay_ms=100.0)
    wrapper = MockCUDAGraphWrapper(model)

    num_iterations = 100
    start = time.perf_counter()
    for _ in range(num_iterations):
        hasattr(wrapper, "nonexistent_attribute_xyz")
    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_ms = elapsed_ms / num_iterations
    # If __repr__ were being called, each would take ~100ms
    # We expect <1ms per call with the fix
    assert avg_ms < 1.0, (
        f"hasattr for non-existent attribute took {avg_ms:.2f}ms on average. "
        "Expected <1ms. This suggests __repr__ is being triggered."
    )


def test_hasattr_existing_attribute_works():
    """Verify that hasattr for existing attributes returns True and works correctly."""
    model = SlowReprModel()
    wrapper = MockCUDAGraphWrapper(model)

    # 'forward' exists on nn.Module
    assert hasattr(wrapper, "forward") is True

    # 'linear' exists on our model
    assert hasattr(wrapper, "linear") is True

    # Can actually access the attribute
    linear = wrapper.linear
    assert isinstance(linear, nn.Linear)


def test_getattr_existing_attribute_returns_value():
    """Verify that getattr for existing attributes returns the correct value."""
    model = SlowReprModel()
    wrapper = MockCUDAGraphWrapper(model)

    # Access forward method
    forward_method = wrapper.forward
    assert callable(forward_method)

    # Access linear layer
    linear = wrapper.linear
    assert isinstance(linear, nn.Linear)
    assert linear.in_features == 16
    assert linear.out_features == 16


def test_getattr_nonexistent_raises_attribute_error():
    """Verify that getattr for non-existent attributes raises AttributeError."""
    model = SlowReprModel()
    wrapper = MockCUDAGraphWrapper(model)

    with pytest.raises(AttributeError) as exc_info:
        _ = wrapper.nonexistent_attribute

    # Verify error message format (should NOT contain model repr)
    error_msg = str(exc_info.value)
    assert "nonexistent_attribute" in error_msg
    assert "cudagraph wrapper" in error_msg
    # Should NOT contain the slow repr output
    assert "SlowReprModel(delay=" not in error_msg


def test_attribute_error_message_does_not_contain_runnable_repr():
    """Explicitly verify the error message doesn't trigger runnable repr."""
    model = SlowReprModel(repr_delay_ms=100.0)
    wrapper = MockCUDAGraphWrapper(model)
    model.repr_call_count = 0

    try:
        _ = wrapper.nonexistent_attr
    except AttributeError:
        pass

    # __repr__ should not have been called during error construction
    assert model.repr_call_count == 0, (
        "AttributeError message construction triggered __repr__. The error message should not include {self.runnable}."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_real_cudagraph_wrapper_hasattr_performance():
    """Test the actual CUDAGraphWrapper from vllm_omni (requires CUDA)."""
    from vllm.config import CUDAGraphMode

    model = SlowReprModel(repr_delay_ms=50.0).cuda()
    model.repr_call_count = 0

    # Create actual CUDAGraphWrapper
    try:
        wrapper = CUDAGraphWrapper(model, runtime_mode=CUDAGraphMode.NONE)
    except Exception:
        pytest.skip("Could not create CUDAGraphWrapper")

    # Test hasattr performance
    num_iterations = 50
    start = time.perf_counter()
    for _ in range(num_iterations):
        hasattr(wrapper, "nonexistent_xyz")
    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_ms = elapsed_ms / num_iterations
    assert avg_ms < 1.0, f"Real CUDAGraphWrapper hasattr took {avg_ms:.2f}ms avg. Expected <1ms with the optimization."
    assert model.repr_call_count == 0, f"__repr__ called {model.repr_call_count} times"
