# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.ernie_image.ernie_image_transformer import _apply_rotary_emb
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def reset_fused_rope_state():
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    fused_rope._FAILED_KEYS.clear()
    yield
    fused_rope._FAILED_KEYS.clear()


def _inputs(shape: tuple[int, int, int, int]):
    batch, sequence, _, head_dim = shape
    torch.manual_seed(17)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    freqs_cos = torch.randn(
        batch,
        sequence,
        head_dim // 2,
        device="cuda",
        dtype=torch.float32,
    )
    freqs_sin = torch.randn_like(freqs_cos)
    return query, key, freqs_cos, freqs_sin


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 128),
        (2, 257, 7, 128),
        (2, 4608, 32, 128),
    ],
)
def test_fused_qk_rope_is_bit_exact(shape):
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    query, key, freqs_cos, freqs_sin = _inputs(shape)
    with torch.inference_mode():
        expected_query = _apply_rotary_emb(query, freqs_cos, freqs_sin)
        expected_key = _apply_rotary_emb(key, freqs_cos, freqs_sin)
        actual = fused_rope.try_fused_qk_rotary_emb(query, key, freqs_cos, freqs_sin)

    assert actual is not None
    actual_query, actual_key = actual
    assert torch.equal(actual_query, expected_query)
    assert torch.equal(actual_key, expected_key)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_launch_failure_disables_runtime_key(monkeypatch):
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    query, key, freqs_cos, freqs_sin = _inputs((1, 17, 3, 128))
    launch_count = 0

    def failing_launch(*args):
        nonlocal launch_count
        launch_count += 1
        raise RuntimeError("injected launch failure")

    monkeypatch.setattr(
        fused_rope,
        "_launch_fused_qk_rotary_emb",
        failing_launch,
    )
    with torch.inference_mode():
        for _ in range(2):
            assert fused_rope.try_fused_qk_rotary_emb(query, key, freqs_cos, freqs_sin) is None
    assert launch_count == 1
    assert len(fused_rope._FAILED_KEYS) == 1


def test_non_cuda_platform_does_not_launch_fused_kernel(monkeypatch):
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    monkeypatch.setattr(fused_rope, "HAS_TRITON", True)
    monkeypatch.setattr(fused_rope.current_omni_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)

    def unexpected_launch(*args):
        raise AssertionError("non-CUDA platforms must remain on the native path")

    monkeypatch.setattr(fused_rope, "_launch_fused_qk_rotary_emb", unexpected_launch)
    query = torch.zeros((1, 1, 1, 128), dtype=torch.bfloat16)
    freqs = torch.zeros((1, 1, 64), dtype=torch.float32)
    assert fused_rope.try_fused_qk_rotary_emb(query, query, freqs, freqs) is None
    assert not fused_rope._FAILED_KEYS


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_failed_runtime_key_cache_is_bounded():
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    for index in range(fused_rope._FAILED_KEYS_MAX_SIZE + 1):
        fused_rope._record_failed_runtime_key((index,))

    assert len(fused_rope._FAILED_KEYS) == fused_rope._FAILED_KEYS_MAX_SIZE
    assert (0,) not in fused_rope._FAILED_KEYS
    assert (fused_rope._FAILED_KEYS_MAX_SIZE,) in fused_rope._FAILED_KEYS


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_compile_path_does_not_launch_fused_kernel(monkeypatch):
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    query, key, freqs_cos, freqs_sin = _inputs((1, 17, 3, 128))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def unexpected_launch(*args):
        raise AssertionError("compile path must remain native")

    monkeypatch.setattr(
        fused_rope,
        "_launch_fused_qk_rotary_emb",
        unexpected_launch,
    )
    with torch.inference_mode():
        assert fused_rope.try_fused_qk_rotary_emb(query, key, freqs_cos, freqs_sin) is None


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("requires_grad_input", ["query", "key", "cos", "sin"])
def test_gradient_inputs_do_not_launch_fused_kernel(monkeypatch, requires_grad_input):
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    values = list(_inputs((1, 17, 3, 128)))
    input_index = {"query": 0, "key": 1, "cos": 2, "sin": 3}[requires_grad_input]
    values[input_index].requires_grad_(True)

    def unexpected_launch(*args):
        raise AssertionError("autograd inputs must remain on the native path")

    monkeypatch.setattr(
        fused_rope,
        "_launch_fused_qk_rotary_emb",
        unexpected_launch,
    )
    assert (
        fused_rope.try_fused_qk_rotary_emb(
            *values,
        )
        is None
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_mismatched_key_dtype_or_device_does_not_launch(monkeypatch):
    from vllm_omni.diffusion.models.ernie_image import fused_rope

    query, key, freqs_cos, freqs_sin = _inputs((1, 17, 3, 128))

    def unexpected_launch(*args):
        raise AssertionError("mismatched Q/K inputs must remain native")

    monkeypatch.setattr(
        fused_rope,
        "_launch_fused_qk_rotary_emb",
        unexpected_launch,
    )
    for unsupported_key in (key.float(), key.cpu()):
        assert (
            fused_rope.try_fused_qk_rotary_emb(
                query,
                unsupported_key,
                freqs_cos,
                freqs_sin,
            )
            is None
        )
