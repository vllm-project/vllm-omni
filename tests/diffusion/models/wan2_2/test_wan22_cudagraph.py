# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Eager and CUDA-graph Wan2.2 DiT step equivalence."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.wan2_2.cudagraph_wrapper import Wan22DiTCUDAGraphWrapper

pytestmark = [pytest.mark.core_model]


class _StubDiT(nn.Module):
    """Records calls and returns a shape-preserving transform of the input."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append({"hidden_states": hidden_states, **kwargs})
        return (hidden_states * 2.0,)


def _inputs(device: str = "cpu", dtype: torch.dtype = torch.float32):
    torch.manual_seed(7)
    return dict(
        hidden_states=torch.randn(2, 48, 21, 60, 104, device=device, dtype=dtype),
        timestep=torch.tensor([1000, 1000], device=device, dtype=dtype),
        encoder_hidden_states=torch.randn(2, 512, 4096, device=device, dtype=dtype),
    )


@pytest.mark.cpu
def test_full_graph_cache_retires_as_one_generation() -> None:
    wrapper = Wan22DiTCUDAGraphWrapper(_StubDiT(), max_graphs=2)
    first = ((2, 48, 21, 60, 104), (2,), (2, 512, 4096))
    second = ((2, 48, 21, 30, 52), (2,), (2, 512, 4096))
    wrapper._cache[first] = object()

    wrapper._retire_graph_generation_if_full()
    assert list(wrapper._cache) == [first]

    wrapper._cache[second] = object()
    wrapper._retire_graph_generation_if_full()
    assert not wrapper._cache


@pytest.mark.cpu
def test_key_uses_exact_shapes() -> None:
    wrapper = Wan22DiTCUDAGraphWrapper(_StubDiT())
    inputs = _inputs()
    key = wrapper._key(inputs["hidden_states"], inputs["timestep"], inputs["encoder_hidden_states"])
    assert key == ((2, 48, 21, 60, 104), (2,), (2, 512, 4096))


@pytest.mark.cpu
def test_cpu_calls_run_through_the_transformer() -> None:
    dit = _StubDiT()
    wrapper = Wan22DiTCUDAGraphWrapper(dit)
    inputs = _inputs()
    attention_kwargs = None

    out = wrapper(**inputs, attention_kwargs=attention_kwargs)

    torch.testing.assert_close(out, inputs["hidden_states"] * 2.0)
    assert len(dit.calls) == 1
    # ``None``-valued kwargs are dropped before reaching the transformer.
    assert "attention_kwargs" not in dit.calls[0]
    assert "return_dict" not in dit.calls[0]


@pytest.mark.cpu
def test_unexpected_kwarg_falls_back_to_eager() -> None:
    dit = _StubDiT()
    wrapper = Wan22DiTCUDAGraphWrapper(dit)
    inputs = _inputs()

    out = wrapper(**inputs, attention_kwargs={"backend": "x"})

    torch.testing.assert_close(out, inputs["hidden_states"] * 2.0)
    # The passthrough keeps the original kwargs untouched.
    assert dit.calls[0]["attention_kwargs"] == {"backend": "x"}


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
def test_graph_replay_matches_eager_and_updates_inputs() -> None:
    model, wrapper = _make_gpu_wrapper(max_graphs=4)
    inputs = _inputs("cuda", torch.bfloat16)

    with torch.no_grad():
        eager = model(**inputs, return_dict=False)[0]
        first = wrapper(**inputs, return_dict=False)
        torch.testing.assert_close(first, eager)

        # A replay must pick up new values in every static input, including
        # the mutable timestep buffer.
        moved = {
            "hidden_states": inputs["hidden_states"] * 0.9,
            "timestep": torch.tensor([500, 500], device="cuda", dtype=torch.bfloat16),
            "encoder_hidden_states": inputs["encoder_hidden_states"] * 1.1,
        }
        replayed = wrapper(**moved, return_dict=False)
        reference = model(**moved, return_dict=False)[0]
    torch.testing.assert_close(replayed, reference)
    assert len(wrapper._cache) == 1


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
def test_graph_capture_failure_is_propagated(mocker) -> None:
    model, wrapper = _make_gpu_wrapper()
    inputs = _inputs("cuda", torch.bfloat16)
    capture = mocker.patch.object(wrapper, "_capture", side_effect=RuntimeError("capture failed"))

    with pytest.raises(RuntimeError, match="capture failed"):
        wrapper(**inputs, return_dict=False)

    assert capture.call_count == 1
    assert not wrapper.enabled
    assert not wrapper._cache

    # The disabled wrapper still serves the request in eager mode. Shape
    # only: numeric equivalence is covered by the replay test, and two eager
    # runs already differ by the attention backend's nondeterminism.
    with torch.no_grad():
        out = wrapper(**inputs, return_dict=False)
    assert out.shape == inputs["hidden_states"].shape
    assert torch.isfinite(out).all()


def _make_gpu_wrapper(max_graphs: int = 8):
    """Small real transformer on CUDA with the runtime contexts it needs."""
    from contextlib import ExitStack

    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

    global _GPU_STACK
    try:
        _GPU_STACK
    except NameError:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29677",
        )
        initialize_model_parallel()
        _cfg = VllmConfig()
        _GPU_STACK = ExitStack()
        _GPU_STACK.enter_context(set_current_vllm_config(_cfg))
        _GPU_STACK.enter_context(set_forward_context(vllm_config=_cfg, omni_diffusion_config=OmniDiffusionConfig()))

    torch.manual_seed(42)
    model = (
        WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=4,
            attention_head_dim=128,
            in_channels=48,
            out_channels=48,
            text_dim=4096,
            freq_dim=256,
            ffn_dim=1024,
            num_layers=2,
            cross_attn_norm=True,
            eps=1e-06,
        )
        .cuda()
        .bfloat16()
        .eval()
    )
    with torch.no_grad():
        for _, param in model.named_parameters():
            # Freshly allocated vLLM parameters hold uninitialized memory;
            # re-init everything so results never depend on what a previous
            # test left behind in the allocator.
            if param.ndim >= 2:
                param.normal_(0.0, 0.02)
            else:
                param.zero_()
    return model, Wan22DiTCUDAGraphWrapper(model, max_graphs=max_graphs)
