# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regional torch.compile parity for MammothModa2 ``Transformer2DModel``.

Uses ``fullgraph=True`` so a graph break in ``TransformerBlock`` fails the test
instead of silently falling back to eager.
"""

import copy

import pytest
import torch

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import Transformer2DModel
from vllm_omni.diffusion.models.mammoth_moda2.rope_real import RotaryPosEmbedReal

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# Small enough for CPU; still builds every block kind (refiners and main layers).
HIDDEN, HEADS, KV_HEADS = 96, 4, 2
AXES_DIM_ROPE = (12, 6, 6)  # sum == head_dim == HIDDEN // HEADS
AXES_LENS = (128, 128, 128)
TEXT_FEAT_DIM = 64
_SDPA_CONFIG = OmniDiffusionConfig(diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}})


def _build_model() -> Transformer2DModel:
    torch.manual_seed(0)
    with set_current_diffusion_config(_SDPA_CONFIG):
        model = Transformer2DModel(
            patch_size=2,
            in_channels=16,
            hidden_size=HIDDEN,
            num_layers=2,
            num_refiner_layers=1,
            num_attention_heads=HEADS,
            num_kv_heads=KV_HEADS,
            multiple_of=16,
            ffn_dim_multiplier=1.0,
            norm_eps=1e-5,
            axes_dim_rope=AXES_DIM_ROPE,
            axes_lens=AXES_LENS,
            text_feat_dim=TEXT_FEAT_DIM,
        ).eval()
    return model


def _freqs_cis():
    return RotaryPosEmbedReal.get_freqs_real(AXES_DIM_ROPE, AXES_LENS, theta=10000)


def _inputs(*, batch: int, text_len: int, h_latent: int, w_latent: int):
    torch.manual_seed(text_len * 100 + batch * 10 + h_latent + w_latent)
    hidden_states = torch.randn(batch, 16, h_latent, w_latent)
    timestep = torch.rand(batch)
    text_hidden_states = torch.randn(batch, text_len, TEXT_FEAT_DIM)
    text_attention_mask = torch.ones(batch, text_len, dtype=torch.bool)
    return hidden_states, timestep, text_hidden_states, text_attention_mask


def _call(model: Transformer2DModel, hidden_states, timestep, text_hidden_states, text_attention_mask, freqs_cis):
    with torch.no_grad(), set_current_diffusion_config(_SDPA_CONFIG):
        return model(
            hidden_states=hidden_states,
            timestep=timestep,
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            freqs_cis=freqs_cis,
            ref_image_hidden_states=None,
        )


class _CompileCounter:
    """Count Dynamo frame compilations."""

    def __init__(self) -> None:
        import torch._dynamo.convert_frame as convert_frame

        self._module = convert_frame
        self._original = convert_frame._compile
        self.count = 0

    def __enter__(self) -> "_CompileCounter":
        outer = self

        def _tracked(*args, **kwargs):
            outer.count += 1
            return outer._original(*args, **kwargs)

        self._module._compile = _tracked
        return self

    def __exit__(self, *_exc) -> None:
        self._module._compile = self._original


@pytest.mark.parametrize("text_len", [0, 4, 77], ids=["empty_text", "short_text", "recipe_text"])
def test_forward_regionally_compiled_matches_eager(text_len: int):
    """``text_len=0`` covers the empty-text skip in ``_apply_refiners``."""
    eager = _build_model()
    compiled = copy.deepcopy(eager)
    with set_current_diffusion_config(_SDPA_CONFIG):
        regionally_compile(compiled, dynamic=True, fullgraph=True)

    hidden_states, timestep, text_hidden_states, text_attention_mask = _inputs(
        batch=1, text_len=text_len, h_latent=32, w_latent=32
    )
    freqs_cis = _freqs_cis()

    want = _call(eager, hidden_states, timestep, text_hidden_states, text_attention_mask, freqs_cis)
    got = _call(compiled, hidden_states, timestep, text_hidden_states, text_attention_mask, freqs_cis)

    assert want.shape == got.shape
    # Inductor fusion reorders FMAs, so compare with a tight fp32 tolerance.
    assert torch.allclose(want, got, rtol=1e-5, atol=1e-5), f"max abs diff: {(want - got).abs().max().item()}"


def test_forward_no_shape_driven_recompile_across_resolutions_and_text_len():
    """With ``dynamic=True``, new resolutions and text lengths reuse the first graph."""
    model = _build_model()
    with set_current_diffusion_config(_SDPA_CONFIG):
        regionally_compile(model, dynamic=True, fullgraph=True)

    freqs_cis = _freqs_cis()
    # (h_latent, w_latent, text_len)
    shapes = [(32, 32, 77), (32, 64, 77), (64, 32, 4)]

    with _CompileCounter() as counter:
        for i, (h, w, t_len) in enumerate(shapes):
            hidden_states, timestep, text_hidden_states, text_attention_mask = _inputs(
                batch=1, text_len=t_len, h_latent=h, w_latent=w
            )
            _call(model, hidden_states, timestep, text_hidden_states, text_attention_mask, freqs_cis)
            if i == 0:
                first_call_count = counter.count
                assert first_call_count >= 1, "regional compile did not trigger on first shape"

    assert counter.count == first_call_count, (
        f"shape-driven recompile detected: {counter.count - first_call_count} extra frame(s) "
        f"compiled across shapes {shapes[1:]}"
    )
