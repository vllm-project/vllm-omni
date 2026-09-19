# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end torch.compile parity for MammothModa2 ``Transformer2DModel``.

Drives the same code path the runner uses — ``regionally_compile`` over the
model — but tightens the config to ``fullgraph=True`` so any graph break in
the compiled per-block region surfaces as a compile error rather than a silent
perf regression. Bitwise parity is asserted against an eager copy of the model
across three text-length regimes (including the ``T=0`` case that exercises
the ``_apply_refiners`` empty-seq skip introduced alongside this compile
enablement).
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

# Shrunk config: small enough to run in tens of milliseconds on CPU, wide
# enough to exercise every block class (context/noise/ref-image refiners plus
# the main layer stack).
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
    # Reuse the model's own routine so the shape / dtype contract matches the
    # production pipeline (``pipeline_mammothmoda2_dit.py:106`` computes this
    # once at construction time and hands it to ``forward`` every step).
    return RotaryPosEmbedReal.get_freqs_real(AXES_DIM_ROPE, AXES_LENS, theta=10000)


def _inputs(*, batch: int, text_len: int, h_latent: int, w_latent: int):
    torch.manual_seed(text_len * 100 + batch * 10 + h_latent + w_latent)
    hidden_states = torch.randn(batch, 16, h_latent, w_latent)
    timestep = torch.rand(batch)
    text_hidden_states = torch.randn(batch, text_len, TEXT_FEAT_DIM) if text_len > 0 else torch.zeros(
        batch, 0, TEXT_FEAT_DIM
    )
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
    """Same probe as ``test_dit_compile_block.py``: watch Dynamo's frame
    compiler and fail if a later shape triggers an extra compilation."""

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
    """Fullgraph regional compile of every ``TransformerBlock`` still yields
    the exact eager output. ``text_len=0`` covers the moved empty-seq guard
    (``_apply_refiners`` skips ``context_refiner`` on zero-length text)."""
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
    # Strict fp32 tolerance rather than torch.equal: Inductor's kernel fusion
    # legitimately reorders FMAs. The bitwise contract from #7139 applies to
    # the AR→DiT hidden-state boundary — which the compile change does not
    # touch (same tensor object flows through ``Transformer2DModel.forward``).
    assert torch.allclose(want, got, rtol=1e-5, atol=1e-5), (
        f"max abs diff: {(want - got).abs().max().item()}"
    )


def test_forward_no_shape_driven_recompile_across_resolutions_and_text_len():
    """Sweep resolution and text-length variations through the same compiled
    model at batch=1. This mirrors the production hot path in
    ``pipeline_mammothmoda2_dit.py`` where CFG runs as two SEPARATE batch=1
    forwards (positive then unconditional) — batch never actually varies.
    What DOES vary is (a) latent resolution across requests and (b) tokenized
    text length. Under ``dynamic=True`` the per-block graph must generalize
    across both without a recompile."""
    model = _build_model()
    with set_current_diffusion_config(_SDPA_CONFIG):
        regionally_compile(model, dynamic=True, fullgraph=True)

    freqs_cis = _freqs_cis()
    # (h_latent, w_latent, text_len). All batch=1. First entry primes the
    # graph; the rest must reuse it.
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
