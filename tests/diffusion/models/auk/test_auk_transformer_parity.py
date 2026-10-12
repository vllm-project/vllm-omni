#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK transformer parity: the correctness oracle for the in-tree port.

Checks that :class:`AuKTransformer` reproduces the upstream ``auk`` reference backbone
on the released ``auk_base`` weights. Both models are built from the same
state dict and fed the same inputs, under the reference's own inference
configuration: fp32 weights with ``torch.autocast(bfloat16)``. That
configuration matters. The reference keeps its rotary frequencies in a buffer,
so casting its weights to bf16 would round both the frequencies and the
integer positions and move the numbers for reasons unrelated to the port.

Cases:

* ``single`` - one conditional forward
* ``cfg`` - the doubled ``cfg_infer`` forward, both branches
* ``masked`` - batch of 2 with padded targets, references and text
* ``rope`` - the local rotary implementation against ``x_transformers``, and
  its survival of a cast to bf16
* ``sample`` - an 8-step Euler trajectory against the reference backbone
  driven by a transcription of upstream's ODE loop
* ``generator`` - the sampler's two noise paths agreeing
* ``cache`` - the text projection cache and ``clear_cache()``
* ``mask flag`` - ``attn_mask_enabled=False`` reproducing the reference with
  its own flag off

This directory deliberately has no ``__init__.py``, unlike its siblings.
pytest's prepend import mode puts the nearest non-package directory on
``sys.path``, and an ``auk`` package here would shadow the upstream ``auk``
distribution this test compares against.

Needs the released checkpoint and a GPU, so it runs locally rather than in
CI. Point ``AUK_CKPT_DIR`` at a directory holding ``auk_base.safetensors`` and
``config.yaml``; the test skips when that is unset or when the upstream
``auk`` package is not importable::

    AUK_CKPT_DIR=/path/to/ckpts/AuK CUDA_VISIBLE_DEVICES=6 \
        python -m pytest tests/diffusion/models/auk/test_auk_transformer_parity.py -v -s
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import pytest
import torch
import yaml

from vllm_omni.diffusion.models.auk.auk_transformer import (
    AuKTransformer,
    Rotary,
    _apply_rope,
    dit_state_dict,
    sample_latents,
)

CKPT_DIR = os.environ.get("AUK_CKPT_DIR")

# Tolerances. The port evaluates rotary embeddings in fp32 and skips the
# reference's redundant uncond text projection, so the two graphs are not
# bit-identical under autocast; they agree well inside bf16 noise.
FORWARD_ATOL = 3e-2
FORWARD_COS = 0.999
SAMPLE_ATOL = 5e-2
SAMPLE_COS = 0.999
ROPE_ATOL = 1e-5

TARGET_FRAMES = 300
REF_FRAMES = 150
TEXT_TOKENS = 40
SHORT_LENS = (220, 90, 25)  # target, reference, text for the padded batch row
TIME = 0.37
SAMPLE_STEPS = 8
SWAY_COEF = -1.0
SEED = 1234
DEVICE = "cuda:0"
AUTOCAST_DTYPE = torch.bfloat16


def _skip_reason() -> str | None:
    if CKPT_DIR is None:
        return "AUK_CKPT_DIR is not set"
    if not Path(CKPT_DIR, "auk_base.safetensors").is_file():
        return f"no auk_base.safetensors under {CKPT_DIR}"
    try:
        # Resolving the submodule, not just "auk", also catches the shadow an
        # __init__.py in this directory would create.
        upstream = importlib.util.find_spec("auk.model")
    except ModuleNotFoundError:
        upstream = None
    if upstream is None:
        return "the upstream 'auk' package is not importable or is shadowed by this directory"
    if not torch.cuda.is_available():
        return "no CUDA device"
    return None


SKIP_REASON = _skip_reason()

pytestmark = [
    pytest.mark.local_model,
    pytest.mark.diffusion,
    pytest.mark.gpu,
    pytest.mark.cuda,
    pytest.mark.skipif(SKIP_REASON is not None, reason=str(SKIP_REASON)),
]


def _arch() -> dict:
    """Read the backbone geometry out of the checkpoint's config."""
    config = yaml.safe_load(Path(CKPT_DIR, "config.yaml").read_text())["model"]
    return {
        "dim": config["arch"]["dim"],
        "heads": config["arch"]["heads"],
        "dim_head": config["arch"].get("dim_head", 64),
        "ff_mult": config["arch"]["ff_mult"],
        "text_hidden_dim": config["arch"]["text_hidden_dim"],
        "num_layers": config["arch"]["num_layers"],
        "num_single_layers": config["arch"]["num_single_layers"],
        "latent_dim": config["vae"]["latent_dim"],
    }


def _stats(port: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """Return max abs diff, cosine similarity and the reference's RMS."""
    a = port.flatten().float()
    b = ref.flatten().float()
    return (
        (a - b).abs().max().item(),
        torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        b.pow(2).mean().sqrt().item(),
    )


def _assert_close(name: str, port: torch.Tensor, ref: torch.Tensor, atol: float, min_cos: float) -> None:
    max_abs, cos, rms = _stats(port, ref)
    print(f"[{name}] max|Δ|={max_abs:.4e} cos={cos:.8f} ref_rms={rms:.4f} rel={max_abs / rms:.3e}")
    assert max_abs < atol, f"{name}: max|Δ|={max_abs:.4e} >= {atol:.1e} (reference rms {rms:.4f})"
    assert cos > min_cos, f"{name}: cos={cos:.8f} <= {min_cos}"


def _autocast():
    return torch.autocast("cuda", dtype=AUTOCAST_DTYPE)


@pytest.fixture(scope="module")
def models() -> tuple[torch.nn.Module, AuKTransformer]:
    """The reference and the port, same weights, fp32 on the accelerator."""
    from auk.model.flux2_edit import Flux2Edit
    from safetensors.torch import load_file

    arch = _arch()
    device = torch.device(DEVICE)
    state = dit_state_dict(load_file(str(Path(CKPT_DIR, "auk_base.safetensors")), device="cpu"))

    reference = Flux2Edit(dropout=0.0, attn_backend="torch", attn_mask_enabled=True, **arch)
    reference.load_state_dict(state, strict=True)
    reference = reference.to(device).eval().requires_grad_(False)

    port = AuKTransformer(**arch)
    # strict=True is the point of the port: every released tensor lands.
    port.load_state_dict(state, strict=True)
    port = port.to(device).eval().requires_grad_(False)

    del state
    return reference, port


@pytest.fixture(scope="module")
def inputs() -> dict[str, torch.Tensor]:
    """Fixed random inputs, as a batch of 2 whose second row is padded."""
    device = torch.device(DEVICE)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    arch = _arch()

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(device)

    def pad_mask(width: int, short: int) -> torch.Tensor:
        mask = torch.zeros(2, width, dtype=torch.bool, device=device)
        mask[0, :] = True
        mask[1, :short] = True
        return mask

    mask2 = pad_mask(TARGET_FRAMES, SHORT_LENS[0])
    ref_mask2 = pad_mask(REF_FRAMES, SHORT_LENS[1])
    c_mask2 = pad_mask(TEXT_TOKENS, SHORT_LENS[2])

    # Padding rows are all-zero in the text encoder's output, so keep the
    # inputs consistent with the masks.
    x2 = randn(2, TARGET_FRAMES, arch["latent_dim"]) * mask2.unsqueeze(-1)
    ref2 = randn(2, REF_FRAMES, arch["latent_dim"]) * ref_mask2.unsqueeze(-1)
    text2 = randn(2, TEXT_TOKENS, arch["text_hidden_dim"]) * c_mask2.unsqueeze(-1)

    return {
        "x": x2[:1].contiguous(),
        "ref": ref2[:1].contiguous(),
        "text": text2[:1].contiguous(),
        "c_mask": c_mask2[:1].contiguous(),
        "x2": x2,
        "ref2": ref2,
        "text2": text2,
        "mask2": mask2,
        "ref_mask2": ref_mask2,
        "c_mask2": c_mask2,
        "time": torch.tensor(TIME, dtype=torch.float32, device=device),
    }


def test_single_forward(models, inputs) -> None:
    """One conditional forward: reference against port."""
    reference, port = models
    kwargs = dict(text=inputs["text"], time=inputs["time"], c_mask=inputs["c_mask"], ref=inputs["ref"])
    with torch.no_grad(), _autocast():
        want = reference(x=inputs["x"], **kwargs)
        got = port(inputs["x"], **kwargs)

    assert got.shape == want.shape == (1, TARGET_FRAMES, port.latent_dim)
    _assert_close("single", got, want, FORWARD_ATOL, FORWARD_COS)


def test_cfg_forward(models, inputs) -> None:
    """The doubled cfg_infer forward, cond branch first."""
    reference, port = models
    kwargs = dict(
        text=inputs["text"],
        time=inputs["time"],
        c_mask=inputs["c_mask"],
        ref=inputs["ref"],
        cfg_infer=True,
    )
    with torch.no_grad(), _autocast():
        want = reference(x=inputs["x"], **kwargs)
        got = port(inputs["x"], **kwargs)

    assert got.shape == want.shape == (2, TARGET_FRAMES, port.latent_dim)
    _assert_close("cfg-cond", got[:1], want[:1], FORWARD_ATOL, FORWARD_COS)
    _assert_close("cfg-uncond", got[1:], want[1:], FORWARD_ATOL, FORWARD_COS)


def test_masked_batch_forward(models, inputs) -> None:
    """Batch of 2 with padded targets, references and text.

    Only valid target frames are compared. Padded frames are unconstrained:
    the reference leaves them to whatever masked attention emits there.
    """
    reference, port = models
    kwargs = dict(
        text=inputs["text2"],
        time=inputs["time"],
        mask=inputs["mask2"],
        c_mask=inputs["c_mask2"],
        ref=inputs["ref2"],
        ref_mask=inputs["ref_mask2"],
    )
    with torch.no_grad(), _autocast():
        want = reference(x=inputs["x2"], **kwargs)
        got = port(inputs["x2"], **kwargs)

    assert got.shape == want.shape == (2, TARGET_FRAMES, port.latent_dim)
    valid = inputs["mask2"].unsqueeze(-1).expand_as(got)
    _assert_close("masked", got[valid], want[valid], FORWARD_ATOL, FORWARD_COS)


def _set_mask_flag(reference, port, enabled: bool) -> None:
    """Flip attn_mask_enabled on both models in place, so no second copy is loaded."""
    for module in port.modules():
        if hasattr(module, "attn_mask_enabled"):
            module.attn_mask_enabled = enabled
    for blocks in (reference.transformer_blocks, reference.single_transformer_blocks):
        for block in blocks:
            block.attn.processor.attn_mask_enabled = enabled


def test_mask_flag_disabled(models, inputs) -> None:
    """``attn_mask_enabled=False`` must track the reference with its own flag off.

    With the flag off both models run attention unmasked and still zero the
    padded outputs, so the padded batch is compared in full.
    """
    reference, port = models
    kwargs = dict(
        text=inputs["text2"],
        time=inputs["time"],
        mask=inputs["mask2"],
        c_mask=inputs["c_mask2"],
        ref=inputs["ref2"],
        ref_mask=inputs["ref_mask2"],
    )
    with torch.no_grad(), _autocast():
        masked = port(inputs["x2"], **kwargs)
        _set_mask_flag(reference, port, False)
        try:
            want = reference(x=inputs["x2"], **kwargs)
            got = port(inputs["x2"], **kwargs)
        finally:
            _set_mask_flag(reference, port, True)

    _assert_close("mask-flag-off", got, want, FORWARD_ATOL, FORWARD_COS)
    # Guard against a vacuous pass: the flag has to actually change something.
    assert (got.float() - masked.float()).abs().max().item() > 1e-3


def test_sampler_generator_matches_seed(models, inputs) -> None:
    """A generator seeded with 0 must draw exactly what ``seed=0`` draws."""
    _, port = models
    common = dict(
        gen_frames=TARGET_FRAMES,
        nfe=2,
        cfg_strength=0.0,
        sway_sampling_coef=SWAY_COEF,
        text=inputs["text"],
        c_mask=inputs["c_mask"],
        ref=inputs["ref"],
        ref_mask=None,
    )
    with torch.no_grad(), _autocast():
        by_seed = sample_latents(port, seed=0, **common)
        by_generator = sample_latents(port, generator=torch.Generator(device=DEVICE).manual_seed(0), **common)

    print(f"[generator] max|Δ|={(by_generator.float() - by_seed.float()).abs().max().item():.4e}")
    torch.testing.assert_close(by_generator, by_seed, rtol=0, atol=0)


def test_cache_needs_clearing(models, inputs) -> None:
    """The cached text projection is reused until ``clear_cache()``.

    ``sample_latents`` clears on the way out, so a pipeline that reuses the
    module across requests is safe without doing anything. A caller driving
    ``forward(cache=True)`` itself has to clear when the text changes.
    """
    _, port = models
    other_text = inputs["text"] * 0.5
    kwargs = dict(time=inputs["time"], c_mask=inputs["c_mask"], ref=inputs["ref"], cfg_infer=True)

    port.clear_cache()
    with torch.no_grad(), _autocast():
        want = port(inputs["x"], inputs["text"], **kwargs)
        port(inputs["x"], other_text, cache=True, **kwargs)
        stale = port(inputs["x"], inputs["text"], cache=True, **kwargs)
        port.clear_cache()
        fresh = port(inputs["x"], inputs["text"], cache=True, **kwargs)
        port.clear_cache()

    # The cache is real: the second call reused the first call's projection.
    assert (stale.float() - want.float()).abs().max().item() > 1e-3
    torch.testing.assert_close(fresh, want, rtol=0, atol=0)
    assert port.text_cond is None and port.text_uncond is None


def test_rope_matches_x_transformers() -> None:
    """The local rotary implementation against x_transformers, in fp32."""
    from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

    device = torch.device(DEVICE)
    dim_head, seq_len = 64, TEXT_TOKENS + REF_FRAMES + TARGET_FRAMES
    q = torch.randn(2, 24, seq_len, dim_head, dtype=torch.float32, device=device)

    freqs = RotaryEmbedding(dim_head).to(device).forward_from_seq_len(seq_len)[0]
    want = apply_rotary_pos_emb(q, freqs)
    got = _apply_rope(q, Rotary(dim_head).to(device)(seq_len))

    max_abs, cos, _ = _stats(got, want)
    print(f"[rope] max|Δ|={max_abs:.4e} cos={cos:.10f}")
    assert max_abs < ROPE_ATOL, f"rope: max|Δ|={max_abs:.4e} >= {ROPE_ATOL:.1e}"


def test_rope_survives_half_precision_cast() -> None:
    """Casting the module to bf16 must not round the rotary frequencies.

    The reference stores them in a buffer, so ``.to(bfloat16)`` rounds both the
    frequencies and the integer positions and silently changes every rotation.
    The port recomputes them instead.
    """
    dim_head, seq_len = 64, TEXT_TOKENS + REF_FRAMES + TARGET_FRAMES
    exact = Rotary(dim_head)(seq_len)
    got = Rotary(dim_head).to(torch.bfloat16)(seq_len)

    assert got.dtype == torch.float32
    torch.testing.assert_close(got, exact, rtol=0, atol=0)


def _reference_euler(reference, *, x, text, c_mask, ref, steps, cfg_strength, sway) -> torch.Tensor:
    """Upstream's ODE loop, transcribed.

    ``CFMEdit.sample`` hands the backbone to ``torchdiffeq.odeint(method=
    "euler")``, which on a fixed grid is ``y += (t1 - t0) * f(t0, y)``.
    Constructing a real ``CFMEdit`` would drag in the Qwen-Omni text encoder,
    so only the backbone here comes from upstream.
    """
    t = torch.linspace(0, 1, steps + 1, device=x.device, dtype=torch.float32)
    if sway is not None:
        t = t + sway * (torch.cos(math.pi / 2 * t) - 1 + t)

    shared = dict(text=text, c_mask=c_mask, ref=ref, mask=None, ref_mask=None, cache=True)
    for i in range(steps):
        if cfg_strength < 1e-5:
            v = reference(x=x, time=t[i], **shared)
        else:
            pred = reference(x=x, time=t[i], cfg_infer=True, **shared)
            v_cond, v_uncond = torch.chunk(pred, 2, dim=0)
            v = v_cond + (v_cond - v_uncond) * cfg_strength
        x = x + (t[i + 1] - t[i]) * v
    reference.clear_cache()
    return x


@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_euler_sample(models, inputs, cfg_strength: float) -> None:
    """An 8-step Euler trajectory, with and without guidance."""
    reference, port = models
    shared = dict(text=inputs["text"], c_mask=inputs["c_mask"], ref=inputs["ref"])

    with torch.no_grad(), _autocast():
        got = sample_latents(
            port,
            gen_frames=TARGET_FRAMES,
            ref_mask=None,
            nfe=SAMPLE_STEPS,
            cfg_strength=cfg_strength,
            sway_sampling_coef=SWAY_COEF,
            seed=SEED,
            **shared,
        )
        # Re-seeding reproduces the noise sample_latents drew for itself.
        torch.manual_seed(SEED)
        noise = torch.randn(
            TARGET_FRAMES, port.latent_dim, device=inputs["ref"].device, dtype=inputs["ref"].dtype
        ).unsqueeze(0)
        want = _reference_euler(
            reference, x=noise, steps=SAMPLE_STEPS, cfg_strength=cfg_strength, sway=SWAY_COEF, **shared
        )

    assert got.shape == want.shape == (1, TARGET_FRAMES, port.latent_dim)
    _assert_close(f"sample-cfg{cfg_strength:g}", got, want, SAMPLE_ATOL, SAMPLE_COS)
