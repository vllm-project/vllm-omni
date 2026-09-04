# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Equivalence matrix: BooguAutoencoderKL vs diffusers AutoencoderKL.

The whole point of the Boogu-private VAE (#6686, D-15) is that it is a
faithful copy of diffusers' with the GroupNorm+SiLU fusion baked into an
owned forward — so the core test obligation is submodule-by-submodule
equivalence against diffusers, with shared weights.

CPU-only: ``HAS_TRITON`` is forced off (autouse fixture), so the fused op's
native fallback is ``F.silu(F.group_norm(...))`` — mathematically the same
functional calls the eager modules make, hence every comparison below can
demand ``torch.equal`` (bit-exact), which is stronger than the task's
fp32-equal/bf16-envelope requirement.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.vae import Decoder as DiffusersDecoder
from diffusers.models.autoencoders.vae import Encoder as DiffusersEncoder
from diffusers.models.downsampling import Downsample2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.upsampling import Upsample2D
from torch import nn

from vllm_omni.diffusion.models.boogu_image.vae import (
    BooguAutoencoderKL,
    BooguDecoder,
    BooguDownsample2D,
    BooguEncoder,
    BooguResnetBlock2D,
    BooguUpsample2D,
    vae_group_norm_silu_fusion_enabled,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _force_native_fallback(monkeypatch: pytest.MonkeyPatch):
    # HAS_TRITON is a platform property, not per-tensor: on a CUDA host a CPU
    # tensor would be handed to the Triton path. Force the native fallback.
    op_module = importlib.import_module("vllm_omni.model_executor.models.common.ops.fused_group_norm_silu")
    monkeypatch.setattr(op_module, "HAS_TRITON", False)


# Tiny config used everywhere below (kept divisible-by-groups); the real
# Boogu architecture is asserted separately in the site-count test.
_TINY = dict(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
    block_out_channels=(32, 64),
    layers_per_block=1,
    latent_channels=4,
    norm_num_groups=8,
    sample_size=32,
    mid_block_add_attention=True,
    use_quant_conv=False,
    use_post_quant_conv=False,
)

# The real Boogu decoder architecture (vae/config.json @ 334ad7e5).
_BOOGU = dict(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D",) * 4,
    up_block_types=("UpDecoderBlock2D",) * 4,
    block_out_channels=(128, 256, 512, 512),
    layers_per_block=2,
    latent_channels=16,
    norm_num_groups=32,
    sample_size=1024,
    mid_block_add_attention=True,
    use_quant_conv=False,
    use_post_quant_conv=False,
)


# --- 1. ResnetBlock vs diffusers ResnetBlock2D ------------------------------


@pytest.mark.parametrize("in_ch,out_ch", [(32, 32), (32, 64)])
@pytest.mark.parametrize("fused", [False, True])
def test_resnet_block_matches_diffusers(in_ch: int, out_ch: int, fused: bool) -> None:
    torch.manual_seed(0)
    theirs = ResnetBlock2D(
        in_channels=in_ch, out_channels=out_ch, temb_channels=None, groups=8, eps=1e-6
    ).eval()
    ours = BooguResnetBlock2D(in_channels=in_ch, out_channels=out_ch, groups=8, eps=1e-6, act_fn="silu").eval()
    ours.load_state_dict(theirs.state_dict(), strict=True)
    ours.fuse_group_norm_silu = fused

    x = torch.randn(2, in_ch, 16, 16)
    with torch.no_grad():
        assert torch.equal(ours(x), theirs(x, None))


# --- 2. Upsample / Downsample vs diffusers ----------------------------------


def test_upsample_matches_diffusers() -> None:
    torch.manual_seed(0)
    theirs = Upsample2D(32, use_conv=True, out_channels=32).eval()
    ours = BooguUpsample2D(32).eval()
    ours.load_state_dict(theirs.state_dict(), strict=True)
    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        assert torch.equal(ours(x), theirs(x))


def test_downsample_matches_diffusers() -> None:
    torch.manual_seed(0)
    # The encoder builds Downsample2D with padding=0 / name="op" (asymmetric pad).
    theirs = Downsample2D(32, use_conv=True, out_channels=32, padding=0, name="op").eval()
    ours = BooguDownsample2D(32).eval()
    ours.load_state_dict(theirs.state_dict(), strict=True)
    x = torch.randn(1, 32, 9, 9)  # odd spatial exercises the (0,1,0,1) pad
    with torch.no_grad():
        assert torch.equal(ours(x), theirs(x))


# --- 3./4. Decoder / Encoder vs diffusers -----------------------------------


def _decoder_kwargs(cfg):
    return dict(
        in_channels=cfg["latent_channels"],
        out_channels=cfg["out_channels"],
        block_out_channels=cfg["block_out_channels"],
        layers_per_block=cfg["layers_per_block"],
        norm_num_groups=cfg["norm_num_groups"],
        act_fn="silu",
        mid_block_add_attention=cfg["mid_block_add_attention"],
    )


@pytest.mark.parametrize("latent_hw", [8, 16])
@pytest.mark.parametrize("fused", [False, True])
def test_decoder_matches_diffusers(latent_hw: int, fused: bool) -> None:
    torch.manual_seed(0)
    theirs = DiffusersDecoder(up_block_types=_TINY["up_block_types"], **_decoder_kwargs(_TINY)).eval()
    ours = BooguDecoder(**_decoder_kwargs(_TINY)).eval()
    ours.load_state_dict(theirs.state_dict(), strict=True)
    for resnet in list(ours.mid_block.resnets) + [r for b in ours.up_blocks for r in b.resnets]:
        resnet.fuse_group_norm_silu = fused
    ours.fuse_conv_norm_out = fused

    z = torch.randn(1, _TINY["latent_channels"], latent_hw, latent_hw)
    with torch.no_grad():
        assert torch.equal(ours(z), theirs(z))


def test_encoder_matches_diffusers() -> None:
    torch.manual_seed(0)
    kwargs = dict(
        in_channels=_TINY["in_channels"],
        out_channels=_TINY["latent_channels"],
        block_out_channels=_TINY["block_out_channels"],
        layers_per_block=_TINY["layers_per_block"],
        norm_num_groups=_TINY["norm_num_groups"],
        act_fn="silu",
        mid_block_add_attention=_TINY["mid_block_add_attention"],
    )
    theirs = DiffusersEncoder(down_block_types=_TINY["down_block_types"], double_z=True, **kwargs).eval()
    ours = BooguEncoder(double_z=True, **kwargs).eval()
    ours.load_state_dict(theirs.state_dict(), strict=True)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(ours(x), theirs(x))


# --- 5./6. Full model: decode/encode parity + weight-load completeness ------


def _full_pair():
    torch.manual_seed(0)
    theirs = AutoencoderKL(**_TINY).eval()
    ours = BooguAutoencoderKL(**_TINY).eval()
    ours.load_state_dict(theirs.state_dict(), strict=True)
    return ours, theirs


def test_full_decode_and_encode_match_diffusers() -> None:
    ours, theirs = _full_pair()
    z = torch.randn(1, _TINY["latent_channels"], 8, 8)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(ours.decode(z, return_dict=False)[0], theirs.decode(z, return_dict=False)[0])
        ours_post = ours.encode(x).latent_dist
        theirs_post = theirs.encode(x).latent_dist
        assert torch.equal(ours_post.mean, theirs_post.mean)
        assert torch.equal(ours_post.logvar, theirs_post.logvar)


def test_weight_load_completeness_both_directions() -> None:
    # The #1 silent-bug guard: zero missing and zero unexpected keys, both ways.
    torch.manual_seed(0)
    theirs = AutoencoderKL(**_TINY)
    ours = BooguAutoencoderKL(**_TINY)
    assert set(ours.state_dict().keys()) == set(theirs.state_dict().keys())
    ours.load_state_dict(theirs.state_dict(), strict=True)
    theirs.load_state_dict(ours.state_dict(), strict=True)


# --- 7. Fusion on == off (native fallback) ----------------------------------


def test_fusion_on_equals_off_on_fallback() -> None:
    ours, _ = _full_pair()
    z = torch.randn(1, _TINY["latent_channels"], 8, 8)
    assert ours.set_group_norm_silu_fusion(True) == 13
    with torch.no_grad():
        fused = ours.decode(z, return_dict=False)[0]
    assert ours.set_group_norm_silu_fusion(False) == 0
    with torch.no_grad():
        plain = ours.decode(z, return_dict=False)[0]
    assert torch.equal(fused, plain)


# --- 8. Site counts (real architecture + tiny) -------------------------------


def test_site_counts() -> None:
    with torch.device("meta"):
        vae = BooguAutoencoderKL(**_BOOGU)
    # Decoder: 4 up blocks x 3 resnets x 2 + mid 2 x 2 + conv_norm_out = 29.
    assert vae.group_norm_silu_fusion_site_count() == 29  # armed at construction (act_fn silu)
    assert vae.set_group_norm_silu_fusion(True) == 29
    assert vae.set_group_norm_silu_fusion(False) == 0
    # Encoder is plain by construction: its blocks are never armed.
    assert all(not r.fuse_group_norm_silu for b in vae.encoder.down_blocks for r in b.resnets)
    assert all(not r.fuse_group_norm_silu for r in vae.encoder.mid_block.resnets)


# --- 9. Non-SiLU activation: fusion refuses, output == diffusers ------------


def test_non_silu_act_fn_disables_fusion_and_matches_diffusers() -> None:
    cfg = dict(_TINY, act_fn="mish")
    torch.manual_seed(0)
    theirs = AutoencoderKL(**cfg).eval()
    ours = BooguAutoencoderKL(**cfg).eval()
    ours.load_state_dict(theirs.state_dict(), strict=True)

    # Fusion can never arm for a non-SiLU activation.
    assert ours.group_norm_silu_fusion_site_count() == 0
    assert ours.set_group_norm_silu_fusion(True) == 0

    z = torch.randn(1, _TINY["latent_channels"], 8, 8)
    with torch.no_grad():
        assert torch.equal(ours.decode(z, return_dict=False)[0], theirs.decode(z, return_dict=False)[0])


# --- 10. Kill-switch resolution ----------------------------------------------


def test_kill_switch_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_VAE_GN_SILU_FUSION", raising=False)

    on_by_default = SimpleNamespace(additional_config=None)
    assert vae_group_norm_silu_fusion_enabled(on_by_default) is True

    config_off = SimpleNamespace(additional_config={"vae_group_norm_silu_fusion": False})
    assert vae_group_norm_silu_fusion_enabled(config_off) is False

    config_off_str = SimpleNamespace(additional_config={"vae_group_norm_silu_fusion": "false"})
    assert vae_group_norm_silu_fusion_enabled(config_off_str) is False

    monkeypatch.setenv("VLLM_OMNI_VAE_GN_SILU_FUSION", "0")
    assert vae_group_norm_silu_fusion_enabled(on_by_default) is False


# --- guardrails ---------------------------------------------------------------


def test_tiling_and_slicing_refuse() -> None:
    with torch.device("meta"):
        vae = BooguAutoencoderKL(**_TINY)
    with pytest.raises(NotImplementedError):
        vae.enable_tiling()
    with pytest.raises(NotImplementedError):
        vae.enable_slicing()


def test_unsupported_block_types_refuse() -> None:
    with pytest.raises(NotImplementedError):
        BooguAutoencoderKL(**dict(_TINY, up_block_types=("AttnUpDecoderBlock2D", "UpDecoderBlock2D")))


# --- strict loading -----------------------------------------------------------


def test_strict_from_pretrained_raises_on_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from diffusers.models.modeling_utils import ModelMixin

    from vllm_omni.diffusion.models.boogu_image.vae import BooguVaeStrictLoadError

    monkeypatch.setattr(
        ModelMixin,
        "from_pretrained",
        classmethod(
            lambda cls, *a, **k: (
                object(),
                {"missing_keys": ["decoder.foo"], "unexpected_keys": [], "mismatched_keys": []},
            )
        ),
    )
    with pytest.raises(BooguVaeStrictLoadError, match="decoder.foo"):
        BooguAutoencoderKL.from_pretrained("x")


def test_from_pretrained_roundtrip_ok(tmp_path) -> None:
    # Happy path: the strict override must not false-positive on a clean
    # save/load round trip.
    torch.manual_seed(0)
    vae = BooguAutoencoderKL(**_TINY)
    vae.save_pretrained(tmp_path)
    loaded = BooguAutoencoderKL.from_pretrained(tmp_path)
    src, dst = vae.state_dict(), loaded.state_dict()
    assert set(src) == set(dst)
    assert all(torch.equal(src[k], dst[k]) for k in src)
