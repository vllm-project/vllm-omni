# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the family-based Wan VAE fast path: Qwen-Image and Qwen-Image-2.1 share the Wan forwards."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLQwenImage, AutoencoderKLWan
from torch import nn

from tests.diffusion.distributed.wan_vae_fastpath_helpers import (  # noqa: F401
    original_wan_rms_norm,
    unpatched_wan_rms_norm,
)
from vllm_omni.diffusion import registry as registry_module
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    ENCODER_REPORT_ATTR,
    REPORT_ATTR,
    find_family,
    install_wan_vae_encoder_fastpath,
    install_wan_vae_fastpath,
    register_family,
    uninstall_wan_vae_encoder_fastpath,
    uninstall_wan_vae_fastpath,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import families as fam
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import forwards as fp
from vllm_omni.diffusion.models.qwen_image.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage as OmniAutoencoderKLQwenImage,
)
from vllm_omni.diffusion.models.qwen_image_21.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.diffusion,
    pytest.mark.usefixtures("unpatched_wan_rms_norm"),
]


TINY_QWEN_IMAGE = dict(
    base_dim=8,
    z_dim=4,
    dim_mult=[1, 2],
    num_res_blocks=1,
    temperal_downsample=[False, True],
    latents_mean=[0.0] * 4,
    latents_std=[1.0] * 4,
)


def _qwen_image(dtype: torch.dtype) -> nn.Module:
    return AutoencoderKLQwenImage(**TINY_QWEN_IMAGE)


def _qwen_image_layered(dtype: torch.dtype) -> nn.Module:
    return OmniAutoencoderKLQwenImage(**TINY_QWEN_IMAGE)


def _qwen_image_21(dtype: torch.dtype) -> nn.Module:
    return AutoencoderKLQwenImage21(
        base_dim=8,
        decoder_base_dim=12,
        z_dim=4,
        dim_mult=[1, 2, 2],
        num_res_blocks=1,
        temperal_downsample=[False, False],
        is_residual=True,
        in_channels=4,
        out_channels=4,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
        scale_factor_spatial=None,
        scale_factor_temporal=None,
    )


# name -> (builder, image channels, spatial compression, latent frames)
BUILDERS = {
    "qwen_image": (_qwen_image, 3, 2, 2),
    "qwen_image_layered": (_qwen_image_layered, 3, 2, 2),
    "qwen_image_21": (_qwen_image_21, 4, 4, 1),
}


def _build_pair(name: str, dtype: torch.dtype) -> tuple[nn.Module, nn.Module]:
    builder = BUILDERS[name][0]
    torch.manual_seed(0)
    reference = builder(dtype)
    with torch.no_grad():
        for module in reference.modules():
            if type(module).__name__.endswith("RMS_norm"):
                module.gamma.uniform_(0.5, 1.5)  # away from the all-ones init, so a dropped gamma shows
    reference = reference.eval().to(dtype)
    candidate = builder(dtype).eval().to(dtype)
    candidate.load_state_dict(reference.state_dict())
    return reference, candidate


@pytest.fixture(params=[False, True], ids=["fallbacks", "kernel_paths"])
def kernel_paths(request, monkeypatch):
    """``kernel_paths`` runs the fast-path forwards on CPU, where each device kernel declines to its exact fallback."""
    if request.param:
        monkeypatch.setattr(fp, "_kernels_allowed", lambda x: not torch.is_grad_enabled())
        # CPU convolutions add the bias inside the accumulator, like oneDNN on XPU, so follow the XPU rule: defer
        # a conv bias to the next kernel only under the tolerance-based channels_last level.
        monkeypatch.setattr(
            fp,
            "_may_defer_conv_bias",
            lambda owner, x: getattr(getattr(owner, fp.CFG_ATTR, None), "channels_last", False),
        )
    return request.param


def _force_fused_silu(vae: nn.Module, dtype: torch.dtype) -> None:
    # The installer enables the fused SiLU epilogue only where it is proven exact on a kernel device.
    for module in vae.modules():
        cfg = getattr(module, fp.CFG_ATTR, None)
        if cfg is not None:
            setattr(module, fp.CFG_ATTR, replace(cfg, fused_silu_dtypes=frozenset({dtype})))


def test_families_resolve_by_vae_class():
    assert find_family(AutoencoderKLWan(base_dim=8, z_dim=4, dim_mult=[1, 1])).name == "wan"
    assert find_family(_qwen_image(torch.float32)).name == "qwen_image"
    assert find_family(_qwen_image_layered(torch.float32)).name == "qwen_image_layered"
    assert find_family(_qwen_image_21(torch.float32)).name == "qwen_image_21"
    assert find_family(nn.Linear(2, 2)) is None


@pytest.mark.parametrize("name", sorted(BUILDERS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lossless_decode_and_encode_are_bitwise_exact(name: str, dtype: torch.dtype, kernel_paths: bool) -> None:
    _, channels, compression, frames = BUILDERS[name]
    reference, candidate = _build_pair(name, dtype)
    decoder = install_wan_vae_fastpath(candidate, level="lossless")
    encoder = install_wan_vae_encoder_fastpath(candidate, level="lossless")
    assert decoder.installed and decoder.family == name, decoder
    assert encoder.installed and encoder.family == name, encoder
    if kernel_paths:
        _force_fused_silu(candidate, dtype)
    torch.manual_seed(1)
    latents = torch.randn(1, 4, frames, 6, 8).to(dtype)
    pixel_frames = 1 if frames == 1 else 1 + 4 * (frames - 1)
    image = torch.randn(1, channels, pixel_frames, 6 * compression, 8 * compression).to(dtype)
    with torch.no_grad():
        expected = reference.decode(latents, return_dict=False)[0]
        actual = candidate.decode(latents, return_dict=False)[0]
        expected_z = reference.encode(image).latent_dist.mode()
        actual_z = candidate.encode(image).latent_dist.mode()
    assert torch.equal(actual, expected)
    assert torch.equal(actual_z, expected_z)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_lossless_binds_the_family_roles(name: str) -> None:
    _, candidate = _build_pair(name, torch.float32)
    patched = install_wan_vae_fastpath(candidate, level="lossless").patched
    family = find_family(candidate)
    for cls in (family.decoder, family.causal_conv, family.residual_block, family.resample, family.upsample):
        assert patched.get(cls.__name__, 0) > 0, (cls.__name__, patched)
    if family.rms_norm is not None:
        assert family.rms_norm.__name__ in patched
    else:  # its norm has other numerics and keeps its own forward
        assert not any(name.endswith("RMS_norm") for name in patched)
    assert family.attention_block.__name__ not in patched  # chunked attention is channels_last only


def test_qwen_image_decoder_does_not_pass_first_chunk(kernel_paths: bool) -> None:
    # QwenImageUpBlock.forward takes no ``first_chunk``; the decode above would raise TypeError if it were passed.
    _, candidate = _build_pair("qwen_image", torch.float32)
    install_wan_vae_fastpath(candidate, level="lossless")
    assert not getattr(candidate.decoder, fp.CFG_ATTR).up_blocks_take_first_chunk
    with torch.no_grad():
        candidate.decode(torch.randn(1, 4, 2, 4, 4), return_dict=False)


def test_single_frame_conv_probes_padding_once_per_shape(monkeypatch, kernel_paths: bool) -> None:
    if not kernel_paths:
        pytest.skip("the probe runs on the kernel path only")
    _, candidate = _build_pair("qwen_image_21", torch.float32)
    install_wan_vae_fastpath(candidate, level="lossless")
    calls = []
    real = fp._equal_on_random_operands

    def counting(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(fp, "_equal_on_random_operands", counting)
    conv = candidate.decoder.conv_in
    x = torch.randn(1, conv.in_channels, 1, 6, 8)
    with torch.no_grad():
        conv(x)
        conv(x)
        monkeypatch.setattr(fp, "_equal_on_random_operands", lambda *a: False)
        conv(torch.randn(1, conv.in_channels, 1, 5, 7))  # new shape, declined verdict: pre-padded input
    assert len(calls) == 1


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_channels_last_is_close(name: str, kernel_paths: bool) -> None:
    _, channels, compression, frames = BUILDERS[name]
    reference, candidate = _build_pair(name, torch.float32)
    decoder = install_wan_vae_fastpath(candidate, level="channels_last")
    install_wan_vae_encoder_fastpath(candidate, level="channels_last")
    family = find_family(candidate)
    if family.chunked_attention:
        assert decoder.patched[family.attention_block.__name__] == 1
    torch.manual_seed(1)
    latents = torch.randn(1, 4, frames, 6, 8)
    pixel_frames = 1 if frames == 1 else 1 + 4 * (frames - 1)
    image = torch.randn(1, channels, pixel_frames, 6 * compression, 8 * compression)
    with torch.no_grad():
        torch.testing.assert_close(
            candidate.decode(latents, return_dict=False)[0],
            reference.decode(latents, return_dict=False)[0],
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            candidate.encode(image).latent_dist.mode(),
            reference.encode(image).latent_dist.mode(),
            atol=1e-4,
            rtol=1e-4,
        )


def test_chunked_attention_matches_sdpa() -> None:
    q, k, v = (torch.randn(1, 1, 37, 16) for _ in range(3))
    torch.testing.assert_close(
        fp._chunked_attention(q, k, v, rows=8),
        nn.functional.scaled_dot_product_attention(q, k, v),
        atol=1e-5,
        rtol=1e-5,
    )


def test_chunked_attention_fp16_scores_do_not_overflow() -> None:
    # Unscaled, every score is 384 * 20 * 20 = 153600, beyond the fp16 maximum of 65504.
    q = torch.full((1, 1, 16, 384), 20.0)
    k = q.clone()
    v = torch.randn(1, 1, 16, 384)
    out = fp._chunked_attention(q.half(), k.half(), v.half(), rows=8)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), nn.functional.scaled_dot_product_attention(q, k, v), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_uninstall_restores_forwards_and_layouts(name: str) -> None:
    _, candidate = _build_pair(name, torch.float32)
    conv = candidate.decoder.conv_in
    install_wan_vae_fastpath(candidate, level="channels_last")
    install_wan_vae_encoder_fastpath(candidate, level="channels_last")
    uninstall_wan_vae_fastpath(candidate)
    uninstall_wan_vae_encoder_fastpath(candidate)
    assert REPORT_ATTR not in candidate.__dict__ and ENCODER_REPORT_ATTR not in candidate.__dict__
    for module in candidate.modules():
        assert "forward" not in module.__dict__ and fp.CFG_ATTR not in module.__dict__
    assert conv.weight.is_contiguous()


def test_custom_forward_is_refused() -> None:
    _, candidate = _build_pair("qwen_image_21", torch.float32)
    candidate.decoder.conv_in.forward = lambda x, cache_x=None: x
    report = install_wan_vae_fastpath(candidate, level="lossless")
    assert not report.installed and "custom forward" in report.reason


def test_registering_a_family_is_all_a_new_implementation_needs(monkeypatch) -> None:
    layered = next(f for f in fam.families() if f.name == "qwen_image_layered")
    monkeypatch.setattr(fam, "_FAMILIES", [f for f in fam._FAMILIES if f is not layered])
    fam._role_types.cache_clear()
    reference, candidate = _build_pair("qwen_image_layered", torch.float32)
    assert not install_wan_vae_fastpath(candidate, level="lossless").installed
    register_family(layered)
    assert install_wan_vae_fastpath(candidate, level="lossless").installed
    latents = torch.randn(1, 4, 2, 6, 8)
    with torch.no_grad():
        assert torch.equal(
            candidate.decode(latents, return_dict=False)[0], reference.decode(latents, return_dict=False)[0]
        )
    fam._role_types.cache_clear()


def test_registry_installs_on_qwen_vaes(monkeypatch) -> None:
    monkeypatch.setattr(registry_module.current_omni_platform, "is_xpu", lambda: True)
    for name in sorted(BUILDERS):
        model = nn.Module()
        model.vae = _build_pair(name, torch.float32)[1]
        registry_module._apply_wan_vae_fastpath_if_enabled(
            model, SimpleNamespace(vae_fast_path="lossless", vae_encode_fast_path="channels_last")
        )
        assert getattr(model.vae, REPORT_ATTR).family == name
        assert getattr(model.vae, ENCODER_REPORT_ATTR).level == "channels_last"


def test_most_specific_family_wins_and_lazy_families_keep_earlier_registrations(monkeypatch) -> None:
    class MyQwenVae(AutoencoderKLQwenImage):
        pass

    builtin = next(f for f in fam.families() if f.name == "qwen_image")
    mine = replace(builtin, name="my_qwen", vaes=(MyQwenVae,))
    monkeypatch.setattr(fam, "_FAMILIES", list(fam._FAMILIES))
    register_family(mine)
    # Registered after `qwen_image` but for a subclass: an instance of the subclass gets it either way.
    assert fam.find_family(MyQwenVae(**TINY_QWEN_IMAGE)) is mine
    assert fam.find_family(AutoencoderKLQwenImage(**TINY_QWEN_IMAGE)) is builtin
    monkeypatch.setattr(fam, "_FAMILIES", [mine, *(f for f in fam._FAMILIES if f is not mine)])
    assert fam.find_family(MyQwenVae(**TINY_QWEN_IMAGE)) is mine  # registration order does not matter

    # A family registered before the lazy built-ins load is not replaced by the built-in of the same name.
    users = replace(builtin, vaes=(MyQwenVae,))
    monkeypatch.setattr(fam, "_FAMILIES", [f for f in fam._FAMILIES if f.name in ("wan",)])
    monkeypatch.setattr(fam, "_LAZY_LOADED", False)
    register_family(users)
    assert [f for f in fam.families() if f.name == "qwen_image"] == [users]

    # A differently named family for a built-in's class, registered before first use, is not shadowed by it.
    other = replace(builtin, name="my_other_qwen")
    monkeypatch.setattr(fam, "_FAMILIES", [f for f in fam._FAMILIES if f.name in ("wan",)])
    monkeypatch.setattr(fam, "_LAZY_LOADED", False)
    register_family(other)
    assert fam.find_family(AutoencoderKLQwenImage(**TINY_QWEN_IMAGE)) is other
    fam._role_types.cache_clear()


def test_fp32_channel_norm_is_the_upcast_reduction_and_probes_once(monkeypatch) -> None:
    from vllm_omni.diffusion.layers import vae_rms_norm as vrn

    monkeypatch.setattr(vrn, "_FP32_NORM_VERDICTS", {})
    x = torch.randn(1, 96, 2, 5, 7).to(torch.bfloat16)
    expected = x.float().norm(2, 1, keepdim=True)
    probes = []
    probe = vrn._fp32_norm_probe

    def counting_probe(t):
        probes.append(True)
        return probe(t)

    monkeypatch.setattr(vrn, "_fp32_norm_probe", counting_probe)
    for _ in range(2):
        assert torch.equal(vrn.fp32_channel_norm(x).view(torch.int32), expected.view(torch.int32))
    assert len(probes) == 1  # one verdict per (device, dtype, shape, layout)

    # A backend whose direct reduction differs keeps the upcast; a probe that does not fit records nothing.
    monkeypatch.setattr(vrn, "_FP32_NORM_VERDICTS", {})
    monkeypatch.setattr(vrn, "_fp32_norm_probe", lambda t: False)
    assert torch.equal(vrn.fp32_channel_norm(x).view(torch.int32), expected.view(torch.int32))
    monkeypatch.setattr(vrn, "_FP32_NORM_VERDICTS", {})
    monkeypatch.setattr(vrn, "_fp32_norm_probe", lambda t: None)
    assert torch.equal(vrn.fp32_channel_norm(x).view(torch.int32), expected.view(torch.int32))
    assert not vrn._FP32_NORM_VERDICTS


def test_fp32_channel_norm_traces_under_fullgraph_compile(monkeypatch) -> None:
    from vllm_omni.diffusion.layers import vae_rms_norm as vrn

    monkeypatch.setattr(vrn, "_FP32_NORM_VERDICTS", {})  # a shape no eager call has probed
    x = torch.randn(1, 96, 2, 5, 7).to(torch.bfloat16)
    compiled = torch.compile(vrn.fp32_channel_norm, fullgraph=True, backend="eager")
    out = compiled(x)
    assert torch.equal(out.view(torch.int32), x.float().norm(2, 1, keepdim=True).view(torch.int32))
    assert not vrn._FP32_NORM_VERDICTS  # nothing probed while tracing
