# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the pluggable VAE fast-path layer (``vllm_omni.diffusion.vae_fast_path``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType, SimpleNamespace

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan
from torch import nn

from tests.diffusion.distributed.test_wan_vae_encoder_fastpath import CONFIG
from vllm_omni.diffusion import vae_fast_path as vfp
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import forwards as fastpath_forwards
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import (
    install_wan_vae_encoder_fastpath,
    is_encoder_installed,
    reference_forward_copy,
)
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath.entry import WAN_FAMILY_FAST_PATH

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _ToyVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Linear(2, 2)


@dataclass(frozen=True)
class _Report:
    installed: bool = True
    reason: str | None = None


@dataclass(eq=False)
class _ToyFastPath:
    name: str = "toy"
    platform_ok: bool = True
    fail: str | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def matches(self, vae: nn.Module) -> bool:
        return isinstance(vae, _ToyVae)

    def supported(self, ctx: vfp.VaeFastPathContext) -> bool:
        return self.platform_ok

    def install(self, vae, component, level, ctx):
        if component == self.fail:
            raise RuntimeError("injected failure")
        self.calls.append((component, level))
        return _Report()


@pytest.fixture
def toy_fast_path():
    fast_path = _ToyFastPath()
    vfp.register_vae_fast_path(fast_path)
    yield fast_path
    vfp.unregister_vae_fast_path(fast_path.name)


def _pipeline(vae: nn.Module) -> nn.Module:
    model = nn.Module()
    model.vae = vae
    return model


def test_wan_family_is_built_in() -> None:
    assert WAN_FAMILY_FAST_PATH in vfp.vae_fast_paths()
    assert vfp.find_vae_fast_path(AutoencoderKLWan(**CONFIG)) is WAN_FAMILY_FAST_PATH
    assert vfp.find_vae_fast_path(nn.Linear(2, 2)) is None


def test_registered_fast_path_gets_its_vae_and_both_levels(toy_fast_path) -> None:
    assert vfp.find_vae_fast_path(_ToyVae()) is toy_fast_path
    vfp.apply_vae_fast_paths(
        _pipeline(_ToyVae()),
        SimpleNamespace(vae_fast_path="channels_last", vae_encode_fast_path="lossless"),
        platform=None,
    )
    assert toy_fast_path.calls == [("decoder", "channels_last"), ("encoder", "lossless")]


@pytest.mark.parametrize(
    ("decode", "encode", "expected"),
    [
        ("off", "lossless", [("encoder", "lossless")]),
        ("lossless", "off", [("decoder", "lossless")]),
        ("off", "off", []),
    ],
)
def test_off_components_are_not_installed(toy_fast_path, decode, encode, expected) -> None:
    vfp.apply_vae_fast_paths(
        _pipeline(_ToyVae()), SimpleNamespace(vae_fast_path=decode, vae_encode_fast_path=encode), platform=None
    )
    assert toy_fast_path.calls == expected


def test_unsupported_platform_and_failures_keep_the_reference(toy_fast_path) -> None:
    config = SimpleNamespace(vae_fast_path="lossless", vae_encode_fast_path="lossless")
    toy_fast_path.platform_ok = False
    matched = []
    toy_fast_path.matches = lambda vae: matched.append(vae) or True  # type: ignore[method-assign]
    vfp.apply_vae_fast_paths(_pipeline(_ToyVae()), config, platform=None)
    assert toy_fast_path.calls == []
    # Platform support is checked before matching, which may load a fast path's families (imports).
    assert matched == []
    del toy_fast_path.matches

    toy_fast_path.platform_ok = True
    toy_fast_path.fail = "decoder"
    vfp.apply_vae_fast_paths(_pipeline(_ToyVae()), config, platform=None)
    assert toy_fast_path.calls == [("encoder", "lossless")]


def test_latest_registration_wins_and_names_replace() -> None:
    first, second = _ToyFastPath(name="toy_a"), _ToyFastPath(name="toy_b")
    try:
        vfp.register_vae_fast_path(first)
        vfp.register_vae_fast_path(second)
        assert vfp.find_vae_fast_path(_ToyVae()) is second
        replacement = _ToyFastPath(name="toy_a")
        vfp.register_vae_fast_path(replacement)
        assert not any(fp is first for fp in vfp.vae_fast_paths())
        assert vfp.find_vae_fast_path(_ToyVae()) is replacement
    finally:
        vfp.unregister_vae_fast_path("toy_a")
        vfp.unregister_vae_fast_path("toy_b")
    assert vfp.find_vae_fast_path(_ToyVae()) is None


def test_context_reads_the_stage_config() -> None:
    ctx = vfp.VaeFastPathContext.from_config(SimpleNamespace(vae_encode_fast_path="off", enforce_eager=True), "p")
    assert (ctx.level("decoder"), ctx.level("encoder"), ctx.platform, ctx.enforce_eager) == (
        "lossless",
        "off",
        "p",
        True,
    )
    with pytest.raises(ValueError, match="component"):
        ctx.level("vae")


class _Platform:
    def __init__(self, answer):
        self.answer = answer

    def supports_torch_inductor(self) -> bool:
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.mark.parametrize(
    ("enforce_eager", "platform", "expected"),
    [
        (False, _Platform(True), True),
        (True, _Platform(True), False),  # --enforce-eager vetoes before the platform is asked
        (False, _Platform(False), False),
        (False, _Platform(NotImplementedError()), False),
        (False, None, False),
    ],
)
def test_compile_allowed_precedence(enforce_eager, platform, expected) -> None:
    assert vfp.vae_compile_allowed(enforce_eager, platform) is expected
    assert vfp.VaeFastPathContext(platform=platform, enforce_eager=enforce_eager).compile_allowed() is expected


def _is_reference_forward(module: nn.Module) -> bool:
    forward = module.forward
    return isinstance(forward, MethodType) and forward.__func__ is type(module).forward


def test_reference_forward_copy_drops_bindings_and_shares_weights() -> None:
    torch.manual_seed(0)
    vae = AutoencoderKLWan(**CONFIG).eval()
    assert install_wan_vae_encoder_fastpath(vae, level="lossless").installed
    bound = [m for m in vae.encoder.modules() if fastpath_forwards.CFG_ATTR in m.__dict__]
    assert bound

    copied = reference_forward_copy(vae.encoder)

    assert all(_is_reference_forward(m) for m in copied.modules())
    assert not any(fastpath_forwards.CFG_ATTR in m.__dict__ for m in copied.modules())
    # The original keeps its installation and shares every tensor with the copy.
    assert is_encoder_installed(vae)
    assert all(fastpath_forwards.CFG_ATTR in m.__dict__ for m in bound)
    for (name, a), (_, b) in zip(vae.encoder.named_parameters(), copied.named_parameters(), strict=True):
        assert a is b, name
    assert copied.state_dict().keys() == vae.encoder.state_dict().keys()
