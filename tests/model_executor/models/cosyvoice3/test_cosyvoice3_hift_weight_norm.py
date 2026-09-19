# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for folding frozen HiFT generator weight normalization."""

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core import hifigan

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    # The legacy weight_norm API warns on construction; the "legacy" params
    # below exercise it on purpose.
    pytest.mark.filterwarnings("ignore:.*weight_norm.*:FutureWarning"),
]


@pytest.fixture(autouse=True)
def fixed_seed():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


@pytest.fixture(params=["parametrized", "legacy"])
def weight_norm_api(request, monkeypatch):
    if request.param == "legacy":
        monkeypatch.setattr(hifigan, "weight_norm", nn.utils.weight_norm)
    return request.param


def _has_weight_norm(module):
    return parametrize.is_parametrized(module, "weight") or (
        hasattr(module, "weight_g") and hasattr(module, "weight_v")
    )


def _make_hift(causal=True, sampling_rate=24000):
    # Note: pairing the non-causal generator with CausalConvRNNF0Predictor is
    # not a shipped configuration; it exists only to exercise the shared
    # ResBlock/conv fold code on both generator classes.
    cls = hifigan.CausalHiFTGenerator if causal else hifigan.HiFTGenerator
    return cls(
        base_channels=32,
        sampling_rate=sampling_rate,
        # Keep the real CosyVoice3 upsample ratio and SineGen2 (24 kHz),
        # while shrinking channels and residual blocks for CPU tests.
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]],
        source_resblock_kernel_sizes=[3, 3, 3],
        source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
        f0_predictor=hifigan.CausalConvRNNF0Predictor(cond_channels=16),
    ).eval()


@pytest.mark.parametrize("causal", [False, True])
def test_resblock_fold_is_exact_and_idempotent(weight_norm_api, causal):
    block = hifigan.ResBlock(channels=4, causal=causal).eval()
    x = torch.randn(1, 4, 12)
    conv = block.convs1[0]
    # Guard: the fixture really selected the intended API, so the legacy
    # cases cannot silently re-test the parametrizations path.
    if weight_norm_api == "legacy":
        assert hasattr(conv, "weight_g") and not parametrize.is_parametrized(conv, "weight")
    else:
        assert parametrize.is_parametrized(conv, "weight")

    with torch.no_grad():
        before = block(x)
    # Fold in normal grad mode, like the production load path (vLLM's loader
    # does not wrap load_weights in no_grad/inference_mode): the materialized
    # weight must be a real Parameter.
    first = block.remove_weight_norm()
    assert isinstance(conv.weight, nn.Parameter)
    second = block.remove_weight_norm()
    with torch.no_grad():
        once = block(x)
        twice = block(x)

    assert (first, second) == (6, 0)  # 3 dilations x {convs1, convs2}; no-op after
    assert not any(_has_weight_norm(module) for module in block.modules())
    torch.testing.assert_close(once, before, rtol=0, atol=0)
    torch.testing.assert_close(twice, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "causal,sampling_rate,finalize",
    [(False, 22050, True), (True, 22050, True), (True, 24000, True), (True, 24000, False)],
)
def test_hift_fold_preserves_audio_and_leaves_f0_unchanged(weight_norm_api, causal, sampling_rate, finalize):
    hift = _make_hift(causal, sampling_rate)
    mel = torch.randn(1, 80, 20)
    if weight_norm_api == "legacy":
        assert hasattr(hift.conv_pre, "weight_g") and not parametrize.is_parametrized(hift.conv_pre, "weight")
    normalized = {name: module for name, module in hift.named_modules() if _has_weight_norm(module)}
    f0_state = {name: value.clone() for name, value in hift.f0_predictor.state_dict().items()}
    assert any(name.startswith("source_resblocks.") for name in normalized)
    assert any(name.startswith("resblocks.") for name in normalized)
    # The C5-scope layers must start out weight-normed, or the "f0 untouched"
    # asserts below would pass vacuously.
    assert any(name.startswith("f0_predictor.") for name in normalized)

    # Non-causal HiFT draws fresh source noise on each invocation.
    rng = torch.get_rng_state()
    kwargs = {"finalize": finalize} if causal else {}
    before = hift.inference(mel, **kwargs)[0]
    # 3 ups + conv_pre + conv_post + (3 resblocks + 3 source resblocks) x 6 convs
    assert hift.remove_weight_norm() == 41
    assert hift.remove_weight_norm() == 0
    torch.set_rng_state(rng)
    after = hift.inference(mel, **kwargs)[0]

    assert before.numel() > 0
    assert torch.isfinite(before).all()
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    for name, module in normalized.items():
        # F0 is a separate RFC workstream; do not silently fold it here.
        assert _has_weight_norm(module) == name.startswith("f0_predictor.")
    for name, value in hift.f0_predictor.state_dict().items():
        torch.testing.assert_close(value, f0_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("legacy_checkpoint", [False, True])
def test_load_weights_folds_loaded_not_initial_weights(tmp_path, legacy_checkpoint):
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    # Exercise the real loader with a small vocoder and synthetic checkpoints.
    # __new__ skips CosyVoice3Code2Wav.__init__ (which would build the DiT and
    # flow decoder); load_weights only touches .flow_model and .hift.
    model = CosyVoice3Code2Wav.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.flow_model = nn.Linear(2, 2)
    model.hift = _make_hift()
    state = {name: value.clone() for name, value in model.hift.state_dict().items()}
    state["conv_pre.parametrizations.weight.original0"].mul_(2)
    expected_weight = model.hift.conv_pre.weight.detach().clone() * 2
    flow_state = {name: torch.ones_like(value) for name, value in model.flow_model.state_dict().items()}
    if legacy_checkpoint:
        state = {
            name.replace("parametrizations.weight.original0", "weight_g").replace(
                "parametrizations.weight.original1", "weight_v"
            ): value
            for name, value in state.items()
        }
    torch.save(flow_state, tmp_path / "flow.pt")
    torch.save({"generator." + name: value for name, value in state.items()}, tmp_path / "hift.pt")

    model.load_weights(str(tmp_path), torch.device("cpu"))

    torch.testing.assert_close(model.hift.conv_pre.weight, expected_weight, rtol=0, atol=0)
    # Production loads in normal grad mode, so the folded weight is a Parameter.
    assert isinstance(model.hift.conv_pre.weight, nn.Parameter)
    assert not any(
        _has_weight_norm(module) for name, module in model.hift.named_modules() if not name.startswith("f0_predictor.")
    )
    assert _has_weight_norm(model.hift.f0_predictor.condnet[0])
    assert not model.hift.training
    assert not model.flow_model.training
    for name, value in model.flow_model.state_dict().items():
        torch.testing.assert_close(value, flow_state[name], rtol=0, atol=0)


def test_fold_under_inference_mode_stays_usable_outside_it():
    """Folding inside inference_mode must not leak inference tensors: the
    materialized weight stays a frozen Parameter and forward keeps working
    in a normal/no_grad context (pre-hardening this raised RuntimeError)."""
    block = hifigan.ResBlock(channels=4).eval()
    x = torch.randn(1, 4, 12)
    with torch.inference_mode():
        assert block.remove_weight_norm() == 6
    assert isinstance(block.convs1[0].weight, nn.Parameter)
    assert not block.convs1[0].weight.requires_grad
    with torch.no_grad():
        out = block(x)
    assert torch.isfinite(out).all()


def test_shipped_config_generator_folds_exactly_77_layers():
    """Pin the RFC #6870 C4 count against the shipped config, so silent
    config drift (or a fold that quietly stops covering layers) shows up
    here: conv_pre 1 + ups 3 + 9 resblocks x 6 + 3 source resblocks x 6
    + conv_post 1 = 77. Construction only (no forward), so the full
    512-channel model stays cheap on CPU."""
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    hift_cfg = dict(CosyVoice3Config().hift)
    f0_cfg = hift_cfg.pop("f0_predictor")
    f0 = hifigan.CausalConvRNNF0Predictor(
        num_class=f0_cfg["num_class"],
        in_channels=f0_cfg["in_channels"],
        cond_channels=f0_cfg["cond_channels"],
    )
    generator = hifigan.CausalHiFTGenerator(f0_predictor=f0, **hift_cfg)

    assert generator.remove_weight_norm() == 77
    assert generator.remove_weight_norm() == 0
    remaining = [name for name, m in generator.named_modules() if _has_weight_norm(m)]
    assert len(remaining) == 5
    assert all(name.startswith("f0_predictor.") for name in remaining)
