# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""F0 folding and joint C4/C5 loading regressions with the existing CPU F0 policy."""

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core import hifigan

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.filterwarnings("ignore:.*weight_norm.*:FutureWarning"),
]


@pytest.fixture(autouse=True)
def fixed_seed():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(5)
        yield


@pytest.fixture(params=["parametrized", "legacy"])
def weight_norm_api(request, monkeypatch):
    if request.param == "legacy":
        monkeypatch.setattr(hifigan, "weight_norm", nn.utils.weight_norm)
    return request.param


@pytest.fixture(
    params=[torch.enable_grad, torch.no_grad, torch.inference_mode], ids=["grad", "no_grad", "inference_mode"]
)
def fold_context(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=pytest.mark.cuda),
    ]
)
def runtime(request):
    """Only the generator may run on CUDA; production F0 still runs on CPU."""
    on_cuda = request.param == "cuda"
    if on_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the generator/device-placement regressions")
    with torch.random.fork_rng(devices=[0] if on_cuda else []):
        torch.manual_seed(5)
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True, allow_tf32=False):
            yield torch.device("cuda:0" if on_cuda else "cpu")


def _has_weight_norm(module):
    return parametrize.is_parametrized(module, "weight") or (
        hasattr(module, "weight_g") and hasattr(module, "weight_v")
    )


def _f0_layers(f0):
    return [layer for layer in f0.condnet if isinstance(layer, hifigan.CausalConv1d)]


def _make_hift():
    # Source noise/phase tensors are not in state_dict. Give independent
    # instances identical construction RNG without changing the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(5)
        return hifigan.CausalHiFTGenerator(
            base_channels=32,
            sampling_rate=24000,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 3, 5]],
            source_resblock_kernel_sizes=[3, 3, 3],
            source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
            f0_predictor=hifigan.CausalConvRNNF0Predictor(cond_channels=16),
        )


def _make_loader():
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

    # Real load_weights and vocoder; only the unrelated Flow is reduced.
    model = CosyVoice3Code2Wav.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.flow_model = nn.Linear(2, 2)
    model.hift = _make_hift().train()
    model._hift_window_len = 64
    return model


def _write_checkpoint(model, directory, legacy_checkpoint=False):
    state = {name: value.detach().clone() for name, value in model.hift.state_dict().items()}
    # Change conv_pre and F0 gains to detect stale constructor weights without
    # amplifying every generator layer and overflowing its magnitude exp().
    for name, value in state.items():
        if name.startswith(("conv_pre.", "f0_predictor.")) and name.endswith(
            ("weight_g", "parametrizations.weight.original0")
        ):
            value.mul_(2)
    if legacy_checkpoint:
        state = {
            name.replace("parametrizations.weight.original0", "weight_g").replace(
                "parametrizations.weight.original1", "weight_v"
            ): value
            for name, value in state.items()
        }
    torch.save(model.flow_model.state_dict(), directory / "flow.pt")
    torch.save({"generator." + name: value for name, value in state.items()}, directory / "hift.pt")
    return state


@pytest.mark.cpu
@pytest.mark.parametrize("finalize", [False, True])
def test_f0_fold_is_exact_and_idempotent(weight_norm_api, finalize):
    f0 = hifigan.CausalConvRNNF0Predictor(cond_channels=16).eval()
    layers = _f0_layers(f0)
    assert len(layers) == 5 and all(_has_weight_norm(layer) for layer in layers)
    assert hasattr(layers[0], "weight_g") == (weight_norm_api == "legacy")
    classifier = {name: value.detach().clone() for name, value in f0.classifier.state_dict().items()}
    context = f0.left_context_frames
    mel = torch.randn(1, 80, 24)
    with torch.no_grad():
        before = f0(mel, finalize=finalize)
    assert f0.remove_weight_norm() == 5
    weights = [layer.weight for layer in layers]
    assert f0.remove_weight_norm() == 0
    assert all(layer.weight is weight for layer, weight in zip(layers, weights))
    assert not any(_has_weight_norm(module) for module in f0.modules())
    assert f0.left_context_frames == context
    with torch.no_grad():
        after = f0(mel, finalize=finalize)
    assert before.numel() > 0 and torch.isfinite(before).all()
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    for name, value in f0.classifier.state_dict().items():
        torch.testing.assert_close(value, classifier[name], rtol=0, atol=0)


@pytest.mark.cpu
def test_f0_fold_preserves_autograd_usability(weight_norm_api, fold_context):
    f0 = hifigan.CausalConvRNNF0Predictor(cond_channels=16).eval()
    mel = torch.randn(1, 80, 24, requires_grad=True)
    with torch.no_grad():
        before = f0(mel)
    with fold_context():
        assert f0.remove_weight_norm() == 5
        assert f0.remove_weight_norm() == 0
    for layer in _f0_layers(f0):
        assert isinstance(layer.weight, nn.Parameter)
        assert not layer.weight.is_inference()
    out = f0(mel)
    torch.testing.assert_close(out, before, rtol=0, atol=0)
    out.sum().backward()
    assert mel.grad is not None and torch.isfinite(mel.grad).all()


@pytest.mark.cpu
def test_f0_fold_leaves_generator_norms_unchanged(weight_norm_api):
    hift = _make_hift()
    generator_norms = {
        name
        for name, module in hift.named_modules()
        if not name.startswith("f0_predictor.") and _has_weight_norm(module)
    }
    assert len(generator_norms) == 41
    assert sum(_has_weight_norm(module) for module in hift.f0_predictor.modules()) == 5
    assert hift.f0_predictor.remove_weight_norm() == 5
    assert {name for name, module in hift.named_modules() if _has_weight_norm(module)} == generator_norms


@pytest.mark.parametrize(
    "weight_norm_api,legacy_checkpoint",
    [("parametrized", False), ("parametrized", True), ("legacy", True)],
    indirect=["weight_norm_api"],
)
def test_load_folds_loaded_generator_and_f0_weights(
    tmp_path, weight_norm_api, legacy_checkpoint, runtime, fold_context
):
    device = runtime
    model = _make_loader()
    assert model.hift.training and model.hift.f0_predictor.training and model.flow_model.training
    assert hasattr(model.hift.f0_predictor.condnet[0], "weight_g") == (weight_norm_api == "legacy")
    state = _write_checkpoint(model, tmp_path, legacy_checkpoint)
    # Do not forward the loader instance before loading. A separate unfolded
    # reference refreshes legacy cached weights from the changed checkpoint.
    reference = _make_hift()
    reference.load_state_dict(state, strict=True)
    reference.to(device).eval()
    reference.f0_predictor.to(device="cpu", dtype=torch.float32)
    normalized = {name: module for name, module in reference.named_modules() if _has_weight_norm(module)}
    assert len(normalized) == 46  # 41 reduced-fixture generator layers + 5 F0
    mel = torch.randn(1, 80, 24, device=device)
    expected = reference.inference(mel, finalize=True)[0]

    with fold_context():
        model.load_weights(str(tmp_path), device)

    assert not model.hift.training and not model.hift.f0_predictor.training and not model.flow_model.training
    assert not any(_has_weight_norm(module) for module in model.hift.modules())
    for name, ref_layer in normalized.items():
        layer = model.hift.get_submodule(name)
        assert isinstance(layer.weight, nn.Parameter) and not layer.weight.is_inference()
        assert layer.weight.device == (torch.device("cpu") if name.startswith("f0_predictor.") else device)
        assert layer.weight.dtype == torch.float32
        torch.testing.assert_close(layer.weight, ref_layer.weight, rtol=0, atol=0)
    assert {p.device for p in model.hift.f0_predictor.parameters()} == {torch.device("cpu")}
    assert {p.dtype for p in model.hift.f0_predictor.parameters()} == {torch.float32}
    assert model.hift.remove_weight_norm() == 0
    assert model.hift.f0_predictor.remove_weight_norm() == 0
    actual = model.hift.inference(mel, finalize=True)[0]
    assert expected.numel() > 0 and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_invalid_checkpoint_does_not_fold_either_component(tmp_path, weight_norm_api, monkeypatch):
    model = _make_loader()
    state = _write_checkpoint(model, tmp_path)
    state.pop("f0_predictor.classifier.bias")
    torch.save({"generator." + name: value for name, value in state.items()}, tmp_path / "hift.pt")

    def unexpected_fold(*args, **kwargs):
        pytest.fail("Folding must not run before strict checkpoint validation")

    monkeypatch.setattr(model.hift, "remove_weight_norm", unexpected_fold)
    monkeypatch.setattr(model.hift.f0_predictor, "remove_weight_norm", unexpected_fold)
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_weights(str(tmp_path), torch.device("cpu"))
    assert sum(_has_weight_norm(module) for module in model.hift.modules()) == 46


@pytest.mark.parametrize("reference_fold_generator", [False, True], ids=["unfolded", "c4_only"])
def test_joint_fold_preserves_streaming_audio_and_caches(tmp_path, weight_norm_api, runtime, reference_fold_generator):
    device = runtime
    model = _make_loader()
    state = _write_checkpoint(model, tmp_path)
    reference = _make_loader()
    reference.hift.load_state_dict(state, strict=True)
    reference.hift.to(device).eval()
    reference.hift.f0_predictor.to(device="cpu", dtype=torch.float32)
    # Compare both against no folding and against C4 alone, isolating C5's
    # incremental effect while exercising the real loader/streaming path.
    if reference_fold_generator:
        assert reference.hift.remove_weight_norm() == 41
    model.load_weights(str(tmp_path), device)
    assert not any(_has_weight_norm(module) for module in model.hift.modules())
    expected_norms = 5 if reference_fold_generator else 46
    assert sum(_has_weight_norm(module) for module in reference.hift.modules()) == expected_norms
    chunks = torch.randn(1, 80, 144, device=device).split(24, dim=-1)
    cache = ref_cache = None
    shifted = False
    for i, chunk in enumerate(chunks):
        finalize = i == len(chunks) - 1
        expected, ref_cache = reference._stream_hift_from_feat(chunk, cache_state=ref_cache, finalize=finalize)
        actual, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=finalize)
        assert expected.numel() > 0 and torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if finalize:
            assert cache is None and ref_cache is None
        else:
            assert cache.keys() == ref_cache.keys()
            for key, value in ref_cache.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(cache[key], value, rtol=0, atol=0)
                else:
                    assert cache[key] == value
            shifted |= cache["mel_offset"] > 0
    assert shifted


@pytest.mark.cpu
def test_shipped_f0_config_folds_exactly_five_layers(weight_norm_api):
    from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

    f0 = hifigan.CausalConvRNNF0Predictor(**CosyVoice3Config().hift["f0_predictor"])
    assert sum(_has_weight_norm(module) for module in f0.modules()) == 5
    assert f0.remove_weight_norm() == 5
    assert f0.remove_weight_norm() == 0
    assert not any(_has_weight_norm(module) for module in f0.modules())
