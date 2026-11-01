# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram
from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import _build_hift

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

flashcosyvoice_hifigan = pytest.importorskip("flashcosyvoice.modules.hifigan")
flashcosyvoice_audio = pytest.importorskip("flashcosyvoice.utils.audio")


def test_vendored_hift_matches_flashcosyvoice() -> None:
    reference = flashcosyvoice_hifigan.HiFTGenerator().eval()
    vendored = _build_hift().eval()

    # Strict loading verifies that the vendored architecture remains compatible
    # with the Step-Audio2 checkpoint produced for flashcosyvoice.
    vendored.load_state_dict(reference.state_dict(), strict=True)

    speech_feat = torch.randn(1, 80, 4)
    cache_source = torch.zeros(1, 1, 0)
    with torch.inference_mode():
        torch.manual_seed(0)
        expected_speech, expected_source = reference(speech_feat, cache_source)
        torch.manual_seed(0)
        actual_speech, actual_source = vendored.inference(speech_feat, cache_source)

    torch.testing.assert_close(actual_source, expected_source)
    torch.testing.assert_close(actual_speech, expected_speech)


def test_vendored_mel_spectrogram_matches_flashcosyvoice() -> None:
    reference = flashcosyvoice_audio.mel_spectrogram
    vendored = mel_spectrogram

    audio = torch.randn(1, 48000)
    kwargs = {
        "n_fft": 1920,
        "num_mels": 80,
        "sampling_rate": 24000,
        "hop_size": 480,
        "win_size": 1920,
        "fmin": 0,
        "fmax": 8000,
    }
    with torch.inference_mode():
        expected_spec = reference(audio, **kwargs)
        actual_spec = vendored(audio, **kwargs)

    torch.testing.assert_close(actual_spec, expected_spec)


def test_weight_norm_folding_is_exact_and_idempotent() -> None:
    """``StepAudio2Token2WavCore`` folds HiFT weight norms at load time.

    Folding must not change the generator output: it only rewrites
    ``weight = weight_g * weight_v / ||weight_v||`` into a plain weight. It must
    also be a no-op on a second call, so a reloaded core cannot silently
    re-fold an already folded checkpoint.
    """
    reference = _build_hift().eval()
    folded = _build_hift().eval()
    folded.load_state_dict(reference.state_dict(), strict=True)

    # The deterministic generator subpath (no SineGen phase noise).
    speech_feat = torch.randn(1, 80, 6)
    with torch.inference_mode():
        f0 = reference.f0_predictor(speech_feat)
        source = reference.m_source(reference.f0_upsamp(f0[:, None]).transpose(1, 2))[0].transpose(1, 2)
        expected = reference._decode_pre_istft(speech_feat, source)

        assert folded.remove_weight_norm() > 0
        actual = folded._decode_pre_istft(speech_feat, source)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)

    # Already folded: nothing left to remove.
    assert folded.remove_weight_norm() == 0


def _has_weight_norm(module: nn.Module) -> bool:
    return parametrize.is_parametrized(module, "weight") or (
        hasattr(module, "weight_g") and hasattr(module, "weight_v")
    )


def test_ensure_models_loaded_folds_loaded_hift_weights(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_ensure_models_loaded`` must fold the checkpoint after it is on device.

    The parity test above calls ``HiFTGenerator.remove_weight_norm`` itself, so
    deleting the loader's fold would leave that test green.
    """
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator
    from vllm_omni.model_executor.models.step_audio2 import step_audio2_token2wav as token2wav

    source = _build_hift().eval()
    conv = source.conv_pre
    state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    if hasattr(conv, "weight_g"):
        gain_key = "conv_pre.weight_g"
    else:
        assert parametrize.is_parametrized(conv, "weight")
        gain_key = "conv_pre.parametrizations.weight.original0"
    state[gain_key].mul_(2)
    expected_weight = conv.weight.detach().clone() * 2
    assert not torch.equal(conv.weight, expected_weight)

    flow = nn.Linear(2, 2)
    flow_state = {name: torch.ones_like(value) for name, value in flow.state_dict().items()}
    torch.save(flow_state, tmp_path / "flow.pt")
    torch.save({"generator." + name: value for name, value in state.items()}, tmp_path / "hift.pt")
    (tmp_path / "flow.yaml").write_text("flow: unused\n")

    class _Tokenizer:
        encoder = None

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(token2wav.s3tokenizer, "load_model", lambda path: _Tokenizer())
    monkeypatch.setattr(token2wav, "load_hyperpyyaml", lambda stream: {"flow": flow})

    events: list[str] = []
    orig_load = HiFTGenerator.load_state_dict
    orig_to = HiFTGenerator.to
    orig_fold = HiFTGenerator.remove_weight_norm

    def _load(self, *args, **kwargs):
        events.append("load")
        return orig_load(self, *args, **kwargs)

    def _to(self, *args, **kwargs):
        events.append("to")
        return orig_to(self, *args, **kwargs)

    def _fold(self):
        # The idempotent second call happens after the loader's fold.
        if "fold" not in events:
            assert events == ["load", "to"]
        events.append("fold")
        return orig_fold(self)

    monkeypatch.setattr(HiFTGenerator, "load_state_dict", _load)
    monkeypatch.setattr(HiFTGenerator, "to", _to)
    monkeypatch.setattr(HiFTGenerator, "remove_weight_norm", _fold)

    core = token2wav.StepAudio2Token2WavCore.__new__(token2wav.StepAudio2Token2WavCore)
    nn.Module.__init__(core)
    core.model_path = str(tmp_path)
    core.float16 = False
    core.device = torch.device("cpu")
    core._models_loaded = False
    core._drop_chunk_att_buffers = False

    core._ensure_models_loaded()

    assert events == ["load", "to", "fold"]
    folded_conv = core._hift.conv_pre
    torch.testing.assert_close(folded_conv.weight, expected_weight, rtol=0, atol=0)
    assert isinstance(folded_conv.weight, nn.Parameter)
    assert not _has_weight_norm(folded_conv)
    remaining = [name for name, module in core._hift.named_modules() if _has_weight_norm(module)]
    assert remaining
    assert all(name.startswith("f0_predictor.") for name in remaining)
    assert core._hift.remove_weight_norm() == 0
