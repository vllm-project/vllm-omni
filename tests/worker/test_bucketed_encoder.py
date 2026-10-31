# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for bucketed-encoder length publication (worker side).

Reviewer request: "Add a regression that goes through the real installer and
embed_multimodal..." — these tests drive the REAL ``install_bucketed_encoders``
and the REAL wrapped ``embed_multimodal``; the toy model mirrors the CovoAudio
call convention (``self.encoder(audio_features)`` with no positional lengths)
that commit 9ab317a ("publish lengths to installed bucketed encoder wrappers")
fixes.

* test_real_installer_publishes_lengths_through_embed_multimodal
    lengths published via embed_multimodal must ARRIVE at the installed
    wrapper. Pre-fix the publisher held no wrapper instances (the installer
    assigns the callable to ``module.forward``, which is not in the module
    tree), so ``_lengths`` stayed ``None`` and every call fell back.
* test_cumsum_encoder_runs_bucketed_path_not_fallback
    a position-dependent (cumsum) encoder proves the bucketed path ran and is
    not the eager fallback: per-row prefix equals the solo result, the bucketed
    tail semantics hold, the output width is the largest rung (not raw T), and
    the call replayed captured graphs without re-entering Python forward.
* test_published_lengths_apply_alongside_positional_mask
    call sites that pass ``(features, mask)`` positionally (no lengths
    argument) must still bucket using the published lengths, and stay exact
    for a mask-consuming encoder.
* test_module_pattern_env_override
    ``VLLM_OMNI_AUDIO_BUCKET_MODULES`` selects which module names wrap, so
    encoders outside the built-in name pattern can opt in without model
    changes.

All encoder calls run under ``torch.inference_mode()`` — the production
context: graph capture stores inference tensors, so replay outside
inference_mode raises and the wrapper silently falls back.
"""

import dataclasses

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_marks
from vllm_omni.worker.bucketed_encoder import install_bucketed_encoders

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]

torch.manual_seed(0)

LENGTHS = [40, 120, 200, 75]
T = 240  # feature frames; longer than the largest rung (200)
D = 128
# VLLM_OMNI_AUDIO_BUCKET_SECONDS is parsed with 100 frames per second.
RUNGS = (40, 100, 200)


@dataclasses.dataclass
class EncOutput:
    last_hidden_state: torch.Tensor


class RecordingEncoder(nn.Module):
    """cumsum encoder: output[t] = sum_{s<=t} x[s] (position-dependent)."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []

    def forward(self, x: torch.Tensor) -> EncOutput:
        self.calls.append(tuple(x.shape))
        return EncOutput(last_hidden_state=x.cumsum(dim=1))


class ToyModel(nn.Module):
    """CovoAudio-style embed_multimodal: encoder called without lengths."""

    def __init__(self):
        super().__init__()
        self.encoder = RecordingEncoder()

    def embed_multimodal(self, **kwargs) -> torch.Tensor:
        audio_features = kwargs["audio_features"]
        return self.encoder(audio_features).last_hidden_state


def make_features(lengths=LENGTHS, t: int = T, dim: int = D) -> torch.Tensor:
    """Rows carry real data before their true length and garbage after."""
    feats = torch.randn(len(lengths), t, dim, device="cuda")
    for row, length in enumerate(lengths):
        feats[row, length:] += 3.0  # garbage tail: makes fallback outputs differ
    return feats


def rung_of(length: int) -> int:
    for rung in RUNGS:
        if rung >= length:
            return rung
    return RUNGS[-1]


def expected_bucketed(feats: torch.Tensor, lengths=LENGTHS) -> torch.Tensor:
    """The wrapper contract: per row cumsum over (true prefix + zero pad to
    its rung), scattered into a container as wide as the largest rung used;
    rows with a shorter rung stay zero beyond it."""
    width = max(rung_of(length) for length in lengths)
    out = torch.zeros(len(lengths), width, D, device="cuda")
    for row, length in enumerate(lengths):
        rung = rung_of(length)
        window = torch.zeros(rung, D, device="cuda")
        window[:length] = feats[row, :length]
        out[row, :rung] = window.cumsum(dim=0)
    return out


@pytest.fixture()
def bucket_env(monkeypatch):
    # Real env names from the code (not the drifted names in the commit message).
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_SECONDS", "0.4,1,2")
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_BATCH_SIZES", "1,2,4")
    if not torch.cuda.is_available():
        pytest.skip("bucketed encoder requires CUDA")


def test_real_installer_publishes_lengths_through_embed_multimodal(bucket_env):
    from vllm_omni.inputs.mm_processor import OmniMultiModalProcessor

    model = ToyModel().cuda()
    wrapped = install_bucketed_encoders(model)
    assert wrapped == ["encoder"], f"unexpected wrapped modules: {wrapped}"
    assert getattr(OmniMultiModalProcessor, "_bucket_frames_installed", False)

    encoder_wrapper = model.encoder.forward  # installer assigns the _BucketedEncoder here
    assert type(encoder_wrapper).__name__ == "_BucketedEncoder"
    assert encoder_wrapper._lengths is None  # nothing published yet

    feats = make_features()
    with torch.inference_mode():
        model.embed_multimodal(
            audio_features=feats,
            encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
        )

    # Core regression assertion: the published lengths ARRIVE at the wrapper.
    assert encoder_wrapper._lengths == tuple(LENGTHS)

    # Functional echo: a follow-up call without positional lengths is grouped
    # by the published lengths -> per-rung padded executions hit the pool.
    assert encoder_wrapper.pool is not None, "graph pool was not captured"
    seen: list[tuple] = []
    original_execute = encoder_wrapper.pool.execute

    def counting_execute(*args, **kwargs):
        seen.append(tuple(args[0].shape))
        return original_execute(*args, **kwargs)

    encoder_wrapper.pool.execute = counting_execute
    calls_before = len(model.encoder.calls)
    with torch.inference_mode():
        model.encoder(feats)
    # Grouping evidence: one padded execution per length rung present.
    assert sorted(seen) == sorted([(1, 40, D), (2, 200, D), (1, 100, D)]), seen
    # Replay does not re-enter Python forward; a fallback would have.
    assert len(model.encoder.calls) == calls_before


def test_cumsum_encoder_runs_bucketed_path_not_fallback(bucket_env):
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    feats = make_features()
    with torch.inference_mode():
        # First eligible call: eager (learns output type + captures the pool).
        model.embed_multimodal(
            audio_features=feats,
            encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
        )
        calls_after_prime = len(model.encoder.calls)
        out = model.encoder(feats).last_hidden_state
        calls_after_bucketed = len(model.encoder.calls)

    expected = expected_bucketed(feats)

    # (a) Position-dependent proof, per row: the cumsum signature of each row's
    #     TRUE prefix must land in its own row (scatter is row-aligned) ...
    for row, length in enumerate(LENGTHS):
        solo_prefix = feats[row, :length].cumsum(dim=0)
        torch.testing.assert_close(out[row, :length], solo_prefix, rtol=0, atol=1e-6)

    # ... and the bucketed tail semantics hold: flat to the row's rung, zeros after.
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)

    # (b) Bucketed output is structurally DIFFERENT from the fallback's:
    #     width = largest rung (200), not the raw T (240).
    assert out.shape == (len(LENGTHS), 200, D), out.shape
    # The fallback computes cumsum over each full garbage tail; its values in
    # the tail region differ from the bucketed (zero-padded) ones.
    fallback_tail = feats.cumsum(dim=1)[:, 40:200]
    assert not torch.allclose(out[0, 40:200], fallback_tail[0], atol=1e-3)

    # (c) Execution evidence: the run happened as per-rung graph replays
    #     (checked before any solo runs below that call the original forward).
    assert wrapper.pool is not None and wrapper.pool.entries, "no captured graphs"
    for key in ((1, 40), (2, 200), (1, 100)):
        assert key in wrapper.pool.entries, f"missing captured entry {key}"
    assert calls_after_bucketed == calls_after_prime, (
        "Python forward re-entered during the bucketed call (fallback path)"
    )


class MaskConsumingEncoder(nn.Module):
    """Mask-aware encoder: output depends on both features and the mask."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.unsqueeze(-1).to(x.dtype)
        return (x * w).cumsum(dim=1) + w.cumsum(dim=1)


class MaskToyModel(nn.Module):
    """Embed path that calls the encoder as ``encoder(feats, mask)``."""

    def __init__(self):
        super().__init__()
        self.encoder = MaskConsumingEncoder()

    def embed_multimodal(self, **kwargs) -> torch.Tensor:
        return self.encoder(kwargs["audio_features"], kwargs["feature_attention_mask"])


def test_published_lengths_apply_alongside_positional_mask(bucket_env):
    """(features, mask) call sites must bucket via the published lengths and
    stay numerically exact for a mask-consuming encoder."""
    model = MaskToyModel().cuda()
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    feats = make_features()
    mask = torch.zeros(len(LENGTHS), T, dtype=torch.int32, device="cuda")
    for row, length in enumerate(LENGTHS):
        mask[row, :length] = 1

    with torch.inference_mode():
        model.embed_multimodal(
            audio_features=feats,
            feature_attention_mask=mask,
            encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
        )
        assert wrapper._lengths == tuple(LENGTHS)
        out = model.encoder(feats, mask)

    assert out.shape == (len(LENGTHS), 200, D), out.shape
    for row, length in enumerate(LENGTHS):
        solo = wrapper._original(feats[row : row + 1, :length], mask[row : row + 1, :length])[0]
        torch.testing.assert_close(out[row, :length], solo, rtol=0, atol=1e-6)


def test_module_pattern_env_override(bucket_env, monkeypatch):
    """VLLM_OMNI_AUDIO_BUCKET_MODULES selects which module names wrap."""
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_MODULES", r"(?:^|\.)my_frontend$")

    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.my_frontend = RecordingEncoder()
            self.encoder = RecordingEncoder()

        def embed_multimodal(self, **kwargs):
            return self.my_frontend(kwargs["audio_features"]).last_hidden_state

    model = CustomModel().cuda()
    assert install_bucketed_encoders(model) == ["my_frontend"]


def test_fixed_window_encoder_contract_engages(bucket_env, monkeypatch):
    """The tencent/Covo-Audio-Chat call contract (community model, added in
    PR #2293): a single positional ``(B, mel, T)`` input to a fixed-window HF
    WhisperEncoder, lengths published via embed_multimodal. The module's
    time-axis handling and fixed-window recipe must engage the bucketed path
    with zero model-side changes: fixed-window forward untouched, windows
    captured per rung, and exact output where rung == true length."""
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_SECONDS", "3,8,30")  # rungs 300/800/3000
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_BATCH_SIZES", "1,2,4")
    from transformers.models.whisper.configuration_whisper import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    raw_calls = []
    original_forward = WhisperEncoder.forward

    def counting_forward(self, *args, **kwargs):
        raw_calls.append(tuple(args[0].shape))
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(WhisperEncoder, "forward", counting_forward)

    class CovoContractModel(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = WhisperConfig(
                d_model=64,
                encoder_layers=1,
                encoder_attention_heads=2,
                encoder_ffn_dim=128,
                num_mel_bins=80,
                max_source_positions=1500,
            )
            self.encoder = WhisperEncoder(cfg).cuda().eval()

        def embed_multimodal(self, **kwargs):  # covo_audio_llm.py:116 contract
            return self.encoder(kwargs["audio_features"]).last_hidden_state

    model = CovoContractModel()
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    lengths = [300, 800, 2400]
    feats = torch.randn(len(lengths), 80, 3000, device="cuda")  # (B, mel, T) padded to 30s
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, encoder_bucket_frames=torch.tensor(lengths))
        assert wrapper._lengths == tuple(lengths)
        out = model.encoder(feats)

    # Engaged: per-rung windows were captured and replayed (fallback never
    # captures a 300/800-frame window, and calls the fixed-window forward).
    assert wrapper.pool is not None and wrapper.pool.entries
    for key in ((1, 300), (1, 800), (1, 3000)):
        assert key in wrapper.pool.entries, f"missing captured entry {key}"
    # The untouched forward may run once at the eager prime (full 3000-frame
    # input); the bucketed rung windows must never hit it (they use the shim).
    assert raw_calls == [(3, 80, 3000)], f"fixed-window forward calls: {raw_calls}"

    # Exact where the rung equals the true length (no padding involved).
    hidden = out.last_hidden_state
    assert hidden.shape[0] == len(lengths) and hidden.shape[2] == 64
    solo_row0 = wrapper._shim(feats[0:1, :, :300])
    torch.testing.assert_close(hidden[0, :150], solo_row0.last_hidden_state[0], rtol=0, atol=1e-5)


def test_qwen2audio_keyword_call_keeps_pooling_and_true_fallback(bucket_env):
    """P1: Qwen2AudioEncoder must NOT carry the Whisper recipe, so an installed
    module's keyword-argument call (Audex contract) runs the untouched forward:
    pooling preserved (750 emb per 3000-frame clip, not 1500) and feature-off
    parity against a pristine encoder, not the replaced _original."""
    from transformers.models.qwen2_audio.configuration_qwen2_audio import Qwen2AudioEncoderConfig
    from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder

    cfg = Qwen2AudioEncoderConfig(
        d_model=64,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=128,
        num_mel_bins=80,
        max_source_positions=1500,
    )

    class AudexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_tower = Qwen2AudioEncoder(cfg).cuda().eval()

    model = AudexModel()
    assert install_bucketed_encoders(model) == ["audio_tower"]
    wrapper = model.audio_tower.forward
    assert wrapper._shim is None, "Qwen2AudioEncoder must not carry the Whisper recipe"

    pristine = Qwen2AudioEncoder(cfg).cuda().eval()
    pristine.load_state_dict(model.audio_tower.state_dict())

    feats = torch.randn(1, 80, 3000, device="cuda")
    with torch.inference_mode():
        out = model.audio_tower(input_features=feats, return_dict=True).last_hidden_state
        ref = pristine(input_features=feats, return_dict=True).last_hidden_state

    assert out.shape == (1, 750, 64), out.shape  # avg_pooler applied: 3000 -> 750
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_covo_processor_carries_bucket_frames():
    """P2: the concrete CovoAudio processor emits encoder_bucket_frames (from
    raw audio sample counts) and registers it, so lengths survive the
    raw-audio -> concrete-processor boundary."""
    import numpy as np

    from vllm_omni.model_executor.models.covo_audio.covo_audio import CovoAudioMultiModalProcessor

    class _Tok:
        def encode(self, s):
            return [1, 2, 3]

    class _FeatExt:
        def __call__(self, raw_audios, sampling_rate, return_tensors):
            return {"input_features": torch.zeros(len(raw_audios), 128, 3000)}

    class _Info:
        def get_tokenizer(self):
            return _Tok()

        def get_feature_extractor(self):
            return _FeatExt()

    proc = CovoAudioMultiModalProcessor.__new__(CovoAudioMultiModalProcessor)
    proc.info = _Info()
    proc.dummy_inputs = None

    hf = proc._call_hf_processor("hi", {"audios": [(np.zeros(16000 * 3, dtype=np.float32), 16000)]}, {}, {})
    assert torch.equal(hf["encoder_bucket_frames"], torch.tensor([300], dtype=torch.int64)), hf["encoder_bucket_frames"]

    fields = proc._get_mm_fields_config(hf, {})
    assert "encoder_bucket_frames" in fields


def test_whisper_shim_short_audio_uses_channels_first_axis(bucket_env):
    """A short clip (100 frames < 128 mel bins) must still bucket along the
    channels-first time axis — the Whisper recipe implies (B, mel, T) by
    construction, so the length-scale heuristic must not flip the axis."""
    from transformers.models.whisper.configuration_whisper import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    cfg = WhisperConfig(
        d_model=64,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=128,
        num_mel_bins=128,
        max_source_positions=1500,
    )

    class CovoLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = WhisperEncoder(cfg).cuda().eval()

        def embed_multimodal(self, **kwargs):
            return self.encoder(kwargs["audio_features"]).last_hidden_state

    model = CovoLikeModel()
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    feats = torch.randn(1, 128, 3000, device="cuda")
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, encoder_bucket_frames=torch.tensor([100]))
        out = model.encoder(feats).last_hidden_state

    # Bucketed: rung 100 -> 50 output frames; fallback would be 3000/2 = 1500.
    assert out.shape == (1, 50, 64), out.shape
    assert wrapper._disabled is False, "bucketed path fell back for a short clip"


def test_overlong_batch_falls_back_to_original(bucket_env):
    """A batch whose max length exceeds the largest rung is routed through the
    original forward — not truncated (which would silently change the item's
    output length) and not permanently disabled."""
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    lengths = [240, 100]  # 240 > largest rung 200
    feats = torch.randn(2, 240, D, device="cuda")
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, encoder_bucket_frames=torch.tensor(lengths))
        out = model.encoder(feats).last_hidden_state

    assert wrapper._disabled is False, "over-long batch must not disable the wrapper"
    assert out.shape == (2, 240, D), out.shape  # full width = untouched forward


def test_positional_lengths_clamp_to_rung(bucket_env):
    """The positional CPU-lengths path must never tell the encoder a window is
    longer than its rung (an un-clamped copy would index past the window)."""
    model = ToyModel().cuda()
    install_bucketed_encoders(model)
    wrapper = model.encoder.forward
    out = wrapper._lengths_for(torch.tensor([240, 100]), [0, 1], 200, 2)
    assert torch.equal(out, torch.tensor([200, 100]))


def test_stale_lengths_are_cleared(bucket_env):
    """A later embed_multimodal call without the length field must clear stale
    lengths instead of reusing the previous request's."""
    model = ToyModel().cuda()
    install_bucketed_encoders(model)
    wrapper = model.encoder.forward

    feats = make_features()
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, encoder_bucket_frames=torch.tensor(LENGTHS))
        assert wrapper._lengths == tuple(LENGTHS)
        model.embed_multimodal(audio_features=feats)  # no length field
        assert wrapper._lengths is None


def test_empty_batch_falls_back_to_original(bucket_env):
    """An empty batch must fall back to the original forward instead of raising
    inside max(lengths)."""
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    feats = torch.empty(0, 240, D, device="cuda")
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, encoder_bucket_frames=torch.empty(0, dtype=torch.int64))
        out = model.encoder(feats).last_hidden_state

    assert wrapper._disabled is False
    assert out.shape == (0, 240, D), out.shape
