# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Functional correctness tests for bucketed-encoder length plumbing (worker side).

Per-item lengths are the model's own published field (StepAudio2's
``audio_lens``, Covo's ``audio_num_tokens``) and they travel AS AN ARGUMENT on
the encoder call: StepAudio2 passes them to the forward itself, Covo hands
them to the wrapper's explicit ``lengths`` argument. These tests drive the
REAL ``install_bucketed_encoders`` and the REAL wrapped ``forward``; the toy
models hand the encoder their length field exactly like the real call sites.
The wrapper holds no state: every call carries its own lengths, so nothing can
leak across calls and no call can consume another call's lengths.

Coverage is functional only — WHAT the bucketed path computes (each row's
true-length prefix equals its solo result, the rung window keeps the input's
own padding tail, never zero-filled), WHEN it engages (mixed-length batches),
when it must fall back (missing lengths, over-long batches, no-waste batches,
output-contract mismatch), and what the whitelist guarantees (context margin,
time axis, frames-per-length, per-model ladders). Everything is asserted
through observable behavior — outputs, encoder call shapes, and logs — never
through wrapper internals.

All encoder calls run under ``torch.inference_mode()`` (the production
context); the bucketed path is eager — no CUDA graphs.
"""

import dataclasses
import gc
import logging
import weakref
from contextlib import contextmanager

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
# additional_config.audio_bucket_seconds is converted with 100 frames per second.
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
    """CovoAudio-style embed_multimodal: the model's own per-item length field
    travels to the encoder as the wrapper's explicit ``lengths`` argument
    (covo_audio_llm.py's ``audio_num_tokens`` contract)."""

    def __init__(self, lengths_field: str = "audio_lens"):
        super().__init__()
        self.encoder = RecordingEncoder()
        self.lengths_field = lengths_field

    def embed_multimodal(self, **kwargs) -> torch.Tensor:
        audio_features = kwargs["audio_features"]
        return self.encoder(audio_features, lengths=kwargs.get(self.lengths_field)).last_hidden_state


def make_features(lengths=LENGTHS, t: int = T, dim: int = D) -> torch.Tensor:
    """Rows carry real data before their true length and a distinctive tail
    after (the producer's own padding stand-in)."""
    feats = torch.randn(len(lengths), t, dim, device="cuda")
    for row, length in enumerate(lengths):
        feats[row, length:] += 3.0  # distinctive tail: preserved by the wrapper, never zero-filled
    return feats


def rung_of(length: int) -> int:
    for rung in RUNGS:
        if rung >= length:
            return rung
    return RUNGS[-1]


def expected_bucketed(feats: torch.Tensor, lengths=LENGTHS) -> torch.Tensor:
    """The wrapper contract: per row cumsum over the row's rung window (true
    content + the input's own padding tail preserved, never zero-filled),
    scattered into a container as wide as the largest rung used;
    rows with a shorter rung stay zero beyond it."""
    width = max(rung_of(length) for length in lengths)
    out = torch.zeros(len(lengths), width, D, device="cuda")
    for row, length in enumerate(lengths):
        rung = rung_of(length)
        out[row, :rung] = feats[row, :rung].cumsum(dim=0)  # window tail kept as-is
    return out


def _mock_vllm_config(model_type: str = "toy", *seconds):
    """A ``vllm_config`` stand-in: ``model_type`` drives ``_verified_knobs``
    and ``additional_config.audio_bucket_seconds`` is the feature's opt-in
    knob (no seconds = the knob is unset = the feature is off, the default)."""
    from types import SimpleNamespace

    additional = {"audio_bucket_seconds": list(seconds)} if seconds else {}
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type)),
        additional_config=additional,
    )


def _knobs(**overrides):
    """The verified-knobs stub (the whitelist's shape), overridable per test:
    ``output_kind`` declares what the test's toy encoder returns."""
    knobs = {
        "frames_per_second": 100,
        "frames_per_length": 1,
        "context_margin": 0,
        "time_axis": 1,
        "output_kind": "last_hidden_state",
    }
    knobs.update(overrides)
    return knobs


def _use_knobs(monkeypatch, **overrides):
    """Re-stub ``_verified_knobs`` with ``_knobs(**overrides)`` (replacing
    ``bucket_cfg``'s defaults for one test)."""
    from vllm_omni.worker import bucketed_encoder

    monkeypatch.setattr(bucketed_encoder, "_verified_knobs", lambda vllm_config: _knobs(**overrides))


@contextmanager
def capture_encoder_logs(caplog):
    """Route the wrapper's logger into ``caplog``.

    vLLM loggers propagate to the root logger, so while the capture handler is
    attached we disable propagation — otherwise every record is captured twice
    (once here, once via the root) and "exactly once" assertions over-count.
    """
    target_logger = logging.getLogger("vllm_omni.worker.bucketed_encoder")
    target_logger.addHandler(caplog.handler)
    previous_level = target_logger.level
    previous_propagate = target_logger.propagate
    target_logger.setLevel(logging.DEBUG)
    target_logger.propagate = False
    try:
        yield
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(previous_level)
        target_logger.propagate = previous_propagate


@pytest.fixture()
def bucket_cfg(monkeypatch):
    # The stage's additional_config.audio_bucket_seconds is the opt-in switch;
    # this fixture arms the 0.4/1/2 s ladder (rungs 40/100/200 at 100 fps).
    # These tests exercise the wrapper logic on toy models, which are not in the
    # production verified-model whitelist; make the gate permissive so the tests
    # drive the wrapper directly. Individual tests pin margin/time axis via
    # monkeypatched knobs, which override these defaults.
    from vllm_omni.worker import bucketed_encoder

    monkeypatch.setattr(
        bucketed_encoder,
        "_verified_knobs",
        lambda vllm_config: _knobs(),
    )
    if not torch.cuda.is_available():
        pytest.skip("bucketed encoder requires CUDA")
    return _mock_vllm_config("toy", 0.4, 1, 2)


def test_real_installer_receives_lengths_on_the_encoder_call(bucket_cfg):
    model = ToyModel().cuda()
    wrapped = install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg)
    assert wrapped == ["encoder"], f"unexpected wrapped modules: {wrapped}"
    # Only module.forward is wrapped; embed_multimodal is the model's own.
    assert getattr(model.embed_multimodal, "__self__", None) is model

    feats = make_features()
    with torch.inference_mode():
        # The lengths argument arrives at the wrapper on the call itself, so
        # the bucketed path groups by rung from the very first call.
        model.embed_multimodal(
            audio_features=feats,
            audio_lens=torch.tensor(LENGTHS, dtype=torch.int64),
        )
    new_shapes = model.encoder.calls
    assert sorted(new_shapes) == sorted([(1, 40, D), (2, 200, D), (1, 100, D)]), new_shapes


def test_cumsum_encoder_runs_bucketed_path_not_fallback(bucket_cfg):
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    feats = make_features()
    with torch.inference_mode():
        out = model.embed_multimodal(
            audio_features=feats,
            audio_lens=torch.tensor(LENGTHS, dtype=torch.int64),
        )

    expected = expected_bucketed(feats)

    # (a) Position-dependent proof, per row: the cumsum signature of each row's
    #     TRUE prefix must land in its own row (scatter is row-aligned) ...
    for row, length in enumerate(LENGTHS):
        solo_prefix = feats[row, :length].cumsum(dim=0)
        torch.testing.assert_close(out[row, :length], solo_prefix, rtol=0, atol=1e-6)

    # ... and the bucketed tail semantics hold: each row keeps the input's own
    # tail up to its rung (the rung window is a slice of the carried input —
    # producer padding preserved, never zero-filled), zeros after the rung.
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)

    # (b) Bucketed output is structurally DIFFERENT from the fallback's:
    #     width = largest rung (200), not the raw T (240), and a row stops at
    #     its own rung: row 0's rung is exactly its length (40), so its output
    #     is zero past the rung while the fallback keeps accumulating the
    #     carried tail over the full width.
    assert out.shape == (len(LENGTHS), 200, D), out.shape
    fallback_tail = feats.cumsum(dim=1)[:, 40:200]
    assert not torch.allclose(out[0, 40:200], fallback_tail[0], atol=1e-3)

    # (c) Execution evidence: the bucketed call re-entered the encoder with
    #     per-rung window shapes (eager path), not the full-batch fallback.
    bucketed_shapes = model.encoder.calls
    assert sorted(bucketed_shapes) == sorted([(1, 40, D), (2, 200, D), (1, 100, D)]), bucketed_shapes


class MaskConsumingEncoder(nn.Module):
    """Mask-aware encoder: output depends on both features and the mask."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.unsqueeze(-1).to(x.dtype)
        return (x * w).cumsum(dim=1) + w.cumsum(dim=1)


class MaskToyModel(nn.Module):
    """Embed path that calls the encoder as ``encoder(feats, mask, lengths=...)``."""

    def __init__(self):
        super().__init__()
        self.encoder = MaskConsumingEncoder()

    def embed_multimodal(self, **kwargs) -> torch.Tensor:
        return self.encoder(
            kwargs["audio_features"],
            kwargs["feature_attention_mask"],
            lengths=kwargs.get("audio_lens"),
        )


def test_explicit_lengths_apply_alongside_positional_mask(bucket_cfg, monkeypatch):
    """(features, mask) call sites must bucket via the explicit ``lengths``
    argument and stay numerically exact for a mask-consuming encoder."""
    _use_knobs(monkeypatch, output_kind="tensor")  # the mask encoder returns a bare tensor
    model = MaskToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    feats = make_features()
    mask = torch.zeros(len(LENGTHS), T, dtype=torch.int32, device="cuda")
    for row, length in enumerate(LENGTHS):
        mask[row, :length] = 1

    with torch.inference_mode():
        model.embed_multimodal(
            audio_features=feats,
            feature_attention_mask=mask,
            audio_lens=torch.tensor(LENGTHS, dtype=torch.int64),
        )
        out = model.embed_multimodal(
            audio_features=feats,
            feature_attention_mask=mask,
            audio_lens=torch.tensor(LENGTHS, dtype=torch.int64),
        )

    assert out.shape == (len(LENGTHS), 200, D), out.shape
    for row, length in enumerate(LENGTHS):
        # The un-bucketed reference is the encoder's own forward on the same
        # weights over the row's true window.
        solo = MaskConsumingEncoder.forward(model.encoder, feats[row : row + 1, :length], mask[row : row + 1, :length])[0]
        torch.testing.assert_close(out[row, :length], solo, rtol=0, atol=1e-6)


class LengthReturningEncoder(nn.Module):
    """Mask-aware encoder returning ``(features, per-item output lengths)`` —
    the Step-Audio2 / Ming ``forward_padded`` contract."""

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_len = x_len.to(x.device)  # host-side length contract
        w = (torch.arange(x.size(1), device=x.device) < x_len.unsqueeze(1)).to(x.dtype)
        features = (x * w.unsqueeze(-1)).cumsum(dim=1)
        out_len = (x_len + 1) // 2
        return features, out_len


class LengthReturningToyModel(nn.Module):
    """Embed path that calls the encoder as ``encoder(feats, host_lengths)``."""

    def __init__(self):
        super().__init__()
        self.encoder = LengthReturningEncoder()

    def embed_multimodal(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(kwargs["audio_features"], kwargs["audio_lens"])


def test_length_returning_encoder_buckets_losslessly(bucket_cfg, monkeypatch):
    """(features, lengths) encoders must bucket losslessly: each row's true-length
    prefix equals its solo result and the output lengths are the true per-item
    lengths (not rung-padded)."""
    _use_knobs(monkeypatch, output_kind="lengths_tuple")
    model = LengthReturningToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    feats = make_features()
    lens = torch.tensor(LENGTHS, dtype=torch.long)  # host-side 1-D lengths

    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_lens=lens)
        out_features, out_lens = model.embed_multimodal(audio_features=feats, audio_lens=lens)

    assert out_features.shape[0] == len(LENGTHS)
    assert out_lens.shape == (len(LENGTHS),)
    for row, length in enumerate(LENGTHS):
        # The un-bucketed reference is the encoder's own forward on the same
        # weights over the row's true window.
        solo_features, solo_len = LengthReturningEncoder.forward(model.encoder, feats[row : row + 1, :length], lens[row : row + 1])
        torch.testing.assert_close(out_features[row, :length], solo_features[0], rtol=0, atol=1e-6)
        assert out_lens[row].item() == solo_len.item()


def test_model_forward_error_surfaces_and_keeps_bucketing_armed(bucket_cfg):
    """A model-side forward error must surface as-is: it is not swallowed into
    a silent eager fallback, and the forward is NOT re-run a second time by a
    fallback handler. A later valid call must still bucket (the failure does
    not disarm the feature)."""

    class FlakyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls: list[tuple] = []
            self.fail = False

        def forward(self, x: torch.Tensor) -> EncOutput:
            self.calls.append(tuple(x.shape))
            if self.fail:
                raise RuntimeError("model-side failure")
            return EncOutput(last_hidden_state=x.cumsum(dim=1))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = FlakyEncoder()

        def embed_multimodal(self, **kwargs) -> torch.Tensor:
            return self.encoder(kwargs["audio_features"], lengths=kwargs.get("audio_lens")).last_hidden_state

    model = Model().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    feats = make_features()
    lens = torch.tensor(LENGTHS, dtype=torch.long)
    model.encoder.fail = True
    with torch.inference_mode(), pytest.raises(RuntimeError, match="model-side failure"):
        model.embed_multimodal(audio_features=feats, audio_lens=lens)

    assert len(model.encoder.calls) == 1, "the failing forward must run exactly once (no fallback re-run)"

    # The failure did not disarm bucketing: the next valid call buckets normally.
    model.encoder.fail = False
    calls_before = len(model.encoder.calls)
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_lens=lens)
    assert sorted(model.encoder.calls[calls_before:]) == sorted([(1, 40, D), (2, 200, D), (1, 100, D)])


def test_context_margin_preserves_conv_tail(bucket_cfg, monkeypatch):
    """A conv front that reads ``length + context_margin`` frames must not have
    its context tail clipped or zero-filled: the wrapper's rung window covers
    the extra frames, so the bucketed output matches the eager conv over the
    full real window."""
    from torch.nn import functional as F

    # The verified whitelist supplies the context margin (2 for this toy conv).
    _use_knobs(monkeypatch, context_margin=2, output_kind="lengths_tuple")

    class ConvContextEncoder(nn.Module):
        def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x_len = x_len.to(x.device)
            pad = F.pad(x, (0, 0, 1, 1))  # zero-pad the time axis by 1 each side
            out = pad[:, : x.size(1)] + pad[:, 1 : x.size(1) + 1] + pad[:, 2 : x.size(1) + 2]
            return out, (x_len + 1) // 2

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ConvContextEncoder()

        def embed_multimodal(self, **kwargs):
            return self.encoder(kwargs["audio_features"], kwargs["audio_lens"])

    model = Model().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    # Real content is [0, length + margin); producer-style padding after the
    # conv context (kept as-is by the wrapper, never zero-filled).
    lengths = [40, 120, 150, 75]  # max + margin <= largest rung (200)
    feats = torch.randn(len(lengths), T, D, device="cuda")
    for row, length in enumerate(lengths):
        feats[row, length + 2 :] += 3.0
    lens = torch.tensor(lengths, dtype=torch.long)

    with torch.inference_mode():
        out, out_lens = model.embed_multimodal(audio_features=feats, audio_lens=lens)

    assert out.shape[0] == len(lengths)
    for row, length in enumerate(lengths):
        # The eager reference feeds the full real window (length + context margin).
        solo, solo_len = ConvContextEncoder.forward(model.encoder, feats[row : row + 1, : length + 2], lens[row : row + 1])
        torch.testing.assert_close(out[row, :length], solo[0, :length], rtol=0, atol=1e-6)
        assert out_lens[row].item() == solo_len.item()


def test_time_axis_contract_wins_for_all_short_batch(bucket_cfg, monkeypatch):
    """A channels-first encoder (mel on dim 1, time on dim 2) must keep bucketing
    for an all-short batch via its verified ``time_axis=2`` from the whitelist."""
    _use_knobs(monkeypatch, time_axis=2, output_kind="lengths_tuple")

    class ChannelsFirstConvEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(128, 16, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x_len = x_len.to(x.device)
            out = self.conv(x)  # (B, 128, T) -> (B, 16, T)
            out = out.transpose(1, 2)  # (B, T, 16) time-major output
            return out, (x_len + 1) // 2

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ChannelsFirstConvEncoder()

        def embed_multimodal(self, **kwargs):
            return self.encoder(kwargs["audio_features"], kwargs["audio_lens"])

    model = Model().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    # All items shorter than the 128 mel bins: a wrong axis-1 guess would hand
    # the conv the wrong channel count and the call would fail. No length lands
    # exactly on a rung, so the whitelist margin 0 never clips a conv boundary
    # read here.
    mel, T_feat = 128, 120
    lengths = [30, 60, 45, 80]
    feats = torch.randn(len(lengths), mel, T_feat, device="cuda")
    for row, length in enumerate(lengths):
        feats[row, :, length:] += 3.0
    lens = torch.tensor(lengths, dtype=torch.long)

    with torch.inference_mode():
        out, out_lens = model.embed_multimodal(audio_features=feats, audio_lens=lens)

    for row, length in enumerate(lengths):
        # On/off parity: the reference is the un-bucketed forward over the full
        # carried row (the globally padded input feature-off computes); the rung
        # window keeps the input's own tail, so the true-length prefix matches
        # it exactly (the conv's boundary reads land on that tail).
        solo, solo_len = ChannelsFirstConvEncoder.forward(model.encoder, feats[row : row + 1], lens[row : row + 1])
        torch.testing.assert_close(out[row, :length], solo[0, :length], rtol=0, atol=1e-6)
        assert out_lens[row].item() == solo_len.item()


def test_whisper_fixed_window_recipe_matches_reference(bucket_cfg, monkeypatch):
    """The tencent/Covo-Audio-Chat call contract (community model, added in
    PR #2293): a single positional ``(B, mel, T)`` input to a fixed-window HF
    WhisperEncoder, the model's own per-item length field passed on the call.
    The variable-length recipe must engage for an eligible batch (the
    fixed-window forward never runs) and reproduce the reference forward on the
    same weights at both the full fixed window and a short rung window."""
    from transformers.models.whisper.configuration_whisper import WhisperConfig
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    cfg = _mock_vllm_config("toy", 3, 8, 30)  # rungs 300/800/3000 at 100 fps
    whisper_cfg = WhisperConfig(
        d_model=64,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=128,
        num_mel_bins=80,
        max_source_positions=1500,  # fixed window: 1500 * 2 = 3000 mel frames
    )

    raw_calls = []
    original_forward = WhisperEncoder.forward

    def counting_forward(self, *args, **kwargs):
        raw_calls.append(tuple(args[0].shape))
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(WhisperEncoder, "forward", counting_forward)

    class CovoContractModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = WhisperEncoder(whisper_cfg).cuda().eval()

        def embed_multimodal(self, **kwargs):  # covo_audio_llm.py contract
            return self.encoder(kwargs["audio_features"], lengths=kwargs.get("audio_lens")).last_hidden_state

    model = CovoContractModel()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=cfg) == ["encoder"]

    lengths = [3000, 300]
    feats = torch.randn(len(lengths), 80, 3000, device="cuda")  # (B, mel, T) padded to the fixed window
    with torch.inference_mode():
        hidden = model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(lengths))

    # Engaged: the fixed-window forward never ran for the eligible batch (the
    # output kind is declared, so there is no learn pass to serve eagerly).
    assert raw_calls == [], f"fixed-window forward calls: {raw_calls}"

    # Row 1 buckets to its 300-frame rung: 150 output frames, zeros after.
    assert hidden.shape == (2, 1500, 64), hidden.shape
    assert torch.count_nonzero(hidden[1, 150:]) == 0

    # The recipe reproduces the reference forward on the same weights: row 0 at
    # the full fixed window (the only window the untouched forward accepts) and
    # row 1 over its short rung window. Upstream ``transformers`` forward drift
    # fails here, not in production audio output.
    pristine = WhisperEncoder(whisper_cfg).cuda().eval()
    pristine.load_state_dict(model.encoder.state_dict())
    with torch.inference_mode():
        full = original_forward(pristine, feats[0:1]).last_hidden_state
        short = original_forward(pristine, feats[1:2, :, :300]).last_hidden_state
    torch.testing.assert_close(hidden[0], full[0], rtol=0, atol=1e-5)
    torch.testing.assert_close(hidden[1, :150], short[0], rtol=0, atol=1e-5)


def test_whisper_shim_short_audio_uses_channels_first_axis(bucket_cfg):
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
            return self.encoder(kwargs["audio_features"], lengths=kwargs.get("audio_lens")).last_hidden_state

    model = CovoLikeModel()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    feats = torch.randn(1, 128, 3000, device="cuda")
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor([100]))
        out = model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor([100]))

    # Bucketed: rung 100 -> 50 output frames; fallback would be 3000/2 = 1500.
    assert out.shape == (1, 50, 64), out.shape


def test_reads_model_published_length_field_scaled(bucket_cfg, monkeypatch):
    """The model passes its own published field (Covo's ``audio_num_tokens``)
    as the ``lengths`` argument and the wrapper converts its unit to feature
    frames: each rung window is token_count * 16 frames."""
    _use_knobs(monkeypatch, frames_per_length=16)

    model = ToyModel(lengths_field="audio_num_tokens").cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    tokens = torch.tensor([2, 6, 12, 5], dtype=torch.int64)  # * 16 -> 32/96/192/80 frames
    feats = make_features([32, 96, 192, 80])
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_num_tokens=tokens)

    # 32 -> rung 40, 96 -> rung 100, 192 -> rung 200, 80 -> rung 100.
    assert sorted(model.encoder.calls) == sorted([(1, 40, D), (2, 100, D), (1, 200, D)])


def test_missing_length_field_falls_back_to_original(bucket_cfg, caplog):
    """A request whose mm batch carries no length field runs the untouched
    forward (nothing can leak from an earlier call — the wrapper holds no
    state). The silent-death case (bucketing never engages, e.g. a wrong length
    field) surfaces as exactly one warning, then goes quiet at debug level."""
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    feats = make_features()
    with capture_encoder_logs(caplog):
        with torch.inference_mode():
            model.embed_multimodal(audio_features=feats)  # first fallback: warns once
            calls_before = len(model.encoder.calls)
            out = model.embed_multimodal(audio_features=feats)  # later fallbacks: debug
    assert out.shape == feats.shape
    assert model.encoder.calls[calls_before:] == [(len(LENGTHS), 240, D)]
    missing_logs = [r for r in caplog.records if "no per-item lengths" in r.getMessage()]
    assert len(missing_logs) == 2, "the fallback must log every time"
    assert sum(r.levelno == logging.WARNING for r in missing_logs) == 1, "the warning must fire exactly once"


def test_positional_host_lengths_fallback_still_buckets(bucket_cfg, monkeypatch):
    """A direct encoder call whose forward takes lengths positionally
    (StepAudio2's audio_lens contract) still buckets via the positional source
    in _host_lengths."""
    _use_knobs(monkeypatch, output_kind="lengths_tuple")
    model = LengthReturningToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    feats = make_features()
    lens = torch.tensor(LENGTHS, dtype=torch.long)
    with torch.inference_mode():
        out_features, out_lens = model.encoder(feats, lens)

    # Bucketed container: width = largest rung used (200), not the raw T (240).
    assert out_features.shape == (len(LENGTHS), 200, D), out_features.shape
    for row, length in enumerate(LENGTHS):
        solo_features, solo_len = LengthReturningEncoder.forward(model.encoder, feats[row : row + 1, :length], lens[row : row + 1])
        torch.testing.assert_close(out_features[row, :length], solo_features[0], rtol=0, atol=1e-6)
        assert out_lens[row].item() == solo_len.item()


def test_mixed_frame_rate_models_each_get_their_own_ladder(monkeypatch):
    """Lengths come from each model's own field, so a second model with a
    different mel frame rate needs no process-wide refusal: each install
    converts the seconds ladder with its own ``frames_per_second`` — the same
    seconds produce different window shapes per model."""
    from vllm_omni.worker import bucketed_encoder

    cfg = _mock_vllm_config("toy", 0.4, 1)  # rungs 40/100 at 100 fps, 20/50 at 50 fps

    def knobs(frames_per_second):
        return lambda vllm_config: _knobs(frames_per_second=frames_per_second)

    monkeypatch.setattr(bucketed_encoder, "_verified_knobs", knobs(100))
    model_a = ToyModel().cuda()
    assert install_bucketed_encoders(model_a, {"encoder": model_a.encoder}, vllm_config=cfg) == ["encoder"]

    monkeypatch.setattr(bucketed_encoder, "_verified_knobs", knobs(50))
    model_b = ToyModel().cuda()
    assert install_bucketed_encoders(model_b, {"encoder": model_b.encoder}, vllm_config=cfg) == ["encoder"]

    lengths = [15, 45]
    feats = make_features(lengths, t=120)
    with torch.inference_mode():
        model_a.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(lengths))
        model_b.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(lengths))

    # The same 0.4/1 s ladder lands on 40/100 frames at 100 fps and 20/50 at 50 fps.
    assert sorted(model_a.encoder.calls) == sorted([(1, 40, D), (1, 100, D)]), model_a.encoder.calls
    assert sorted(model_b.encoder.calls) == sorted([(1, 20, D), (1, 50, D)]), model_b.encoder.calls


def test_overlong_batch_falls_back_to_original(bucket_cfg):
    """A batch whose max length exceeds the largest rung is routed through the
    original forward — not truncated (which would silently change the item's
    output length) and not permanently disabled."""
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    lengths = [240, 100]  # 240 > largest rung 200
    feats = torch.randn(2, 240, D, device="cuda")
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(lengths))
        out = model.encoder(feats, lengths=torch.tensor(lengths)).last_hidden_state

    assert out.shape == (2, 240, D), out.shape  # full width = untouched forward


def test_each_call_carries_its_own_lengths(bucket_cfg):
    """The wrapper holds no per-request state, so nothing can leak across
    calls: a call without lengths falls back to the full batch, and a
    same-batch-size call with different lengths buckets by ITS OWN lengths —
    the state channel's silent-mismatch failure mode (matched by batch size)
    is structurally impossible."""
    model = ToyModel().cuda()
    install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg)

    feats = make_features()
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(LENGTHS))

        # A direct call without lengths falls back to the full width. It also
        # pins the strip contract: lengths=None must never reach the raw
        # forward (its signature is forward(x) and would raise).
        calls_before = len(model.encoder.calls)
        out = model.encoder(feats, lengths=None).last_hidden_state
    assert out.shape == (len(LENGTHS), 240, D), out.shape
    assert model.encoder.calls[calls_before:] == [(len(LENGTHS), 240, D)]

    # Same batch size (4 items), different lengths: each call buckets by the
    # lengths it carries, never by a previous call's.
    first = [30, 30, 150, 150]
    second = [200, 40, 40, 40]
    feats4 = make_features(first)
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats4, audio_lens=torch.tensor(first))  # buckets by ITS lengths
        calls_before = len(model.encoder.calls)
        model.embed_multimodal(audio_features=feats4, audio_lens=torch.tensor(second))
    shapes = model.encoder.calls[calls_before:]
    # second's own lengths: 200 -> rung 200; 40 -> rung 40 for the other three.
    assert sorted(shapes) == sorted([(1, 200, D), (3, 40, D)]), shapes


def test_empty_batch_falls_back_to_original(bucket_cfg):
    """An empty batch must fall back to the original forward instead of raising
    inside max(lengths)."""
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    feats = torch.empty(0, 240, D, device="cuda")
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_lens=torch.empty(0, dtype=torch.int64))
        out = model.encoder(feats, lengths=torch.empty(0, dtype=torch.int64)).last_hidden_state

    assert out.shape == (0, 240, D), out.shape


def test_no_waste_batch_serves_original_untouched(bucket_cfg):
    """Waste gate: when the rung windows cannot compute fewer frames than the
    untouched forward — a single item whose carried length is already the true
    length (B=1), or a uniform batch — the wrapper serves the original forward
    unchanged. Enabling the feature outside mixed-length batches is a no-op,
    never a regression, and the wrapper stays armed for later batches."""
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    for lengths in ([50], [50, 50, 50]):
        feats = make_features(lengths, t=50)  # carried extent == true length
        with torch.inference_mode():
            model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(lengths))
            calls_before = len(model.encoder.calls)
            out = model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(lengths))
        # The untouched forward ran once at the full extent — no rung windows.
        assert model.encoder.calls[calls_before:] == [(len(lengths), 50, D)], model.encoder.calls[calls_before:]
        torch.testing.assert_close(out, feats.cumsum(dim=1), rtol=0, atol=0)


def test_rung_windows_are_capped_at_input_extent(bucket_cfg):
    """A rung larger than the batch's carried time extent is capped to it: the
    wrapper never computes more frames per row than the untouched forward, and
    the batch still buckets down wherever a rung can shrink it."""
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    lengths = [30, 120, 140]
    feats = make_features(lengths, t=150)  # extent 150 < rung 200
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(lengths))
        calls_before = len(model.encoder.calls)
        out = model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(lengths))
    shapes = model.encoder.calls[calls_before:]
    # 30 -> rung 40; 120/140 -> rung 200 capped to the 150-frame extent.
    assert sorted(shapes) == sorted([(1, 40, D), (2, 150, D)]), shapes
    assert all(shape[1] <= 150 for shape in shapes), shapes
    for row, length in enumerate(lengths):
        torch.testing.assert_close(out[row, :length], feats[row, :length].cumsum(dim=0), rtol=0, atol=1e-6)


def test_length_field_read_is_one_bounded_host_read(bucket_cfg):
    """CPU length fields (the keep_on_cpu delivery) are read without any D2H
    sync; a device-delivered field is read with a single batched host read (the
    rung decision is host-side) and bucketing still engages."""
    model = ToyModel().cuda()
    install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg)
    feats = make_features()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.inference_mode():
            # CPU lengths (the keep_on_cpu delivery) must not sync: two
            # bucketed calls pass under sync-debug-error.
            model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(LENGTHS, dtype=torch.int64))
            calls_before = len(model.encoder.calls)
            model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(LENGTHS, dtype=torch.int64))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert sorted(model.encoder.calls[calls_before:]) == sorted([(1, 40, D), (2, 200, D), (1, 100, D)])

    # A device-delivered field is read once on the host and still buckets.
    with torch.inference_mode():
        calls_before = len(model.encoder.calls)
        model.embed_multimodal(
            audio_features=feats,
            audio_lens=torch.tensor(LENGTHS, dtype=torch.int64, device="cuda"),
        )
    assert sorted(model.encoder.calls[calls_before:]) == sorted([(1, 40, D), (2, 200, D), (1, 100, D)])


def test_bucketing_stays_off_without_the_config_knob():
    """additional_config.audio_bucket_seconds is the opt-in master switch: with
    the knob unset — or no vllm_config at all — the wrapper installs as a
    pass-through (so the model's ``lengths`` call contract stays valid) and the
    original forward runs unchanged: the feature is off by default. The strip
    contract is pinned by the raw forward's strict ``forward(x)`` signature: a
    leaked ``lengths`` kwarg would raise TypeError here."""
    for cfg in (_mock_vllm_config(), None):
        model = ToyModel().cuda()
        assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=cfg) == ["encoder"]
        feats = make_features()
        with torch.inference_mode():
            calls_before = len(model.encoder.calls)
            out = model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(LENGTHS, dtype=torch.int64))
        assert out.shape == feats.shape
        assert model.encoder.calls[calls_before:] == [(len(LENGTHS), 240, D)]


def test_unverified_model_feature_is_gated_off():
    """A model not in the verified whitelist must never silently bucket even
    when additional_config.audio_bucket_seconds is set. This is the safety
    guard: an un-adapted encoder must never silently bucket. The wrapper stays
    a pass-through (the ``lengths`` call contract remains valid) and serves the
    call at full width."""
    model = ToyModel().cuda()
    cfg = _mock_vllm_config("unverified_model", 0.4, 1, 2)
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=cfg) == ["encoder"]

    # No rung windows: the untouched forward serves the call at full width.
    feats = make_features()
    with torch.inference_mode():
        calls_before = len(model.encoder.calls)
        model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(LENGTHS, dtype=torch.int64))
    assert model.encoder.calls[calls_before:] == [(len(LENGTHS), 240, D)]


def test_uninstall_restores_the_module_forward(bucket_cfg):
    """The installer's patch manifest rolls back cleanly: after uninstall the
    module's own forward is back and a re-install buckets again from a clean
    slate (test-isolation contract)."""
    from vllm_omni.worker.bucketed_encoder import uninstall_bucketed_encoders

    model = ToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    assert uninstall_bucketed_encoders(model) == ["encoder"]
    assert model.encoder.forward.__self__ is model.encoder
    assert model.encoder.forward.__func__ is RecordingEncoder.forward

    # Re-install after rollback wraps again and buckets from a clean slate.
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]
    feats = make_features()
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(LENGTHS, dtype=torch.int64))
    assert sorted(model.encoder.calls) == sorted([(1, 40, D), (2, 200, D), (1, 100, D)])


def test_rebuild_carries_only_last_hidden_state(bucket_cfg):
    """The rebuilt output is a fresh ``BaseModelOutput`` carrying only
    ``last_hidden_state`` — the declared output contract (see ``_REBUILDERS``):
    no other field of any request's output can ride along."""
    from transformers.modeling_outputs import BaseModelOutput

    model = ToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    feats = make_features()
    with torch.inference_mode():
        out = model.encoder(feats, lengths=torch.tensor(LENGTHS, dtype=torch.int64))
    assert type(out) is BaseModelOutput
    assert out.hidden_states is None and out.attentions is None
    assert out.last_hidden_state.shape == (len(LENGTHS), 200, D)


def test_rebuild_retains_no_request_outputs(bucket_cfg):
    """No request's output instance may outlive its call (the review note:
    "This closure keeps the first request's output alive"): the rebuild
    captures no per-request instance, so every result is collected once its
    caller drops it."""
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    feats = make_features()
    refs = []
    with torch.inference_mode():
        for _ in range(3):
            out = model.encoder(feats, lengths=torch.tensor(LENGTHS, dtype=torch.int64))
            refs.append(weakref.ref(out))
            del out
    gc.collect()
    assert [ref() for ref in refs] == [None, None, None], "a request's output outlived its call"


def test_output_with_per_request_fields_refuses_to_bucket(bucket_cfg, caplog):
    """An output carrying per-request fields (beyond ``last_hidden_state``)
    cannot be rebuilt from the regrouped features. Instead of silently copying
    a stale request's values into later results, the wrapper refuses to bucket
    and serves the eager result — so every request's fields stay its own."""

    @dataclasses.dataclass
    class StampedOutput:
        last_hidden_state: torch.Tensor
        stamp: str = ""

    class StampedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> StampedOutput:
            self.calls += 1
            return StampedOutput(last_hidden_state=x.cumsum(dim=1), stamp=f"req{self.calls}")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = StampedEncoder()

        def embed_multimodal(self, **kwargs):
            return self.encoder(kwargs["audio_features"], lengths=kwargs.get("audio_lens"))

    model = Model().cuda()
    assert install_bucketed_encoders(model, {"encoder": model.encoder}, vllm_config=bucket_cfg) == ["encoder"]

    feats = make_features()
    with capture_encoder_logs(caplog):
        with torch.inference_mode():
            first = model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(LENGTHS, dtype=torch.int64))
            second = model.embed_multimodal(audio_features=feats, audio_lens=torch.tensor(LENGTHS, dtype=torch.int64))

    assert "stamp" in caplog.text and "cannot reproduce" in caplog.text, "the refusal must be logged"
    # Each result carries ITS OWN request's stamp — never another's.
    assert first.stamp != second.stamp
    torch.testing.assert_close(second.last_hidden_state, feats.cumsum(dim=1), rtol=0, atol=0)
