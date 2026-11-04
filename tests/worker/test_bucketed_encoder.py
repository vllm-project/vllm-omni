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
    the call computed each rung eagerly with window shapes, not the full batch.
* test_published_lengths_apply_alongside_positional_mask
    call sites that pass ``(features, mask)`` positionally (no lengths
    argument) must still bucket using the published lengths, and stay exact
    for a mask-consuming encoder.
* test_module_pattern_env_override
    ``VLLM_OMNI_AUDIO_BUCKET_MODULES`` selects which module names wrap, so
    encoders outside the built-in name pattern can opt in without model
    changes.

All encoder calls run under ``torch.inference_mode()`` (the production
context); the bucketed path is eager — no CUDA graphs.
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
        # First call: eager prime (learns the output type). Lengths are
        # published for this call and consumed on return.
        model.embed_multimodal(
            audio_features=feats,
            encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
        )
        # Consume-once: lengths do not outlive the embed_multimodal call.
        assert encoder_wrapper._lengths is None

        # Second call: lengths are re-published, so the bucketed path groups by
        # rung — proving the published lengths ARRIVE at the wrapper.
        calls_before = len(model.encoder.calls)
        model.embed_multimodal(
            audio_features=feats,
            encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
        )
    new_shapes = model.encoder.calls[calls_before:]
    assert sorted(new_shapes) == sorted([(1, 40, D), (2, 200, D), (1, 100, D)]), new_shapes


def test_cumsum_encoder_runs_bucketed_path_not_fallback(bucket_env):
    model = ToyModel().cuda()
    assert install_bucketed_encoders(model) == ["encoder"]

    feats = make_features()
    with torch.inference_mode():
        # First eligible call: eager (learns the output type).
        model.embed_multimodal(
            audio_features=feats,
            encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
        )
        calls_after_prime = len(model.encoder.calls)
        out = model.embed_multimodal(
            audio_features=feats,
            encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
        )
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

    # (c) Execution evidence: the bucketed call re-entered the encoder with
    #     per-rung window shapes (eager path), not the full-batch fallback.
    bucketed_shapes = model.encoder.calls[calls_after_prime:calls_after_bucketed]
    assert sorted(bucketed_shapes) == sorted([(1, 40, D), (2, 200, D), (1, 100, D)]), bucketed_shapes


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
        out = model.embed_multimodal(
            audio_features=feats,
            feature_attention_mask=mask,
            encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
        )

    assert out.shape == (len(LENGTHS), 200, D), out.shape
    for row, length in enumerate(LENGTHS):
        solo = wrapper._original(feats[row : row + 1, :length], mask[row : row + 1, :length])[0]
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


def test_length_returning_encoder_buckets_losslessly(bucket_env):
    """(features, lengths) encoders must bucket losslessly: each row's true-length
    prefix equals its solo result and the output lengths are the true per-item
    lengths (not rung-padded)."""
    model = LengthReturningToyModel().cuda()
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    feats = make_features()
    lens = torch.tensor(LENGTHS, dtype=torch.long)  # host-side 1-D lengths

    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_lens=lens)
        out_features, out_lens = model.embed_multimodal(audio_features=feats, audio_lens=lens)

    assert out_features.shape[0] == len(LENGTHS)
    assert out_lens.shape == (len(LENGTHS),)
    for row, length in enumerate(LENGTHS):
        solo_features, solo_len = wrapper._original(feats[row : row + 1, :length], lens[row : row + 1])
        torch.testing.assert_close(out_features[row, :length], solo_features[0], rtol=0, atol=1e-6)
        assert out_lens[row].item() == solo_len.item()


def test_length_returning_encoder_shim_nontuple_falls_back(bucket_env):
    """A length-carrying ``_aux_spec`` (learned from a ``(features, lengths)``
    prime call) must not silently slice a bare-tensor shim output: the wrapper
    disables and returns the eager (original) result instead."""
    model = LengthReturningToyModel().cuda()
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    feats = make_features()
    lens = torch.tensor(LENGTHS, dtype=torch.long)

    with torch.inference_mode():
        # Prime call: the real (features, lengths) output teaches _aux_spec.
        model.embed_multimodal(audio_features=feats, audio_lens=lens)
        assert wrapper._aux_spec is not None and wrapper._aux_spec[0] == "lengths"

        eager_features, eager_lens = wrapper._original(feats, lens)

        # A shim that returns a bare tensor (not a (features, lengths) tuple),
        # which the guarded _tensor_forward must reject rather than slice.
        wrapper._shim = lambda *args, **kwargs: args[0]

        out = model.embed_multimodal(audio_features=feats, audio_lens=lens)

    assert wrapper._disabled is True
    torch.testing.assert_close(out[0], eager_features, rtol=0, atol=1e-6)
    torch.testing.assert_close(out[1], eager_lens, rtol=0, atol=1e-6)


def test_context_margin_preserves_conv_tail(bucket_env):
    """A conv front that reads ``length + bucketing_context_margin`` frames must
    not have its context tail zeroed: the wrapper copies the extra frames, so the
    bucketed output matches the eager conv over the full real window."""
    from torch.nn import functional as F

    class ConvContextEncoder(nn.Module):
        # Mirrors StepAudio2's padding_mels: the reported length is 2 less than
        # the number of real mel frames, which the 3-tap conv still reads.
        bucketing_context_margin = 2

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
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    # Real content is [0, length + margin); garbage tail after the conv context.
    lengths = [40, 120, 150, 75]  # max + margin <= largest rung (200)
    feats = torch.randn(len(lengths), T, D, device="cuda")
    for row, length in enumerate(lengths):
        feats[row, length + 2 :] += 3.0
    lens = torch.tensor(lengths, dtype=torch.long)

    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_lens=lens)  # prime
        out, out_lens = model.embed_multimodal(audio_features=feats, audio_lens=lens)

    assert out.shape[0] == len(lengths)
    for row, length in enumerate(lengths):
        # The eager reference feeds the full real window (length + context margin).
        solo, solo_len = wrapper._original(feats[row : row + 1, : length + 2], lens[row : row + 1])
        torch.testing.assert_close(out[row, :length], solo[0, :length], rtol=0, atol=1e-6)
        assert out_lens[row].item() == solo_len.item()


def test_time_axis_contract_wins_for_all_short_batch(bucket_env, monkeypatch):
    """A channels-first encoder (mel on dim 1, time on dim 2) must keep
    bucketing for an all-short batch via its declared ``bucketing_time_axis=2``.
    The length-scale heuristic would otherwise guess axis 1 (longest <= mel bins)
    and hand the conv the wrong channel count, disabling the wrapper."""
    monkeypatch.delenv("VLLM_OMNI_AUDIO_BUCKET_TIME_AXIS", raising=False)

    class ChannelsFirstConvEncoder(nn.Module):
        bucketing_time_axis = 2  # (B, mel, T): time on dim 2

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
    assert install_bucketed_encoders(model) == ["encoder"]
    wrapper = model.encoder.forward

    # All items shorter than the 128 mel bins: the heuristic (longest > dim1)
    # would wrongly pick axis 1 and conv1 would see rung channels, not 128.
    mel, T_feat = 128, 120
    lengths = [30, 60, 40, 80]
    feats = torch.randn(len(lengths), mel, T_feat, device="cuda")
    for row, length in enumerate(lengths):
        feats[row, :, length:] += 3.0
    lens = torch.tensor(lengths, dtype=torch.long)

    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, audio_lens=lens)  # prime
        out, out_lens = model.embed_multimodal(audio_features=feats, audio_lens=lens)

    assert wrapper._disabled is False, "an all-short batch must not disable bucketing"
    assert wrapper._axis == 2, "declared bucketing_time_axis=2 must win over the heuristic"
    for row, length in enumerate(lengths):
        solo, solo_len = wrapper._original(feats[row : row + 1, :, :length], lens[row : row + 1])
        torch.testing.assert_close(out[row, :length], solo[0, :length], rtol=0, atol=1e-6)
        assert out_lens[row].item() == solo_len.item()


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
    computed per rung, and exact output where rung == true length."""
    monkeypatch.setenv("VLLM_OMNI_AUDIO_BUCKET_SECONDS", "3,8,30")  # rungs 300/800/3000
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
        hidden = model.embed_multimodal(audio_features=feats, encoder_bucket_frames=torch.tensor(lengths))

    # Engaged: the bucketed rung windows use the variable-length recipe; the
    # fixed-window forward only runs once, at the eager prime, with the full
    # 3000-frame input.
    assert raw_calls == [(3, 80, 3000)], f"fixed-window forward calls: {raw_calls}"

    # Exact where the rung equals the true length (no padding involved).
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
        out = model.embed_multimodal(audio_features=feats, encoder_bucket_frames=torch.tensor([100]))

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


def test_lengths_do_not_outlive_embed_multimodal(bucket_env):
    """Published lengths are consumed within the embed_multimodal call; a direct
    encoder call afterwards must not reuse them (it falls back to the full
    batch)."""
    model = ToyModel().cuda()
    install_bucketed_encoders(model)
    wrapper = model.encoder.forward

    feats = make_features()
    with torch.inference_mode():
        model.embed_multimodal(audio_features=feats, encoder_bucket_frames=torch.tensor(LENGTHS))
        assert wrapper._lengths is None  # consumed on return

        # A direct call without fresh lengths falls back to the full width.
        calls_before = len(model.encoder.calls)
        out = model.encoder(feats).last_hidden_state
    assert out.shape == (len(LENGTHS), 240, D), out.shape
    assert model.encoder.calls[calls_before:] == [(len(LENGTHS), 240, D)]


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


def test_encoder_bucket_frames_keep_on_cpu(bucket_env):
    """encoder_bucket_frames must be registered keep_on_cpu, so the serving path
    never moves it to device and frames.tolist() stays a CPU op (no D2H sync)."""

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

    # Concrete CovoAudio processor registers the field keep_on_cpu.
    proc = CovoAudioMultiModalProcessor.__new__(CovoAudioMultiModalProcessor)
    proc.info = _Info()
    proc.dummy_inputs = None
    fields = proc._get_mm_fields_config({}, {})
    assert fields["encoder_bucket_frames"].field.keep_on_cpu is True

    # The D2H hazard this avoids: .tolist() on a CUDA tensor syncs.
    model = ToyModel().cuda()
    install_bucketed_encoders(model)
    feats = make_features()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.inference_mode():
            # CPU frames (the keep_on_cpu delivery) must not sync.
            model.embed_multimodal(
                audio_features=feats,
                encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64),
            )
            # CUDA frames would sync -> raises under sync-debug-error.
            with pytest.raises(RuntimeError):
                model.embed_multimodal(
                    audio_features=feats,
                    encoder_bucket_frames=torch.tensor(LENGTHS, dtype=torch.int64, device="cuda"),
                )
    finally:
        torch.cuda.set_sync_debug_mode("default")
