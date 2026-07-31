# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage-0 audio ingress tests for the Qwen3-Omni duplex path.

These use the checkpoint's REAL ``WhisperFeatureExtractor`` with a stubbed
audio tower, so they verify the mel geometry and the reservation invariant
without needing GPU weights. They skip when the checkpoint is unavailable.

The mel-shape assertion is the important one: Whisper's extractor pads to its
30 s ``n_samples`` by default, which yields 3000 mel frames for a 1 s chunk
instead of 100 -- a 230x over-run of the 13-slot reservation that the model
runner would absorb silently.
"""

from __future__ import annotations

import base64

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from vllm_omni.experimental.fullduplex.qwen3omni.policy import (  # noqa: E402
    Qwen3OmniDuplexPolicy,
)
from vllm_omni.experimental.fullduplex.qwen3omni.stage0 import (  # noqa: E402
    Qwen3OmniStage0DuplexRuntime,
)

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
HIDDEN = 2048
CHUNK = Qwen3OmniDuplexPolicy.CHUNK_SAMPLES


class _FakeTower(torch.nn.Module):
    """Stands in for ``Qwen3OmniMoeAudioEncoder``.

    Returns ``sum(aftercnn_lens)`` rows, which is the contract
    ``_process_audio_input`` relies on when it splits the tower output by
    ``audio_output_lengths``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.dtype = torch.float32
        self.calls: list[tuple[tuple[int, ...], int, int]] = []

    def forward(self, input_features, feature_lens, aftercnn_lens):
        self.calls.append((tuple(input_features.shape), int(feature_lens[0]), int(aftercnn_lens[0])))
        return torch.ones(int(aftercnn_lens.sum()), HIDDEN)


class _FakeThinker:
    def __init__(self) -> None:
        self.audio_tower = _FakeTower()


class _FakeModel:
    def __init__(self) -> None:
        self.thinker = _FakeThinker()


@pytest.fixture(scope="module")
def feature_extractor():
    transformers = pytest.importorskip("transformers")
    try:
        processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - offline / no checkpoint
        pytest.skip(f"{MODEL_ID} processor unavailable: {exc}")
    return processor.feature_extractor


@pytest.fixture
def runtime(feature_extractor):
    rt = Qwen3OmniStage0DuplexRuntime(_FakeModel())
    rt._feature_extractor = feature_extractor
    return rt


def _payload(num_samples: int) -> dict[str, object]:
    return {
        "audio": base64.b64encode(np.zeros(num_samples, dtype="<f4").tobytes()).decode("ascii"),
        "format": Qwen3OmniDuplexPolicy.PCM_FORMAT,
        "sample_rate_hz": Qwen3OmniDuplexPolicy.SAMPLE_RATE_HZ,
        "num_samples": num_samples,
    }


def _duplex(num_samples: int, *, seq: int = 1, epoch: int = 0, session_id: str = "s1") -> dict[str, object]:
    return {
        "data_plane": True,
        "session_id": session_id,
        "incarnation": 1,
        "epoch": epoch,
        "seq": seq,
        "payload": _payload(num_samples),
    }


def test_one_second_chunk_yields_thirteen_embeddings(runtime) -> None:
    """1 s -> mel (128, 100) -> 13 embeddings, matching the reservation."""
    embeds = runtime.build_append_embeddings(duplex=_duplex(CHUNK), token_offset=0, prompt_len=13)
    assert embeds is not None
    assert embeds.shape == (13, HIDDEN)

    mel_shape, feature_lens, aftercnn_lens = runtime._model.thinker.audio_tower.calls[-1]
    # Guards the Whisper default-padding trap: 3000 here instead of 100 means
    # padding="longest"/truncation=False was lost.
    assert mel_shape == (128, 100), f"tower saw mel {mel_shape}, expected (128, 100)"
    assert feature_lens == 100
    assert aftercnn_lens == 13


def test_partial_chunks_accumulate(runtime) -> None:
    assert runtime.build_append_embeddings(duplex=_duplex(CHUNK // 2), token_offset=0, prompt_len=13) is None
    embeds = runtime.build_append_embeddings(
        duplex=_duplex(CHUNK // 2, seq=2),
        token_offset=0,
        prompt_len=13,
    )
    assert embeds is not None
    assert embeds.shape[0] == 13


def test_multi_chunk_append_encodes_with_context(runtime) -> None:
    """The whole turn is encoded once, not each second in isolation.

    The tower attends across 8 one-second chunks, so an isolated chunk sees no
    context. Measured against a whole-utterance encode of the same 4 s audio:
    per-chunk cosine 0.844 (min 0.154) vs cumulative 0.949 (min 0.727).
    """
    embeds = runtime.build_append_embeddings(duplex=_duplex(CHUNK * 3), token_offset=0, prompt_len=39)
    assert embeds.shape[0] == 39
    calls = runtime._model.thinker.audio_tower.calls
    assert len(calls) == 1, "one encode over the whole span, not three"
    mel_shape, feature_lens, _ = calls[-1]
    assert feature_lens == 300, "3 s of audio at 100 mel frames per second"
    assert mel_shape == (128, 300)


def test_successive_appends_re_encode_the_whole_turn(runtime) -> None:
    """Each append re-encodes the turn so far and returns only the new rows."""
    first = runtime.build_append_embeddings(duplex=_duplex(CHUNK, seq=1), token_offset=0, prompt_len=13)
    second = runtime.build_append_embeddings(duplex=_duplex(CHUNK, seq=2), token_offset=13, prompt_len=13)
    assert first.shape[0] == 13
    assert second.shape[0] == 13, "only the newly completed second is returned"
    calls = runtime._model.thinker.audio_tower.calls
    assert calls[-1][1] == 200, "second append encodes 2 s of context"


def test_replay_of_same_append_is_memoized(runtime) -> None:
    duplex = _duplex(CHUNK, seq=7)
    first = runtime.build_append_embeddings(duplex=duplex, token_offset=0, prompt_len=13)
    calls = len(runtime._model.thinker.audio_tower.calls)
    second = runtime.build_append_embeddings(duplex=duplex, token_offset=0, prompt_len=13)
    assert torch.equal(first, second)
    assert len(runtime._model.thinker.audio_tower.calls) == calls, "replay must not re-encode"


def test_sessions_have_independent_buffers(runtime) -> None:
    half = CHUNK // 2
    assert runtime.build_append_embeddings(duplex=_duplex(half, session_id="A"), token_offset=0, prompt_len=13) is None
    assert runtime.build_append_embeddings(duplex=_duplex(half, session_id="B"), token_offset=0, prompt_len=13) is None
    assert len(runtime.sessions) == 2


def test_embedding_reservation_mismatch_raises(runtime) -> None:
    """A miscount must fail loudly; the model runner would truncate silently."""

    class _OverProducingTower(_FakeTower):
        def forward(self, input_features, feature_lens, aftercnn_lens):
            super().forward(input_features, feature_lens, aftercnn_lens)
            return torch.ones(int(aftercnn_lens.sum()) + 1, HIDDEN)

    runtime._model.thinker.audio_tower = _OverProducingTower()
    with pytest.raises(RuntimeError, match="reserved"):
        runtime.build_append_embeddings(duplex=_duplex(CHUNK), token_offset=0, prompt_len=13)


def test_non_pcm_f32le_payload_is_rejected(runtime) -> None:
    duplex = _duplex(16)
    duplex["payload"]["format"] = "pcm_s16le"
    with pytest.raises(ValueError, match="unsupported duplex audio format"):
        runtime.build_append_embeddings(duplex=duplex, token_offset=0, prompt_len=1)
