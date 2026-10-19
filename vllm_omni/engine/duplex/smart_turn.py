# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Smart Turn v3.2 end-of-turn classifier for ``turn_detection: semantic_vad``.

Silero answers *is someone speaking*; Smart Turn answers *has the speaker
finished*. The classifier is a Whisper-Tiny encoder with a linear head that
takes the last 8 s of the current turn and returns the probability that the
turn is complete (pipecat-ai/smart-turn, BSD-2-Clause).

The split follows :mod:`vllm_omni.engine.duplex.vad`: :class:`SmartTurnBackend`
is stateless per call and shared by every session in the engine, and
:class:`SmartTurnBackendProvider` resolves, verifies and loads it once. The
per-session state (turn audio, pause bookkeeping) lives in
``SemanticTurnDetector``.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
from vllm.logger import init_logger

from vllm_omni.engine.duplex.vad import ServerVADUnavailableError

logger = init_logger(__name__)

SMART_TURN_REPO_ID = "pipecat-ai/smart-turn-v3"
SMART_TURN_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
SMART_TURN_FILENAME = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"

SMART_TURN_SAMPLE_RATE_HZ = 16_000
SMART_TURN_WINDOW_SECONDS = 8
SMART_TURN_WINDOW_SAMPLES = SMART_TURN_SAMPLE_RATE_HZ * SMART_TURN_WINDOW_SECONDS


class SemanticVADUnavailableError(ServerVADUnavailableError):
    """Smart Turn could not be loaded; a subclass so server-VAD handlers still catch it."""


def smart_turn_window(audio: np.ndarray) -> np.ndarray:
    """The model's input window: the last 8 s, zero-padded at the *beginning*.

    Padding goes in front so the speech always ends at the end of the window,
    which is how the model was trained (``audio_utils.truncate_audio_to_last_n_seconds``).
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= SMART_TURN_WINDOW_SAMPLES:
        return np.ascontiguousarray(audio[-SMART_TURN_WINDOW_SAMPLES:])
    window = np.zeros(SMART_TURN_WINDOW_SAMPLES, dtype=np.float32)
    if audio.size:
        window[-audio.size :] = audio
    return window


class SmartTurnBackend:
    """Shared ONNX Runtime Smart Turn v3 classifier running on CPU."""

    sample_rate_hz = SMART_TURN_SAMPLE_RATE_HZ

    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - platform packaging supplies ORT.
            raise SemanticVADUnavailableError("semantic_vad requires ONNX Runtime") from exc
        try:
            from transformers import WhisperFeatureExtractor
        except ImportError as exc:  # pragma: no cover - vLLM depends on transformers.
            raise SemanticVADUnavailableError("semantic_vad requires transformers") from exc

        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1

        self.model_path = Path(model_path)
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        input_names = {item.name for item in self._session.get_inputs()}
        if "input_features" not in input_names:
            raise SemanticVADUnavailableError(f"Unsupported Smart Turn ONNX input contract: {sorted(input_names)}")
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=SMART_TURN_WINDOW_SECONDS)
        self._warm_up()

    def _warm_up(self) -> None:
        self.predict(np.zeros(SMART_TURN_SAMPLE_RATE_HZ, dtype=np.float32))

    def features(self, audio: np.ndarray) -> np.ndarray:
        """Whisper log-mel features of the 8 s window, shaped ``[1, 80, 800]``."""
        inputs = self._feature_extractor(
            smart_turn_window(audio),
            sampling_rate=self.sample_rate_hz,
            return_tensors="np",
            padding="max_length",
            max_length=SMART_TURN_WINDOW_SAMPLES,
            truncation=True,
            do_normalize=True,
        )
        return np.ascontiguousarray(inputs.input_features, dtype=np.float32)

    def score(self, features: np.ndarray) -> float:
        """Run the classifier on precomputed ``features`` (see :meth:`features`)."""
        output = self._session.run(None, {"input_features": features})
        return float(np.asarray(output[0]).reshape(-1)[0])

    def predict(self, audio: np.ndarray) -> float:
        """Probability that the turn ending at the end of ``audio`` (16 kHz) is complete."""
        # ONNX Runtime permits concurrent Run calls on one CPU session, and the
        # feature extractor keeps no per-call state, so sessions can share this
        # instance. (With torch installed the extractor computes the STFT with
        # torch on CPU: about 2 ms for the 8 s window.)
        return self.score(self.features(audio))


class SmartTurnBackendProvider:
    """Resolve, verify and load the one Smart Turn backend of an engine process."""

    def __init__(self, *, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._backend: SmartTurnBackend | None = None
        self._lock = threading.Lock()

    def get(self) -> SmartTurnBackend:
        """The shared backend, loaded on first use.

        Unlike Silero there is no package fallback, so a missing artifact is an
        error either way; the message says how to provide one.
        """
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is None:
                try:
                    path = self._resolve_local_artifact()
                    self._verify_checksum(path)
                    self._backend = SmartTurnBackend(path)
                except SemanticVADUnavailableError:
                    raise
                except Exception as exc:
                    # Filesystem, ORT initialization and warm-up failures all
                    # disable this detector through the same serving contract.
                    raise SemanticVADUnavailableError(
                        "Could not initialize Smart Turn semantic VAD; check artifact access and "
                        "duplex_session.semantic_vad_model_path, ONNX Runtime and transformers"
                    ) from exc
                logger.info("Loaded Smart Turn semantic VAD model from %s", path)
        return self._backend

    def _resolve_local_artifact(self) -> Path:
        if self.model_path:
            path = Path(self.model_path).expanduser()
            if path.is_file():
                return path
            raise SemanticVADUnavailableError(f"Configured Smart Turn model does not exist: {path}")
        from vllm.transformers_utils.repo_utils import try_get_local_file

        cached = try_get_local_file(
            model=SMART_TURN_REPO_ID,
            file_name=SMART_TURN_FILENAME,
            revision=SMART_TURN_REVISION,
        )
        if isinstance(cached, Path) and cached.is_file():
            return cached
        raise SemanticVADUnavailableError(
            "semantic_vad needs the Smart Turn ONNX artifact: set duplex_session.semantic_vad_model_path "
            f"or pre-download {SMART_TURN_REPO_ID}/{SMART_TURN_FILENAME} at revision {SMART_TURN_REVISION} "
            "into the Hugging Face cache"
        )

    def _verify_checksum(self, path: Path) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SMART_TURN_SHA256:
            raise SemanticVADUnavailableError(
                f"Smart Turn checksum mismatch for {path}: expected {SMART_TURN_SHA256}, got {digest}"
            )
