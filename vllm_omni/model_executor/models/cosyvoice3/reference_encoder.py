# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosy reference artifacts produced once by a server's shared encoder worker."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from vllm_omni.utils.reference_encoding import content_key
from vllm_omni.utils.shared_reference_encoder import WireFormat

REFERENCE_CONDITIONING_KEY = "reference_conditioning"
_FIELDS = ("speech_feat", "speech_token", "speech_token_len", "embedding")


@dataclass(frozen=True)
class CosyReferenceArtifacts:
    speech_feat: torch.Tensor
    speech_token: torch.Tensor
    speech_token_len: torch.Tensor
    embedding: torch.Tensor

    @property
    def nbytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors())

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.speech_feat, self.speech_token, self.speech_token_len, self.embedding

    def processor_kwargs(self) -> dict[str, torch.Tensor]:
        # Match the local cache's ownership: a request cannot mutate the next hit.
        return {name: tensor.clone() for name, tensor in zip(_FIELDS, self.tensors(), strict=True)}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> CosyReferenceArtifacts:
        tensors = []
        for name in _FIELDS:
            tensor = value.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                raise ValueError("Cosy reference conditioning must contain four CPU tensors")
            tensors.append(tensor)
        artifact = cls(*tensors)
        feat, token, length, embedding = artifact.tensors()
        if (
            feat.ndim != 3
            or feat.shape[0] != 1
            or feat.dtype not in (torch.float16, torch.float32, torch.float64)
            or token.ndim != 2
            or token.shape[0] != 1
            or token.dtype not in (torch.int32, torch.int64)
            or length.numel() != 1
            or length.dtype not in (torch.int32, torch.int64)
            or int(length.reshape(-1)[0]) != token.shape[1]
            or embedding.ndim != 2
            or embedding.shape[0] != 1
            or embedding.dtype not in (torch.float16, torch.float32, torch.float64)
        ):
            raise ValueError("Invalid Cosy reference conditioning shapes or dtypes")
        return artifact


def _artifacts_from_arrays(arrays: list[np.ndarray]) -> CosyReferenceArtifacts:
    if len(arrays) != len(_FIELDS):
        raise ValueError("Cosy reference wire requires four arrays")
    return CosyReferenceArtifacts.from_mapping(
        {name: torch.from_numpy(array.copy()) for name, array in zip(_FIELDS, arrays, strict=True)}
    )


COSY_REFERENCE_WIRE = WireFormat(
    clip_to_array=lambda clip: (clip[0], clip[1]),
    clip_from_array=lambda array, sr: (array.copy(), sr),
    artifact_to_arrays=lambda artifact: [tensor.numpy() for tensor in artifact.tensors()],
    artifact_from_arrays=_artifacts_from_arrays,
)


class CosyVoice3ReferenceEncoder:
    """One worker owns the encoder weights and the authoritative bounded cache.

    The shared host submits one batch at a time to this worker. Duplicate clips
    in that batch share a future; later batches hit the same artifact cache,
    including when they originate from a different API process. Threads retain
    Cosy's existing S3 speech-token microbatcher for different cold references.
    """

    def __init__(self, model_path: str, config: dict, cache_bytes: int) -> None:
        from vllm.multimodal.parse import MultiModalDataParser

        from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3MultiModalProcessor
        from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config
        from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

        self.model_path = model_path
        # Cosy's constructor writes defaults after PretrainedConfig.__init__;
        # restore the resolved frontend overrides after that initialization.
        self.config = CosyVoice3Config()
        self.config.update(config)
        self.processor = object.__new__(CosyVoice3MultiModalProcessor)
        self.processor._ensure_cached_runtime_components(model_path, self.config)
        self.processor._speaker_cache = SpeakerEmbeddingCache(max_bytes=cache_bytes)
        self.processor._ensure_s3_model()
        self.resampler = MultiModalDataParser(target_sr=self.config.target_sr).audio_resampler
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="cosy-shared-ref")

    def _encode_one(self, clip: tuple[np.ndarray, int]) -> CosyReferenceArtifacts:
        from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import _audio_conditioning

        wav, sr = clip
        if sr <= 0 or wav.ndim != 1 or wav.size == 0:
            raise ValueError("Reference audio requires a positive sample rate and a non-empty mono waveform")
        audio = self.resampler.resample(wav, orig_sr=sr)
        with torch.inference_mode():
            result = _audio_conditioning(
                self.processor, (audio, self.config.target_sr), self.model_path, self.config, {}
            )
        return CosyReferenceArtifacts.from_mapping(result)

    def encode_batch(self, clips: list[tuple[np.ndarray, int]]) -> list[CosyReferenceArtifacts | Exception]:
        futures = {}
        ordered = []
        for wav, sr in clips:
            key = content_key(self.model_path, wav, sr)
            if key not in futures:
                futures[key] = self._executor.submit(self._encode_one, (wav, sr))
            ordered.append(futures[key])
        results = []
        for future in ordered:
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - one malformed clip must not fail the other references
                results.append(exc)
        return results
