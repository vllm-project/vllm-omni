# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load MiniCPM-o 4.5 vocoder assets and reference features for BatchedToken2Wav.

The in-tree StepAudio2 core owns model loading and prompt extraction, including
its device-specific implementations. BatchedToken2Wav owns streaming execution.
"""

from __future__ import annotations

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
    StepAudio2Token2WavCore,
)

logger = init_logger(__name__)


def _resolve_device(device: str | torch.device | None) -> str:
    if device is not None:
        return str(device)
    try:
        from vllm_omni.platforms import current_omni_platform

        plat_dev = getattr(current_omni_platform, "device_type", None)
        if plat_dev:
            return str(plat_dev)
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cpu"


class MiniCPMO45Token2wav:
    """Asset and prompt-feature adapter around ``StepAudio2Token2WavCore``."""

    def __init__(
        self,
        model_path: str,
        float16: bool = False,
        n_timesteps: int = 10,
        device: str | torch.device | None = None,
        drop_upstream_chunk_att_buffers: bool = False,
    ):
        self.float16 = float16
        self.n_timesteps = n_timesteps
        self.device = _resolve_device(device)
        self._core = StepAudio2Token2WavCore(
            model_path=model_path,
            float16=float16,
            device=self.device,
            n_timesteps=n_timesteps,
            drop_upstream_chunk_att_buffers=drop_upstream_chunk_att_buffers,
        )
        # Eager-load so construction failures surface at init time (same as
        # the external Token2wav package), not on the first request.
        self._core._ensure_models_loaded()

        # The external ``stepaudio2.Token2wav`` class builds ``speech_window``
        # eagerly in ``__init__``. ``StepAudio2Token2WavCore`` only creates it
        # lazily inside ``setup_stream_for``, so materialize it here: the
        # 3-stage ``BatchedToken2Wav`` wrapper clones it at construction time.
        if self._core.speech_window is None:
            self._core.speech_window = torch.from_numpy(np.hamming(2 * self._core.source_cache_len)).to(
                device=self.device, dtype=torch.float32
            )

    # Assets consumed by BatchedToken2Wav; request state lives in that wrapper.

    @property
    def flow(self) -> torch.nn.Module:
        return self._core.flow

    @property
    def hift(self) -> torch.nn.Module:
        return self._core.hift

    @property
    def mel_cache_len(self) -> int:
        return self._core.mel_cache_len

    @property
    def source_cache_len(self) -> int:
        return self._core.source_cache_len

    @property
    def speech_window(self) -> torch.Tensor:
        return self._core.speech_window

    def _prepare_prompt_cpu(self, prompt_wav: str):
        return self._core._prepare_prompt_cpu(prompt_wav)

    def _prepare_prompt(self, prompt_wav: str, *, cpu_features=None):
        """Delegate prompt feature extraction to the wrapped core."""
        return self._core._prepare_prompt(prompt_wav, cpu_features=cpu_features)

    def enable_trt_spk_embedding(self, *, cpu_reference: bool = False) -> None:
        """Run the campplus speaker-embedding model on TensorRT."""
        self._core.enable_trt_spk_embedding(cpu_reference=cpu_reference)

    def enable_device_spk_embedding(self, *, use_trt: bool = True) -> None:
        """Run CAM++ on device, optionally without TensorRT on CUDA."""
        self._core.enable_device_spk_embedding(use_trt=use_trt)

    def enable_speaker_cuda_graph(self, vllm_config: VllmConfig) -> bool:
        """Replay the current Torch speaker forward without changing its math."""
        from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_torch import CampplusTorch
        from vllm_omni.model_executor.models.minicpmo_4_5.code2wav.reference_audio_graph import SpeakerCudaGraph

        speaker = self._core._spk_model
        if not isinstance(speaker, CampplusTorch) or speaker.device.type != "cuda":
            logger.info("MiniCPM-o: skipping speaker CUDA graphs; selected speaker is not CUDA Torch CAM++")
            return False
        self._core._spk_model = SpeakerCudaGraph(speaker, vllm_config)
        return True

    def enable_s3_cuda_graph(self, vllm_config: VllmConfig) -> bool:
        """Replay short reference tokenization with unchanged S3 weights/math."""
        from .reference_audio_graph import S3CudaGraph

        if self._core.device.type != "cuda":
            logger.info("MiniCPM-o: skipping S3 CUDA graphs off CUDA")
            return False
        tokenizer = self._core.audio_tokenizer
        if next(tokenizer.parameters()).device.type != "cuda":
            logger.info("MiniCPM-o: skipping S3 CUDA graphs; selected tokenizer is not on CUDA")
            return False
        self._core._s3_cuda_graph = S3CudaGraph(tokenizer, vllm_config)
        return True
