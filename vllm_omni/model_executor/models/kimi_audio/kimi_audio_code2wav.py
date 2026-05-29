# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-Audio code2wav stage: audio-token IDs -> 24 kHz waveform via the
vendored flow-matching detokenizer + BigVGAN vocoder. The flow-matching
half loads from ``<model_path>/audio_detokenizer/``; the BigVGAN vocoder
loads weight_norm-folded weights from a separate HF repo (see
``modeling_bigvgan.DEFAULT_HF_REPO``). Neither is routed through vLLM's
weight loader."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.kimi_audio.cuda_graph_decoder_wrapper import (
    KimiAudioCudaGraphDecoderWrapper,
)
from vllm_omni.model_executor.models.kimi_audio.detokenizer import (
    detokenize_noref,
    get_audio_detokenizer,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Audio token IDs occupy [KIMIA_TOKEN_OFFSET, vocab_size); subtract the
# offset before handing tokens to the detokenizer, which expects local
# audio-code IDs in [0, 16384).
KIMIA_TOKEN_OFFSET = 152064
NUM_AUDIO_SPECIAL_TOKENS = 512
OUTPUT_SAMPLE_RATE = 24000


class KimiAudioCode2Wav(nn.Module):
    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._detokenizer: Any | None = None
        self._first_chunk_tokens = 100
        self._chunk_tokens = 150
        self._cuda_graph_wrapper: KimiAudioCudaGraphDecoderWrapper | None = None

    def _ensure_detokenizer_loaded(self) -> None:
        if self._detokenizer is not None:
            return
        # ``get_audio_detokenizer`` pins onto torch.cuda.current_device() —
        # the stage process must own the right CUDA device first.
        dtype = self.vllm_config.model_config.dtype
        self._detokenizer = get_audio_detokenizer(self.model_path, dtype=dtype)
        logger.info("KimiAudioCode2Wav detokenizer loaded from %s (dtype=%s)", self.model_path, dtype)
        self._maybe_install_cuda_graph_wrapper()

    def _maybe_install_cuda_graph_wrapper(self) -> None:
        """Wrap ``vocoder.decode_mel`` with a CUDA graph capture for the
        async-chunk path. Skipped on CPU, when no chunk size is
        configured (sync path), or if warmup fails.

        Note: the stage's ``enforce_eager: true`` only governs vLLM's
        outer generation-loop graph capture (which doesn't apply to the
        custom code2wav forward); ``decode_mel`` is a fixed-shape inner
        call we can capture independently."""
        if not torch.cuda.is_available():
            return

        # Async-chunk YAML supplies these via connector ``extra``; sync
        # YAML omits the block. With no chunk frames configured the
        # wrapper has no shapes to capture, so skip cleanly.
        connector_cfg = getattr(self.vllm_config.model_config, "stage_connector_config", None) or {}
        extra = connector_cfg.get("extra", {})
        codec_chunk_frames = int(extra.get("codec_chunk_frames", 0))
        codec_left = int(extra.get("codec_left_context_frames", 0))
        if codec_chunk_frames == 0:
            logger.info("KimiAudioCode2Wav: no codec_chunk_frames configured; skipping CUDA Graph capture")
            return

        vocoder = self._detokenizer.vocoder

        wrapper = KimiAudioCudaGraphDecoderWrapper(vocoder=vocoder, enabled=True)
        wrapper.warmup(
            device=torch.device(f"cuda:{torch.accelerator.current_device_index()}"),
            dtype=self._detokenizer.dtype,
            codec_chunk_frames=codec_chunk_frames,
            codec_left_context_frames=codec_left,
        )

        self._original_decode_mel = vocoder.decode_mel
        vocoder.decode_mel = wrapper.decode_mel
        self._cuda_graph_wrapper = wrapper
        logger.info("KimiAudioCode2Wav: CUDA Graph wrapper installed on vocoder.decode_mel")

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Dummy — stage ignores token embeddings but the runner needs a shape."""
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None,
    ) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        return [ids]

    def _decode_one(self, audio_codes: torch.Tensor) -> torch.Tensor:
        self._detokenizer.clear_states()
        tokens = audio_codes.unsqueeze(0)
        wav = detokenize_noref(self._detokenizer, tokens)
        return wav.squeeze(0).float()

    def _decode_chunk_streaming(
        self,
        audio_codes: torch.Tensor,
        is_final: bool,
        ode_step: int = 30,
    ) -> torch.Tensor:
        """Decode one chunk via the streaming path. Detokenizer keeps
        per-request state (KV cache, mel/wav history) across calls; caller
        must ``clear_states`` between requests.

        ``upsample_factor=4`` and ``ode_step=30`` match upstream
        ``KimiAudio.detokenize_audio`` — the 12.5 Hz semantic tokens expand
        to the 50 Hz mel rate BigVGAN expects, and the flow-matching ODE
        gets the full integration budget."""
        assert self._detokenizer is not None
        tokens = audio_codes.unsqueeze(0)
        wav = self._detokenizer.detokenize_streaming(
            tokens,
            ode_step=ode_step,
            is_final=is_final,
            upsample_factor=4,
        )
        return wav.squeeze(0).float()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        self._ensure_detokenizer_loaded()
        sr_tensor = torch.tensor(OUTPUT_SAMPLE_RATE, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        # input_ids may arrive on CPU (stage-0 → stage-1 hop in sync mode
        # without a GPU-resident connector). The detokenizer submodules
        # (DiT + BigVGAN) live on CUDA, so downstream ops would mix
        # devices. Normalize to the model's device here so every call path
        # below sees CUDA tensors.
        target_device = next(self._detokenizer.semantic_fm.parameters()).device
        ids = input_ids.reshape(-1).to(device=target_device, dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))
        async_chunk = bool(self.vllm_config.model_config.async_chunk)

        # Per-request is_final hint set by kimi2code2wav_async_chunk.
        finished_flags: list[bool] = [not async_chunk] * len(request_ids_list)
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(finished_flags):
                    break
                fin = info.get("finished")
                if isinstance(fin, torch.Tensor):
                    finished_flags[i] = bool(fin.item())
                elif fin is not None:
                    finished_flags[i] = bool(fin)

        wav_outputs: list[torch.Tensor] = []
        for i, req_ids in enumerate(request_ids_list):
            if req_ids.numel() == 0:
                wav_outputs.append(empty)
                continue
            audio_mask = req_ids >= KIMIA_TOKEN_OFFSET
            audio_codes = req_ids[audio_mask] - KIMIA_TOKEN_OFFSET
            # Drop special-token region so control codes (BOS/EOS/etc.)
            # don't reach the flow-matching model.
            audio_codes = audio_codes[audio_codes < (16896 - NUM_AUDIO_SPECIAL_TOKENS)]
            if audio_codes.numel() == 0:
                wav_outputs.append(empty)
                continue
            if async_chunk:
                wav = self._decode_chunk_streaming(audio_codes, is_final=finished_flags[i])
            else:
                wav = self._decode_one(audio_codes)
            wav_outputs.append(wav)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": [w.reshape(1, -1) for w in wav_outputs],
                "sr": [sr_tensor] * len(wav_outputs),
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # No-op: detokenizer loads its own checkpoints. Drain the iterator
        # so the upstream loader doesn't complain about unconsumed weights.
        loaded: set[str] = set()
        for name, _ in weights:
            loaded.add(name)
        return loaded
