# SPDX-License-Identifier: Apache-2.0
"""Fun-Audio-Chat-8B Stage 1: CRQ token IDs → WAV via CosyVoice3 token2wav.

Analogous to FishSpeechDACDecoder — a LLM_GENERATION stage that takes CRQ token
IDs from Stage 0 and produces audio via cosyvoice_detokenizer.token2wav().

CosyVoice3 is loaded lazily on first forward() call from _VOCODER_PATH.
The fun_audio_chat main model weights (Stage 0) are NOT loaded here — load_weights()
is a no-op.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
# Default to the reference repo clone + its CosyVoice submodule, and the
# HF-downloaded Fun-CosyVoice3 weights checked in under pretrained_models/.
# All three are override-able via env vars.
_FUN_REF = os.environ.get(
    "FUN_AUDIO_REF_PATH",
    str(Path(__file__).parents[5] / "src" / "funaudiochat"),
)
_COSY_SRC = os.environ.get(
    "FUN_AUDIO_COSYVOICE_PATH",
    str(Path(_FUN_REF) / "third_party" / "CosyVoice"),
)
_VOCODER_PATH = os.environ.get(
    "FUN_AUDIO_VOCODER_PATH",
    str(Path(__file__).parents[5] / "pretrained_models" / "Fun-CosyVoice3-0.5B-2512"),
)


class FunAudioChatToken2Wav(nn.Module):
    """Stage 1: CRQ token IDs → WAV via CosyVoice3.model.token2wav().

    Used by the funaudiochat pipeline as the final audio-generation stage.
    No vllm PagedAttention or weight loading — CosyVoice3 loads its own weights.
    """

    input_modalities = "audio"
    have_multimodal_outputs: bool = True
    has_preprocess: bool = False
    has_postprocess: bool = False
    requires_raw_input_tokens: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self._cosyvoice3: Any = None

    # ── Required by vllm model runner ─────────────────────────────────────────

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> None:
        return None

    def make_empty_intermediate_tensors(self, *args: Any, **kwargs: Any) -> IntermediateTensors:
        return IntermediateTensors({})

    # ── Forward ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """CRQ token IDs → WAV via token2wav(cosyvoice3, tokens).

        input_ids: flat CRQ token IDs from stage input processor, shape [N].
        Returns OmniOutput with multimodal_outputs={"audio": [wav_tensor], "sr": [sr_tensor]}.
        """
        empty_out = OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio": [torch.zeros(1, 0)], "sr": [torch.tensor(24000)]},
        )

        # Short-circuit dummy/empty inputs BEFORE loading CosyVoice. vllm's
        # worker warmup (_dummy_run) feeds raw-token stages a non-empty tensor
        # of dummy IDs (all zeros). Those tokens pass the later 0<=t<6561
        # codebook filter, so a naive `numel()==0` check does NOT exclude them
        # and CosyVoice ends up loading + running during warmup. Detect:
        #   a) truly empty input
        #   b) all-zero input (dummy-run pattern)
        # Any real CRQ stream will contain a mix of non-zero codebook indices
        # almost immediately; a stream that is all zeros is never a real
        # generation from our decoder (reference greedy emits BOS=6561 first).
        if input_ids is None or input_ids.numel() == 0:
            logger.warning("FunAudioChatToken2Wav: empty input_ids — returning empty audio")
            return empty_out
        if bool((input_ids == 0).all().item()):
            logger.debug(
                "FunAudioChatToken2Wav: all-zero input (dummy warmup) — skipping CosyVoice load"
            )
            return empty_out

        # Now it's a real request — lazy-load CosyVoice3.
        self._ensure_cosyvoice3()

        tokens = input_ids.reshape(-1).tolist()
        valid_tokens = [int(t) for t in tokens if 0 <= int(t) < 6561]
        if not valid_tokens:
            logger.warning("FunAudioChatToken2Wav: no valid CRQ tokens (0-6560) in input")
            return empty_out

        try:
            wav = self._run_token2wav(valid_tokens)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # [1, T]
            sr = torch.tensor(self._cosyvoice3.sample_rate, dtype=torch.int32)
            logger.info(
                "FunAudioChatToken2Wav.forward: synthesised wav shape=%s sr=%d",
                tuple(wav.shape), int(sr.item()),
            )
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": [wav.cpu()], "sr": [sr]},
            )
        except Exception as exc:
            logger.error("FunAudioChatToken2Wav.forward: token2wav failed: %s", exc)
            raise

    # ── Weight loading (no-op) ─────────────────────────────────────────────────

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _name, _weight in weights:
            pass  # consume iterator; CosyVoice3 loads from _VOCODER_PATH in forward()
        return set()

    # ── Private ───────────────────────────────────────────────────────────────

    def _ensure_cosyvoice3(self) -> None:
        if self._cosyvoice3 is not None:
            return
        for p in [_FUN_REF, _COSY_SRC, str(Path(_COSY_SRC) / "third_party" / "Matcha-TTS")]:
            if p not in sys.path:
                sys.path.insert(0, p)
        from cosyvoice.cli.cosyvoice import CosyVoice3  # type: ignore[import]
        self._cosyvoice3 = CosyVoice3(_VOCODER_PATH, load_trt=False, load_vllm=False, fp16=False)
        logger.info("FunAudioChatToken2Wav: CosyVoice3 loaded from %s", _VOCODER_PATH)

    def _run_token2wav(self, tokens: list[int]) -> torch.Tensor:
        from utils.cosyvoice_detokenizer import token2wav  # type: ignore[import]
        speech = token2wav(
            self._cosyvoice3, tokens, embedding=None, token_hop_len=25 * 30, pre_lookahead_len=3
        )
        return speech
