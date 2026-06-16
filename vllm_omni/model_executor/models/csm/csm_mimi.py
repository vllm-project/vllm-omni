# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSM-1B Stage-1 Mimi vocoder (code2wav). Frames of 32 codes -> PCM audio.

Stage 1 of the 2-stage CSM pipeline (A3 design §4.5). It is a pure, stateless-
per-frame decoder of the completed 32-code frames produced by Stage 0: the
backbone<->depth feedback is fully resolved in Stage 0, so nothing crosses back.

Body: REUSE ``_mimi_decode`` (csm.py:729 @ d92e35f7) -- float32 decode (autocast
corrupts codec audio, SKILL Phase-2 dtype rule), reserved-id clamp [0, 2047] on a
COPY before decode (ids 2048/2049/2050 must NEVER reach Mimi; its real codebook
size is 2048).

Invariants honored:
  * I1 (delta streaming): ``forward`` decodes ONLY this chunk's frames and emits
    the delta waveform; left context is trimmed off the front so chunk boundaries
    are seamless without re-emitting. Documented in ``forward``'s docstring.
  * I5 (per-request state): Mimi's streaming conv-state is the known
    "first-request-clean, later-requests-flicker" leak point (#233/#234). We use a
    per-request streaming context keyed by req id; a shared buffer would cross-talk
    under concurrency. Freed in ``on_requests_finished``.
  * "All return paths emit model_outputs" (SKILL): every branch returns an
    ``OmniOutput`` with a ``model_outputs`` list of matching length.

Shape contract from Stage 0 / the stage input processor (A3 §4.4): ``input_ids``
is a flat CODEBOOK-MAJOR ``[32 * F]`` int tensor per request; reshape to
``[32, F]`` -> Mimi ``decode`` wants ``[B, num_quantizers, codes_length]``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.csm.configuration_csm import CsmConfig
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

__all__ = ["CsmMimiVocoder"]

_NUM_CODEBOOKS = 32
_MIMI_CODEBOOK_SIZE = 2048  # reserved ids 2048/2049/2050 must not reach Mimi


class CsmMimiVocoder(nn.Module):
    """Stage-1 Mimi codec decoder for CSM (GenerationModelRunner)."""

    input_modalities = "audio"

    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config: CsmConfig = vllm_config.model_config.hf_config
        self.model_path: str = vllm_config.model_config.model

        self.num_codebooks = int(getattr(self.config, "num_codebooks", _NUM_CODEBOOKS))
        self.sample_rate = int(getattr(self.config, "codec_sample_rate", 24000))

        self._mimi_codec: nn.Module | None = None  # MimiModel
        self._device: torch.device | None = None
        self._lock = threading.Lock()

        # I5: per-request Mimi streaming conv-state, keyed by req id.
        self._stream_state_by_req: dict[str, Any] = {}

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # Vocoder ignores token embeddings; stable dummy for the runner.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        # Vocoder is non-AR (qwen3_tts_code2wav.py:89).
        return None

    # ------------------------------------------------------------------
    # Weight loading: codec_model.* from the same checkpoint
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the Mimi codec from the ``codec_model.*`` checkpoint prefix.

        REUSE shape: csm.py:404-410 codec load. Built lazily (after distributed
        init). Every returned name is a real registered parameter name under
        ``_mimi_codec`` (B2 validator contract).
        """
        with self._lock:
            if self._mimi_codec is None:
                try:
                    self._device = next(self.parameters()).device
                except StopIteration:
                    self._device = self.vllm_config.device_config.device
                self._build_codec()

        codec_weights: list[tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        for name, w in weights:
            if name.startswith("codec_model."):
                codec_weights.append((name[len("codec_model.") :], w))
            # All non-codec prefixes belong to Stage 0; drain + ignore here.

        if self._mimi_codec is not None and codec_weights:
            _, mc_unexpected = self._mimi_codec.load_state_dict(dict(codec_weights), strict=False)
            if list(mc_unexpected):
                logger.warning("CSM Stage-1 codec unexpected keys: %s", list(mc_unexpected)[:8])
            loaded |= {n for n, _ in self._mimi_codec.named_parameters(prefix="_mimi_codec")}
        return loaded

    def _build_codec(self) -> None:
        from transformers import AutoConfig, AutoModel

        hf_cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        codec_cfg = hf_cfg.codec_config
        self._mimi_codec = AutoModel.from_config(codec_cfg)
        # Codec MUST run in float32 (autocast corrupts audio; SKILL Phase-2).
        self._mimi_codec.to(device=self._device, dtype=torch.float32)
        self._mimi_codec.eval()
        for p in self._mimi_codec.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def _mimi_decode(self, frame_codes_qf: torch.Tensor) -> torch.Tensor:
        """Mimi decode: ``[Q=32, F]`` codes -> ``(samples,)`` fp32.

        REUSE: csm.py:729 ``_mimi_decode``. Reserved guard: clamp to [0, 2047] on a
        COPY (the reserved ids are valid frame-embed inputs upstream but must not
        reach the codec). ``decode`` wants ``[B, num_quantizers, codes_length]``.
        """
        codes = frame_codes_qf.clamp(min=0, max=_MIMI_CODEBOOK_SIZE - 1)
        audio_codes = codes.unsqueeze(0)  # (1, Q, F)
        with torch.no_grad():
            out = self._mimi_codec.decode(audio_codes)
        audio_values = out.audio_values  # (B, channels, samples) or (B, samples)
        wav = audio_values.reshape(audio_values.shape[0], -1)[0]
        return wav.to(torch.float32)

    def _split_request_ids(self, ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments (qwen3 pattern)."""
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = getattr(get_forward_context(), "ubatch_slices", None)
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode flat codebook-major ``[32*F]`` codes per request into audio.

        Streaming contract (I1): DELTA. Each call decodes only this chunk's frames
        and trims the leading ``left_context_size`` frames (decoder context only),
        so the emitted waveform is the new audio for this chunk -- never a re-decode
        of the full sequence. All return paths emit a ``model_outputs`` list whose
        length matches the request count.
        """
        sr_tensor = torch.tensor(self.sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        per_req = self._split_request_ids(ids, kwargs.get("seq_token_counts"))
        num_req = len(per_req)

        left_context_size = [0] * num_req
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= num_req:
                    break
                meta = info.get("meta", {}) if isinstance(info, dict) else {}
                if "left_context_size" in meta:
                    left_context_size[i] = int(meta["left_context_size"] or 0)

        q = self.num_codebooks
        device = self._device or input_ids.device
        audios: list[torch.Tensor] = [empty] * num_req
        srs: list[torch.Tensor] = [sr_tensor] * num_req

        for i, req_ids in enumerate(per_req):
            n = int(req_ids.numel())
            if n == 0 or n % q != 0:
                if n > 0:
                    logger.warning(
                        "CSM Stage-1 input_ids length %d not divisible by num_codebooks %d; skipping.",
                        n,
                        q,
                    )
                continue
            frames = n // q
            # [Q*F] codebook-major -> [Q, F]
            codes_qf = req_ids.reshape(q, frames).to(device)
            wav = self._mimi_decode(codes_qf)  # (samples,) fp32

            # I1 delta: trim the leading left-context frames (decoder context).
            ctx = left_context_size[i]
            samples_per_frame = int(getattr(self.config, "codec_samples_per_frame", 1920))
            start = max(0, ctx * samples_per_frame)
            if start < wav.shape[0]:
                wav = wav[start:]
            else:
                wav = empty
            audios[i] = wav.reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        """Passthrough (forward returns OmniOutput). REUSE shape: qwen3_tts_code2wav.py:425."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"CsmMimiVocoder expected OmniOutput, got {type(model_outputs)}")

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Free per-request Mimi streaming state on finish (I5, #233/#234)."""
        for req_id in finished_req_ids:
            self._stream_state_by_req.pop(str(req_id), None)
