# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SongGen single-stage model for vLLM-Omni.

SongGen (ICML 2025) is a 1.3B auto-regressive transformer for text-to-song
generation. It accepts music description text and lyrics as conditioning, plus
an optional reference voice for voice conditioning.

Architecture: SongGenForCausalLM (AR decoder with T5 text encoder) paired
with an X-Codec audio decoder. Both components run in a single AR worker
stage, following the VoxCPM-style single-stage generator pattern.

The upstream model's ``generate()`` handles the full pipeline:
  1. AR decoding produces X-Codec token sequences (8 codebooks).
  2. X-Codec immediately decodes tokens to a 16 kHz waveform.
  3. The waveform is returned directly (no separate decode stage needed).

Because ``generate()`` is a single blocking call, all audio arrives in one
shot (no incremental streaming). The generator yields one ``(waveform, True)``
tuple per request.

Prerequisites:
  pip install git+https://github.com/LiuZH-19/SongGen.git

Note on transformers version:
  SongGen's setup.py pins transformers<=4.43.3, while vllm-omni requires
  >=4.56.0. The pin reflects SongGen's tested range; if incompatibilities
  arise at runtime, install SongGen in a compatible environment or apply
  patches to the upstream SongGen code.
"""

from __future__ import annotations

import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Default sampling parameters matching the upstream README examples.
_DEFAULT_DO_SAMPLE = True
_DEFAULT_DESCRIPTION = "a pop song"


def _pick(info: dict, key: str, default):
    """Extract a scalar from an additional_information dict (list or plain value)."""
    val = info.get(key, default)
    if isinstance(val, list | tuple) and len(val) > 0:
        return val[0]
    return val if val is not None else default


class SongGenForGeneration(nn.Module):
    """Single-stage SongGen model with streaming-compatible audio output.

    Uses the VoxCPM-style generator pattern: ``generate()`` is wrapped in a
    per-request generator stored in ``self._stream_gens``. The AR scheduler
    keeps the request alive until ``compute_logits()`` emits EOS (which
    happens on the first and only yield, since ``generate()`` is synchronous).

    Per ``forward()`` docstring: this model emits **delta** output
    (``model_outputs`` contains the complete waveform on the single yielded
    step). Offline consumers receive the full waveform via consolidation;
    streaming consumers receive it as a single final chunk.
    """

    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True
    inject_omni_request_id_into_runtime_info = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_path: str = vllm_config.model_config.model

        self._model: nn.Module | None = None
        self._processor = None
        self._device: torch.device | None = None
        self._lock = threading.Lock()

        # Per-request streaming generators keyed by global_request_id.
        # The single-threaded AR worker serialises forward() calls, so no
        # additional locking is needed here.
        self._stream_gens: dict[str, Any] = {}
        # Per-row EOS mask used by compute_logits() to signal request finish.
        self._ar_last_chunk_flags: list[bool] = []

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        with self._lock:
            if self._model is not None:
                return None

            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device

            logger.info("Loading SongGen from %s on %s", self.model_path, device)

            try:
                from songgen import SongGenMixedForConditionalGeneration, SongGenProcessor
            except ImportError as exc:
                raise ImportError(
                    "SongGen requires the 'songgen' package. "
                    "Install it from: pip install git+https://github.com/LiuZH-19/SongGen.git"
                ) from exc

            model = SongGenMixedForConditionalGeneration.from_pretrained(
                self.model_path,
                attn_implementation="sdpa",
                torch_dtype=torch.float32,
            )
            model.to(device=device)
            model.eval()
            self._model = model
            logger.info("SongGen AR model loaded on %s", device)

            processor = SongGenProcessor(self.model_path, device)
            self._processor = processor
            logger.info("SongGenProcessor loaded")

        # Exhaust the weights iterator (vLLM protocol requirement).
        for _ in weights:
            pass
        return None

    # ------------------------------------------------------------------
    # Dummy run support
    # ------------------------------------------------------------------

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict]:
        return [{"lyrics": "la la la", "_is_dummy": True}] * num_reqs

    # ------------------------------------------------------------------
    # Streaming generator management
    # ------------------------------------------------------------------

    def _create_stream_gen(self, info: dict[str, Any]):
        """Yield (waveform_tensor, is_last) for one request.

        Calls ``model.generate()`` synchronously and yields the complete
        waveform as a single (waveform, True) tuple. Output semantics: delta
        (the single yield contains the full audio).
        """
        sr = int(getattr(self.config, "sampling_rate", 16000))
        lyrics: str = str(_pick(info, "lyrics", "") or "")
        description: str = str(_pick(info, "text_description", _DEFAULT_DESCRIPTION) or _DEFAULT_DESCRIPTION)
        ref_voice_array = _pick(info, "ref_voice_array", None)

        if not lyrics.strip():
            logger.warning("SongGen received empty lyrics; yielding silence.")
            yield torch.zeros((sr,), dtype=torch.float32), True
            return

        ref_voice_tmp: str | None = None
        ref_voice_path: str | None = None

        if ref_voice_array is not None:
            try:
                import numpy as np
                import soundfile as sf

                wav_list, sr_in = ref_voice_array
                wav_np = np.asarray(wav_list, dtype=np.float32)
                if wav_np.ndim > 1:
                    wav_np = np.mean(wav_np, axis=-1)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    ref_voice_tmp = tmp.name
                sf.write(ref_voice_tmp, wav_np, int(sr_in))
                ref_voice_path = ref_voice_tmp
            except Exception:
                logger.exception("SongGen failed to stage ref_voice_array to temp file")
                ref_voice_tmp = None

        try:
            model_inputs = self._processor(
                text=description,
                lyrics=lyrics,
                ref_voice_path=ref_voice_path,
                # Skip Demucs vocal separation; callers should provide
                # pre-separated audio when voice conditioning is needed.
                separate=False,
                return_tensors="pt",
            )
            model_inputs = {
                k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in model_inputs.items()
            }

            output = self._model.generate(**model_inputs, do_sample=_DEFAULT_DO_SAMPLE)
            # output shape: (batch_size, audio_samples) at 16 kHz
            waveform = output.squeeze(0)
            if waveform.ndim > 1:
                # Mix down to mono (channels, samples) -> (samples,)
                waveform = waveform.mean(dim=0)
            yield waveform.cpu().float().contiguous(), True
        except Exception:
            logger.exception("SongGen inference failed for lyrics=%r", lyrics[:80])
            yield torch.zeros((sr,), dtype=torch.float32), True
        finally:
            if ref_voice_tmp is not None:
                try:
                    Path(ref_voice_tmp).unlink(missing_ok=True)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Core forward pass (VoxCPM-style generator pattern)
    # ------------------------------------------------------------------

    def _make_dummy_hidden(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        device = self._device or torch.device("cpu")
        hidden = int(getattr(self.config, "hidden_size", 1024))
        n = 1 if input_ids is None else max(1, input_ids.shape[0])
        return torch.zeros((n, hidden), device=device, dtype=torch.float32)

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
        if self._model is None or self._processor is None:
            self.load_weights([])

        sr = int(getattr(self.config, "sampling_rate", 16000))
        sr_tensor = torch.tensor(sr, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)
        hidden = self._make_dummy_hidden(input_ids)

        infos = runtime_additional_information or [{}]

        if not runtime_additional_information or all(info.get("_is_dummy") for info in infos):
            self._ar_last_chunk_flags = [True] * len(infos)
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={
                    "model_outputs": [empty] * len(infos),
                    "sr": [sr_tensor] * len(infos),
                },
            )

        outputs: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []
        last_chunk_flags: list[bool] = []

        for info in infos:
            if info.get("_is_dummy"):
                outputs.append(empty)
                srs.append(sr_tensor)
                last_chunk_flags.append(True)
                continue

            request_key = str(info.get("global_request_id") or info.get("_omni_req_id") or id(info))

            if request_key not in self._stream_gens:
                self._stream_gens[request_key] = self._create_stream_gen(info)

            generator = self._stream_gens[request_key]
            try:
                chunk, is_last = next(generator)
            except StopIteration:
                self._stream_gens.pop(request_key, None)
                outputs.append(empty)
                last_chunk_flags.append(True)
            else:
                if is_last:
                    self._stream_gens.pop(request_key, None)
                outputs.append(chunk)
                last_chunk_flags.append(bool(is_last))

            srs.append(sr_tensor)

        self._ar_last_chunk_flags = last_chunk_flags

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"model_outputs": outputs, "sr": srs},
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Release streaming generators for cancelled/aborted requests.

        ``forward()`` only pops generators on normal completion. Abnormal
        termination (cancel, timeout, preempt) would otherwise leak
        generators and skip the ``finally`` cleanup block (temp files, etc.).
        """
        for req_id in finished_req_ids:
            gen = self._stream_gens.pop(str(req_id), None)
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    logger.exception("SongGen failed to close stream gen for request %s", req_id)

    # ------------------------------------------------------------------
    # AR runner interface
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        """Emit per-row EOS/non-EOS logits to control AR scheduler lifetime.

        Rows whose ``_ar_last_chunk_flags`` entry is True get EOS-dominant
        logits so the scheduler finishes that request. Other rows get a
        non-EOS token to stay alive for the next generator step.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if hidden_states is None:
            device = self._device or torch.device("cpu")
            hidden_states = torch.zeros((0, 1), device=device, dtype=torch.float32)
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(-1)
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        vocab_size = int(getattr(self.config, "vocab_size", 32000))
        num_rows = int(hidden_states.shape[0])
        logits = torch.zeros(
            (num_rows, vocab_size),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        eos_id = 2 if vocab_size > 2 else 0
        safe_id = 1 if vocab_size > 1 and 1 != eos_id else 0

        flags = self._ar_last_chunk_flags
        for row in range(num_rows):
            is_last = flags[row] if row < len(flags) else True
            if is_last:
                logits[row, eos_id] = 1.0e6
            else:
                logits[row, eos_id] = -1.0e9
                logits[row, safe_id] = 1.0e6
        return logits

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        hidden = int(getattr(self.config, "hidden_size", 1024))
        return torch.zeros(
            (input_ids.shape[0], hidden),
            device=input_ids.device,
            dtype=torch.float32,
        )
