# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MOSS-TTS-Nano single-stage model for vLLM-Omni.

Runs in a single AR worker stage.  The 0.1B AR LM and the
MOSS-Audio-Tokenizer-Nano codec are both loaded here.

Streaming is supported via the VoxCPM-style generator pattern:
  - On first forward() for a request, inference_stream() is started as
    a Python generator and stored in self._stream_gens[request_key].
  - Each subsequent forward() call pops one audio chunk from the generator
    and returns it as multimodal_outputs.
  - compute_logits() emits EOS only when the last chunk has been yielded,
    telling the AR scheduler to finish the request.

Weight loading deliberately happens inside load_weights() -- NOT __init__ --
so that vLLM initialises distributed state before any CUDA allocations occur.
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
from vllm_omni.model_executor.models.utils import reinit_rotary_inv_freq

logger = init_logger(__name__)


def _patch_torchaudio_load() -> None:
    """Patch torchaudio.load to use soundfile if torchcodec is unavailable."""
    try:
        import torchaudio

        torchaudio  # noqa
        import torchcodec  # noqa: F401

        return
    except Exception:
        pass

    import soundfile as sf

    def _soundfile_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if frame_offset > 0:
            data = data[frame_offset:]
        if num_frames > 0:
            data = data[:num_frames]
        waveform = torch.from_numpy(data)
        if channels_first:
            waveform = waveform.T
        return waveform, sr

    def _soundfile_save(path, src, sample_rate, channels_first=True, **kwargs):
        wav = src.detach().cpu().float().numpy()
        if channels_first and wav.ndim == 2:
            wav = wav.T
        sf.write(str(path), wav, sample_rate)

    try:
        import torchaudio

        torchaudio.load = _soundfile_load
        torchaudio.save = _soundfile_save
        logger.info("Patched torchaudio.load/save to use soundfile (torchcodec unavailable)")
    except Exception as e:
        logger.warning("Could not patch torchaudio: %s", e)


# Default sampling parameters matching the upstream demo defaults.
_DEFAULT_TEXT_TEMPERATURE = 1.0
_DEFAULT_TEXT_TOP_P = 1.0
_DEFAULT_TEXT_TOP_K = 50
_DEFAULT_AUDIO_TEMPERATURE = 0.8
_DEFAULT_AUDIO_TOP_P = 0.95
_DEFAULT_AUDIO_TOP_K = 25
_DEFAULT_AUDIO_REPETITION_PENALTY = 1.2
_DEFAULT_MAX_NEW_FRAMES = 375
# MOSS-TTS-Nano upstream supports two modes (voice_clone / continuation).
# voice_clone is the recommended workflow; the serving layer routes by
# whether ref_text was supplied (see _build_moss_tts_params in
# vllm_omni/entrypoints/openai/serving_speech.py).
_DEFAULT_MODE = "voice_clone"


def _to_mono_1d(waveform: Any) -> torch.Tensor:
    """Convert an upstream waveform event to a 1-D float32 mono tensor.

    The MOSS audio tokenizer is configured for stereo (number_channels=2)
    and emits ``(channels, samples)`` tensors. The downstream serving path
    writes a single-channel WAV at the tokenizer's sample rate, so we mix
    multi-channel audio down to mono via mean. Naively flattening with
    ``.T.reshape(-1)`` interleaves L/R into a 2N-length stream that gets
    re-interpreted at 1× the sample rate — playback ends up 2× too slow.
    """
    chunk = torch.as_tensor(waveform, dtype=torch.float32).cpu()
    if chunk.ndim == 2:
        if chunk.shape[0] > 1:
            return chunk.mean(dim=0).contiguous()
        return chunk.reshape(-1)
    return chunk.reshape(-1)


def _pick(info: dict, key: str, default):
    """Extract scalar from additional_information dict (list or plain value)."""
    val = info.get(key, default)
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val[0]
    return val if val is not None else default


class MossTTSNanoForGeneration(nn.Module):
    """Single-stage MOSS-TTS-Nano model with streaming audio output.

    Uses the VoxCPM-style generator pattern: inference_stream() is stored
    per-request and yields one audio chunk per forward() call.  The AR
    scheduler keeps the request alive until compute_logits() emits EOS.
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

        self._lm: nn.Module | None = None
        self._audio_tokenizer: nn.Module | None = None
        self._device: torch.device | None = None
        self._lock = threading.Lock()

        # Per-request streaming generators (VoxCPM pattern).
        # Single-threaded AR worker: mutations happen only inside forward(),
        # so we rely on the worker's serial execution instead of self._lock.
        self._stream_gens: dict[str, Any] = {}
        # Per-row EOS mask aligned with the most recent forward() batch.
        # compute_logits() emits EOS for rows flagged True, keeps others alive.
        self._ar_last_chunk_flags: list[bool] = []

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        with self._lock:
            if self._lm is not None:
                return set()
            _patch_torchaudio_load()

            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device

            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                tts_dtype = torch.bfloat16
            elif device.type == "cuda":
                tts_dtype = torch.float16
            else:
                tts_dtype = torch.float32

            logger.info("Loading MOSS-TTS-Nano LM from %s (dtype=%s)", self.model_path, tts_dtype)
            from transformers import AutoModelForCausalLM

            lm = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=tts_dtype,
            )
            if device.type == "cuda":
                try:
                    import flash_attn  # noqa: F401

                    lm._set_attention_implementation("flash_attention_2")
                    logger.info("MOSS-TTS-Nano using flash_attention_2")
                except ImportError:
                    lm._set_attention_implementation("sdpa")
                    logger.info("MOSS-TTS-Nano using sdpa (flash_attn not installed)")
            lm.to(device=device)
            lm.eval()

            # ``trust_remote_code`` custom RoPE classes that register
            # ``inv_freq`` with ``persistent=False`` and aren't in
            # ``ROPE_INIT_FUNCTIONS`` come out of ``from_pretrained``'s
            # post-init chain holding garbage (~ -1.7e38 / 9.9e33). The
            # very first text-LM forward then emits NaN logits.
            # See vllm_omni.model_executor.models.utils.reinit_rotary_inv_freq
            # for the full mechanism and reproduction.
            n_fixed = reinit_rotary_inv_freq(lm, base=10000.0)
            if n_fixed > 0:
                logger.info("MOSS-TTS-Nano: re-initialised %d rotary_emb.inv_freq buffers", n_fixed)

            self._lm = lm
            logger.info("MOSS-TTS-Nano LM loaded on %s", device)

            codec_path: str = getattr(
                self.config,
                "audio_tokenizer_pretrained_name_or_path",
                "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
            )
            logger.info("Loading MOSS-Audio-Tokenizer-Nano from %s", codec_path)
            from transformers import AutoModel

            audio_tokenizer = AutoModel.from_pretrained(
                codec_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            audio_tokenizer.to(device=device)
            audio_tokenizer.eval()
            self._audio_tokenizer = audio_tokenizer
            logger.info("MOSS-Audio-Tokenizer-Nano loaded on %s", device)

        # MOSS-TTS-Nano loads weights inline via from_pretrained() above;
        # vLLM's weight-loading protocol still requires us to exhaust the
        # iterator so the underlying stream is closed cleanly.
        for _ in weights:
            pass
        return set()

    # ------------------------------------------------------------------
    # Dummy run support
    # ------------------------------------------------------------------

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict]:
        return [{"text": "hello", "_is_dummy": True}] * num_reqs

    # ------------------------------------------------------------------
    # Streaming generator management
    # ------------------------------------------------------------------

    def _create_stream_gen(self, info: dict[str, Any]):
        """Create an inference_stream() generator for a request.

        Yields (waveform_tensor, is_last) tuples.
        """
        text: str = str(_pick(info, "text", "") or "")
        if not text.strip():
            logger.warning("MOSS-TTS-Nano received empty text; yielding silence.")
            sr = getattr(self.config, "audio_tokenizer_sample_rate", 48000)
            yield torch.zeros((sr,), dtype=torch.float32), True
            return

        mode: str = str(_pick(info, "mode", _DEFAULT_MODE))
        prompt_audio_path: str | None = _pick(info, "prompt_audio_path", None)
        if prompt_audio_path is not None:
            prompt_audio_path = str(prompt_audio_path)
        # Voice-cloning path: serving layer passes the resolved waveform as
        # (wav_list, sample_rate) so we can avoid re-decoding base64 and the
        # model owns temp-file lifecycle.
        prompt_audio_array = _pick(info, "prompt_audio_array", None)
        prompt_text: str | None = _pick(info, "prompt_text", None)
        if prompt_text is not None:
            prompt_text = str(prompt_text)
        max_new_frames: int = int(_pick(info, "max_new_frames", _DEFAULT_MAX_NEW_FRAMES))
        seed: int | None = _pick(info, "seed", None)
        if seed is not None:
            # Upstream inference_stream relies on global RNG state. With
            # max_num_seqs > 1, concurrent seeded requests will race to set
            # the global seed; snapshot and restore on exit so we at least
            # don't leak state back to other components of the engine.
            seed = int(seed)
            _cpu_rng_state = torch.get_rng_state()
            _cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        else:
            _cpu_rng_state = None
            _cuda_rng_state = None

        sampling = {
            "text_temperature": float(_pick(info, "text_temperature", _DEFAULT_TEXT_TEMPERATURE)),
            "text_top_p": float(_pick(info, "text_top_p", _DEFAULT_TEXT_TOP_P)),
            "text_top_k": int(_pick(info, "text_top_k", _DEFAULT_TEXT_TOP_K)),
            "audio_temperature": float(_pick(info, "audio_temperature", _DEFAULT_AUDIO_TEMPERATURE)),
            "audio_top_p": float(_pick(info, "audio_top_p", _DEFAULT_AUDIO_TOP_P)),
            "audio_top_k": int(_pick(info, "audio_top_k", _DEFAULT_AUDIO_TOP_K)),
            "audio_repetition_penalty": float(
                _pick(info, "audio_repetition_penalty", _DEFAULT_AUDIO_REPETITION_PENALTY)
            ),
        }

        device = self._device or torch.device("cpu")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        # Materialise resolved reference audio to a temp WAV so upstream's
        # prompt_audio_path (which expects a filesystem path) can consume it.
        prompt_audio_tmp: str | None = None
        if prompt_audio_array is not None and prompt_audio_path is None:
            try:
                import numpy as np
                import soundfile as sf

                wav_list, sr_in = prompt_audio_array
                wav_np = np.asarray(wav_list, dtype=np.float32)
                if wav_np.ndim > 1:
                    wav_np = np.mean(wav_np, axis=-1)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
                    prompt_audio_tmp = tmp_in.name
                sf.write(prompt_audio_tmp, wav_np, int(sr_in))
                prompt_audio_path = prompt_audio_tmp
            except Exception:
                logger.exception("MOSS-TTS-Nano failed to stage prompt_audio_array to temp file")
                prompt_audio_tmp = None

        # Stream audio chunks from inference_stream as they arrive. A final
        # "result" event is only used as a fallback when no audio events
        # were emitted (e.g. very short utterances on some backends).
        audio_chunks: list[torch.Tensor] = []
        try:
            for event in self._lm.inference_stream(
                text=text,
                output_audio_path=output_path,
                mode=mode,
                prompt_text=prompt_text,
                prompt_audio_path=prompt_audio_path,
                text_tokenizer_path=self.model_path,
                audio_tokenizer=self._audio_tokenizer,
                device=device,
                nq=None,
                max_new_frames=max_new_frames,
                do_sample=True,
                use_kv_cache=True,
                **sampling,
            ):
                event_type = str(event.get("type", ""))
                if event_type == "audio":
                    waveform = event.get("waveform")
                    if waveform is not None:
                        chunk = _to_mono_1d(waveform)
                        audio_chunks.append(chunk)
                        # Yield each chunk as it arrives (is_last=False).
                        yield chunk, False
                elif event_type == "result":
                    if not audio_chunks:
                        waveform = event.get("waveform")
                        if waveform is not None:
                            chunk = _to_mono_1d(waveform)
                            yield chunk, True
                            return
        except Exception:
            logger.exception("MOSS-TTS-Nano inference failed for text=%r", text[:80])
        finally:
            try:
                Path(output_path).unlink(missing_ok=True)
            except Exception:
                pass
            if prompt_audio_tmp is not None:
                try:
                    Path(prompt_audio_tmp).unlink(missing_ok=True)
                except Exception:
                    pass
            if _cpu_rng_state is not None:
                torch.set_rng_state(_cpu_rng_state)
            if _cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(_cuda_rng_state)

        # Signal completion. If we yielded audio chunks above, the last
        # yield was is_last=False, so emit a final empty sentinel.
        yield torch.zeros((0,), dtype=torch.float32), True

    # ------------------------------------------------------------------
    # Core forward pass (streaming, VoxCPM pattern)
    # ------------------------------------------------------------------

    def _make_dummy_hidden(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        """Return a dummy hidden_states tensor for the AR runner."""
        device = self._device or torch.device("cpu")
        hidden = int(getattr(self.config, "hidden_size", 768))
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
        sr = getattr(self.config, "audio_tokenizer_sample_rate", 48000)
        sr_tensor = torch.tensor(sr, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)
        hidden = self._make_dummy_hidden(input_ids)

        infos = runtime_additional_information or [{}]

        if not runtime_additional_information or all(info.get("_is_dummy") for info in infos):
            # Dummy/warmup path: finish immediately for every row.
            self._ar_last_chunk_flags = [True] * len(infos)
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={
                    "model_outputs": [empty] * len(infos),
                    "sr": [sr_tensor] * len(infos),
                },
            )

        if self._lm is None or self._audio_tokenizer is None:
            raise RuntimeError("MOSS-TTS-Nano model not loaded.  Was load_weights() called?")

        outputs: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []
        last_chunk_flags: list[bool] = []

        for info in infos:
            if info.get("_is_dummy"):
                outputs.append(empty)
                srs.append(sr_tensor)
                last_chunk_flags.append(True)
                continue

            # Per-request key so concurrent / consecutive requests don't
            # share a generator. ``global_request_id`` is set by the engine
            # (see info keys ``['text', 'mode', 'prompt_audio_array',
            # 'global_request_id', 'omni_final_stage_id', 'generated_len']``).
            # ``_omni_req_id`` is a legacy fallback that is never set in
            # the current engine; falling back to a constant collapses all
            # requests onto one generator and lets a stale generator from
            # request N replay its remaining chunks for request N+1, which
            # surfaces as request N+1's audio matching request N's input.
            request_key = str(info.get("global_request_id") or info.get("_omni_req_id") or id(info))

            # Create generator on first call for this request.
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

        # Per-row EOS mask: compute_logits() finishes each request the step
        # its last chunk was yielded, without waiting for the slowest peer.
        self._ar_last_chunk_flags = last_chunk_flags

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"model_outputs": outputs, "sr": srs},
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Release streaming generators for requests the scheduler finished.

        forward() only pops generators on normal completion (is_last or
        StopIteration). Abnormal termination (cancel, timeout, preempt) would
        otherwise leak the generator and skip its ``finally`` block, stranding
        temp WAV files. Closing here raises GeneratorExit inside the generator
        so the cleanup block runs.
        """
        for req_id in finished_req_ids:
            gen = self._stream_gens.pop(str(req_id), None)
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    logger.exception("MOSS-TTS-Nano failed to close stream gen for request %s", req_id)

    # ------------------------------------------------------------------
    # AR runner interface
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        """Emit per-row EOS / non-EOS logits to control AR scheduler lifetime.

        Rows whose ``_ar_last_chunk_flags`` entry is True get EOS-dominant
        logits so the scheduler finishes that request; other rows get a
        non-EOS token so they stay alive for the next streaming chunk.
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
        # If we lost alignment with the forward batch (e.g. scheduler dropped
        # a row), conservatively treat missing rows as "finished" to avoid
        # stranding requests.
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
        hidden = int(getattr(self.config, "hidden_size", 768))
        return torch.zeros(
            (input_ids.shape[0], hidden),
            device=input_ids.device,
            dtype=torch.float32,
        )
