# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
"""MiniCPM-o 4.5 Talker + Token2Wav: MiniCPMTTS with hidden_text_merge condition.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Run MiniCPMTTS.generate() -> discrete audio tokens
  5. Run Token2wav(tokens) -> waveform bytes -> numpy array
"""

import io
import logging
import os
import sys
from collections.abc import Iterable

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsPP

from vllm_omni.platforms import current_omni_platform

# Preserve the established external vocoder on CUDA. Ascend uses the in-tree
# adapter because ``stepaudio2-minicpmo`` hard-codes CUDA device placement.
if current_omni_platform.is_npu():
    try:
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav import (
            MiniCPMO45Token2wav as _Token2wav,
        )

        _token2wav_backend = "step_audio2_core"
    except ImportError:
        try:
            from stepaudio2 import Token2wav as _Token2wav

            _token2wav_backend = "stepaudio2_pkg"
        except ImportError:
            _Token2wav = None
            _token2wav_backend = None
else:
    try:
        from stepaudio2 import Token2wav as _Token2wav

        _token2wav_backend = "stepaudio2_pkg"
    except ImportError:
        _Token2wav = None
        _token2wav_backend = None

_stepaudio2_available = _Token2wav is not None

logger = logging.getLogger(__name__)


def _install_torchaudio_soundfile_shim() -> None:
    """Monkey-patch torchaudio.load to use soundfile instead of the default
    torchcodec backend, which requires libtorchcodec/ffmpeg shared libs that
    may be missing on the deployment machine."""
    try:
        import torchaudio

        if getattr(torchaudio, "_soundfile_shim_installed", False):
            return
        _orig_load = torchaudio.load

        def _patched_load(uri, *args, **kwargs):
            try:
                return _orig_load(uri, *args, **kwargs)
            except Exception:
                import numpy as _np
                import soundfile as _sf

                data, sr = _sf.read(uri, dtype="float32", always_2d=True)
                wav = torch.from_numpy(_np.ascontiguousarray(data.T))
                return wav, sr

        torchaudio.load = _patched_load
        torchaudio._soundfile_shim_installed = True
        logger.info("Installed torchaudio.load soundfile shim")
    except Exception as _e:
        logger.warning("Could not install torchaudio shim: %s", _e)


_install_torchaudio_soundfile_shim()


class MiniCPMO45OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """MiniCPM-o 4.5 Talker: MiniCPMTTS + Token2wav in a single forward pass."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMOConfig

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config

        self.tts = None
        self.audio_tokenizer = None
        self._assets_loaded = False

        tts_config = getattr(config, "tts_config", None)
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = getattr(tts_config, "audio_bos_token_id", 151687)
            self._text_eos_id = getattr(tts_config, "text_eos_token_id", 151692)
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
        else:
            self._tts_config = None

    def _lazy_init_tts(self):
        if self._assets_loaded or self._tts_config is None:
            return
        try:
            model_path = self.vllm_config.model_config.model

            if model_path not in sys.path:
                sys.path.insert(0, model_path)
            from transformers import AutoImageProcessor
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            # openbmb/MiniCPM-o-4_5/processing_minicpmo.py registers via a
            # string: AutoImageProcessor.register("MiniCPMVImageProcessor", ...),
            # which crashes on transformers>=5 (register reads key.__module__).
            # Loading MiniCPMTTS imports that module, so no-op the string form
            # (unused by the standalone talker) while it runs, then restore.
            original_register = AutoImageProcessor.register
            AutoImageProcessor.register = (  # type: ignore[method-assign]
                lambda key, *a, **k: None if isinstance(key, str) else original_register(key, *a, **k)
            )
            try:
                MiniCPMTTS = get_class_from_dynamic_module("modeling_minicpmo.MiniCPMTTS", model_path)
            finally:
                AutoImageProcessor.register = original_register  # type: ignore[method-assign]

            # MiniCPMTTS.__init__ reads `config.top_p / top_k / repetition_penalty`
            # directly (modeling_minicpmo.py L4112-4114), but the model repo's
            # config.json `tts_config` block does not declare these fields and
            # PretrainedConfig in recent transformers no longer surfaces
            # generation-style params on `self.config`. Inject the defaults the
            # upstream code itself ships with (modeling_minicpmo.py L2212-2214,
            # L3132-3133) so attribute access does not raise.
            for _attr, _default in (("top_p", 0.8), ("top_k", 100), ("repetition_penalty", 1.02)):
                if not hasattr(self._tts_config, _attr):
                    setattr(self._tts_config, _attr, _default)

            # The copied Hugging Face flash_attention_2 setting is not valid
            # for this standalone MiniCPMTTS path. Use PyTorch SDPA on every
            # backend until a dedicated flash-attention implementation exists.
            self._tts_config.attn_implementation = "sdpa"

            prev_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                self.tts_obj = MiniCPMTTS(config=self._tts_config, audio_tokenizer=None)
            finally:
                torch.set_default_dtype(prev_dtype)
            self.emb_text = self.tts_obj.emb_text
            self.projector_semantic = self.tts_obj.projector_semantic

            token2wav_dir = os.path.join(model_path, "assets", "token2wav")
            if os.path.isdir(token2wav_dir):
                if not _stepaudio2_available:
                    raise ImportError(
                        "MiniCPM-o 4.5 token2wav stage requires the `stepaudio2` Python "
                        "module (a MiniCPM-o-flavored Token2wav vocoder, NOT the upstream "
                        "stepfun-ai/Step-Audio2 — the upstream signature does not accept "
                        "n_timesteps and will fail at __init__). Install via:\n"
                        "    pip install 'vllm-omni[minicpmo]'   # recommended, declared as PR extra\n"
                        "Equivalent direct installs of the same `from stepaudio2 import Token2wav`\n"
                        "entry point used by openbmb/MiniCPM-o-4_5/modeling_minicpmo.py:\n"
                        "    pip install stepaudio2-minicpmo     # bare token2wav package\n"
                        "    pip install 'minicpmo-utils[all]'   # MiniCPM-o umbrella (also brings image/video deps)"
                    )
                prev_dtype2 = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                try:
                    # NB: this must be the MiniCPM-o-flavored Token2wav from
                    # the `stepaudio2-minicpmo` PyPI package (or the
                    # `minicpmo-utils[all]` umbrella), not the upstream
                    # `stepfun-ai/Step-Audio2` repo. The MiniCPM-o variant's
                    # __init__ accepts n_timesteps; the upstream signature is
                    # (model_path, float16=False) and will raise
                    # TypeError on n_timesteps. See ImportError message below
                    # for installation guidance.
                    self.audio_tokenizer = _Token2wav(token2wav_dir, float16=False, n_timesteps=10)
                finally:
                    torch.set_default_dtype(prev_dtype2)
                self.tts_obj.audio_tokenizer = self.audio_tokenizer
                logger.info(
                    "Loaded Token2wav from %s (backend=%s)",
                    token2wav_dir,
                    _token2wav_backend,
                )
            # Only mark init as complete after every step succeeds, so a
            # partial failure leaves the next call free to retry the full
            # init instead of short-circuiting back to a silent empty path.
            self._assets_loaded = True
        except ImportError:
            # Surface missing dependencies directly so users can act on them
            # instead of getting a silent None waveform downstream.
            raise
        except Exception:
            # Re-raise non-import init failures (bad token2wav assets, missing
            # weights, OOM during Token2wav construction, etc.) so the server
            # fails loudly at startup / first request instead of returning
            # silent empty audio for every subsequent request.
            logger.error("Failed to init 4.5 TTS", exc_info=True)
            raise

    def generate_speech(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
    ) -> np.ndarray | None:
        """Run full 4.5 TTS pipeline using original MiniCPMTTS.generate."""
        self._lazy_init_tts()
        if not hasattr(self, "tts_obj") or self.tts_obj is None:
            logger.warning("generate_speech: tts_obj not initialized")
            return None

        tts = self.tts_obj
        device = tts.emb_text.weight.device
        # MiniCPMTTS AR backbone uses FlashAttention (fp16/bf16 only). The
        # submodule is constructed under float32 default dtype during lazy init,
        # so pin the condition embeddings to bfloat16 explicitly rather than
        # inheriting the (float32) parameter dtype — a float32 condition breaks
        # the CUDA FA2 path and wastes memory on the NPU sdpa path.
        ar_dtype = torch.bfloat16

        llm_embeds = tts.emb_text(tts_token_ids.to(device))
        hidden_embeds = tts.projector_semantic(tts_hidden_states.to(device=device, dtype=ar_dtype))
        if getattr(tts.config, "normalize_projected_hidden", False):
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        tts_embeds = (llm_embeds + hidden_embeds).to(dtype=ar_dtype)

        text_eos = tts.emb_text(torch.tensor([tts.config.text_eos_token_id], device=device, dtype=torch.long))
        audio_bos = tts.emb_text(torch.tensor([tts.audio_bos_token_id], device=device, dtype=torch.long))
        spk_embeds = torch.zeros(0, tts.config.hidden_size, device=device, dtype=ar_dtype)

        inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos, audio_bos], dim=0).unsqueeze(0)
        inputs_embeds = inputs_embeds.to(dtype=ar_dtype)
        logger.info("generate_speech: inputs_embeds shape=%s", list(inputs_embeds.shape))

        # Scale max_new_token with input text length to avoid mid-stream truncation on long
        # responses (default 2048 can only cover ~300 text tokens at ~6x audio/text ratio).
        # Empirically 511 text tokens → 1951 audio tokens (~3.8x) finishes cleanly, so use 10x
        # as a safe upper bound with a floor of 2048 and a hard cap of 16384 to bound latency/mem.
        num_text = int(tts_token_ids.shape[-1]) if tts_token_ids.ndim > 0 else 0
        max_new_token = max(2048, min(16384, num_text * 10))

        eos_token = torch.tensor([tts.config.num_audio_tokens - 1], dtype=torch.long, device=device)
        outputs = tts.generate(
            inputs_embeds=inputs_embeds,
            eos_token=eos_token,
            max_new_token=max_new_token,
            show_tqdm=False,
        )
        generated_tokens = outputs.new_ids.squeeze(-1)
        logger.info(
            "generate_speech: generated %d audio tokens (cap=%d, text_tokens=%d)",
            generated_tokens.shape[-1],
            max_new_token,
            num_text,
        )

        if self.audio_tokenizer is None:
            logger.warning("No audio_tokenizer")
            return None

        import torchaudio

        model_path = self.vllm_config.model_config.model
        default_ref = os.path.join(model_path, "assets", "HT_ref_audio.wav")
        prompt_wav_path = default_ref if os.path.exists(default_ref) else None

        _orig_save = torchaudio.save

        def _patched_save(uri, src, sample_rate, **kw):
            kw.pop("backend", None)
            if hasattr(uri, "write"):
                sf.write(uri, src.cpu().numpy().T, sample_rate, format="WAV")
                return
            return _orig_save(uri, src, sample_rate, backend="soundfile", **kw)

        torchaudio.save = _patched_save
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            # Vocoder path is float32; use the platform abstraction because
            # torch.amp.autocast validates unsupported device types even when
            # autocast is disabled.
            autocast_device = device.type if isinstance(device, torch.device) else str(device)
            with current_omni_platform.create_autocast_context(
                device_type=autocast_device,
                dtype=torch.float32,
                enabled=False,
            ):
                token_list = generated_tokens.squeeze(0).tolist()
                num_tokens = len(token_list)

                # For long outputs, the one-shot vocoder path
                # (Token2wav.__call__ -> flow.inference) runs full O(N^2) self-
                # attention over all audio tokens and OOMs on a 24GB card once
                # N exceeds a few thousand (e.g. 4964 tokens needs ~3GiB for a
                # single attention matmul). Switch to the chunked / streaming
                # vocoder (set_stream_cache + stream) which truncates the flow
                # attention caches to prompt_len + 100 steps on every chunk,
                # keeping peak memory bounded regardless of total length.
                STREAM_THRESHOLD = int(os.environ.get("MINICPMO45_TTS_STREAM_THRESHOLD", "2500"))  # ~100s @ 25Hz
                CHUNK_SIZE = int(os.environ.get("MINICPMO45_TTS_STREAM_CHUNK", "50"))  # ~2s per chunk
                MIN_TAIL = 6  # must exceed flow.pre_lookahead_len (typically 3)

                if num_tokens <= STREAM_THRESHOLD:
                    wav_bytes = self.audio_tokenizer(token_list, prompt_wav_path)
                    waveform, sr = sf.read(io.BytesIO(wav_bytes))
                    waveform = waveform.astype(np.float32)
                else:
                    # Build chunk boundaries, merging a too-small tail into the
                    # previous chunk so every chunk satisfies MIN_TAIL.
                    boundaries = []
                    i = 0
                    while i < num_tokens:
                        end = min(i + CHUNK_SIZE, num_tokens)
                        if 0 < num_tokens - end < MIN_TAIL:
                            end = num_tokens
                        boundaries.append((i, end))
                        i = end

                    logger.info(
                        "generate_speech: streaming vocoder, %d tokens -> %d chunks (chunk=%d)",
                        num_tokens,
                        len(boundaries),
                        CHUNK_SIZE,
                    )

                    stream_cache, hift_cache_dict = self.audio_tokenizer.set_stream_cache(prompt_wav_path)
                    self.audio_tokenizer.stream_cache = stream_cache
                    self.audio_tokenizer.hift_cache_dict = hift_cache_dict

                    try:
                        pieces = []
                        for idx, (s, e) in enumerate(boundaries):
                            is_last = idx == len(boundaries) - 1
                            wav_np = self.audio_tokenizer.stream(
                                token_list[s:e],
                                prompt_wav_path,
                                last_chunk=is_last,
                                return_waveform=True,
                            )
                            pieces.append(np.asarray(wav_np).reshape(-1))
                        waveform = np.concatenate(pieces, axis=0).astype(np.float32)
                        sr = 24000
                    finally:
                        # Free per-request streaming state so the next request starts clean
                        self.audio_tokenizer.stream_cache = None
                        self.audio_tokenizer.hift_cache_dict = {}
        finally:
            torch.set_default_dtype(prev_dtype)
            torchaudio.save = _orig_save

        logger.info("generate_speech: waveform %d samples, sr=%d", waveform.shape[0], sr)
        return waveform

    def _generate_tokens(self, inputs_embeds: torch.Tensor, max_new_token: int = 2048) -> torch.Tensor | None:
        """Autoregressive generation of audio tokens using the TTS LlamaModel."""
        device = inputs_embeds.device
        eos_token = self._num_audio_tokens - 1
        condition_length = inputs_embeds.shape[1]
        num_vq = len(self.emb_code)

        new_tokens = torch.zeros(1, max_new_token, num_vq, device=device, dtype=torch.long)
        past_key_values = None
        finished = False

        for t in range(max_new_token):
            if t == 0:
                emb = inputs_embeds
                position_ids = torch.arange(condition_length, device=device).unsqueeze(0)
            else:
                code_emb = [self.emb_code[q](new_tokens[:, t - 1 : t, q]) for q in range(num_vq)]
                emb = torch.stack(code_emb, -1).sum(-1)
                position_ids = torch.tensor([[condition_length + t - 1]], device=device)

            outputs = self.tts_model(
                inputs_embeds=emb,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            logits = torch.stack([self.head_code[q](hidden[:, -1]) for q in range(num_vq)], dim=-1)
            logits = logits.float() / 0.8

            if t < 50:
                logits[:, eos_token, :] = -float("inf")

            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs.view(-1, probs.shape[1]), 1).view(1, num_vq)
            new_tokens[:, t] = idx

            if (idx == eos_token).any():
                finished = True
                break

        return new_tokens[:, : t + 1 if finished else t, :]

    def _dummy_hidden_states(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shape-correct zero tensor for vllm KV cache profiling.

        vllm's gpu_model_runner._dummy_run takes forward()'s return value as
        ``hidden_states`` and does ``hidden_states[logit_indices_device]``;
        returning None on the dummy path crashes with
        ``TypeError: 'NoneType' object is not subscriptable``.
        """
        for ref in (input_ids, positions, inputs_embeds):
            if isinstance(ref, torch.Tensor):
                num_tokens = int(ref.shape[0]) if ref.ndim >= 1 else 1
                device = ref.device
                break
        else:
            num_tokens = 1
            device = current_omni_platform.get_torch_device()
        hidden_size = int(getattr(self, "_hidden_size", 768) or 768)
        return torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        additional_information=None,
        **kwargs,
    ):
        if additional_information is None:
            additional_information = {}

        tts_token_ids = additional_information.get("tts_token_ids")
        tts_hidden_states = additional_information.get("tts_hidden_states")
        tts_text = additional_information.get("llm_output_text", [""])
        if isinstance(tts_text, list):
            tts_text = tts_text[0] if tts_text else ""

        if tts_token_ids is None or tts_hidden_states is None:
            # KV cache profiling / dummy run path — no real TTS input yet.
            logger.debug("4.5 Talker: dummy forward (missing tts_token_ids/tts_hidden_states)")
            return self._dummy_hidden_states(input_ids, positions, inputs_embeds)

        logger.info("4.5 Talker: generating speech for %d tokens", tts_token_ids.shape[0])
        waveform = self.generate_speech(tts_token_ids, tts_hidden_states)
        # Tuple layout: (mel_spec, waveform). 4.5 talker emits only waveform,
        # so mel_spec stays None; the wrapper unpacks in this order and
        # packages the waveform into ``multimodal_outputs["model_outputs"]``.
        if waveform is not None:
            return None, torch.tensor(waveform, dtype=torch.float32)
        return None, None

    def compute_logits(self, hidden_states, *args, **kwargs):
        # Placeholder logits: one row per sampled request (the scheduler
        # indexes sampled_token_ids by req_index). Hardcoding a single row
        # breaks batched/concurrent decoding with IndexError. The values are
        # discarded — real output is the waveform via multimodal_outputs.
        if isinstance(hidden_states, torch.Tensor):
            device = hidden_states.device
            num_reqs = hidden_states.shape[0] if hidden_states.ndim >= 1 else 1
        else:
            device = current_omni_platform.get_torch_device()
            num_reqs = 1
        return torch.zeros(num_reqs, 2, device=device)

    def sample(self, logits, sampling_metadata):
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loaded = set()
        tts_weights = {}
        for k, v in weights:
            if k.startswith("tts."):
                tts_weights[k.replace("tts.", "", 1)] = v
                # vllm sanity-checks `loaded` against `named_parameters()`.
                # The submodule is attached at `self.tts_obj`, not `self.tts`,
                # so report the loaded name under the on-module path.
                loaded.add(k.replace("tts.", "tts_obj.", 1))

        if tts_weights and self._tts_config is not None:
            self._lazy_init_tts()
            if hasattr(self, "tts_obj") and self.tts_obj is not None:
                missing, unexpected = self.tts_obj.load_state_dict(tts_weights, strict=False)
                if missing:
                    logger.warning("TTS missing keys (%d): %s", len(missing), missing[:5])
                if unexpected:
                    logger.warning("TTS unexpected keys (%d): %s", len(unexpected), unexpected[:5])
                # Move the AR backbone to the active device (cuda / npu / …) and
                # cast to bfloat16: MiniCPMTTS AR uses FlashAttention (fp16/bf16
                # only) and is built under a float32 default dtype during lazy
                # init, so an uncast float32 backbone breaks CUDA FA2 and wastes
                # memory on the NPU sdpa path. Detach the Token2wav vocoder first
                # so the cast does not drag it onto the accelerator or downcast
                # its float32 flow/HiFT weights: it manages its own device
                # placement and may not be an nn.Module.
                device = current_omni_platform.get_torch_device()
                audio_tok = getattr(self.tts_obj, "audio_tokenizer", None)
                if audio_tok is not None:
                    self.tts_obj.audio_tokenizer = None
                try:
                    self.tts_obj = self.tts_obj.to(device=device, dtype=torch.bfloat16)
                finally:
                    if audio_tok is not None:
                        self.tts_obj.audio_tokenizer = audio_tok
                        self.audio_tokenizer = audio_tok
                self.emb_text = self.tts_obj.emb_text
                self.projector_semantic = self.tts_obj.projector_semantic
                logger.info(
                    "Loaded %d TTS weights, moved to %s (bfloat16)",
                    len(tts_weights),
                    device,
                )

        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        if hasattr(self, "emb_text") and self.emb_text is not None:
            return self.emb_text(input_ids)
        return torch.zeros(input_ids.shape[0], 1)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
