# coding=utf-8
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
import tempfile
from typing import Iterable, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsPP

logger = logging.getLogger(__name__)


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
        self._assets_loaded = True
        try:
            model_path = self.vllm_config.model_config.model
            import sys, os
            if model_path not in sys.path:
                sys.path.insert(0, model_path)
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            MiniCPMTTS = get_class_from_dynamic_module(
                "modeling_minicpmo.MiniCPMTTS", model_path
            )

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
                from stepaudio2 import Token2wav
                prev_dtype2 = torch.get_default_dtype()
                torch.set_default_dtype(torch.float32)
                try:
                    self.audio_tokenizer = Token2wav(token2wav_dir, float16=False, n_timesteps=10)
                finally:
                    torch.set_default_dtype(prev_dtype2)
                self.tts_obj.audio_tokenizer = self.audio_tokenizer
                logger.info("Loaded Token2wav from %s", token2wav_dir)
        except Exception as e:
            logger.error("Failed to init 4.5 TTS: %s", e, exc_info=True)

    def generate_speech(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
    ) -> Optional[np.ndarray]:
        """Run full 4.5 TTS pipeline using original MiniCPMTTS.generate."""
        self._lazy_init_tts()
        if not hasattr(self, "tts_obj") or self.tts_obj is None:
            logger.warning("generate_speech: tts_obj not initialized")
            return None

        tts = self.tts_obj
        device = tts.emb_text.weight.device
        dtype = tts.emb_text.weight.dtype

        llm_embeds = tts.emb_text(tts_token_ids.to(device))
        hidden_embeds = tts.projector_semantic(tts_hidden_states.to(device=device, dtype=dtype))
        if getattr(tts.config, "normalize_projected_hidden", False):
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        tts_embeds = llm_embeds + hidden_embeds

        text_eos = tts.emb_text(torch.tensor([tts.config.text_eos_token_id], device=device, dtype=torch.long))
        audio_bos = tts.emb_text(torch.tensor([tts.audio_bos_token_id], device=device, dtype=torch.long))
        spk_embeds = torch.zeros(0, tts.config.hidden_size, device=device, dtype=tts_embeds.dtype)

        inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos, audio_bos], dim=0).unsqueeze(0)
        logger.info("generate_speech: inputs_embeds shape=%s", list(inputs_embeds.shape))

        eos_token = torch.tensor([tts.config.num_audio_tokens - 1], dtype=torch.long, device=device)
        outputs = tts.generate(inputs_embeds=inputs_embeds, eos_token=eos_token, show_tqdm=False)
        generated_tokens = outputs.new_ids.squeeze(-1)
        logger.info("generate_speech: generated %d audio tokens", generated_tokens.shape[-1])

        if self.audio_tokenizer is None:
            logger.warning("No audio_tokenizer")
            return None

        import os, torchaudio
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
            with torch.amp.autocast("cuda", enabled=False):
                token_list = generated_tokens.squeeze(0).tolist()
                wav_bytes = self.audio_tokenizer(token_list, prompt_wav_path)
        finally:
            torch.set_default_dtype(prev_dtype)
            torchaudio.save = _orig_save

        waveform, sr = sf.read(io.BytesIO(wav_bytes))
        logger.info("generate_speech: waveform %d samples, sr=%d", waveform.shape[0], sr)
        return waveform.astype(np.float32)

    def _generate_tokens(self, inputs_embeds: torch.Tensor, max_new_token: int = 2048) -> Optional[torch.Tensor]:
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
                code_emb = [self.emb_code[q](new_tokens[:, t-1:t, q]) for q in range(num_vq)]
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

        return new_tokens[:, :t+1 if finished else t, :]

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
        tts_text = additional_information.get("thinker_output_text", [""])
        if isinstance(tts_text, list):
            tts_text = tts_text[0] if tts_text else ""

        if tts_token_ids is None or tts_hidden_states is None:
            logger.warning("4.5 Talker: missing tts_token_ids or tts_hidden_states")
            return None, None

        logger.info("4.5 Talker: generating speech for %d tokens", tts_token_ids.shape[0])
        waveform = self.generate_speech(tts_token_ids, tts_hidden_states)
        if waveform is not None:
            return torch.tensor(waveform, dtype=torch.float32), None
        return None, None

    def compute_logits(self, hidden_states, *args, **kwargs):
        return torch.zeros(1, 2, device=hidden_states.device if isinstance(hidden_states, torch.Tensor) else "cuda")

    def sample(self, logits, sampling_metadata):
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loaded = set()
        tts_weights = {}
        for k, v in weights:
            if k.startswith("tts."):
                tts_weights[k.replace("tts.", "", 1)] = v
                loaded.add(k)

        if tts_weights and self._tts_config is not None:
            self._lazy_init_tts()
            if hasattr(self, "tts_obj") and self.tts_obj is not None:
                missing, unexpected = self.tts_obj.load_state_dict(tts_weights, strict=False)
                if missing:
                    logger.warning("TTS missing keys (%d): %s", len(missing), missing[:5])
                if unexpected:
                    logger.warning("TTS unexpected keys (%d): %s", len(unexpected), unexpected[:5])
                self.tts_obj = self.tts_obj.to("cuda")
                self.emb_text = self.tts_obj.emb_text
                self.projector_semantic = self.tts_obj.projector_semantic
                logger.info("Loaded %d TTS weights, moved to cuda", len(tts_weights))

        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        if hasattr(self, "emb_text") and self.emb_text is not None:
            return self.emb_text(input_ids)
        return torch.zeros(input_ids.shape[0], 1)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)
