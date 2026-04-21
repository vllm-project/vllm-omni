"""MiniCPM-o-4_5 Stage 1: text token IDs → WAV via MiniCPMTTS + stepaudio2.Token2wav.

Text-only conditioning: uses tts.emb_text(text_token_ids) only, without
LM hidden state projection. Avoids Stage 0 hidden-state accumulation at
the cost of flat prosody. Sufficient for the E2E smoke test.
"""

import io
import math
import os
import struct
import sys
import tempfile
import wave
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

_MODEL_DIR = os.environ.get(
    "MINICPMO_MODEL_PATH",
    "/wbl-fast/usrs/ye/benchmarks/_data/_hf_cache/hub/models--openbmb--MiniCPM-o-4_5"
    "/snapshots/44151b35f1b232a280bda5a87ea1a7575d5433fc",
)
_TOKEN2WAV_DIR = os.path.join(_MODEL_DIR, "assets", "token2wav")


def _load_tts_class():
    """Dynamically import MiniCPMTTS from the HF model's modeling file.

    Uses a fake package namespace (_minicpmo_model_pkg) so that relative
    imports inside modeling_minicpmo.py (e.g. `from .configuration_minicpmo
    import ...`) resolve correctly without needing a real __init__.py.
    """
    import importlib.util
    import types

    PKG = "_minicpmo_model_pkg"

    if PKG not in sys.modules:
        pkg = types.ModuleType(PKG)
        pkg.__path__ = [_MODEL_DIR]
        pkg.__package__ = PKG
        sys.modules[PKG] = pkg

    def _load_submod(name: str):
        full_name = f"{PKG}.{name}"
        if full_name in sys.modules:
            return sys.modules[full_name]
        fpath = os.path.join(_MODEL_DIR, f"{name}.py")
        spec = importlib.util.spec_from_file_location(full_name, fpath)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = PKG
        sys.modules[full_name] = mod  # register before exec to handle circulars
        spec.loader.exec_module(mod)
        return mod

    # Load in dependency order; modeling_minicpmo imports all of the others
    for _name in (
        "modeling_navit_siglip",
        "utils",
        "configuration_minicpmo",
        "processing_minicpmo",
        "modeling_minicpmo",
    ):
        _load_submod(_name)

    return sys.modules[f"{PKG}.modeling_minicpmo"].MiniCPMTTS


def _load_tts_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(_MODEL_DIR, trust_remote_code=True)
    return cfg.tts_config


def _patch_torchaudio() -> None:
    """Replace torchaudio.load/save with soundfile equivalents.

    torchaudio 2.10.0 on this cluster requires torchcodec for both load and
    save, but torchcodec is not installed. stepaudio2 calls torchaudio.load
    (with backend='soundfile') and torchaudio.save — both fail. Patch them
    permanently at first call so stepaudio2 and s3tokenizer work without
    torchcodec.
    """
    import torchaudio as _ta

    if getattr(_ta, "_patched_no_torchcodec", False):
        return

    import soundfile as _sf
    import torch as _torch

    def _load(uri, frame_offset=0, num_frames=-1, normalize=True,
               channels_first=True, format=None, buffer_size=4096, backend=None):
        if hasattr(uri, "read"):
            data, sr = _sf.read(uri, dtype="float32", always_2d=True)
        else:
            data, sr = _sf.read(str(uri), dtype="float32", always_2d=True)
        tensor = _torch.tensor(data.T)  # [C, T]
        if frame_offset > 0:
            tensor = tensor[:, frame_offset:]
        if num_frames > 0:
            tensor = tensor[:, :num_frames]
        if not channels_first:
            tensor = tensor.T
        return tensor, sr

    def _save(uri, src, sample_rate, channels_first=True, compression=None,
              format=None, encoding=None, bits_per_sample=None,
              buffer_size=4096, backend=None):
        if not channels_first:
            src = src.T
        data = src.cpu().numpy().T  # [T, C]
        fmt = (format or "WAV").upper()
        if hasattr(uri, "write"):
            _sf.write(uri, data, sample_rate, format=fmt, subtype="PCM_16")
        else:
            _sf.write(str(uri), data, sample_rate)

    _ta.load = _load
    _ta.save = _save
    _ta._patched_no_torchcodec = True
    logger.info("MiniCPMOToken2Speech: patched torchaudio.load/save to use soundfile")


def _make_dummy_wav_path() -> str:
    """Write a 1s 440 Hz sine wave at 16 kHz as a temporary WAV file."""
    rate = 16000
    pcm = [int(32767 * math.sin(2 * math.pi * 440 * t / rate)) for t in range(rate)]
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(f.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
    return f.name


class MiniCPMOToken2Speech(nn.Module):
    """Stage 1 for MiniCPM-o-4_5 S2S: text token IDs → WAV.

    Loads MiniCPMTTS from the checkpoint's tts.* weights and stepaudio2.Token2wav
    from assets/token2wav/. Lazy-initialized on first forward() call (requires GPU).

    Text-only conditioning: inputs_embeds = tts.emb_text(text_ids) + special tokens.
    No LM hidden states. Sufficient for smoke test; extend later if prosody matters.
    """

    have_multimodal_outputs: bool = True
    has_postprocess: bool = True

    def __init__(self, *, vllm_config: Any, prefix: str = ""):
        super().__init__()
        self._tts = None
        self._token2wav = None

    # ── Required by vllm model runner ─────────────────────────────────────────

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def make_empty_intermediate_tensors(self, *args: Any, **kwargs: Any):
        from vllm.sequence import IntermediateTensors
        return IntermediateTensors({})

    # ── Private ───────────────────────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._tts is not None:
            return

        _patch_torchaudio()
        logger.info("MiniCPMOToken2Speech: loading MiniCPMTTS + Token2wav...")

        MiniCPMTTS = _load_tts_class()
        tts_config = _load_tts_config()

        tts = MiniCPMTTS(config=tts_config, audio_tokenizer=None)

        # Load tts.* weights from safetensors shards
        from safetensors import safe_open
        import glob as _glob
        shard_files = sorted(_glob.glob(os.path.join(_MODEL_DIR, "model*.safetensors")))
        tts_state = {}
        for shard_file in shard_files:
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("tts."):
                        tts_state[key[4:]] = f.get_tensor(key)
        missing, unexpected = tts.load_state_dict(tts_state, strict=False)
        if missing:
            logger.warning("MiniCPMOToken2Speech: missing tts keys: %s", missing[:5])
        tts = tts.to(dtype=torch.bfloat16, device="cuda").eval()
        self._tts = tts

        # Load Token2wav (requires GPU for ONNX session)
        from stepaudio2 import Token2wav
        token2wav = Token2wav(_TOKEN2WAV_DIR, float16=False, n_timesteps=10)
        # Pre-load speaker cache with a dummy reference WAV
        dummy_wav = _make_dummy_wav_path()
        try:
            token2wav.cache = token2wav._prepare_prompt(dummy_wav)
        finally:
            os.unlink(dummy_wav)
        self._token2wav = token2wav

        logger.info("MiniCPMOToken2Speech: ready (text-only TTS conditioning)")

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: Any = None,
        **kwargs: Any,
    ) -> OmniOutput:
        self._ensure_loaded()

        text_ids = input_ids.flatten().to(dtype=torch.long, device="cuda")
        if text_ids.numel() == 0:
            raise RuntimeError("MiniCPMOToken2Speech: received empty text token IDs")

        tts = self._tts
        dev = tts.emb_text.weight.device
        dtype = tts.emb_text.weight.dtype

        # Text embeddings only (no hidden states → zero semantic projection)
        tts_embs = tts.emb_text(text_ids.to(dev))  # [T, H]

        text_eos_emb = tts.emb_text(
            torch.tensor([tts.config.text_eos_token_id], dtype=torch.long, device=dev)
        )  # [1, H]
        audio_bos_emb = tts.emb_text(
            torch.tensor([tts.audio_bos_token_id], dtype=torch.long, device=dev)
        )  # [1, H]

        # [1, T+2, H]
        inputs_embeds_seq = torch.cat(
            [tts_embs, text_eos_emb, audio_bos_emb], dim=0
        ).unsqueeze(0).to(dtype=dtype)

        eos_token = torch.tensor(
            [tts.num_audio_tokens - 1], dtype=torch.long, device=dev
        )

        try:
            gen_out = tts.generate(
                inputs_embeds=inputs_embeds_seq,
                eos_token=eos_token,
                min_new_token=50,
                max_new_token=2048,
                show_tqdm=False,
            )
        except Exception as exc:
            raise RuntimeError(f"MiniCPMOToken2Speech: TTS generate failed: {exc}") from exc

        # new_ids shape: [1, S, num_vq=1] → squeeze to [S]
        speech_tokens = gen_out.new_ids.squeeze(-1).squeeze(0)
        logger.info("MiniCPMOToken2Speech: %d speech tokens generated", speech_tokens.numel())

        if speech_tokens.numel() == 0:
            raise RuntimeError("MiniCPMOToken2Speech: TTS produced no speech tokens")

        try:
            # prompt_wav=None is OK since cache is pre-loaded
            wav_bytes = self._token2wav(speech_tokens.tolist(), prompt_wav=None)
        except Exception as exc:
            raise RuntimeError(f"MiniCPMOToken2Speech: token2wav failed: {exc}") from exc

        import soundfile as sf
        wav_arr, sr = sf.read(io.BytesIO(wav_bytes))
        if wav_arr.ndim == 1:
            wav_arr = wav_arr[np.newaxis, :]  # [1, T]
        wav_tensor = torch.tensor(wav_arr, dtype=torch.float32)  # [1, T]
        sr_tensor = torch.tensor(sr, dtype=torch.int32)

        logger.info("MiniCPMOToken2Speech: %.2fs WAV @ %d Hz", wav_arr.shape[-1] / sr, sr)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"audio": [wav_tensor.cpu()], "sr": [sr_tensor]},
        )

    def load_weights(self, weights):
        for _name, _weight in weights:
            pass  # consume iterator; weights loaded lazily in _ensure_loaded()
        return set()
