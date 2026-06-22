import base64
import io
import logging
import math
import os
import os.path
import threading
import time
from collections.abc import Iterable
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import onnxruntime
import requests
import soundfile as sf
import torch
import whisper
import yaml
from transformers import AutoTokenizer

from .audio_tokenizer.frontend import WavFrontendOnline
from .audio_tokenizer.paraformer import ParaformerStreaming
from .utils import (
    AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL,
    AUDIO_EDIT_SYSTEM_PROMPT,
    energy_norm_fn,
    prepare_data_iterator,
    resample_audio,
    trim_silence,
)

logger = logging.getLogger(__name__)


def estimate_vq02_len(num_samples: int) -> int:
    frame_len = 400
    frame_shift = 160
    chunk_stride = 4 * 960  # 3840
    lfr_m = 7
    lfr_n = 6
    left_ctx = (lfr_m - 1) // 2  # 3

    if num_samples <= 0:
        return 0

    full_chunks = num_samples // chunk_stride
    tail_samples = num_samples % chunk_stride

    # In the default StepAudioEditX streaming frontend, each complete 3840-sample
    # chunk contributes four LFR frames. The first chunk gets a 3-frame left
    # context; subsequent full chunks reuse the 1-frame LFR splice cache.
    total = full_chunks * 4

    if tail_samples == 0:
        return total

    # The final partial chunk is processed with the steady-state caches left by
    # previous complete chunks: 320 waveform samples for fbank and 1 LFR frame.
    input_cache_len = 320 if full_chunks > 0 else 0
    lfr_cache_len = 1 if full_chunks > 0 else left_ctx

    effective_len = input_cache_len + tail_samples
    if effective_len < frame_len:
        return total

    tail_fbank_len = int((effective_len - frame_len) / frame_shift + 1)
    if tail_fbank_len < 1:
        return total

    tail_lfr_input_len = lfr_cache_len + tail_fbank_len

    if tail_lfr_input_len < lfr_m:
        return total

    tail_lfr_len = max(0, math.ceil((tail_lfr_input_len - left_ctx) / lfr_n))

    return total + tail_lfr_len


def estimate_step_audio_editx_prompt_len(
    additional_information: dict[str, Any],
    model_path: str,
) -> int:
    config_path = os.path.dirname(model_path) if model_path.endswith("tokenizer_config.json") else model_path
    text_tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
    try:

        def _first(x, default=None):
            if isinstance(x, list):
                return x[0] if x else default
            return x if x is not None else default

        ref_audio = _first(additional_information.get("ref_audio"), None)
        ref_text = _first(additional_information.get("ref_text"), "")
        text = _first(additional_information.get("text"), "")
        sr = _first(additional_information.get("sr"), 16000)
        edit_type = _first(additional_information.get("edit_type", "clone"))

        ref_audio = StepAudioTokenizer.preprocess_wav(ref_audio, sr)
        ref_audio = ref_audio.squeeze(0)

        audio_chunks = StepAudioTokenizer.split_audio(ref_audio, chunk_duration=30 * 16000)
        vq06_len = 0
        for chunks in audio_chunks:
            duration = round(chunks.shape[0] / 16000, 2)
            vq06_len += math.ceil(duration * 25)

        vq02_len = estimate_vq02_len(len(ref_audio))
        estimated_audio_token_len = min(vq02_len // 2, vq06_len // 3) * 5

        dummy_audio_len = 1
        dummy_audio = "".join(f"<audio_{i}>" for i in range(dummy_audio_len))

        if edit_type == "clone":
            prompt_speaker = "debug"
            prompt = StepAudioTokenizer._build_clone_prompt(
                text,
                ref_text,
                prompt_speaker,
                dummy_audio,
            )
        else:
            edit_info = _first(additional_information.get("edit_info", None))
            instruct_prefix = StepAudioTokenizer._build_audio_edit_instruction(ref_text, edit_type, edit_info, text)
            prompt = StepAudioTokenizer._build_edit_prompt(instruct_prefix, dummy_audio)

        dummy_token_ids = text_tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
        token_ids = dummy_token_ids.get("input_ids")
        prompt_len = len(token_ids) - dummy_audio_len + estimated_audio_token_len
        return max(2, prompt_len)
    except Exception as exc:
        logger.warning("Failed to estimate prompt length, using fallback 2048: %s", exc)
        return 2048


class FunASRModel:
    def __init__(self, model_path):
        self.config_path = os.path.join(model_path, "config.yaml")
        kwargs = self.resolve_config(self.config_path)
        self.frontend = WavFrontendOnline(cmvn_file=os.path.join(model_path, "am.mvn"), **kwargs["frontend_conf"])
        kwargs["frontend"] = self.frontend
        self.model = ParaformerStreaming(
            **kwargs["model_conf"],
            encoder_conf=kwargs["encoder_conf"],
            input_size=self.frontend.output_size(),
        )
        state = torch.load(os.path.join(model_path, "model.pt"), map_location="cpu")

        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))

        self.model.load_weights(state.items())
        self.model.to("cuda").eval()
        self.kwargs = kwargs

    def resolve_config(self, config_path):
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return self._resolve_config(config)

    @staticmethod
    def _resolve_config(config):
        """Resolve the official FunASR config to the local runtime schema."""
        if not isinstance(config, dict):
            raise ValueError("FunASR config must be a YAML mapping")

        model_conf_src = config.get("model_conf") or {}
        encoder_conf_src = config.get("encoder_conf") or {}
        frontend_conf_src = config.get("frontend_conf") or {}

        model_keys = {
            "ctc_weight",
            "lsm_weight",
            "length_normalized_loss",
            "predictor_weight",
            "predictor_bias",
            "sampling_ratio",
        }
        model_conf = {key: model_conf_src[key] for key in model_keys if key in model_conf_src}
        frontend_keys = {
            "fs",
            "window",
            "n_mels",
            "frame_length",
            "frame_shift",
            "filter_length_min",
            "filter_length_max",
            "lfr_m",
            "lfr_n",
            "dither",
            "snip_edges",
            "upsacle_samples",
        }
        frontend_conf = {key: frontend_conf_src[key] for key in frontend_keys if key in frontend_conf_src}

        encoder_conf = {
            "output_size": encoder_conf_src.get("output_size", 256),
            "attention_heads": encoder_conf_src.get("attention_heads", 4),
            "linear_units": encoder_conf_src.get("linear_units", 2048),
            "num_blocks": encoder_conf_src.get("num_blocks", 6),
            "normalize_before": encoder_conf_src.get("normalize_before", True),
            "kernel_size": encoder_conf_src.get("kernel_size", 11),
            "sanm_shift": encoder_conf_src.get("sanm_shift", encoder_conf_src.get("sanm_shift", 0)),
        }

        device = "cuda"
        if not torch.cuda.is_available():
            device = "cpu"

        return {
            "model": config.get("model", "ParaformerStreaming"),
            "model_conf": model_conf,
            "encoder": config.get("encoder", "SANMEncoderChunkOpt"),
            "encoder_conf": encoder_conf,
            "frontend": config.get("frontend", "WavFrontendOnline"),
            "frontend_conf": frontend_conf,
            "device": device,
            "batch_size": 1,
            "data_type": "sound",
            "chunk_size": [0, 4, 5],
            "encoder_chunk_look_back": 4,
            "decoder_chunk_look_back": 1,
        }

    @torch.inference_mode()
    def infer_encoder(self, input, input_len=None, kwargs=None, key=None, **cfg):
        kwargs = self.kwargs if kwargs is None else kwargs
        kwargs.update(cfg)
        batch_size = kwargs.get("batch_size", 1)
        key_list, data_list = prepare_data_iterator(input, data_type=kwargs.get("data_type", None), key=key)
        asr_result_list = []
        num_samples = len(data_list)
        for beg_idx in range(0, num_samples, batch_size):
            end_idx = min(num_samples, beg_idx + batch_size)
            data_batch = data_list[beg_idx:end_idx]
            key_batch = key_list[beg_idx:end_idx]
            batch = {"data_in": data_batch, "key": key_batch}
            if (end_idx - beg_idx) == 1 and kwargs.get("data_type", None) == "fbank":  # fbank
                batch["data_in"] = data_batch[0]
                batch["data_lengths"] = input_len

            results, meta_data, cache = self.model.infer_encoder(**batch, **kwargs)
            asr_result_list.extend(results)

        torch.accelerator.empty_cache()
        return asr_result_list, cache

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.model.load_weights(weights)


class StepAudioTokenizer:
    def __init__(
        self,
        tokenizer_path,
        config_path,
        funasr_model_id="dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
    ):
        if not tokenizer_path:
            raise ValueError("audio_tokenizer_path is not set")
        self.text_tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
        self.funasr_tokenizer_path = os.path.join(tokenizer_path, funasr_model_id)
        self.funasr_model = FunASRModel(model_path=self.funasr_tokenizer_path)
        self.kms_path = os.path.join(tokenizer_path, "linguistic_tokenizer.npy")
        self.cosy_tokenizer_path = os.path.join(tokenizer_path, "speech_tokenizer_v1.onnx")
        self.kms = torch.tensor(np.load(self.kms_path))

        providers = ["CUDAExecutionProvider"]
        session_option = onnxruntime.SessionOptions()
        session_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_option.intra_op_num_threads = 1
        self.ort_session = onnxruntime.InferenceSession(
            self.cosy_tokenizer_path, sess_options=session_option, providers=providers
        )
        self.chunk_size = [0, 4, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1

        self.vq02_sessions = {}
        self.vq02_lock = threading.Lock()
        self.vq06_lock = threading.Lock()

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        # Heuristic: no filesystem path separators and long enough.
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    @staticmethod
    def _is_url(s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    def encode(self, edit_type, audio, prompt, sr):
        """
        output: (token_ids, vq0206_codes)

        supported prompt template:
        • edit:  {prompt_text, edit_type, edit_info, target_text}
        • clone: {prompt_text, target_text}
        """
        audio_tokens, vq0206_codes = self._audio_tokenize(audio, sr)

        token_ids = self._text_tokenize(edit_type, audio_tokens, prompt)
        return token_ids, vq0206_codes

    def _audio_tokenize(self, audio, sr):
        vq0206_codes, vq02_codes_ori, vq06_codes_ori = self.wav2token(audio, sr)
        audio_tokens = self.merge_vq0206_to_token_str(vq02_codes_ori, vq06_codes_ori)
        return audio_tokens, vq0206_codes

    def _text_tokenize(self, edit_type: str, audio_tokens: str, prompt: dict | tuple):
        if edit_type == "clone":
            prompt_text, target_text = prompt
            prompt_speaker = "debug"
            prompt = self._build_clone_prompt(
                target_text,
                prompt_text,
                prompt_speaker,
                audio_tokens,
            )
        else:
            prompt_text, edit_type, edit_info, target_text = prompt
            instruct_prefix = self._build_audio_edit_instruction(prompt_text, edit_type, edit_info, target_text)

            prompt = self._build_edit_prompt(instruct_prefix, audio_tokens)

        return self.text_tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )

    @staticmethod
    def _load_audio(prompt_wav, prompt_wav_sr: int | list[int] | tuple[int, ...] | None = None):
        # Single explicit pair: (audio, sr)
        if isinstance(prompt_wav, tuple) and len(prompt_wav) == 2 and isinstance(prompt_wav[1], (int, np.integer)):
            return StepAudioTokenizer._load_audio(prompt_wav[0], int(prompt_wav[1]))

        # Batch input: [audio1, audio2, ...]
        if isinstance(prompt_wav, list):
            if prompt_wav_sr is None:
                prompt_wav_sr_list = [None] * len(prompt_wav)
            elif isinstance(prompt_wav_sr, (int, np.integer)):
                prompt_wav_sr_list = [int(prompt_wav_sr)] * len(prompt_wav)
            else:
                if len(prompt_wav_sr) != len(prompt_wav):
                    raise ValueError(
                        "prompt_wav_sr length must match prompt_wav length "
                        f"({len(prompt_wav_sr)} != {len(prompt_wav)})."
                    )
                prompt_wav_sr_list = list(prompt_wav_sr)

            return [
                StepAudioTokenizer._load_audio(item, item_sr) for item, item_sr in zip(prompt_wav, prompt_wav_sr_list)
            ]

        # Tuple that is not (audio, sr) is ambiguous.
        if isinstance(prompt_wav, tuple):
            raise TypeError("prompt_wav tuple must be (audio, sample_rate). Use a list for batched audio inputs.")

        if isinstance(prompt_wav, torch.Tensor):
            if prompt_wav_sr is None:
                raise ValueError("Tensor audio requires prompt_wav_sr.")
            wav = prompt_wav.detach().to(dtype=torch.float32)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            elif wav.ndim != 2:
                raise ValueError(f"Tensor audio must be 1D or 2D, got {tuple(wav.shape)}")
            return wav.contiguous(), int(prompt_wav_sr)

        if isinstance(prompt_wav, np.ndarray):
            if prompt_wav_sr is None:
                raise ValueError("NumPy audio requires prompt_wav_sr.")
            wav = torch.from_numpy(prompt_wav.astype(np.float32, copy=False))
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            elif wav.ndim == 2:
                # NumPy audio from callers is often [T, C]; model wants [C, T].
                if wav.shape[0] > wav.shape[1] and wav.shape[1] <= 8:
                    wav = wav.transpose(0, 1)
            else:
                raise ValueError(f"NumPy audio must be 1D or 2D, got {prompt_wav.shape}")
            return wav.contiguous(), int(prompt_wav_sr)

        if isinstance(prompt_wav, str):
            audio = prompt_wav
            parsed = urlparse(audio)

            if audio.startswith("data:"):
                header, sep, payload = audio.partition(",")
                if not sep:
                    raise ValueError("Invalid audio data URL: missing comma separator.")
                if not header.startswith("data:audio/"):
                    raise ValueError(f"Unsupported data URL MIME type: {header}")
                if ";base64" not in header:
                    raise ValueError("Only base64 audio data URLs are supported.")
                audio_bytes = base64.b64decode(payload)
                data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)

            elif parsed.scheme in ("http", "https") and parsed.netloc:
                response = requests.get(audio, timeout=30)
                response.raise_for_status()
                data, sr = sf.read(io.BytesIO(response.content), dtype="float32", always_2d=False)

            elif parsed.scheme == "file":
                data, sr = sf.read(unquote(parsed.path), dtype="float32", always_2d=False)

            else:
                if not os.path.exists(audio):
                    raise FileNotFoundError(f"Audio file not found: {audio}")
                data, sr = sf.read(audio, dtype="float32", always_2d=False)

            wav = torch.from_numpy(data.astype(np.float32, copy=False))
            # soundfile returns mono [T], multi-channel [T, C].
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            elif wav.ndim == 2:
                wav = wav.transpose(0, 1)
            else:
                raise ValueError(f"Decoded audio must be 1D or 2D, got {data.shape}")
            return wav.contiguous(), int(sr)

        raise TypeError(f"Unsupported prompt_wav type: {type(prompt_wav)}")

    @staticmethod
    def preprocess_wav(audio, sample_rate, enable_trim=True, energy_norm=True):
        audio, sample_rate = StepAudioTokenizer._load_audio(audio, sample_rate)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # volume-normalize avoid clipping
        norm = torch.max(torch.abs(audio), dim=1, keepdim=True)[0]
        if norm.item() > 0.6:
            audio = audio / norm * 0.6

        audio = resample_audio(audio, sample_rate, 16000)

        if audio.dim() == 2:
            audio = audio.squeeze(0)
        audio = audio.cpu().to(torch.float32)

        if energy_norm:
            audio = energy_norm_fn(audio)

        if enable_trim:
            audio = trim_silence(audio, 16000)
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio.astype(np.float32))
                if audio.ndim == 1:
                    audio = audio.unsqueeze(0)  # [1, T]
                elif audio.ndim == 2:
                    audio = audio.transpose(0, 1)
        return audio

    def wav2token(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = self.preprocess_wav(audio, sample_rate, enable_trim=enable_trim, energy_norm=energy_norm)
        vq02_ori = self.get_vq02_code(audio)
        vq02 = [int(x) + 65536 for x in vq02_ori]
        vq06_ori = self.get_vq06_code(audio)
        vq06 = [int(x) + 65536 + 1024 for x in vq06_ori]

        chunk = 1
        chunk_nums = min(len(vq06) // (3 * chunk), len(vq02) // (2 * chunk))
        speech_tokens = []
        for idx in range(chunk_nums):
            speech_tokens += vq02[idx * chunk * 2 : (idx + 1) * chunk * 2]
            speech_tokens += vq06[idx * chunk * 3 : (idx + 1) * chunk * 3]
        speech_tokens = torch.tensor([speech_tokens], dtype=torch.long)
        return speech_tokens, vq02_ori, vq06_ori

    def get_vq02_code(self, audio, session_id=None, is_final=True):
        audio_in = io.BytesIO()
        audio_np = audio.squeeze(0).cpu().numpy() if audio.dim() > 1 else audio.cpu().numpy()
        sf.write(audio_in, audio_np, 16000, format="WAV")
        audio_in.seek(0)

        with self.vq02_lock:
            cache = {}
            if session_id in self.vq02_sessions:
                cache = self.vq02_sessions[session_id].get("cache", {})

            res, new_cache = self.funasr_model.infer_encoder(
                input=[audio_in],
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
                device=0,
                is_final=is_final,
                cache=cache,
            )
            c_list = []
            for j, res_ in enumerate(res):
                feat = res_["enc_out"]
                if len(feat) > 0:
                    c_list = self.dump_label([feat], self.kms)[0]

            if is_final:
                if session_id in self.vq02_sessions:
                    self.vq02_sessions.pop(session_id)
            else:
                if isinstance(session_id, str) and len(session_id) > 0:
                    self.vq02_sessions[session_id] = {
                        "cache": new_cache,
                        "update_time": time.time(),
                    }

            return c_list

    @staticmethod
    def split_audio(audio, chunk_duration=480000):
        start = 0
        chunks = []
        while start < len(audio):
            end = min(start + chunk_duration, len(audio))
            chunk = audio[start:end]
            if len(chunk) < 480:
                pass
            else:
                chunks.append(chunk)
            start = end
        return chunks

    def get_vq06_code(self, audio):
        with self.vq06_lock:
            audio = audio.squeeze(0)
            chunk_audios = StepAudioTokenizer.split_audio(audio, chunk_duration=30 * 16000)  # Maximum support 30s
            speech_tokens = []
            for chunk in chunk_audios:
                duration = round(chunk.shape[0] / 16000, 2)
                feat = whisper.log_mel_spectrogram(chunk, n_mels=128)
                feat = feat.unsqueeze(0)
                feat_len = np.array([feat.shape[2]], dtype=np.int32)
                chunk_token = (
                    self.ort_session.run(
                        None,
                        {
                            self.ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                            self.ort_session.get_inputs()[1].name: feat_len,
                        },
                    )[0]
                    .flatten()
                    .tolist()
                )
                assert abs(len(chunk_token) - duration * 25) <= 2
                speech_tokens += chunk_token

            return speech_tokens

    def kmean_cluster(self, samples, means):
        dists = torch.cdist(samples, means)
        indices = dists.argmin(dim=1).cpu().numpy()
        return indices.tolist()

    def dump_label(self, samples, mean):
        dims = samples[0].shape[-1]
        x_lens = [x.shape[1] for x in samples]
        total_len = sum(x_lens)
        x_sel = torch.FloatTensor(1, total_len, dims)
        start_len = 0
        for sample in samples:
            sample_len = sample.shape[1]
            end_len = start_len + sample_len
            x_sel[:, start_len:end_len] = sample
            start_len = end_len
        dense_x = x_sel.squeeze(0).to(mean.device)
        indices = self.kmean_cluster(dense_x, mean)
        indices_list = []
        start_len = 0
        for x_len in x_lens:
            end_len = start_len + end_len
            indices_list.append(indices[start_len:end_len])
        return indices_list

    def merge_vq0206_to_token_str(self, vq02, vq06):
        _vq06 = [1024 + x for x in vq06]
        result = []
        i = 0
        j = 0
        while i < len(vq02) - 1 and j < len(_vq06) - 2:
            sublist = vq02[i : i + 2] + _vq06[j : j + 3]
            result.extend(sublist)
            i += 2
            j += 3
        return "".join([f"<audio_{x}>" for x in result])

    @staticmethod
    def _build_edit_prompt(instruct_prefix: str, audio_token_str: str) -> list[int]:
        sys_prompt = AUDIO_EDIT_SYSTEM_PROMPT
        """Encode audio edit prompt to token sequence"""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{instruct_prefix}\n{audio_token_str}\n"},
        ]

        return messages

    @staticmethod
    def _build_clone_prompt(text: str, prompt_text: str, prompt_speaker: str, prompt_wav_tokens: str):
        sys_prompt = AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL.format(
            speaker=prompt_speaker, prompt_text=prompt_text, prompt_wav_tokens=prompt_wav_tokens
        )
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"{text}"}]

        return messages

    @staticmethod
    def _build_audio_edit_instruction(
        audio_text: str, edit_type: str, edit_info: str | None = None, text: str | None = None
    ) -> str:
        """Build audio editing instruction based on request"""
        audio_text = audio_text.strip() if audio_text else ""
        if edit_type in {"emotion", "speed"}:
            if edit_info == "remove":
                instruct_prefix = f"Remove any emotion in the following audio and the reference text is: {audio_text}\n"
            else:
                instruct_prefix = (
                    f"Make the following audio more {edit_info}. The text corresponding to the audio is: {audio_text}\n"
                )
        elif edit_type == "style":
            if edit_info == "remove":
                instruct_prefix = (
                    f"Remove any speaking styles in the following audio and the reference text is: {audio_text}\n"
                )
            else:
                instruct_prefix = (
                    f"Make the following audio more {edit_info} style. "
                    "The text corresponding to the audio is: "
                    f"{audio_text}\n"
                )
        elif edit_type == "denoise":
            instruct_prefix = (
                "Remove any noise from the given audio while preserving the voice content clearly. "
                "Ensure that the speech quality remains intact with minimal distortion, and "
                "eliminate all noise from the audio.\n"
            )
        elif edit_type == "vad":
            instruct_prefix = (
                "Remove any silent portions from the given audio while preserving the voice content clearly. "
                "Ensure that the speech quality remains intact with minimal distortion, and "
                "eliminate all silence from the audio.\n"
            )
        elif edit_type == "paralinguistic":
            instruct_prefix = (
                f"Add some non-verbal sounds to make the audio more natural, the new text is : {text}\n"
                f"  The text corresponding to the audio is: {audio_text}\n"
            )
        else:
            logger.error("Unsupported audio editing type: %s", edit_type)
        return instruct_prefix

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.funasr_model.load_weights(weights)
