from __future__ import annotations

import io
import os
import sys
import tempfile
import types
from collections.abc import Iterable
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.multimodal.audio import AudioResampler

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _resample_audio(audio: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return np.asarray(audio, dtype=np.float32)
    resampler = AudioResampler(target_sr=int(target_sr))
    return np.asarray(resampler.resample(np.asarray(audio, dtype=np.float32), orig_sr=int(orig_sr)), dtype=np.float32)


def _setup_cosyvoice2_alias() -> None:
    """Register cosyvoice2.* import aliases expected by MiniCPM flow.yaml."""
    if "cosyvoice2.flow.flow" in sys.modules:
        return

    import stepaudio2.cosyvoice2.flow.decoder_dit as step_decoder_dit
    import stepaudio2.cosyvoice2.flow.flow as step_flow
    import stepaudio2.cosyvoice2.flow.flow_matching as step_flow_matching
    import stepaudio2.cosyvoice2.transformer.upsample_encoder_v2 as step_upsample

    cosyvoice2_pkg = types.ModuleType("cosyvoice2")
    cosyvoice2_flow_pkg = types.ModuleType("cosyvoice2.flow")
    cosyvoice2_transformer_pkg = types.ModuleType("cosyvoice2.transformer")

    cosyvoice2_flow_pkg.flow = step_flow
    cosyvoice2_flow_pkg.flow_matching = step_flow_matching
    cosyvoice2_flow_pkg.decoder_dit = step_decoder_dit
    cosyvoice2_transformer_pkg.upsample_encoder_v2 = step_upsample
    cosyvoice2_pkg.flow = cosyvoice2_flow_pkg
    cosyvoice2_pkg.transformer = cosyvoice2_transformer_pkg

    sys.modules["cosyvoice2"] = cosyvoice2_pkg
    sys.modules["cosyvoice2.flow"] = cosyvoice2_flow_pkg
    sys.modules["cosyvoice2.flow.flow"] = step_flow
    sys.modules["cosyvoice2.flow.flow_matching"] = step_flow_matching
    sys.modules["cosyvoice2.flow.decoder_dit"] = step_decoder_dit
    sys.modules["cosyvoice2.transformer"] = cosyvoice2_transformer_pkg
    sys.modules["cosyvoice2.transformer.upsample_encoder_v2"] = step_upsample


def fade_in_out(
    fade_in_mel: torch.Tensor,
    fade_out_mel: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel


class MiniCPMToken2wavCore:
    """Local Token2wav core that avoids torchaudio/TorchCodec prompt I/O."""

    def __init__(
        self,
        model_path: str,
        *,
        float16: bool = False,
        n_timesteps: int = 10,
        device: str = "cuda",
    ) -> None:
        import onnxruntime
        import s3tokenizer
        import torchaudio.compliance.kaldi as kaldi
        from hyperpyyaml import load_hyperpyyaml
        from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator
        from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

        _setup_cosyvoice2_alias()

        self.model_path = model_path
        self.float16 = float16
        self.n_timesteps = n_timesteps
        self.device = torch.device(device)
        self.sample_rate = 24000
        self.silence_token_id = 4218

        self._s3tokenizer = s3tokenizer
        self._kaldi = kaldi
        self._mel_spectrogram = mel_spectrogram

        self.audio_tokenizer = (
            s3tokenizer.load_model(f"{model_path}/speech_tokenizer_v2_25hz.onnx").to(self.device).eval()
        )

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(
            f"{model_path}/campplus.onnx",
            sess_options=option,
            providers=["CPUExecutionProvider"],
        )
        self._spk_input_name = self.spk_model.get_inputs()[0].name

        with open(f"{model_path}/flow.yaml", encoding="utf-8") as f:
            configs = load_hyperpyyaml(f)
        self.flow = configs["flow"]
        if float16:
            self.flow.half()
        self.flow.load_state_dict(
            torch.load(f"{model_path}/flow.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        self.flow.to(self.device).eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {
            k.replace("generator.", ""): v
            for k, v in torch.load(f"{model_path}/hift.pt", map_location="cpu", weights_only=True).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        self.cache = None
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).to(
            device=self.device, dtype=torch.float32
        )
        self.stream_cache = None
        self.hift_cache_dict: dict[str, torch.Tensor] = {}

    @staticmethod
    def _waveform_to_soundfile_array(waveform: torch.Tensor) -> np.ndarray:
        audio_np = np.asarray(waveform.detach().cpu().numpy(), dtype=np.float32)
        if audio_np.ndim == 2:
            if audio_np.shape[0] == 1:
                audio_np = audio_np[0]
            else:
                audio_np = audio_np.T
        elif audio_np.ndim != 1:
            raise ValueError(f"Expected 1-D or 2-D audio tensor, got shape {tuple(audio_np.shape)}")
        return np.asarray(audio_np, dtype=np.float32)

    @staticmethod
    def _write_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
        output = io.BytesIO()
        sf.write(output, MiniCPMToken2wavCore._waveform_to_soundfile_array(waveform), sample_rate, format="WAV")
        return output.getvalue()

    @staticmethod
    def _load_audio(file: str, *, sr: int) -> torch.Tensor:
        audio, sample_rate = sf.read(file, dtype="float32", always_2d=False)
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=-1)
        audio_np = audio_np.reshape(-1)
        if int(sample_rate) != int(sr):
            audio_np = _resample_audio(audio_np, orig_sr=int(sample_rate), target_sr=int(sr))
        return torch.from_numpy(np.asarray(audio_np, dtype=np.float32))

    @staticmethod
    def _load_prompt_audio_24k(file: str) -> torch.Tensor:
        audio, sample_rate = sf.read(file, dtype="float32", always_2d=False)
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=-1)
        audio_np = audio_np.reshape(-1)
        if int(sample_rate) != 24000:
            audio_np = _resample_audio(audio_np, orig_sr=int(sample_rate), target_sr=24000)
        return torch.from_numpy(np.asarray(audio_np, dtype=np.float32)).unsqueeze(0)

    def _prepare_prompt(
        self,
        prompt_wav: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_16k = self._load_audio(prompt_wav, sr=16000)
        mels = self._s3tokenizer.log_mel_spectrogram(audio_16k)
        mels, mels_lens = self._s3tokenizer.padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(
            mels.to(self.device),
            mels_lens.to(self.device),
        )

        spk_feat = self._kaldi.fbank(audio_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(
            self.spk_model.run(None, {self._spk_input_name: spk_feat.unsqueeze(dim=0).cpu().numpy()})[0],
            device=self.device,
        )

        audio_24k = self._load_prompt_audio_24k(prompt_wav)
        prompt_mel = self._mel_spectrogram(audio_24k).transpose(1, 2).squeeze(0)
        prompt_mels = prompt_mel.unsqueeze(0).to(self.device)
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device=self.device)
        prompt_mels = torch.nn.functional.pad(
            prompt_mels,
            (0, 0, 0, prompt_speech_tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]),
            mode="replicate",
        )
        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    def __call__(self, generated_speech_tokens: list[int], prompt_wav: str) -> bytes:
        if self.cache is None:
            self.cache = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache

        generated_speech_tokens_tensor = torch.tensor(
            [generated_speech_tokens],
            dtype=torch.int32,
            device=self.device,
        )
        generated_speech_tokens_lens = torch.tensor(
            [generated_speech_tokens_tensor.shape[1]],
            dtype=torch.int32,
            device=self.device,
        )

        with torch.amp.autocast(str(self.device.type), dtype=torch.float16 if self.float16 else torch.float32):
            mel = self.flow.inference(
                generated_speech_tokens_tensor,
                generated_speech_tokens_lens,
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                prompt_mels,
                prompt_mels_lens,
                spk_emb,
                self.n_timesteps,
            )

        wav, _ = self.hift(speech_feat=mel)
        return self._write_wav_bytes(wav.cpu(), self.sample_rate)

    def set_stream_cache(self, prompt_wav: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.cache = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, _, spk_emb, prompt_mels, _ = self.cache

        right_pad_speech_tokens = torch.full(
            (1, 3),
            self.silence_token_id,
            device=prompt_speech_tokens.device,
            dtype=prompt_speech_tokens.dtype,
        )
        self.stream_cache = self.flow.setup_cache(
            torch.cat([prompt_speech_tokens, right_pad_speech_tokens], dim=1),
            prompt_mels,
            spk_emb,
            n_timesteps=self.n_timesteps,
        )
        self.hift_cache_dict = {
            "mel": torch.zeros(1, prompt_mels.shape[2], 0, device=self.device),
            "source": torch.zeros(1, 1, 0, device=self.device),
            "speech": torch.zeros(1, 0, device=self.device),
        }
        return self.stream_cache, self.hift_cache_dict

    def stream(
        self,
        generated_speech_tokens: list[int],
        prompt_wav: str | None,
        *,
        last_chunk: bool = False,
        return_waveform: bool = False,
    ) -> np.ndarray | bytes:
        if self.cache is None:
            if prompt_wav is None:
                raise ValueError("prompt_wav is required before streaming cache is initialized")
            self.cache = self._prepare_prompt(prompt_wav)
        _, _, spk_emb, prompt_mels, _ = self.cache

        generated_speech_tokens_tensor = torch.tensor(
            [generated_speech_tokens],
            dtype=torch.int32,
            device=self.device,
        )

        if self.stream_cache is None:
            raise ValueError("stream_cache is not set")

        with torch.amp.autocast(str(self.device.type), dtype=torch.float16 if self.float16 else torch.float32):
            chunk_mel, self.stream_cache = self.flow.inference_chunk(
                token=generated_speech_tokens_tensor,
                spk=spk_emb,
                cache=self.stream_cache,
                last_chunk=last_chunk,
                n_timesteps=self.n_timesteps,
            )

        estimator_att_cache = self.stream_cache.get("estimator_att_cache")
        if estimator_att_cache is not None and estimator_att_cache.shape[4] > (prompt_mels.shape[1] + 100):
            self.stream_cache["estimator_att_cache"] = torch.cat(
                [
                    estimator_att_cache[:, :, :, :, : prompt_mels.shape[1]],
                    estimator_att_cache[:, :, :, :, -100:],
                ],
                dim=4,
            )

        conformer_att_cache = self.stream_cache.get("conformer_att_cache")
        if conformer_att_cache is not None and conformer_att_cache.shape[3] > (prompt_mels.shape[1] + 100):
            self.stream_cache["conformer_att_cache"] = torch.cat(
                [
                    conformer_att_cache[:, :, :, : prompt_mels.shape[1], :],
                    conformer_att_cache[:, :, :, -100:, :],
                ],
                dim=3,
            )

        hift_cache_mel = self.hift_cache_dict["mel"]
        hift_cache_source = self.hift_cache_dict["source"]
        hift_cache_speech = self.hift_cache_dict["speech"]
        mel = torch.cat([hift_cache_mel, chunk_mel], dim=2)

        speech, source = self.hift(mel, hift_cache_source)

        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)

        is_first_chunk = hift_cache_speech.shape[-1] == 0
        self.hift_cache_dict = {
            "mel": mel[..., -self.mel_cache_len :].clone().detach(),
            "source": source[:, :, -self.source_cache_len :].clone().detach(),
            "speech": speech[:, -self.source_cache_len :].clone().detach(),
        }

        if not last_chunk:
            if is_first_chunk:
                silence_padding = torch.zeros(1, self.source_cache_len, device=speech.device)
                speech = torch.cat([silence_padding, speech[:, : -self.source_cache_len]], dim=1)
            else:
                speech = speech[:, : -self.source_cache_len]

        wav_np = np.asarray(speech.detach().cpu().numpy(), dtype=np.float32)
        if return_waveform:
            return wav_np

        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767.0).astype("<i2")
        return wav_int16.tobytes()


class MiniCPMO4_5Code2Wav(nn.Module):
    """MiniCPM-o 4.5 code2wav stage.

    Non-async Stage 2 wrapper around the official ``stepaudio2.Token2wav``
    decoder. This stage consumes the finished talker token ids directly and
    returns waveform tensors for the final audio output.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.root_config = vllm_config.model_config.hf_config
        stage_config_name = getattr(vllm_config.model_config, "hf_config_name", None)
        stage_config = getattr(self.root_config, stage_config_name, None) if stage_config_name else None
        if stage_config_name and stage_config is None:
            logger.warning(
                "MiniCPMO4_5 code2wav could not find hf_config.%s; falling back to root hf_config.",
                stage_config_name,
            )
        self.config = stage_config if stage_config is not None else self.root_config
        self.model_path = vllm_config.model_config.model
        self.prefix = prefix

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._token2wav = None
        self._output_sample_rate = 24000
        self._audio_prompt_sample_rate = int(getattr(self.config, "audio_tokenizer_sample_rate", 16000))
        self._audio_eos_token_id = int(getattr(self.config, "num_audio_tokens", 1)) - 1

    def embed_multimodal(self, **kwargs: Any) -> list[torch.Tensor]:
        if not kwargs:
            return []

        logger.warning(
            "MiniCPM code2wav received multimodal encoder inputs during profile run; "
            "returning dummy embeddings because Stage 2 does not consume them."
        )

        hidden_size = int(getattr(self.root_config, "hidden_size", 1) or 1)
        device = torch.device("cpu")
        dtype = torch.float32
        num_items = 0

        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                dtype = value.dtype if value.is_floating_point() else torch.float32
                num_items = int(value.shape[0]) if value.ndim > 0 else 1
                break
            if isinstance(value, list):
                num_items = len(value)
                if value and isinstance(value[0], torch.Tensor):
                    device = value[0].device
                    dtype = value[0].dtype if value[0].is_floating_point() else torch.float32
                break

        return [torch.zeros((1, hidden_size), device=device, dtype=dtype) for _ in range(num_items)]

    def _ensure_token2wav_loaded(self) -> None:
        if self._token2wav is not None:
            return

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"MiniCPM code2wav requires a local model directory, got: {self.model_path!r}")

        asset_dir = os.path.join(self.model_path, "assets", "token2wav")
        if not os.path.isdir(asset_dir):
            raise FileNotFoundError(f"MiniCPM token2wav assets not found under local path: {asset_dir}")

        n_timesteps = int(getattr(self.config, "s3_stream_n_timesteps", 10) or 10)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._token2wav = MiniCPMToken2wavCore(
                asset_dir,
                float16=False,
                n_timesteps=n_timesteps,
                device=device,
            )
        except ImportError as e:
            raise ImportError("Please install Token2wav via: pip install minicpmo-utils[all]") from e
        logger.info("MiniCPM Token2wav loaded from %s", asset_dir)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, bool]]:
        return [{"_is_dummy": True} for _ in range(num_reqs)]

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    @staticmethod
    def _extract_ref_audio(
        model_intermediate_buffer: list[dict[str, Any]] | None,
        index: int,
    ) -> object | None:
        if model_intermediate_buffer is None or index >= len(model_intermediate_buffer):
            return None
        info = model_intermediate_buffer[index]
        return info.get("ref_audio")

    def _normalize_ref_audio(self, ref_audio: object) -> tuple[np.ndarray, int]:
        if not isinstance(ref_audio, dict):
            raise TypeError(f"MiniCPM code2wav expects canonical ref_audio dict or None, got {type(ref_audio)}")

        wav = ref_audio.get("wav")
        sr = ref_audio.get("sr")
        if wav is None or sr is None:
            raise ValueError("MiniCPM canonical ref_audio must contain 'wav' and 'sr'.")
        if isinstance(wav, torch.Tensor):
            raise TypeError("MiniCPM canonical ref_audio['wav'] must not be a tensor.")
        if not isinstance(sr, int):
            raise TypeError(f"MiniCPM canonical ref_audio['sr'] must be int, got {type(sr)}")

        wav_np = np.asarray(wav, dtype=np.float32)
        if wav_np.ndim == 0:
            raise ValueError("MiniCPM canonical ref_audio['wav'] must be a 1-D waveform.")
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=-1)
        wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
        return wav_np, sr

    def _write_prompt_wav(self, ref_audio: object | None) -> str | None:
        target_sr = self._audio_prompt_sample_rate
        if ref_audio is None:
            # stepaudio2.Token2wav still routes through prompt conditioning even
            # when the caller conceptually has no reference audio. Provide a
            # short silent prompt so the downstream mel path always receives a
            # valid waveform.
            wav_np = np.zeros((target_sr,), dtype=np.float32)
            sr = target_sr
        else:
            wav_np, sr = self._normalize_ref_audio(ref_audio)
            if wav_np.size == 0:
                raise ValueError("MiniCPM ref_audio is empty.")
            if sr != target_sr:
                wav_np = _resample_audio(wav_np.astype(np.float32), orig_sr=sr, target_sr=target_sr)
                sr = target_sr

        with tempfile.NamedTemporaryFile(prefix="minicpm_ref_", suffix=".wav", delete=False) as f:
            prompt_wav_path = f.name
        sf.write(prompt_wav_path, wav_np, sr)
        return prompt_wav_path

    def _decode_one(
        self,
        token_ids: torch.Tensor | list[int],
        ref_audio: object | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.reshape(-1).tolist()
        else:
            token_ids = list(token_ids)
        while token_ids and token_ids[-1] == self._audio_eos_token_id:
            token_ids.pop()

        if not token_ids:
            empty = torch.zeros((0,), dtype=torch.float32)
            return empty, torch.tensor(self._output_sample_rate, dtype=torch.int32)

        assert self._token2wav is not None

        prompt_wav_path = self._write_prompt_wav(ref_audio)
        try:
            # stepaudio2.Token2wav caches the prompt-derived conditioning on the
            # instance. Clear it so each request uses its own reference audio
            # instead of reusing the previous request's speaker prompt.
            self._token2wav.cache = None
            wav_bytes = self._token2wav(token_ids, prompt_wav_path)
        finally:
            if prompt_wav_path is not None:
                try:
                    os.unlink(prompt_wav_path)
                except OSError:
                    logger.warning("Failed to remove temporary MiniCPM prompt wav: %s", prompt_wav_path)

        waveform, sr = sf.read(io.BytesIO(wav_bytes))
        waveform_np = np.asarray(waveform, dtype=np.float32)
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.mean(axis=-1)
        audio_tensor = torch.from_numpy(waveform_np.reshape(-1)).to(dtype=torch.float32)
        sr_tensor = torch.tensor(int(sr), dtype=torch.int32)
        return audio_tensor, sr_tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return set()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        empty = torch.zeros((0,), dtype=torch.float32)
        sr_tensor = torch.tensor(self._output_sample_rate, dtype=torch.int32)
        if runtime_additional_information is None:
            runtime_additional_information = kwargs.get("runtime_additional_information")

        if runtime_additional_information and all(info.get("_is_dummy") for info in runtime_additional_information):
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * len(runtime_additional_information),
                    "sr": [sr_tensor] * len(runtime_additional_information),
                },
            )

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        self._ensure_token2wav_loaded()
        ids = input_ids.reshape(-1).to(dtype=torch.long)
        seq_token_counts = kwargs.get("seq_token_counts")
        request_ids_list = self._split_request_ids(ids, seq_token_counts)

        audios: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []
        for i, req_ids in enumerate(request_ids_list):
            ref_audio = self._extract_ref_audio(model_intermediate_buffer, i)
            audio_tensor, sr = self._decode_one(req_ids, ref_audio)
            audios.append(audio_tensor)
            srs.append(sr)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(f"MiniCPMO4_5Code2Wav expected (audio_tensor, sr), got {type(model_outputs)}")
        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audio_tensor, "sr": sr},
        )
