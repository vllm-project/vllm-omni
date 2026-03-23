# Copyright 2025 Xiaomi Corporation.
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torchaudio
from torch import nn
from torchaudio.transforms import MelSpectrogram
from transformers import AutoTokenizer, Qwen2Config
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import TALKER_CODEC_PAD_TOKEN_ID, MiMoAudioConfig
from vllm_omni.model_executor.models.mimo_audio.modeling_audio_tokenizer import MiMoAudioTokenizer

logger = logging.getLogger(__name__)


DUMMY_CODE_SHAPE = 36


class MiMoAudioTokenizerWorker:
    def __init__(
        self,
        device_str: str,
        config_path: str,
        audio_tokenizer_path: str,
    ):
        self.device = device_str
        logger.info("[tokenizer worker] Loading MiMoAudioConfig from %s", config_path)
        load_cfg_start = time.monotonic()
        self.config: MiMoAudioConfig = MiMoAudioConfig.from_pretrained(config_path)
        logger.info(
            "[tokenizer worker] Config loaded in %.2fs",
            time.monotonic() - load_cfg_start,
        )

        if device_str == "cpu":
            model_dtype = torch.float32  # CPU must use float32
        else:
            model_dtype = torch.bfloat16  # GPU uses bfloat16

        logger.info(
            "[tokenizer worker] Loading MiMo-Audio Tokenizer from %s (device=%s, dtype=%s)",
            audio_tokenizer_path,
            self.device,
            model_dtype,
        )
        start_loading_slm_tokenizer_time = time.monotonic()
        logger.info(f"MiMoAudioTokenizer,device:{self.device}")
        self.audio_tokenizer = MiMoAudioTokenizer.from_pretrained(
            audio_tokenizer_path,
            dtype=model_dtype,
            device_map={"": self.device},
        )
        # Move to target device and only cast to bfloat16 on non-CPU devices.
        self.audio_tokenizer.to(self.device)
        if self.device != "cpu" and model_dtype == torch.bfloat16:
            self.audio_tokenizer.to(dtype=torch.bfloat16)
        self.audio_tokenizer.eval()

        logger.info(
            f"Audio Tokenizers loaded in "
            f"{time.monotonic() - start_loading_slm_tokenizer_time:.2f} seconds, "
            f"device: {self.device}"
        )

        logger.info(
            "[tokenizer worker] Building MelSpectrogram transform (sr=%s, n_fft=%s)",
            self.audio_tokenizer.config.sampling_rate,
            self.audio_tokenizer.config.nfft,
        )
        mel_start = time.monotonic()
        self.mel_transform = (
            MelSpectrogram(
                sample_rate=self.audio_tokenizer.config.sampling_rate,
                n_fft=self.audio_tokenizer.config.nfft,
                hop_length=self.audio_tokenizer.config.hop_length,
                win_length=self.audio_tokenizer.config.window_size,
                f_min=self.audio_tokenizer.config.fmin,
                f_max=self.audio_tokenizer.config.fmax,
                n_mels=self.audio_tokenizer.config.n_mels,
                power=1.0,
                center=True,
            )
            .to(self.device)
            .to(torch.float32)
        )
        logger.info(
            "[tokenizer worker] MelSpectrogram ready in %.2fs",
            time.monotonic() - mel_start,
        )

        self.group_size = self.config.group_size
        self.audio_channels = self.config.audio_channels
        self.sample_rate = self.audio_tokenizer.config.sampling_rate

        # Warmup (skip for CPU due to potential shape mismatch issues)
        if device_str != "cpu":
            logger.info("[tokenizer worker] Running warmup encode/decode...")
            warmup_start = time.monotonic()
            warmup_dtype = torch.float32 if device_str == "cpu" else torch.bfloat16
            try:
                self.encode_wav_base(torch.zeros(self.sample_rate, dtype=warmup_dtype))
                self.decode(torch.zeros(self.audio_channels, self.group_size, dtype=torch.long))
                logger.info(
                    "[tokenizer worker] Warmup finished in %.2fs",
                    time.monotonic() - warmup_start,
                )
            except Exception as e:
                logger.warning("[tokenizer worker] Warmup failed (non-critical): %s", str(e))
        else:
            logger.info("[tokenizer worker] Skipping warmup for CPU device")

    def resample_audio_if_needed(self, wav_tensor: torch.Tensor, original_sr: int):
        """Resample audio if sample rate doesn't match config"""
        target_sr = self.sample_rate
        if original_sr != target_sr:
            wav_tensor = torchaudio.functional.resample(wav_tensor, original_sr, target_sr)
        return wav_tensor

    def wav2mel(self, wav: torch.Tensor):
        """Convert waveform to mel spectrogram using consistent processing"""
        wav = wav.to(torch.float32)
        spec = self.mel_transform(wav[None, :])
        return torch.log(torch.clip(spec, min=1e-7)).squeeze()

    def group_by_length(self, features: torch.Tensor, lengths: torch.Tensor, max_length: int):
        if features.size(0) != lengths.sum().item():
            raise ValueError(f"Feature size mismatch: {features.size(0)} vs {lengths.sum().item()}")

        split_points = []
        current_sum = 0

        for i, seq_len in enumerate(lengths):
            if current_sum + seq_len > max_length and current_sum > 0:
                split_points.append(i)
                current_sum = seq_len.item()
            else:
                current_sum += seq_len.item()

        # Convert split points to group sizes
        group_sizes = []
        prev = 0
        for point in split_points:
            group_sizes.append(point - prev)
            prev = point
        if prev < len(lengths):
            group_sizes.append(len(lengths) - prev)

        len_groups = torch.split(lengths, group_sizes)
        feature_sizes = [group.sum().item() for group in len_groups]
        feature_groups = torch.split(features, feature_sizes)

        return feature_groups, len_groups

    @torch.inference_mode()
    def encode_batch_base(
        self,
        feature_groups: list[torch.Tensor],
        len_groups: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run this in cuda stream if available"""
        encoded_parts = []
        for features, lengths in zip(feature_groups, len_groups):
            codes, _ = self.audio_tokenizer.encoder.encode(
                input_features=features.to(self.device),
                input_lens=lengths.to(self.device),
                return_codes_only=True,
            )
            encoded_parts.append(codes)
        codes_packed = torch.cat(encoded_parts, dim=1)

        codes = codes_packed.transpose(0, 1).detach()
        audio_codes = codes[:, : self.audio_channels]

        # Pad the sequence to be a multiple of group_size by repeating the last frame
        T = audio_codes.shape[0]
        if T % self.group_size != 0:
            pad = self.group_size - (T % self.group_size)
            last_tokens = audio_codes[-1, :]  # Keep dim for repeat
            padding_tokens = last_tokens.expand(pad, -1)
            audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)

        audio_codes = audio_codes.transpose(0, 1).cpu()

        return audio_codes  # [audio_channels, T] cpu

    @torch.inference_mode()
    def encode(self, audio: tuple[torch.Tensor, int], max_length: float | None = 256000) -> torch.Tensor:
        """wav: [samples] cpu, sample_rate = 24000"""
        wav, original_sr = audio
        wav = self.resample_audio_if_needed(wav, original_sr)
        wav = wav.to(self.device)

        mel = self.wav2mel(wav).transpose(0, 1)  # (seq_len, n_mels)
        input_len = mel.size(0)
        input_features = mel
        segment_size = 6000
        input_len_seg = [segment_size] * (input_len // segment_size)

        if input_len % segment_size > 0:
            input_len_seg.append(input_len % segment_size)

        input_lens = torch.tensor(input_len_seg, device=self.device)

        feature_groups, len_groups = self.group_by_length(input_features, input_lens, max_length)

        audio_codes = self.encode_batch_base(feature_groups, len_groups)
        return audio_codes  # [audio_channels, T] cpu

    @torch.inference_mode()
    def encode_wav_base(
        self,
        wav: torch.Tensor,  # [samples] cpu
    ) -> torch.Tensor:
        """Run this in cuda stream if available"""
        wav = wav.to(self.device)

        mel = self.wav2mel(wav).transpose(0, 1)  # (seq_len, n_mels)
        input_features = mel  # [seq_len, n_mels]
        input_lens = torch.tensor([mel.shape[0]], device=self.device)
        codes_packed, _ = self.audio_tokenizer.encoder.encode(
            input_features=input_features,
            input_lens=input_lens,
            return_codes_only=True,
        )
        codes = codes_packed.transpose(0, 1).detach()
        audio_codes = codes[:, : self.audio_channels]

        # Pad the sequence to be a multiple of group_size by repeating the last frame
        T = audio_codes.shape[0]
        if T % self.group_size != 0:
            pad = self.group_size - (T % self.group_size)
            last_tokens = audio_codes[-1, :]  # Keep dim for repeat
            padding_tokens = last_tokens.expand(pad, -1)
            audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)

        audio_codes = audio_codes.transpose(0, 1).cpu()

        return audio_codes  # [audio_channels, T] cpu

    @torch.inference_mode()
    def decode(
        self,
        tokens: torch.Tensor,  # [audio_channels, T] cpu
    ) -> torch.Tensor:
        """Decode audio tokens to waveform using the tokenizer's decoder"""
        tokens = tokens.to(self.device)
        with torch.no_grad():
            decoded_audio: torch.Tensor = self.audio_tokenizer.decode(tokens)
        decoded_audio = decoded_audio.float().reshape(-1).detach().cpu()
        return decoded_audio  # [samples] cpu


@dataclass
class AudioStreamerConfig:
    group_size: int
    audio_channels: int


@dataclass
class MiMoAudioCodes:
    sosp: int
    eosp: int
    sostm: int
    eostm: int
    im_end: int
    pad: int
    eot: int
    empty: int

    @classmethod
    def from_tokenizer(cls, tokenizer: AutoTokenizer) -> "MiMoAudioCodes":
        def idx(token: str) -> int:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if not isinstance(token_id, int):
                raise ValueError(f"Token {token} is not mapped to a single id.")
            return token_id

        return cls(
            sosp=idx("<|sosp|>"),
            eosp=idx("<|eosp|>"),
            sostm=idx("<|sostm|>"),
            eostm=idx("<|eostm|>"),
            im_end=idx("<|im_end|>"),
            pad=idx("<|endoftext|>"),
            eot=idx("<|eot|>"),
            empty=idx("<|empty|>"),
        )


def extract_audio_code_tensor(
    flat_codes: torch.Tensor,
    group_size: int,
    audio_channels: int,
    codes: MiMoAudioCodes,
) -> torch.Tensor | None:
    """Convert flattened talker output into [audio_channels, T] codes."""
    if flat_codes.numel() == 0:
        return None

    group_width = group_size * (audio_channels + 1)
    usable = (flat_codes.numel() // group_width) * group_width
    if usable == 0:
        return None

    groups = flat_codes[:usable].view(-1, group_size, audio_channels + 1)
    audio_buffer: list[torch.Tensor] = []

    for group in groups:
        text_token = int(group[0, 0].item())
        if text_token == codes.empty:
            audio_buffer.append(group[:, 1:])
        elif text_token == codes.eostm:
            break

    if not audio_buffer:
        return None

    audio_tokens = torch.cat(audio_buffer, dim=0).transpose(0, 1).contiguous()
    return audio_tokens  # [audio_channels, T]


def _normalize_tokenizer_worker_cache_key(
    device: torch.device,
    config_path: str | None,
    audio_tokenizer_path: str,
) -> tuple[str, str, str]:
    """Normalize cache key so that same tokenizer always hits the same cache entry."""
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    # Use realpath so symlinks / trailing slash don't create duplicate entries
    ap = audio_tokenizer_path or ""
    if ap and os.path.exists(ap):
        ap = os.path.realpath(ap)
    cp = config_path or ""
    if cp and os.path.exists(cp):
        cp = os.path.realpath(cp)

    if not cp and ap:
        cp = os.path.dirname(ap)
    return (device_type, cp, ap)


_TOKENIZER_WORKER_CACHE: dict[tuple[str, str, str], MiMoAudioTokenizerWorker] = {}


def get_tokenizer_worker(
    device: torch.device,
    config_path: str,
    audio_tokenizer_path: str,
) -> MiMoAudioTokenizerWorker:
    key = _normalize_tokenizer_worker_cache_key(device, config_path, audio_tokenizer_path)
    if key not in _TOKENIZER_WORKER_CACHE:
        device_type = key[0]
        _TOKENIZER_WORKER_CACHE[key] = MiMoAudioTokenizerWorker(
            device_str=device_type,
            config_path=config_path,
            audio_tokenizer_path=audio_tokenizer_path,
        )
    logger.info(
        "[tokenizer cache worker] cache size=%s pid=%s",
        len(_TOKENIZER_WORKER_CACHE),
        os.getpid(),
    )
    return _TOKENIZER_WORKER_CACHE[key]


class MiMoAudioToken2WavForConditionalGenerationVLLM(nn.Module, SupportsPP):
    """Decode MiMo audio codes to waveform for the code2wav stage."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        config = MiMoAudioConfig(**vars(config)) if isinstance(config, Qwen2Config) else config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.lora_config = vllm_config.lora_config

        self.logits_processor = LogitsProcessor(config.vocab_size)

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.sample_rate = getattr(config, "audio_sample_rate", 24000)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path, trust_remote_code=True)
        self.codes = MiMoAudioCodes.from_tokenizer(self.tokenizer)
        self.streamer_config = AudioStreamerConfig(
            group_size=self.config.group_size,
            audio_channels=self.config.audio_channels,
        )

        self.audio_tokenizer_path = getattr(vllm_config.model_config, "audio_tokenizer_path", None) or os.environ.get(
            "MIMO_AUDIO_TOKENIZER_PATH"
        )
        if not self.audio_tokenizer_path:
            raise ValueError(
                "Audio tokenizer path is not set. Provide "
                "`model_config.audio_tokenizer_path` in the stage config "
                "or export MIMO_AUDIO_TOKENIZER_PATH."
            )

        self.tokenizer_config_path = (
            getattr(vllm_config.model_config, "audio_tokenizer_config_path", None)
            or os.environ.get("MIMO_AUDIO_TOKENIZER_PATH")
            or self.config.name_or_path
        )

        self._tokenizer_service: MiMoAudioTokenizerWorker | None = get_tokenizer_worker(
            device=self.device,
            config_path=self.tokenizer_config_path,
            audio_tokenizer_path=self.audio_tokenizer_path,
        )
        # samples per codec frame for streaming context strip (same as tokenizer frames_per_token)
        audio_tokenizer_config = self._tokenizer_service.audio_tokenizer.config
        self.total_upsample = (
            getattr(audio_tokenizer_config, "avg_pooler", 2)
            * getattr(audio_tokenizer_config, "stride_size", 2)
            * getattr(audio_tokenizer_config, "hop_length", 240)
        ) * self.config.group_size

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        audio__dict_path: str | None = None,
        **kwargs,
    ) -> set[str]:
        # Decoder has no trainable weights to load in this stage.
        return set()

    def chunked_decode_streaming(
        self,
        codes: torch.Tensor,
        chunk_size: int = 10,
        left_context_size: int = 10,
    ) -> torch.Tensor:
        """
        Decode one chunk of codes and return waveform with left context removed.

        Used when async_chunk is True: each chunk is decoded and we strip
        context_size * total_upsample samples from the start (left context).
        """
        wav_chunk = self._decode_waveform_from_codes(codes)
        if wav_chunk.numel() == 0:
            return wav_chunk
        # Infer number of codec frames: flat format has 36 elements per frame (9*4 with prepend)
        num_flat = codes.numel() if codes.ndim == 1 else codes.shape[-1]
        num_chunks = num_flat // 36
        if num_chunks <= chunk_size:
            context_size = 0
        else:
            context_size = left_context_size
        drop = context_size * self.total_upsample
        if drop >= wav_chunk.numel():
            return wav_chunk[:0].clone()
        return wav_chunk[drop:].contiguous()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        codes: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        # Support both input_ids (from vLLM) and codes (from mimo_audio.py)
        code_tensor = codes if codes is not None else input_ids

        if code_tensor is None:
            raise ValueError("Either input_ids or codes must be provided")

        if code_tensor.numel() == 0:
            raise ValueError("code_tensor is empty.")

        if getattr(self.vllm_config.model_config, "async_chunk", False):
            waveform = self.chunked_decode_streaming(code_tensor, chunk_size=10, left_context_size=10)
        else:
            waveform = self._decode_waveform_from_codes(code_tensor)

        # For warmup/dummy run: if waveform is empty, return dummy hidden_states
        # _dummy_run expects a tensor that can be processed by extract_multimodal_outputs
        if waveform.numel() == 0:
            # Return dummy hidden_states with shape [seq_len, hidden_size]
            # This is needed for warmup to avoid NoneType errors
            hidden_size = getattr(self.config, "hidden_size", None)
            if hidden_size is None:
                # Fallback: try to get from vllm_config.model_config
                hidden_size = getattr(
                    self.vllm_config.model_config,
                    "hidden_size",
                    4096,  # Default fallback
                )

            # Return [seq_len=1, hidden_size] tensor for warmup
            dummy_hidden = torch.zeros(
                (1, hidden_size),
                dtype=torch.bfloat16,
                device=self.device,
            )

            return dummy_hidden

        if waveform.numel() == 1 and waveform.item() == 0:
            # Designed for Text output only
            return None

        return waveform

    def _get_full_code_sequence(
        self,
        code_tensor: torch.Tensor | None,
        kwargs: dict,
    ) -> torch.Tensor | None:
        """Gather the full prompt token ids (code sequence) if available."""
        if code_tensor is not None and code_tensor.numel() > 1:
            return code_tensor

        sampling_metadata = kwargs.get("sampling_metadata")
        prompt_token_ids = getattr(sampling_metadata, "prompt_token_ids", None) if sampling_metadata else None

        if prompt_token_ids is None:
            return code_tensor

        if isinstance(prompt_token_ids, torch.Tensor):
            return prompt_token_ids.detach().to(torch.long).view(-1)

        if isinstance(prompt_token_ids, list):
            if len(prompt_token_ids) == 0:
                return code_tensor
            if isinstance(prompt_token_ids[0], list):
                flat = prompt_token_ids[0]
            else:
                flat = prompt_token_ids
            if len(flat) == 0:
                return code_tensor
            return torch.tensor(flat, dtype=torch.long)

        return code_tensor

    def _check_dummy_code_tensor(self, code_tensor: torch.Tensor) -> bool:
        if code_tensor is not None and code_tensor.numel() == DUMMY_CODE_SHAPE:
            code_groups = code_tensor.view(self.config.group_size, self.config.audio_channels + 1)
            return (
                (code_groups[:, 0] == TALKER_CODEC_PAD_TOKEN_ID).all() and (code_groups[:, 1:].sum() == 0).all()
            ).item()
        return False

    def _decode_waveform_from_codes(self, code_tensor: torch.Tensor) -> torch.Tensor:
        # Check if in CUDA graph capture phase
        is_capturing = torch.cuda.is_current_stream_capturing()

        # During CUDA graph capture, return dummy tensor to avoid operations like .cpu() which are not allowed
        if is_capturing:
            # Return an empty dummy tensor with shape and dtype consistent with normal output
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        if code_tensor is None or code_tensor.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        if self._check_dummy_code_tensor(code_tensor):
            return torch.zeros(1, dtype=torch.float32, device=self.device)

        if code_tensor.ndim > 1:
            if code_tensor.shape[0] == 1:
                code_tensor = code_tensor.squeeze(0)

            elif code_tensor.shape[0] == self.streamer_config.audio_channels + 1:
                code_tensor = code_tensor.transpose(0, 1).contiguous().view(-1)

            else:
                raise ValueError(
                    f"code2wav expects shape [L], [1, L] or [audio_channels+1, T], got {code_tensor.shape}"
                )

        flat_codes = code_tensor.detach().to(torch.long).cpu()

        audio_codes = extract_audio_code_tensor(
            flat_codes,
            self.streamer_config.group_size,
            self.streamer_config.audio_channels,
            self.codes,
        )

        if audio_codes is None or audio_codes.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=self.device)

        decoded_audio = self._tokenizer_service.decode(audio_codes)

        result = decoded_audio.to(self.device)

        return result

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: bool = False,
    ) -> torch.Tensor:
        return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.vllm_config.model_config.get_hidden_size())

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # code2wav stage does not need actual logits, but needs to return a valid tensor
        # Return a dummy logits tensor with shape [batch_size, vocab_size]
        if hidden_states.numel() == 0:
            return None

        batch_size = hidden_states.shape[0] if hidden_states.ndim > 0 else 1
        vocab_size = self.config.vocab_size

        # Create an all-zero logits tensor (sampler will choose the first token, usually pad token)
        # This is safe for the code2wav stage as we do not rely on sampling results
        dummy_logits = torch.zeros(
            (batch_size, vocab_size),
            dtype=hidden_states.dtype,
            device=self.device,
        )
        return dummy_logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        # code2wav stage does not need sampling, but needs to return a valid SamplerOutput to avoid CUDA errors
        # Use Sampler to handle sampling, even though the result will not be used
        if logits is None or logits.numel() == 0:
            return None

        # Use Sampler for sampling (although the result will not be used)
        # This ensures the sampling process can complete normally and avoids CUDA errors
        sampler = Sampler()
        sampler_output = sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return sampler_output
