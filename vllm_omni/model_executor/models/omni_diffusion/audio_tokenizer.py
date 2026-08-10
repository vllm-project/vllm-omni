from __future__ import annotations

import uuid
from collections.abc import Sequence
from enum import Enum, auto
from pathlib import Path

import torch
from torch import Tensor
from torchaudio import transforms
from vllm.logger import init_logger
from vllm.utils.cache import LRUCache

from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_INPUT_SAMPLE_RATE,
    OmniDiffusionModelSpecialTokens,
    OmniDiffusionTokenizerBaseData,
)

_OMNI_DIFFUSION_EXTRA_INSTALL_HINT = (
    "Install the optional Omni-Diffusion runtime dependencies with "
    "`uv pip install -e '.[omni-diffusion]'` or "
    "`pip install 'vllm-omni[omni-diffusion]'` when running from source."
)

# Common input rates are pre-cached only for convenience. Any other input
# sample rate is still supported through a bounded dynamic cache.
_COMMON_INPUT_SAMPLE_RATES = (8000, 22050, 24000, 32000, 44100, 48000)

# Keep a small LRU cache for uncommon sample rates so online hot values can be
# reused without letting arbitrary request sample rates grow memory unbounded.
_DYNAMIC_RESAMPLER_CACHE_SIZE = 1 << 10

logger = init_logger(__name__)


class OmniDiffusionAudioEncodingMode(Enum):
    """Audio encoding modes supported by the Omni-Diffusion tokenizer."""

    DISCRETE = auto()
    CONTIGUOUS = auto()


class OmniDiffusionAudioTokenizer:
    """Loads and runs the audio tokenizer for Omni-Diffusion."""

    def __init__(
        self,
        sensevoice_path: str | None,
        device: torch.device | str,
        flow_path: str | None = None,
    ) -> None:
        self.sensevoice_path = sensevoice_path
        self.flow_path = flow_path
        self.device = torch.device(device)

        self._common_resample_buffer = self._get_precache_common_sample_rates()
        self._dynamic_resample_buffer: LRUCache[int, transforms.Resample] = LRUCache(
            capacity=_DYNAMIC_RESAMPLER_CACHE_SIZE
        )

        logger.info(
            "Initialized OmniDiffusionAudioTokenizer with SenseVoice path=%s, "
            "flow path=%s, and device=%s; weights are not loaded yet.",
            self.sensevoice_path,
            self.flow_path,
            str(self.device),
        )

    def encode(
        self,
        audio: Tensor,
        sample_rate: int,
        *,
        mode: OmniDiffusionAudioEncodingMode = OmniDiffusionAudioEncodingMode.CONTIGUOUS,
    ) -> Tensor:
        """
        Encode request audio into SenseVoice fbank features for Omni-Diffusion.
        """
        if mode == OmniDiffusionAudioEncodingMode.DISCRETE:
            raise NotImplementedError("Discrete Omni-Diffusion audio encoding has not been implemented yet.")

        self._load_sensevoice()

        try:
            from funasr.utils.load_utils import extract_fbank
        except Exception as e:
            raise ImportError(
                "Failed to import FunASR extract_fbank. Make sure FunASR is installed and available. "
                f"{_OMNI_DIFFUSION_EXTRA_INSTALL_HINT}"
            ) from e

        audio = self._normalize_audio_waveform(audio)
        audio = self._resample_to_input_sample_rate(audio, sample_rate)

        frontend = self.sensevoice_kwargs["frontend"]
        # extract_fbank returns batched features; Omni-Diffusion uses the first
        # sample because each request is encoded one audio item at a time.
        # Keep feature extraction on CPU to match the official inference path.
        speech, _ = extract_fbank(
            audio.cpu(),
            data_type="sound",
            frontend=frontend,
        )
        return speech[0]

    def decode(
        self,
        audio_tokens: Tensor | Sequence[int],
        *,
        option_steps: int = 10,
    ) -> Tensor:
        """Decode Omni-Diffusion audio tokens into a CPU float waveform."""
        self._load_glm4_voice_audio_decoder()

        # token2wav follows the official batch-shaped input contract.
        tts_token = torch.tensor(
            audio_tokens,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        # GLM-4-Voice token2wav supports optional prompt conditioning. The
        # official Omni-Diffusion decode path passes empty prompt inputs here,
        # so generated audio tokens are decoded without a reference voice.
        flow_prompt_speech_token = torch.zeros((1, 0), dtype=torch.long, device=self.device)
        prompt_speech_feat = torch.zeros((1, 0, 80), dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            tts_speech, _ = self.audio_decoder.token2wav(
                tts_token,
                uuid=str(uuid.uuid4()),
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                finalize=True,
                option_steps=option_steps,
            )
        # Return a service-friendly waveform tensor for API serialization.
        return tts_speech.squeeze().to(torch.float32).cpu()

    def prepare_contiguous_audio_inputs(
        self,
        input_ids: Sequence[int],
        omni_audios: Tensor | Sequence,
        omni_audio_sample_rates: int | Tensor | Sequence | None,
        tokenizer_base_data: OmniDiffusionTokenizerBaseData,
    ) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
        """
        Replace <|audio|> tags with contiguous audio slots.

        Omni-Diffusion represents user audio with placeholder tokens in the
        prompt and a parallel list of SenseVoice fbank features. The returned
        audio indices point from those prompt slots back to the encoded audio
        features so the model can overwrite placeholder embeddings.
        """
        audio_tensors = self._normalize_audio_tensors(omni_audios)
        sample_rates = self._normalize_audio_sample_rates(omni_audio_sample_rates, len(audio_tensors))

        aud_context_id = tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_CONTEXT)
        aud_tag_id = tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_TAG)
        aud_start_id = tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_START)
        aud_end_id = tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.AUD_END)

        aud_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == aud_tag_id]
        if len(audio_tensors) != len(aud_positions):
            raise ValueError(
                f"Expected {len(aud_positions)} audio tensors to match prompt placeholders, got {len(audio_tensors)}."
            )
        if len(sample_rates) != len(audio_tensors):
            raise ValueError(f"Expected {len(audio_tensors)} audio sample rates, got {len(sample_rates)}.")

        audios: list[torch.Tensor] = []
        audio_indices: list[torch.Tensor] = []
        new_input_ids: list[int] = []
        start = 0

        for audio_idx, aud_pos in enumerate(aud_positions):
            # Each <|audio|> placeholder must consume exactly one audio tensor
            # and one sample rate, preserving request order.
            sample_rate = sample_rates[audio_idx]
            if isinstance(sample_rate, torch.Tensor):
                sample_rate = int(sample_rate.item())

            audio = self.encode(
                audio_tensors[audio_idx],
                sample_rate,
                mode=OmniDiffusionAudioEncodingMode.CONTIGUOUS,
            )
            audios.append(audio)
            # Keep the official Omni-Diffusion slot count: fbank length plus
            # four extra context positions used by the audio encoder alignment.
            audio_token_length = audio.size(0) + 4

            # Copy text tokens before the placeholder, then replace the
            # placeholder itself with begin/context/end audio tokens.
            new_input_ids += input_ids[start:aud_pos]
            new_input_ids.append(aud_start_id)

            # audio_indices stores [batch_index, sequence_index] coordinates
            # for the context slots that will receive audio embeddings.
            audio_indice_b = torch.zeros(1, audio_token_length, dtype=torch.long)
            audio_indice_s = torch.arange(
                len(new_input_ids),
                len(new_input_ids) + audio_token_length,
            ).unsqueeze(0)
            audio_indices.append(torch.stack([audio_indice_b, audio_indice_s], dim=0))

            new_input_ids += [aud_context_id] * audio_token_length
            new_input_ids.append(aud_end_id)
            start = aud_pos + 1

        new_input_ids += input_ids[start:]
        return new_input_ids, audios, audio_indices

    def _load_glm4_voice_audio_decoder(self) -> None:
        """Load the GLM-4-Voice token-to-waveform decoder used by TTS outputs."""
        if hasattr(self, "audio_decoder"):
            return
        if self.flow_path is None:
            raise ValueError("Omni-Diffusion audio decoding requires flow_path.")

        logger.info("begin loading Omni-Diffusion GLM-4-Voice audio decoder")

        self._register_glm4_voice_official_module_aliases()

        try:
            from vllm_omni.model_executor.models.omni_diffusion.third_party.glm4voice.flow_inference import (
                AudioDecoder,
            )
        except Exception as e:
            raise ImportError(
                "Failed to import the GLM-4-Voice audio decoder. "
                "Make sure the Omni-Diffusion optional dependencies are installed and available. "
                f"{_OMNI_DIFFUSION_EXTRA_INSTALL_HINT}"
            ) from e

        # flow_path points at decoder weights, not source code.
        flow_path = Path(self.flow_path).expanduser()

        config_path = flow_path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Expected GLM-4-Voice config.yaml not found at {config_path}.")

        flow_ckpt_path = flow_path / "flow.pt"
        if not flow_ckpt_path.exists():
            raise FileNotFoundError(f"Expected GLM-4-Voice flow.pt checkpoint not found at {flow_ckpt_path}.")

        hift_ckpt_path = flow_path / "hift.pt"
        if not hift_ckpt_path.exists():
            raise FileNotFoundError(f"Expected GLM-4-Voice hift.pt checkpoint not found at {hift_ckpt_path}.")

        logger.info(
            "Loading Omni-Diffusion GLM-4-Voice audio decoder, params: "
            "flow_path=%s, config_path=%s, flow_ckpt_path=%s, hift_ckpt_path=%s, device=%s",
            flow_path,
            config_path,
            flow_ckpt_path,
            hift_ckpt_path,
            str(self.device),
        )

        try:
            self.audio_decoder = AudioDecoder(
                config_path=str(config_path),
                flow_ckpt_path=str(flow_ckpt_path),
                hift_ckpt_path=str(hift_ckpt_path),
                device=str(self.device),
            )
        except Exception as e:
            raise ImportError(
                "Failed to initialize the GLM-4-Voice audio decoder used by Omni-Diffusion TTS. "
                f"{_OMNI_DIFFUSION_EXTRA_INSTALL_HINT}"
            ) from e

        logger.info("Loaded Omni-Diffusion GLM-4-Voice audio decoder successfully.")

    def _register_glm4_voice_official_module_aliases(self) -> None:
        """Register official GLM-4-Voice module names for decoder config loading."""
        logger.info("begin registering vendored GLM-4-Voice module aliases for Omni-Diffusion TTS")
        try:
            from vllm_omni.model_executor.models.omni_diffusion.third_party.glm4voice import (
                register_official_module_aliases,
            )

            register_official_module_aliases()
            logger.info("registered vendored GLM-4-Voice module aliases successfully")
        except Exception as e:
            raise ModuleNotFoundError(
                f"Could not register vendored GLM-4-Voice module aliases. {_OMNI_DIFFUSION_EXTRA_INSTALL_HINT}"
            ) from e

    def _load_sensevoice(self) -> None:
        """Load SenseVoiceSmall used for contiguous audio feature extraction."""
        if hasattr(self, "sensevoice_kwargs"):
            return
        if self.sensevoice_path is None:
            raise ValueError("Omni-Diffusion audio encoding requires sensevoice_path.")

        logger.info("Begin loading Omni-Diffusion SenseVoiceSmall")

        try:
            from funasr.models.sense_voice.model import SenseVoiceSmall
        except Exception as e:
            logger.error("Failed to import FunASR SenseVoiceSmall.")
            raise ImportError(
                "Failed to import FunASR SenseVoiceSmall. Make sure FunASR is installed and available. "
                f"{_OMNI_DIFFUSION_EXTRA_INSTALL_HINT}"
            ) from e

        logger.info(
            "Loading Omni-Diffusion SenseVoiceSmall from: %s, device: %s.", self.sensevoice_path, str(self.device)
        )

        _, self.sensevoice_kwargs = SenseVoiceSmall.from_pretrained(
            model=self.sensevoice_path,
            # FunASR forwards this value through OmegaConf, which only accepts
            # primitive values. Keep the torch.device for tensor placement, but
            # pass a string to the SenseVoice loader.
            device=str(self.device),
        )

        logger.info("Loading Omni-Diffusion SenseVoiceSmall successfully.")

    def _get_precache_common_sample_rates(self) -> dict[int, transforms.Resample]:
        """Create reusable resamplers for common input sample rates."""
        resample_buffer: dict[int, transforms.Resample] = {}
        for sample_rate in _COMMON_INPUT_SAMPLE_RATES:
            resample_buffer[sample_rate] = transforms.Resample(
                orig_freq=sample_rate,
                new_freq=OMNI_DIFFUSION_INPUT_SAMPLE_RATE,
            )
        return resample_buffer

    def _resample_to_input_sample_rate(self, audio: Tensor, sample_rate: int) -> Tensor:
        """Resample waveform to the 16 kHz input expected by SenseVoice."""
        if sample_rate == OMNI_DIFFUSION_INPUT_SAMPLE_RATE:
            return audio

        resampler = self._get_resampler(sample_rate)
        return resampler(audio.to(self.device).unsqueeze(0)).squeeze(0)

    def _normalize_audio_waveform(self, audio: Tensor) -> Tensor:
        """Convert request audio into a non-empty mono float32 waveform."""
        # Audio preprocessing does not need gradients and expects float samples.
        audio = audio.detach().to(dtype=torch.float32)
        match audio.ndim:
            # [T]: already a single-channel waveform.
            case 1:
                pass
            # [C, T]: average channels into one waveform, matching the official script.
            case 2:
                audio = audio.mean(dim=0)
            case _:
                raise ValueError(f"Expected audio tensor with shape [T] or [C, T], got {tuple(audio.shape)}.")

        # Empty audio indicates a bad request or decode failure.
        if audio.numel() == 0:
            raise ValueError("Expected non-empty audio tensor.")

        # Normalize non-standard waveform ranges, e.g. int PCM-like tensors.
        max_abs = torch.max(torch.abs(audio))
        if max_abs > 1.0:
            audio = audio / max_abs

        return audio

    def _get_resampler(self, sample_rate: int) -> transforms.Resample:
        """Get a common-rate resampler or a bounded LRU dynamic resampler."""
        if sample_rate in self._common_resample_buffer:
            return self._common_resample_buffer[sample_rate].to(self.device)

        resampler = self._dynamic_resample_buffer.get(sample_rate)
        if resampler is not None:
            return resampler.to(self.device)

        logger.debug(
            "Creating dynamic Omni-Diffusion audio resampler: input_sample_rate=%s, target_sample_rate=%s.",
            sample_rate,
            OMNI_DIFFUSION_INPUT_SAMPLE_RATE,
        )
        resampler = transforms.Resample(
            orig_freq=sample_rate,
            new_freq=OMNI_DIFFUSION_INPUT_SAMPLE_RATE,
        )
        self._dynamic_resample_buffer.put(sample_rate, resampler)
        return resampler.to(self.device)

    def _normalize_audio_tensors(self, omni_audios: torch.Tensor | Sequence) -> list[torch.Tensor]:
        if isinstance(omni_audios, torch.Tensor):
            if omni_audios.ndim in (1, 2):
                return [omni_audios]
            if omni_audios.ndim == 3:
                return list(omni_audios)
            raise ValueError(
                f"Expected omni_audios tensor with shape [T], [C, T], or [N, C, T], got {tuple(omni_audios.shape)}."
            )
        if isinstance(omni_audios, Sequence) and not isinstance(omni_audios, (str, bytes)):
            audio_tensors = list(omni_audios)
            if not all(isinstance(audio, torch.Tensor) for audio in audio_tensors):
                raise TypeError("Expected every omni_audios item to be a torch.Tensor.")
            return audio_tensors
        raise TypeError(f"Expected omni_audios to be a tensor or sequence of tensors, got {type(omni_audios)!r}.")

    def _normalize_audio_sample_rates(
        self,
        sample_rates: int | str | torch.Tensor | Sequence | None,
        audio_count: int,
    ) -> list[int | torch.Tensor]:
        if sample_rates is None:
            return [OMNI_DIFFUSION_INPUT_SAMPLE_RATE] * audio_count
        if isinstance(sample_rates, torch.Tensor):
            if sample_rates.ndim == 0:
                return [sample_rates] * audio_count
            return [sample_rate for sample_rate in sample_rates.flatten()]
        if isinstance(sample_rates, str):
            return [int(sample_rates)] * audio_count
        if isinstance(sample_rates, Sequence):
            return list(sample_rates)
        return [int(sample_rates)] * audio_count
