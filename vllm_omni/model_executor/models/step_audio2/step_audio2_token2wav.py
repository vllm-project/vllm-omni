import io
from typing import Optional, Iterable
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

import s3tokenizer
from hyperpyyaml import load_hyperpyyaml

# Import from step-audio2 pip package (installed with: pip install -e /path/to/Step-Audio2-main)
from flashcosyvoice.modules.hifigan import HiFTGenerator
from flashcosyvoice.utils.audio import mel_spectrogram


logger = init_logger(__name__)


def fade_in_out(fade_in_mel: torch.Tensor, fade_out_mel: torch.Tensor, window: torch.Tensor):
    """Perform fade_in_out in tensor style"""
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = \
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel


class StepAudio2Token2WavCore(nn.Module):
    """Core Token2Wav model - handles token to waveform conversion"""

    def __init__(self, model_path: str, float16: bool = False, device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.float16 = float16
        self.device = torch.device(device)

        # Load audio tokenizer (ONNX)
        self.audio_tokenizer = s3tokenizer.load_model(
            f"{model_path}/speech_tokenizer_v2_25hz.onnx"
        ).to(self.device).eval()

        # Load speaker embedding model (ONNX)
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(
            f"{model_path}/campplus.onnx",
            sess_options=option,
            providers=["CPUExecutionProvider"]
        )

        # Load flow model (CFM)
        # Now using step-audio2 pip package, cosyvoice2 is directly available
        with open(f"{model_path}/flow.yaml", "r") as f:
            configs = load_hyperpyyaml(f)
        self.flow = configs['flow']
        if float16:
            self.flow.half()
        self.flow.load_state_dict(
            torch.load(f"{model_path}/flow.pt", map_location="cpu", weights_only=True),
            strict=True
        )
        self.flow.to(self.device).eval()

        # Load HiFT Generator (vocoder)
        self.hift = HiFTGenerator()
        hift_state_dict = {
            k.replace('generator.', ''): v
            for k, v in torch.load(f"{model_path}/hift.pt", map_location="cpu", weights_only=True).items()
        }
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        # Cache for prompt processing
        self.cache = {}

        # Stream configuration
        self.mel_cache_len = 8  # 160ms
        self.source_cache_len = int(self.mel_cache_len * 480)  # 50hz mel -> 24kHz wave
        self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).to(self.device)

        # Streaming cache
        self.hift_cache_dict = {}
        self.stream_cache = None

    def _prepare_prompt(self, prompt_wav: str):
        """Prepare prompt audio for conditioning"""
        # Load and process audio for tokenization
        audio = s3tokenizer.load_audio(prompt_wav, sr=16000)  # [T]
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(
            mels.to(self.device), mels_lens.to(self.device)
        )

        # Extract speaker embedding
        spk_feat = kaldi.fbank(
            audio.unsqueeze(0),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(
            self.spk_model.run(
                None,
                {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()}
            )[0],
            device=self.device
        )

        # Load prompt mel spectrogram
        audio, sample_rate = torchaudio.load(prompt_wav, backend='soundfile')
        audio = audio.mean(dim=0, keepdim=True)  # [1, T]
        if sample_rate != 24000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)(audio)
        prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)  # [T, num_mels]
        prompt_mels = prompt_mel.unsqueeze(0).to(self.device)
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device=self.device)
        prompt_mels = torch.nn.functional.pad(
            prompt_mels,
            (0, 0, 0, prompt_speech_tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]),
            mode='replicate'
        )

        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    def forward(
        self,
        generated_speech_tokens: list,
        prompt_wav: str,
        return_bytes: bool = True
    ) -> torch.Tensor | bytes:
        """
        Convert audio tokens to waveform

        Args:
            generated_speech_tokens: List of audio token IDs (0-6560)
            prompt_wav: Path to prompt wav file for speaker conditioning
            return_bytes: If True, return WAV bytes; else return tensor

        Returns:
            Audio waveform as bytes or tensor
        """
        # Cache prompt processing
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = \
            self.cache[prompt_wav]

        # Convert token list to tensor
        generated_speech_tokens = torch.tensor(
            [generated_speech_tokens],
            dtype=torch.int32,
            device=self.device
        )
        generated_speech_tokens_lens = torch.tensor(
            [generated_speech_tokens.shape[1]],
            dtype=torch.int32,
            device=self.device
        )

        # Generate mel spectrogram using flow model
        with torch.amp.autocast(str(self.device.type), dtype=torch.float16 if self.float16 else torch.float32):
            mel = self.flow.inference(
                generated_speech_tokens, generated_speech_tokens_lens,
                prompt_speech_tokens, prompt_speech_tokens_lens,
                prompt_mels, prompt_mels_lens, spk_emb, 10
            )

        # Generate waveform using HiFT vocoder
        wav, _ = self.hift(speech_feat=mel)

        if return_bytes:
            output = io.BytesIO()
            torchaudio.save(output, wav.cpu(), sample_rate=24000, format='wav')
            return output.getvalue()
        else:
            return wav

    def set_stream_cache(self, prompt_wav: str):
        """Initialize streaming cache for a prompt"""
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = \
            self.cache[prompt_wav]

        self.stream_cache = self.flow.setup_cache(
            torch.cat([prompt_speech_tokens, prompt_speech_tokens[:, :3]], dim=1),
            prompt_mels, spk_emb, n_timesteps=10
        )

        # Initialize HiFT cache
        self.hift_cache_dict = dict(
            mel=torch.zeros(1, prompt_mels.shape[2], 0, device=self.device),
            source=torch.zeros(1, 1, 0, device=self.device),
            speech=torch.zeros(1, 0, device=self.device),
        )

    def stream(
        self,
        generated_speech_tokens: list,
        prompt_wav: str,
        last_chunk: bool = False
    ) -> bytes:
        """
        Stream audio generation chunk by chunk

        Args:
            generated_speech_tokens: List of audio tokens for this chunk
            prompt_wav: Path to prompt wav file
            last_chunk: Whether this is the last chunk

        Returns:
            PCM audio bytes for this chunk
        """
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = \
            self.cache[prompt_wav]

        generated_speech_tokens = torch.tensor(
            [generated_speech_tokens],
            dtype=torch.int32,
            device=self.device
        )

        if self.stream_cache is None:
            raise ValueError("stream_cache is not set. Call set_stream_cache() first.")

        # Generate mel chunk
        with torch.amp.autocast(str(self.device.type), dtype=torch.float16 if self.float16 else torch.float32):
            chunk_mel, self.stream_cache = self.flow.inference_chunk(
                token=generated_speech_tokens,
                spk=spk_emb,
                cache=self.stream_cache,
                last_chunk=last_chunk,
                n_timesteps=10,
            )

        # Manage cache size
        if self.stream_cache['estimator_att_cache'].shape[4] > (prompt_mels.shape[1] + 100):
            self.stream_cache['estimator_att_cache'] = torch.cat([
                self.stream_cache['estimator_att_cache'][:, :, :, :, :prompt_mels.shape[1]],
                self.stream_cache['estimator_att_cache'][:, :, :, :, -100:],
            ], dim=4)

        # Vocoder processing with cache
        hift_cache_mel = self.hift_cache_dict['mel']
        hift_cache_source = self.hift_cache_dict['source']
        hift_cache_speech = self.hift_cache_dict['speech']
        mel = torch.concat([hift_cache_mel, chunk_mel], dim=2)

        speech, source = self.hift(mel, hift_cache_source)

        # Smooth overlap
        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)

        # Update cache
        self.hift_cache_dict = dict(
            mel=mel[..., -self.mel_cache_len:].clone().detach(),
            source=source[:, :, -self.source_cache_len:].clone().detach(),
            speech=speech[:, -self.source_cache_len:].clone().detach(),
        )

        if not last_chunk:
            speech = speech[:, :-self.source_cache_len]

        # Convert to PCM bytes
        wav_np = speech.cpu().numpy()
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767.0).astype('<i2')  # 16-bit little-endian PCM
        pcm_bytes = wav_int16.tobytes()
        return pcm_bytes


class StepAudio2Token2WavForConditionalGenerationVLLM(nn.Module, SupportsPP):
    """vLLM-compatible wrapper for Step-Audio2 Token2Wav"""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "token2wav.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        # Get model path from config
        # Assuming model_path is stored in config or passed via model_config
        model_path = getattr(self.config, 'token2wav_path', None)
        if model_path is None:
            # Fallback: try to construct from model name
            model_name_or_path = vllm_config.model_config.model
            model_path = f"{model_name_or_path}/token2wav"

        float16 = getattr(self.config, 'token2wav_float16', False)

        # Get device from vllm_config
        device = "cuda"
        if hasattr(vllm_config, 'device_config') and vllm_config.device_config:
            device = str(vllm_config.device_config.device)

        # Initialize core Token2Wav model
        self.token2wav = StepAudio2Token2WavCore(
            model_path=model_path,
            float16=float16,
            device=device
        )

        # vLLM compatibility
        self.make_empty_intermediate_tensors = lambda: None

    def get_language_model(self) -> torch.nn.Module:
        return self.token2wav

    @property
    def sampler(self):
        return Sampler()

    def forward(
        self,
        generated_speech_tokens: torch.Tensor | list,
        prompt_wav: str,
        return_bytes: bool = True,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> torch.Tensor | bytes:
        """
        Forward pass for Token2Wav

        Args:
            generated_speech_tokens: Audio tokens (batch can be 1)
            prompt_wav: Path to prompt wav file
            return_bytes: Whether to return WAV bytes

        Returns:
            Waveform as bytes or tensor
        """
        # Convert tensor to list if needed
        if isinstance(generated_speech_tokens, torch.Tensor):
            generated_speech_tokens = generated_speech_tokens.squeeze().tolist()

        return self.token2wav(
            generated_speech_tokens=generated_speech_tokens,
            prompt_wav=prompt_wav,
            return_bytes=return_bytes
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        # Token2Wav outputs waveform, not logits
        return hidden_states

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return None

    def load_weights_without_buffers(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Token2Wav weights are loaded from files in __init__
        # This method is called by vLLM but we handle loading ourselves
        logger.info(
            f"[Model Loaded] name={self.__class__.__name__}, "
            f"Note: Weights loaded from model_path in __init__"
        )
        return set()
