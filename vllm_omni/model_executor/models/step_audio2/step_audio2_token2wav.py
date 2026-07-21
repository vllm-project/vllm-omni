import io
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import librosa  # noqa: TID251
import numpy as np
import onnxruntime
import s3tokenizer
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from flashcosyvoice.modules.hifigan import HiFTGenerator
from flashcosyvoice.utils.audio import mel_spectrogram
from hyperpyyaml import load_hyperpyyaml
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import WeightsMapper
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import (
    DEFAULT_STREAM_CONFIG,
    DEFAULT_TOKEN2WAV_CONFIG,
    STEP_AUDIO2_DEFAULT_PROMPT_WAV,
    STREAM_SOURCE_CACHE_LEN,
)

logger = init_logger(__name__)


def fade_in_out(
    fade_in_mel: torch.Tensor,
    fade_out_mel: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    """Cross-fade two overlapping waveform segments using a Hamming window.

    The window is split in half: the first half ramps *up* (fade-in) and
    the second half ramps *down* (fade-out).  The overlap region of
    ``fade_in_mel`` is blended with the tail of ``fade_out_mel``.
    """
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel


@dataclass
class _StreamState:
    """Per-request streaming state for concurrent Token2Wav sessions."""

    setup_done: bool = False
    stream_cache: dict | None = None
    hift_cache_dict: dict = field(default_factory=dict)
    finished: bool = False


class StepAudio2Token2WavCore(nn.Module):
    """Core Token2Wav model - handles token to waveform conversion"""

    def __init__(
        self,
        model_path: str,
        float16: bool = False,
        device: str = "cuda",
        n_timesteps: int = DEFAULT_TOKEN2WAV_CONFIG.n_timesteps,
    ):
        super().__init__()
        self.model_path = model_path
        self.float16 = float16
        self.device = torch.device(device)
        self.n_timesteps = n_timesteps

        self._models_loaded = False
        self._audio_tokenizer = None
        self._spk_model = None
        self._flow = None
        self._hift = None

        self.cache = {}

        # Streaming state (constants from centralised config)
        self.mel_cache_len = DEFAULT_STREAM_CONFIG.mel_cache_len
        self.source_cache_len = STREAM_SOURCE_CACHE_LEN
        self.speech_window: torch.Tensor | None = None  # created lazily on device

        # On Ascend the talker worker never routes through
        # ``NPUOmniPlatform.set_device``, so gate on ``current_omni_platform``
        # and apply the Token2Wav NPU patch here (idempotent) before
        # ``_ensure_models_loaded`` runs. Covers every
        # construction site (Step-Audio2 wrapper + MiniCPM-o facade): patched
        # ``_ensure_models_loaded`` replaces HiFT's failing linear downsample,
        # while patched ``forward`` expands CosyVoice DiT masks and forces
        # MATH SDPA (avoids FA 161001).
        from vllm_omni.platforms import current_omni_platform

        if current_omni_platform.is_npu():
            from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
                apply_step_audio2_token2wav_npu_patch,
            )

            apply_step_audio2_token2wav_npu_patch()

    def _ensure_models_loaded(self):
        """Lazy load models on first use"""
        if self._models_loaded:
            return

        from vllm.logger import init_logger

        logger = init_logger(__name__)
        logger.info(f"Loading Token2Wav models from: {self.model_path}")

        self._audio_tokenizer = (
            s3tokenizer.load_model(f"{self.model_path}/speech_tokenizer_v2_25hz.onnx").to(self.device).eval()
        )

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self._spk_model = onnxruntime.InferenceSession(
            f"{self.model_path}/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"]
        )

        with open(f"{self.model_path}/flow.yaml") as f:
            configs = load_hyperpyyaml(f)
        self._flow = configs["flow"]
        if self.float16:
            self._flow.half()
        self._flow.load_state_dict(
            torch.load(f"{self.model_path}/flow.pt", map_location="cpu", weights_only=True), strict=True
        )
        self._flow.to(self.device).eval()

        self._hift = HiFTGenerator()
        hift_state_dict = {
            k.replace("generator.", ""): v
            for k, v in torch.load(f"{self.model_path}/hift.pt", map_location="cpu", weights_only=True).items()
        }
        self._hift.load_state_dict(hift_state_dict, strict=True)
        self._hift.to(self.device).eval()

        self._models_loaded = True
        logger.info("Token2Wav models loaded successfully")

    @property
    def audio_tokenizer(self):
        """Lazy-loaded audio tokenizer"""
        self._ensure_models_loaded()
        return self._audio_tokenizer

    @property
    def spk_model(self):
        """Lazy-loaded speaker model"""
        self._ensure_models_loaded()
        return self._spk_model

    @property
    def flow(self):
        """Lazy-loaded flow model"""
        self._ensure_models_loaded()
        return self._flow

    @property
    def hift(self):
        """Lazy-loaded HiFT vocoder"""
        self._ensure_models_loaded()
        return self._hift

    def _prepare_prompt(self, prompt_wav: str):
        """Prepare prompt audio for conditioning"""
        # Prefer soundfile/librosa path to avoid torchaudio->torchcodec runtime coupling.
        try:
            audio_np_raw, sample_rate = sf.read(prompt_wav, dtype="float32", always_2d=False)
        except Exception:
            audio_np_raw, sample_rate = librosa.load(prompt_wav, sr=None, mono=True)
        if isinstance(audio_np_raw, np.ndarray) and audio_np_raw.ndim > 1:
            audio_np_raw = np.mean(audio_np_raw, axis=-1)
        sample_rate = int(sample_rate)

        # 16 kHz branch: s3tokenizer + speaker embedding
        audio_16k = audio_np_raw.astype(np.float32)
        if sample_rate != 16000:
            audio_16k = librosa.resample(y=audio_16k, orig_sr=sample_rate, target_sr=16000)
        audio = torch.from_numpy(audio_16k)
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(
            mels.to(self.device), mels_lens.to(self.device)
        )

        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = torch.tensor(
            self.spk_model.run(None, {self.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()})[0],
            device=self.device,
        )

        # 24 kHz branch: mel spectrogram for flow model conditioning
        # Must resample from the ORIGINAL audio, not the 16 kHz version.
        audio_24k = audio_np_raw.astype(np.float32)
        if sample_rate != 24000:
            audio_24k = librosa.resample(y=audio_24k, orig_sr=sample_rate, target_sr=24000)
        audio_24k_t = torch.from_numpy(audio_24k).unsqueeze(0)  # [1, T]
        prompt_mel = mel_spectrogram(audio_24k_t).transpose(1, 2).squeeze(0)  # [T, num_mels]
        prompt_mels = prompt_mel.unsqueeze(0).to(self.device)
        prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device=self.device)
        prompt_mels = torch.nn.functional.pad(
            prompt_mels,
            (0, 0, 0, prompt_speech_tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]),
            mode="replicate",
        )

        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    def forward(
        self, generated_speech_tokens: list, prompt_wav: str, return_bytes: bool = True
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
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device=self.device)
        generated_speech_tokens_lens = torch.tensor(
            [generated_speech_tokens.shape[1]], dtype=torch.int32, device=self.device
        )

        with torch.amp.autocast(str(self.device.type), dtype=torch.float16 if self.float16 else torch.float32):
            mel = self.flow.inference(
                generated_speech_tokens,
                generated_speech_tokens_lens,
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                prompt_mels,
                prompt_mels_lens,
                spk_emb,
                self.n_timesteps,
            )

        wav, _ = self.hift(speech_feat=mel)

        if return_bytes:
            output = io.BytesIO()
            torchaudio.save(output, wav.cpu(), sample_rate=DEFAULT_TOKEN2WAV_CONFIG.sample_rate, format="wav")
            return output.getvalue()
        else:
            return wav

    # ------------------------------------------------------------------
    # Per-request streaming (does NOT mutate self – uses external state)
    # ------------------------------------------------------------------

    def setup_stream_for(self, prompt_wav: str, state: _StreamState) -> None:
        """Initialise flow + HiFT caches into *state* (no self mutation)."""
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, _, spk_emb, prompt_mels, _ = self.cache[prompt_wav]

        if self.speech_window is None or self.speech_window.device != self.device:
            self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).to(
                device=self.device, dtype=torch.float32
            )

        pre_lookahead = DEFAULT_STREAM_CONFIG.pre_lookahead_len
        state.stream_cache = self.flow.setup_cache(
            torch.cat(
                [prompt_speech_tokens, prompt_speech_tokens[:, :pre_lookahead]],
                dim=1,
            ),
            prompt_mels,
            spk_emb,
            n_timesteps=self.n_timesteps,
        )
        state.hift_cache_dict = {
            "mel": torch.zeros(1, prompt_mels.shape[2], 0, device=self.device),
            "source": torch.zeros(1, 1, 0, device=self.device),
            "speech": torch.zeros(1, 0, device=self.device),
        }
        state.setup_done = True

    def stream_chunk_for(
        self,
        audio_tokens: list[int],
        prompt_wav: str,
        last_chunk: bool,
        state: _StreamState,
    ) -> torch.Tensor:
        """Process one chunk using *state* (no self mutation except speech_window)."""
        if state.stream_cache is None:
            raise ValueError("stream_cache not initialised – call setup_stream_for() first")

        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        _, _, spk_emb, prompt_mels, _ = self.cache[prompt_wav]

        token_tensor = torch.tensor([audio_tokens], dtype=torch.int32, device=self.device)

        with torch.amp.autocast(
            str(self.device.type),
            dtype=torch.float16 if self.float16 else torch.float32,
        ):
            chunk_mel, state.stream_cache = self.flow.inference_chunk(
                token=token_tensor,
                spk=spk_emb,
                cache=state.stream_cache,
                last_chunk=last_chunk,
                n_timesteps=self.n_timesteps,
            )

        # Trim estimator attention cache to avoid unbounded growth
        keep = DEFAULT_STREAM_CONFIG.estimator_cache_keep
        est_att = state.stream_cache.get("estimator_att_cache")
        if est_att is not None and est_att.shape[4] > (prompt_mels.shape[1] + keep):
            state.stream_cache["estimator_att_cache"] = torch.cat(
                [
                    est_att[:, :, :, :, : prompt_mels.shape[1]],
                    est_att[:, :, :, :, -keep:],
                ],
                dim=4,
            )

        # ---- HiFT vocoder with overlap-add ----
        hift_cache_mel = state.hift_cache_dict["mel"]
        hift_cache_source = state.hift_cache_dict["source"]
        hift_cache_speech = state.hift_cache_dict["speech"]

        mel = torch.cat([hift_cache_mel, chunk_mel], dim=2)
        speech, source = self.hift(mel, hift_cache_source)

        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)

        state.hift_cache_dict = {
            "mel": mel[..., -self.mel_cache_len :].clone().detach(),
            "source": source[:, :, -self.source_cache_len :].clone().detach(),
            "speech": speech[:, -self.source_cache_len :].clone().detach(),
        }

        if not last_chunk:
            speech = speech[:, : -self.source_cache_len]

        return speech.squeeze(0)

    @staticmethod
    def reset_stream_for(state: _StreamState) -> None:
        """Clear streaming caches for the given state."""
        state.stream_cache = None
        state.hift_cache_dict = {}
        state.setup_done = False
        state.finished = True


class StepAudio2Token2WavForConditionalGeneration(nn.Module, SupportsPP):
    """vLLM-compatible wrapper for Step-Audio2 Token2Wav"""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "token2wav.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        model_path = getattr(self.config, "token2wav_path", None)
        model_name_or_path = vllm_config.model_config.model
        if model_path is None:
            # Resolve HF repo names to local cache path
            if not os.path.isdir(model_name_or_path):
                from huggingface_hub import snapshot_download

                model_name_or_path = snapshot_download(model_name_or_path)
            model_path = f"{model_name_or_path}/token2wav"

        float16 = getattr(self.config, "token2wav_float16", False)

        device = "cuda"
        if hasattr(vllm_config, "device_config") and vllm_config.device_config:
            device = str(vllm_config.device_config.device)

        self.token2wav = StepAudio2Token2WavCore(model_path=model_path, float16=float16, device=device)

        self.have_multimodal_outputs = True
        # Required for the runner to pass async_chunk info via
        # runtime_additional_information on every forward step.
        self.enable_update_additional_information = True

        # Per-request streaming states (ordered list matching batch order).
        self._stream_states: list[_StreamState] = []

        # Resolve default prompt wav path from model directory.
        # Priority: env var > {model_dir}/assets/default_female.wav > bare filename (cwd)
        self._default_prompt_wav = self._resolve_prompt_wav(model_name_or_path)

        self.make_empty_intermediate_tensors = lambda: None

    @staticmethod
    def _resolve_prompt_wav(model_root: str) -> str:
        """Find the default prompt wav, searching model assets directory."""
        env_val = os.environ.get("STEP_AUDIO2_DEFAULT_PROMPT_WAV")
        if env_val and os.path.exists(env_val):
            return env_val
        # Search common locations: model dir, then bundled assets in this package
        pkg_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(model_root, "assets", STEP_AUDIO2_DEFAULT_PROMPT_WAV),
            os.path.join(model_root, STEP_AUDIO2_DEFAULT_PROMPT_WAV),
            os.path.join(pkg_dir, "assets", STEP_AUDIO2_DEFAULT_PROMPT_WAV),
            os.path.join(pkg_dir, STEP_AUDIO2_DEFAULT_PROMPT_WAV),
            STEP_AUDIO2_DEFAULT_PROMPT_WAV,
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        # Return the env var (even if not found) or the bare filename as fallback
        return env_val or STEP_AUDIO2_DEFAULT_PROMPT_WAV

    def get_language_model(self) -> torch.nn.Module:
        # Return the wrapper so vLLM interface helpers resolve
        # embed_input_ids() on this adapter instead of the bare core model.
        return self

    @property
    def sampler(self):
        return Sampler()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional: dict | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | bytes:
        """
        Forward pass for Token2Wav

        Args:
            input_ids: Audio token IDs (from stage input processor)
            positions: Position IDs (unused)
            intermediate_tensors: Intermediate tensors (unused)
            inputs_embeds: Input embeddings (unused)
            additional: Additional information dict (deprecated)
            runtime_additional_information: Per-request dicts from the
                async_chunk processor.  Each dict contains ``audio_tokens``
                (list[int]) and ``last_chunk`` (bool).
            **kwargs: Other kwargs

        Returns:
            OmniOutput with waveform tensor
        """
        # ----- async_chunk streaming path -----
        # Only enter when at least one entry carries ``left_context_size``
        # (the field the framework forwards from the async_chunk payload).
        # The runner always passes runtime_additional_information (possibly
        # [{}]), so a bare truthiness check would incorrectly enter this
        # branch in synchronous mode.
        if runtime_additional_information and any(
            "left_context_size" in info for info in runtime_additional_information
        ):
            return self._forward_async_chunk(input_ids, runtime_additional_information, **kwargs)

        # ----- profiling / warmup guard -----
        # Profile/warmup passes use all-zero token IDs. Real audio tokens
        # are non-zero, so check sum == 0 rather than a length threshold
        # (short utterances may have fewer than 50 tokens).
        if isinstance(input_ids, torch.Tensor):
            is_profile_mode = input_ids.numel() == 0 or input_ids.sum().item() == 0
        else:
            is_profile_mode = not input_ids if not hasattr(input_ids, "__len__") else len(input_ids) == 0

        if is_profile_mode:
            device = torch.device("cpu")
            if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.numel() > 0:
                device = inputs_embeds.device
            elif isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
                device = input_ids.device

            dummy_audio = torch.zeros(1000, device=device, dtype=torch.float32)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [dummy_audio]},
            )

        # ----- synchronous (full-sequence) path -----
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                input_ids_list = input_ids.cpu().tolist()
            else:
                input_ids_list = input_ids[0].cpu().tolist()
        else:
            input_ids_list = input_ids if isinstance(input_ids, list) else [input_ids]

        prompt_wav = self._default_prompt_wav

        if not os.path.exists(prompt_wav):
            raise FileNotFoundError(f"Token2Wav: prompt_wav file not found: {prompt_wav}")

        waveform_tensor = self.token2wav(input_ids_list, prompt_wav=prompt_wav, return_bytes=False)

        if waveform_tensor.dim() == 2 and waveform_tensor.shape[0] == 1:
            waveform_tensor = waveform_tensor.squeeze(0)

        waveform_tensor_cpu = waveform_tensor.detach().cpu().contiguous()

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": [waveform_tensor_cpu]},
        )

    # ------------------------------------------------------------------
    # Async-chunk streaming forward
    # ------------------------------------------------------------------

    def _forward_async_chunk(
        self,
        input_ids: torch.Tensor,
        runtime_additional_information: list[dict[str, Any]],
        **kwargs: Any,
    ) -> OmniOutput:
        """Handle one async_chunk forward step (batch=1).

        The end-of-stream flag is encoded as ``left_context_size == 1``
        (the only field the framework transparently forwards from the
        async_chunk payload to ``request.additional_information``).
        """
        # --- batch=1 guard ---
        batch_size = sum(1 for info in runtime_additional_information if "left_context_size" in info)
        if batch_size != 1:
            raise RuntimeError(
                f"Token2Wav async_chunk only supports batch=1, got {batch_size}. "
                "Batch>1 requires framework support for batch_req_ids."
            )

        info = next(info for info in runtime_additional_information if "left_context_size" in info)
        last_chunk = info.get("left_context_size", 0) == 1

        # --- Manage single stream state ---
        # If the previous request was preempted (setup_done but not finished),
        # its stale caches must be discarded before we start a new session.
        if self._stream_states and not self._stream_states[0].finished:
            state = self._stream_states[0]
            if state.setup_done and not last_chunk:
                # Heuristic: a non-last chunk arriving into an already-active
                # state that we did NOT set up this call means a new request
                # reused the slot.  For a genuinely continuing request the
                # state is fine; we cannot distinguish without request IDs,
                # so we only forcibly reset on an explicit new-session signal
                # (setup_done=True but stream_cache already consumed all
                # previous data -- detected below when stream_chunk fails).
                pass
        if not self._stream_states or self._stream_states[0].finished:
            self._stream_states = [_StreamState()]
        state = self._stream_states[0]

        # --- Extract audio tokens ---
        audio_tokens = input_ids.flatten().cpu().tolist()

        # Empty chunk (e.g. EOF with no audio tokens)
        if not audio_tokens:
            if state.setup_done:
                self.token2wav.reset_stream_for(state)
            else:
                state.finished = True
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [torch.zeros(1, dtype=torch.float32)]},
            )

        prompt_wav = self._default_prompt_wav
        if not os.path.exists(prompt_wav):
            raise FileNotFoundError(f"Token2Wav: prompt_wav file not found: {prompt_wav}")

        try:
            if not state.setup_done:
                logger.info("Token2Wav: initialising streaming caches")
                self.token2wav.setup_stream_for(prompt_wav, state)

            waveform = self.token2wav.stream_chunk_for(
                audio_tokens,
                prompt_wav=prompt_wav,
                last_chunk=last_chunk,
                state=state,
            )
        except Exception:
            logger.exception("Token2Wav: stream error, resetting state")
            self.token2wav.reset_stream_for(state)
            raise

        if last_chunk:
            self.token2wav.reset_stream_for(state)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": [waveform.detach().cpu().contiguous()]},
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return hidden_states

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = set()
        for name, _ in self.token2wav.named_parameters():
            loaded_params.add(f"token2wav.{name}")
        for name, _ in self.token2wav.named_buffers():
            loaded_params.add(f"token2wav.{name}")
        return loaded_params

    def load_weights_without_buffers(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = set()
        for name, _ in self.token2wav.named_parameters():
            loaded_params.add(f"token2wav.{name}")
        return loaded_params
