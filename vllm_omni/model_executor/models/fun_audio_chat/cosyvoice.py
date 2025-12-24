# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CosyVoice3 Token2Wav module for Fun-Audio-Chat speech synthesis.

This is the third stage in the Fun-Audio-Chat S2S pipeline, converting
discrete speech tokens (from CRQ Decoder) into audio waveforms.

Reference:
- CosyVoice3: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- Fun-Audio-Chat: https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B
"""

import os
import sys
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Default speaker embedding path (relative to this file)
DEFAULT_SPK_EMBEDDING_PATH = os.path.join(os.path.dirname(__file__), "new_spk2info.pt")


class FunAudioChatCosyVoice(nn.Module, SupportsPP):
    """
    CosyVoice3 Token-to-Waveform model for speech synthesis.

    This stage takes discrete speech tokens from the CRQ decoder and
    synthesizes audio waveforms using the CosyVoice3 vocoder.

    Supports:
    - Auto-download from HuggingFace (FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
    - User-specified local model path
    - Custom speaker embeddings
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        # Model path configuration
        # Priority: engine_args.cosyvoice_model_path > env var > auto-download
        model_config = vllm_config.model_config
        self.cosyvoice_model_path = getattr(model_config, "cosyvoice_model_path", None)
        if self.cosyvoice_model_path is None:
            self.cosyvoice_model_path = os.environ.get("COSYVOICE_MODEL_PATH", None)

        # Audio config
        self.sample_rate = 24000  # CosyVoice3 sample rate
        self.token_hop_len = 25 * 30  # 30 seconds at 25Hz
        self.pre_lookahead_len = 3

        # Lazy initialization
        self._cosyvoice_model = None
        self._speaker_embedding = None

        # For vLLM interface
        self.make_empty_intermediate_tensors = lambda: None
        self.hidden_size = 1024  # Placeholder

        logger.info("CosyVoice stage initialized (model will be loaded on first use)")

    def _ensure_cosyvoice_loaded(self) -> bool:
        """Lazy-load CosyVoice3 model."""
        if self._cosyvoice_model is not None:
            return True

        try:
            # Try to import CosyVoice
            cosyvoice_available = self._setup_cosyvoice_path()
            if not cosyvoice_available:
                logger.error("CosyVoice not available. Please install CosyVoice or set COSYVOICE_PATH.")
                return False

            from cosyvoice.cli.cosyvoice import CosyVoice3

            # Determine model path
            model_path = self._get_model_path()
            if model_path is None:
                logger.error("CosyVoice model path not found. Set COSYVOICE_MODEL_PATH or download the model.")
                return False

            logger.info(f"Loading CosyVoice3 from: {model_path}")
            self._cosyvoice_model = CosyVoice3(model_path, load_trt=False, load_vllm=False, fp16=False)

            # Set static chunk size for streaming
            self._cosyvoice_model.model.flow.decoder.estimator.static_chunk_size = 2 * self.token_hop_len

            # Load default speaker embedding
            self._load_speaker_embedding()

            logger.info("CosyVoice3 loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load CosyVoice3: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _setup_cosyvoice_path(self) -> bool:
        """Setup CosyVoice import path."""
        import importlib.util

        # Check if already available
        if importlib.util.find_spec("cosyvoice.cli.cosyvoice") is not None:
            return True

        # Try common paths
        possible_paths = [
            os.environ.get("COSYVOICE_PATH", ""),
            os.path.join(os.path.dirname(__file__), "../../../../third_party/CosyVoice"),
            os.path.expanduser("~/CosyVoice"),
            "/opt/CosyVoice",
        ]

        for path in possible_paths:
            if path and os.path.exists(os.path.join(path, "cosyvoice")):
                sys.path.insert(0, path)
                matcha_path = os.path.join(path, "third_party/Matcha-TTS")
                if os.path.exists(matcha_path):
                    sys.path.insert(0, matcha_path)

                # Use find_spec to check if module is available without importing
                import importlib.util

                if importlib.util.find_spec("cosyvoice.cli.cosyvoice") is not None:
                    logger.info(f"Found CosyVoice at: {path}")
                    return True

        return False

    def _get_model_path(self) -> str | None:
        """Get CosyVoice model path, downloading if necessary."""
        # Check user-specified path
        if self.cosyvoice_model_path and os.path.exists(self.cosyvoice_model_path):
            return self.cosyvoice_model_path

        # Try default paths
        default_paths = [
            "pretrained_models/Fun-CosyVoice3-0.5B-2512",
            os.path.expanduser("~/.cache/huggingface/hub/models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512"),
        ]

        for path in default_paths:
            if os.path.exists(path):
                return path

        # Try to download from HuggingFace
        try:
            from huggingface_hub import snapshot_download

            logger.info("Downloading CosyVoice3 from HuggingFace...")
            model_path = snapshot_download(
                repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                local_dir="pretrained_models/Fun-CosyVoice3-0.5B-2512",
            )
            return model_path
        except Exception as e:
            logger.warning(f"Failed to download CosyVoice3: {e}")
            return None

    def _load_speaker_embedding(self):
        """Load default speaker embedding."""
        try:
            # Try to load from Fun-Audio-Chat utils
            spk_path = os.path.join(os.path.dirname(__file__), "../../../../Fun-Audio-Chat/utils/new_spk2info.pt")
            if os.path.exists(spk_path):
                spk_info = torch.load(spk_path, map_location="cpu", weights_only=False)
                self._speaker_embedding = spk_info.get("中文女", {}).get("embedding")
                logger.info("Loaded speaker embedding from Fun-Audio-Chat")
                return

            # Try default path
            if os.path.exists(DEFAULT_SPK_EMBEDDING_PATH):
                spk_info = torch.load(DEFAULT_SPK_EMBEDDING_PATH, map_location="cpu", weights_only=False)
                self._speaker_embedding = spk_info.get("中文女", {}).get("embedding")
                logger.info("Loaded speaker embedding from default path")
                return

            logger.warning("No speaker embedding found, using None (will use CosyVoice default)")
            self._speaker_embedding = None

        except Exception as e:
            logger.warning(f"Failed to load speaker embedding: {e}")
            self._speaker_embedding = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """
        Forward pass: Convert speech tokens to audio waveform.

        Args:
            additional_information: Contains:
                - speech_tokens: [batch, num_tokens] from CRQ decoder

        Returns:
            OmniOutput with audio waveform in multimodal_outputs
        """
        if additional_information is None:
            logger.warning("CosyVoice: No additional_information provided")
            return OmniOutput(
                text_hidden_states=torch.zeros(1, self.hidden_size),
                multimodal_outputs={"audio": None},
            )

        speech_tokens = additional_information.get("speech_tokens")
        if speech_tokens is None:
            logger.warning("CosyVoice: No speech_tokens provided")
            return OmniOutput(
                text_hidden_states=torch.zeros(1, self.hidden_size),
                multimodal_outputs={"audio": None},
            )

        # Ensure model is loaded
        if not self._ensure_cosyvoice_loaded():
            logger.error("CosyVoice model not loaded, returning empty audio")
            return OmniOutput(
                text_hidden_states=torch.zeros(1, self.hidden_size),
                multimodal_outputs={"audio": None, "sample_rate": self.sample_rate},
            )

        # Convert tokens to waveform
        audio_waveform = self.token2wav(speech_tokens)

        return OmniOutput(
            text_hidden_states=torch.zeros(1, self.hidden_size),
            multimodal_outputs={
                "audio": audio_waveform,
                "sample_rate": self.sample_rate,
            },
        )

    def token2wav(
        self,
        speech_tokens: torch.Tensor,
        speaker_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert speech tokens to audio waveform.

        Args:
            speech_tokens: [batch, num_tokens] speech tokens from CRQ decoder
            speaker_embedding: Optional speaker embedding override

        Returns:
            Audio waveform tensor [batch, num_samples]
        """

        if speaker_embedding is None:
            speaker_embedding = self._speaker_embedding

        device = speech_tokens.device

        # Filter valid tokens (0 <= token < 6561)
        batch_size = speech_tokens.shape[0]
        all_audio = []

        for b in range(batch_size):
            tokens = speech_tokens[b].tolist()
            valid_tokens = [t for t in tokens if 0 <= t < 6561]

            if len(valid_tokens) == 0:
                all_audio.append(torch.zeros(1, 0, device=device))
                continue

            audio = self._token2wav_single(valid_tokens, speaker_embedding)
            all_audio.append(audio)

        # Pad and stack
        if all_audio:
            max_len = max(a.shape[-1] for a in all_audio)
            padded = [torch.nn.functional.pad(a, (0, max_len - a.shape[-1])) for a in all_audio]
            return torch.cat(padded, dim=0)

        return torch.zeros(batch_size, 0, device=device)

    def _token2wav_single(
        self,
        tokens: list[int],
        speaker_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        """Convert a single sequence of tokens to waveform."""
        import uuid

        speech_list = []
        tokens_list = []

        # Split into 30-second segments
        time_step = 0
        while time_step * 25 < len(tokens):
            start = time_step * 25
            end = min((time_step + 30) * 25, len(tokens))
            token_segment = tokens[start:end]
            tokens_list.append(token_segment)
            time_step += 30

        # Handle short last segment
        if len(tokens_list) > 1 and len(tokens_list[-1]) < 50:
            last_segment = tokens_list.pop()
            second_last_segment = tokens_list.pop()

            merged = second_last_segment + last_segment
            split_point = len(merged) // 2

            tokens_list.append(merged[:split_point])
            tokens_list.append(merged[split_point:])

        # Process each segment
        for token_segment in tokens_list:
            this_uuid = str(uuid.uuid4())
            self._cosyvoice_model.model.hift_cache_dict[this_uuid] = None

            token_offset = 0
            for i in range(0, len(token_segment), self.token_hop_len):
                this_token = torch.tensor(
                    token_segment[: token_offset + self.token_hop_len + self.pre_lookahead_len]
                ).view(1, -1)
                finalize = this_token.shape[1] == len(token_segment)

                this_speech = self._cosyvoice_model.model.token2wav(
                    this_token,
                    torch.zeros(1, 0, dtype=torch.int32),
                    torch.zeros(1, 0, 80),
                    speaker_embedding,
                    token_offset,
                    this_uuid,
                    stream=False,
                    finalize=finalize,
                    speed=1.0,
                )
                speech_list.append(this_speech)
                token_offset += self.token_hop_len

            del self._cosyvoice_model.model.hift_cache_dict[this_uuid]

        if speech_list:
            return torch.cat(speech_list, dim=1)

        return torch.zeros(1, 0)

    def load_weights(self, weights):
        """No weights to load - CosyVoice is loaded separately."""
        return set()


__all__ = ["FunAudioChatCosyVoice"]
