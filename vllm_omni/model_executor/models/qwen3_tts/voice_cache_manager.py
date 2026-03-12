# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.metadata_manager import MetadataManager

logger = init_logger(__name__)


@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """

    ref_code: torch.Tensor | None  # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_spk_embedding: torch.Tensor  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


class VoiceCacheManager:
    """
    Voice cache manager, responsible for managing custom voice cache functionality.

    Main features:
    1. Load uploaded speaker information from metadata.json
    2. Manage voice clone prompt cache
    3. Update cache status to metadata.json

    Security properties:
    - No pickle / torch.load
    - Safetensors-only
    - Cache path confined to voice samples directory
    """

    def __init__(self, speech_voice_samples_dir: str | None = None, metadata_manager: MetadataManager | None = None):
        """
        Initialize the voice cache manager.

        Args:
            speech_voice_samples_dir: Speech voice samples directory path,
                if None, get from environment variable
            metadata_manager: Optional MetadataManager instance for shared metadata access.
                If not provided, will create its own (less efficient).
        """
        self.speech_voice_samples_dir = speech_voice_samples_dir or os.environ.get(
            "SPEECH_VOICE_SAMPLES", "/tmp/voice_samples"
        )

        # Initialize metadata manager
        if metadata_manager is not None:
            self.metadata_manager = metadata_manager
        else:
            metadata_file = Path(self.speech_voice_samples_dir) / "metadata.json"
            self.metadata_manager = MetadataManager(metadata_file)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def load_uploaded_speakers_from_metadata(self) -> dict[str, Any] | None:
        """Load uploaded speakers from metadata manager."""
        try:
            return self.metadata_manager.get_uploaded_speakers()
        except Exception as e:
            logger.warning(f"Failed to load uploaded speakers from metadata: {e}")
            return None

    def update_metadata_cache_info(self, speaker: str, cache_file_path: Path, status: str = "ready") -> bool:
        """
        Update cache information using metadata manager.

        Args:
            speaker: Speaker name
            cache_file_path: Cache file path
            status: Cache status, default is "ready"

        Returns:
            bool: Whether the update was successful
        """
        try:
            speaker_key = speaker.lower()
            return self.metadata_manager.update_cache_info(
                speaker_key=speaker_key, cache_file_path=cache_file_path, status=status
            )
        except Exception as e:
            logger.error(f"Failed to update metadata cache info: {e}")
            return False

    # ------------------------------------------------------------------
    # Cache save (SAFE)
    # ------------------------------------------------------------------

    def save_voice_cache(
        self,
        speaker: str,
        audio_file_path: Path,
        prompt_items: list[VoiceClonePromptItem],
    ) -> bool:
        """
        Save voice cache using safetensors (no pickle, no RCE).
        """
        try:
            cache_file_path = audio_file_path.with_suffix(".safetensors")

            tensors: dict[str, torch.Tensor] = {}
            metadata: dict[str, str] = {}

            tensors["__len__"] = torch.tensor(len(prompt_items), dtype=torch.int64)

            for i, item in enumerate(prompt_items):
                prefix = f"item_{i}_"

                tensors[prefix + "ref_spk_embedding"] = item.ref_spk_embedding.detach().cpu()

                has_ref_code = item.ref_code is not None
                tensors[prefix + "has_ref_code"] = torch.tensor(int(has_ref_code), dtype=torch.int8)

                if has_ref_code:
                    tensors[prefix + "ref_code"] = item.ref_code.detach().cpu()

                tensors[prefix + "x_vector_only_mode"] = torch.tensor(int(item.x_vector_only_mode), dtype=torch.int8)
                tensors[prefix + "icl_mode"] = torch.tensor(int(item.icl_mode), dtype=torch.int8)

                if item.ref_text is not None:
                    metadata[prefix + "ref_text"] = item.ref_text

            save_file(tensors, str(cache_file_path), metadata=metadata)

            return self.update_metadata_cache_info(
                speaker=speaker,
                cache_file_path=cache_file_path,
                status="ready",
            )

        except Exception as e:
            logger.error(f"Failed to save safetensors cache for speaker {speaker}: {e}")
            self.update_metadata_cache_info(speaker, Path(""), "failed")
            return False

    # ------------------------------------------------------------------
    # Cache load (SAFE)
    # ------------------------------------------------------------------

    def load_cached_voice_prompt(
        self,
        speaker: str,
        device: str | None = None,
    ) -> list[VoiceClonePromptItem] | None:
        """
        Load cached VoiceClonePromptItem list from safetensors.
        """
        try:
            uploaded_speakers = self.load_uploaded_speakers_from_metadata()
            if not uploaded_speakers:
                return None

            speaker_key = speaker.lower()
            if speaker_key not in uploaded_speakers:
                return None

            speaker_info = uploaded_speakers[speaker_key]
            if speaker_info.get("cache_status") != "ready":
                return None

            cache_file_path = Path(speaker_info.get("cache_file", "")).resolve()

            base_dir = Path(self.speech_voice_samples_dir).resolve()

            # ---- Path confinement (critical security check)
            if not str(cache_file_path).startswith(str(base_dir)):
                logger.error(f"Illegal cache path outside base dir: {cache_file_path}")
                return None

            if not cache_file_path.exists():
                return None

            if cache_file_path.suffix != ".safetensors":
                logger.error(f"Legacy or unsafe cache format rejected: {cache_file_path}")
                return None

            with safe_open(cache_file_path, framework="pt", device="cpu") as f:
                meta = f.metadata()

                num_items = int(f.get_tensor("__len__").item())
                result: list[VoiceClonePromptItem] = []

                for i in range(num_items):
                    prefix = f"item_{i}_"

                    has_ref_code = bool(f.get_tensor(prefix + "has_ref_code").item())

                    ref_code = f.get_tensor(prefix + "ref_code").to(device) if has_ref_code else None

                    ref_spk_embedding = f.get_tensor(prefix + "ref_spk_embedding").to(device)

                    x_vector_only_mode = bool(f.get_tensor(prefix + "x_vector_only_mode").item())
                    icl_mode = bool(f.get_tensor(prefix + "icl_mode").item())

                    ref_text = meta.get(prefix + "ref_text")

                    result.append(
                        VoiceClonePromptItem(
                            ref_code=ref_code,
                            ref_spk_embedding=ref_spk_embedding,
                            x_vector_only_mode=x_vector_only_mode,
                            icl_mode=icl_mode,
                            ref_text=ref_text,
                        )
                    )

            logger.info(f"Safetensors cache loaded for speaker: {speaker}")
            return result

        except Exception as e:
            logger.warning(f"Failed to load safetensors cache for speaker {speaker}: {e}")
            return None

    # ------------------------------------------------------------------
    # Audio path helper
    # ------------------------------------------------------------------

    def get_speaker_audio_path(self, speaker: str) -> Path | None:
        """
        Get speaker's audio file path.

        Args:
            speaker: Speaker name

        Returns:
            Optional[Path]: Audio file path, returns None if speaker doesn't exist
        """
        uploaded_speakers = self.load_uploaded_speakers_from_metadata()
        if not uploaded_speakers:
            return None

        speaker_key = speaker.lower()
        if speaker_key not in uploaded_speakers:
            return None

        audio_file_path = Path(uploaded_speakers[speaker_key]["file_path"])
        if audio_file_path.exists():
            return audio_file_path

        logger.warning(f"Audio file not found for speaker {speaker}: {audio_file_path}")
        return None
