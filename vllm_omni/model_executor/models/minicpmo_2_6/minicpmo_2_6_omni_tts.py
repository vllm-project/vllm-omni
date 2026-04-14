# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
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

import math
import os
from collections.abc import Iterable
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.minicpmo_2_6.minicpmo_2_6_omni_llm import (
    ConditionalChatTTSConfig,
    MiniCPMOConfig,
)
from vllm_omni.model_executor.models.minicpmo_2_6.minicpmo_2_6_omni_t2w import (
    ConditionalChatTTS,
    gen_logits,
)

logger = init_logger(__name__)

try:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, BertTokenizerFast
    from vocos import Vocos
    from vocos.pretrained import instantiate_class

    _tts_deps = True
except ImportError:
    _tts_deps = False
    AutoTokenizer = None
    BertTokenizerFast = None
    Vocos = None
    instantiate_class = None


class MiniCPMO26OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """MiniCPM-o Talker model: full TTS pipeline from thinker hidden states to audio waveform.

    Pipeline:
      1. Receive speaker embedding + TTS text from thinker via additional_information
      2. Tokenize text with BertTokenizerFast (ChatTTS tokenizer)
      3. ConditionalChatTTS: prefill_text → generate audio codes
      4. DVAE: decode audio codes → mel spectrogram
      5. Vocos: mel spectrogram → audio waveform

    Checkpoint weight prefix: tts.*
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config

        tts_config = getattr(config, "tts_config", None)
        if tts_config is not None:
            if isinstance(tts_config, dict):
                tts_config = ConditionalChatTTSConfig(**tts_config)
            self.tts = ConditionalChatTTS(tts_config)
        else:
            logger.warning("No tts_config found; talker TTS module will be None")
            self.tts = None

        self._tts_text_tokenizer = None
        self._llm_tokenizer = None
        self._vocos = None
        self._initialized_assets = False

        # Cached special token IDs from the LLM tokenizer
        self._spk_start_id = None
        self._spk_end_id = None
        self._tts_start_token = "<|tts_bos|>"
        self._tts_end_token = "<|tts_eos|>"

        self.make_empty_intermediate_tensors = lambda *a, **kw: None

    @property
    def sampler(self):
        return Sampler()

    def _lazy_init_assets(self):
        """Lazily load LLM tokenizer, BertTokenizerFast, and Vocos from model assets."""
        if self._initialized_assets:
            return
        self._initialized_assets = True

        model_path = self.vllm_config.model_config.model

        # --- LLM tokenizer (for finding spk_bos/spk_eos in prompt_token_ids) ---
        if _tts_deps and AutoTokenizer is not None:
            try:
                self._llm_tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )
                self._spk_start_id = self._llm_tokenizer.convert_tokens_to_ids("<|spk_bos|>")
                self._spk_end_id = self._llm_tokenizer.convert_tokens_to_ids("<|spk_eos|>")
                logger.info(
                    "Loaded LLM tokenizer; spk_start_id=%s, spk_end_id=%s",
                    self._spk_start_id, self._spk_end_id,
                )
            except Exception as e:
                logger.warning("Failed to load LLM tokenizer: %s", e)

        # --- TTS text tokenizer (BertTokenizerFast) ---
        if _tts_deps and BertTokenizerFast is not None:
            tok_path = os.path.join(model_path, "assets/chattts_tokenizer")
            if not os.path.exists(tok_path):
                tok_path = "openbmb/chattts_tokenizer"
            try:
                self._tts_text_tokenizer = BertTokenizerFast.from_pretrained(tok_path)
                logger.info("Loaded ChatTTS tokenizer from %s", tok_path)
            except Exception as e:
                logger.warning("Failed to load ChatTTS tokenizer: %s", e)

        # --- Vocos vocoder ---
        if _tts_deps and Vocos is not None:
            vocos_path = os.path.join(model_path, "assets/Vocos.pt")
            if not os.path.exists(vocos_path):
                try:
                    vocos_path = hf_hub_download(
                        repo_id="openbmb/MiniCPM-o-2_6",
                        subfolder="assets",
                        filename="Vocos.pt",
                    )
                except Exception as e:
                    logger.warning("Failed to download Vocos: %s", e)
                    vocos_path = None
            if vocos_path and os.path.exists(vocos_path):
                try:
                    self._vocos = self._init_vocos(vocos_path)
                    logger.info("Loaded Vocos vocoder from %s", vocos_path)
                except Exception as e:
                    logger.warning("Failed to init Vocos: %s", e)

    def _init_vocos(self, ckpt_path: str):
        feature_extractor = instantiate_class(
            args=(),
            init={
                "class_path": "vocos.feature_extractors.MelSpectrogramFeatures",
                "init_args": {"sample_rate": 24000, "n_fft": 1024, "hop_length": 256, "n_mels": 100},
            },
        )
        backbone = instantiate_class(
            args=(),
            init={
                "class_path": "vocos.models.VocosBackbone",
                "init_args": {"input_channels": 100, "dim": 512, "intermediate_dim": 1536, "num_layers": 8},
            },
        )
        head = instantiate_class(
            args=(),
            init={"class_path": "vocos.heads.ISTFTHead", "init_args": {"dim": 512, "n_fft": 1024, "hop_length": 256}},
        )
        device = self.tts.device if self.tts is not None else torch.device("cuda")
        vocos = Vocos(feature_extractor, backbone, head).to(device).eval().to(torch.float32)
        vocos.load_state_dict(torch.load(ckpt_path, weights_only=True, mmap=True))
        return vocos

    # ===================== Data extraction from thinker output =====================

    def _extract_spk_embeds(
        self,
        prompt_embeds: torch.Tensor,
        prompt_token_ids: List[int],
    ) -> Optional[torch.Tensor]:
        """Extract speaker embedding from thinker hidden states at spk_bos/spk_eos positions."""
        if self._spk_start_id is None or self._spk_end_id is None:
            logger.warning("spk token IDs not available, cannot extract speaker embedding")
            return None

        ids_tensor = torch.tensor(prompt_token_ids, dtype=torch.long)
        start_positions = (ids_tensor == self._spk_start_id).nonzero(as_tuple=False).squeeze(-1)
        end_positions = (ids_tensor == self._spk_end_id).nonzero(as_tuple=False).squeeze(-1)

        if len(start_positions) == 0 or len(end_positions) == 0:
            logger.warning("No spk_bos/spk_eos tokens found in prompt_token_ids")
            return None

        # Use the last speaker span (matching original _get_last_spk_embeds)
        spk_start = int(start_positions[-1].item()) + 1  # +1 to skip spk_bos itself
        spk_end = int(end_positions[-1].item())  # exclusive end

        if spk_start >= spk_end or spk_end > prompt_embeds.shape[0]:
            logger.warning("Invalid spk_bounds: start=%d, end=%d, embeds_len=%d",
                           spk_start, spk_end, prompt_embeds.shape[0])
            return None

        return prompt_embeds[spk_start:spk_end]  # [num_spk_tokens, hidden_dim]

    def _extract_tts_text(self, thinker_output_text: str) -> str:
        """Extract TTS text content from thinker output (between <|tts_bos|> and <|tts_eos|>)."""
        text = thinker_output_text
        if self._tts_start_token in text:
            text = text.split(self._tts_start_token)[-1]
        if self._tts_end_token in text:
            text = text.split(self._tts_end_token)[0]
        return text.strip()

    # ===================== TTS text preparation =====================

    def prepare_tts_text(self, text: str) -> Tuple[str, int]:
        """Format text for ConditionalChatTTS streaming input.

        Format: [Stts][spk_emb]*N + text + padding([Etts][PAD]*) + [Ptts]
        """
        tts_tokens = self._tts_text_tokenizer.encode(text, add_special_tokens=False)
        tts_tokens_len = len(tts_tokens)
        reserved_len = self.tts.streaming_text_reserved_len

        if tts_tokens_len < reserved_len:
            num_pad = reserved_len - tts_tokens_len
            pad_str = "[Etts]" + "[PAD]" * (num_pad - 1)
        else:
            tts_tokens = tts_tokens[:reserved_len]
            tts_tokens_len = len(tts_tokens)
            text = self._tts_text_tokenizer.decode(tts_tokens, add_special_tokens=False)
            pad_str = ""

        spk_placeholder = "[spk_emb]" * self.tts.num_spk_embs
        tts_text = f"[Stts]{spk_placeholder}{text}{pad_str}[Ptts]"
        return tts_text, tts_tokens_len

    def _build_streaming_mask(self, tts_tokens_len: int) -> torch.Tensor:
        seq_len = (
            1 + self.tts.num_spk_embs * self.tts.use_speaker_embedding
            + self.tts.streaming_text_reserved_len + 1
        )
        mask = torch.zeros(seq_len, dtype=torch.int8)
        mask[0: 1 + 1 + tts_tokens_len + 1] = 1
        mask[-1] = 1
        return mask

    # ===================== Full TTS generation =====================

    @torch.inference_mode()
    def generate_speech(
        self,
        spk_embeds: torch.Tensor,
        tts_text: str,
        output_chunk_size: int = 25,
        tts_max_new_tokens: int = 2048,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the TTS pipeline: text → audio codes → mel (→ optional waveform).

        Args:
            spk_embeds: Speaker embedding from thinker hidden states, shape [num_spk_embs, llm_dim]
            tts_text: Raw text to synthesize
            output_chunk_size: Audio codes to generate per chunk
            tts_max_new_tokens: Maximum total audio codes

        Returns:
            Tuple of (mel_spec, waveform).
            mel_spec is always produced; waveform is None if Vocos is unavailable.
        """
        self._lazy_init_assets()

        if self.tts is None:
            logger.error("TTS module not loaded")
            return torch.zeros(1, 100, 1), None
        if self._tts_text_tokenizer is None:
            logger.error("TTS text tokenizer not loaded")
            return torch.zeros(1, 100, 1), None

        device = self.tts.device
        dtype = self.tts.emb_text.weight.dtype

        # Ensure spk_embeds is [1, num_spk_embs, llm_dim]
        if spk_embeds.ndim == 2:
            spk_embeds = spk_embeds.unsqueeze(0)
        spk_embeds = spk_embeds.to(device=device)

        # Prepare TTS text
        formatted_text, tts_token_lens = self.prepare_tts_text(tts_text)
        tts_input_ids = torch.tensor(
            self._tts_text_tokenizer.encode(formatted_text, add_special_tokens=False),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        streaming_mask = self._build_streaming_mask(tts_token_lens).to(device=device)

        logits_warpers, logits_processors = gen_logits(
            num_code=self.tts.num_audio_tokens,
            top_P=self.tts.top_p,
            top_K=self.tts.top_k,
            repetition_penalty=self.tts.repetition_penalty,
        )

        condition_length = (
            1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs
            + self.tts.streaming_text_reserved_len + 1
        )

        # Initialize KV cache
        head_dim = self.tts.config.hidden_size // self.tts.config.num_attention_heads
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = [
            (
                torch.zeros(1, self.tts.config.num_attention_heads, condition_length - 1, head_dim, dtype=dtype, device=device),
                torch.zeros(1, self.tts.config.num_attention_heads, condition_length - 1, head_dim, dtype=dtype, device=device),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        ]

        audio_input_ids = torch.zeros(1, condition_length, self.tts.num_vq, dtype=torch.long, device=device)

        tts_start_token_len = 1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs

        # Prefill text in chunks and generate audio
        eos_reached = False
        for chunk_idx in range(math.ceil(condition_length / self.tts.streaming_text_chunk_size)):
            if chunk_idx == 0:
                begin = 0
                end = (chunk_idx + 1) * self.tts.streaming_text_chunk_size + tts_start_token_len
            else:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + tts_start_token_len
                end = min(
                    (chunk_idx + 1) * self.tts.streaming_text_chunk_size + tts_start_token_len,
                    condition_length - 1,
                )

            if end - begin <= 0:
                continue

            text_input_ids = tts_input_ids[:, begin:end]
            position_ids = torch.arange(begin, end, dtype=torch.long, device=device).unsqueeze(0)

            if begin == 0:
                past_key_values = self.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    lm_spk_emb_last_hidden_states=spk_embeds,
                )
            else:
                past_key_values = self.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                )

            outputs = self.tts.generate(
                input_ids=audio_input_ids,
                past_key_values=past_key_values,
                streaming_tts_text_mask=streaming_mask,
                max_new_token=output_chunk_size,
                force_no_stop=False,
                temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=device),
                eos_token=torch.tensor([625], dtype=torch.long, device=device),
                logits_warpers=logits_warpers,
                logits_processors=logits_processors,
            )
            audio_input_ids = outputs.audio_input_ids
            past_key_values = outputs.past_key_values

            if outputs.finished:
                eos_reached = True
                break

        # Continue generating if text chunks are exhausted but not finished
        if not eos_reached:
            while True:
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=False,
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )
                audio_input_ids = outputs.audio_input_ids
                past_key_values = outputs.past_key_values

                if outputs.finished:
                    break
                if outputs.new_ids.shape[1] > tts_max_new_tokens:
                    logger.debug("TTS generation exceeded %d tokens, stopping", tts_max_new_tokens)
                    break

        # Decode audio codes → mel spectrogram
        mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids)


        waveform = None
        if self._vocos is not None:
            waveform = self._vocos.decode(mel_spec.float()).cpu().squeeze()

        return mel_spec, waveform

    # ===================== vLLM model interface =====================

    def get_input_embeddings(self, input_ids: torch.Tensor, multimodal_embeddings=None) -> torch.Tensor:
        if self.tts is not None:
            return self.tts.emb_text(input_ids)
        return torch.zeros(input_ids.shape[0], 1)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        additional_information: Optional[dict] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], IntermediateTensors]:
        """Run full TTS pipeline.

        The additional_information (from stage_input_processor) carries:
          - prompt_embeds: thinker hidden states for prompt portion
          - prompt_token_ids: list of prompt token IDs
          - thinker_output_text: decoded text from thinker
        """
        self._lazy_init_assets()

        if additional_information is None:
            additional_information = {}

        prompt_embeds = additional_information.get("prompt_embeds")
        prompt_token_ids = additional_information.get("prompt_token_ids")
        thinker_output_text_raw = additional_information.get("thinker_output_text", [""])
        thinker_output_text = thinker_output_text_raw[0] if isinstance(thinker_output_text_raw, list) else thinker_output_text_raw

        # If data from thinker is available, extract spk_embeds and tts_text
        spk_embeds = None
        tts_text = ""

        if prompt_embeds is not None and prompt_token_ids is not None:
            if isinstance(prompt_embeds, torch.Tensor):
                spk_embeds = self._extract_spk_embeds(prompt_embeds, prompt_token_ids)
        if thinker_output_text:
            tts_text = self._extract_tts_text(thinker_output_text)

        if spk_embeds is None or not tts_text:
            logger.warning(
                "Talker forward: missing spk_embeds (got %s) or tts_text='%s', "
                "returning dummy hidden states",
                type(spk_embeds).__name__ if spk_embeds is not None else "None",
                tts_text[:50],
            )
            return None, None

        logger.info("Talker generating speech for text: '%s' (spk_embeds shape: %s)",
                     tts_text[:80], list(spk_embeds.shape))
        mel_spec, waveform = self.generate_speech(spk_embeds, tts_text)
        return mel_spec, waveform

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        # Dummy logits: the real output is the audio waveform from forward()
        # Return a single logit vector that will produce any token (doesn't matter)
        return torch.zeros(1, 2, device=hidden_states.device if isinstance(hidden_states, torch.Tensor) else "cuda")

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load tts.* weights into ConditionalChatTTS."""
        loaded_weights: set[str] = set()
        if self.tts is None:
            logger.warning("TTS module is None, skipping talker weight loading")
            return loaded_weights

        tts_weights = {}
        for k, v in weights:
            if k.startswith("tts."):
                clean_k = k.replace("tts.", "", 1)
                tts_weights[clean_k] = v
            else:
                logger.debug("Skipping non-TTS weight: %s", k)

        if tts_weights:
            missing, unexpected = self.tts.load_state_dict(tts_weights, strict=False)
            if missing:
                logger.warning("TTS missing keys (%d): %s", len(missing), missing[:10])
            if unexpected:
                logger.warning("TTS unexpected keys (%d): %s", len(unexpected), unexpected[:10])
            loaded_weights.update("tts." + k for k in tts_weights)

        return loaded_weights
