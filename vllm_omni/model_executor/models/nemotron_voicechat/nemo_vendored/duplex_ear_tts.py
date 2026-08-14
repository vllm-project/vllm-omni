# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Vendored from NVIDIA-NeMo/Speech (branch nemotron-labs-voicechat) nemo/collections/speechlm2/models/duplex_ear_tts.py; training-only code removed.
# Reduced for inference: base class changed LightningModule+HFHubMixin -> nn.Module;
# removed training_step/validation_step/test_step, epoch hooks, log_model_stats,
# run_evaluation_one_batch, configure_optimizers/backward/configure_model (FSDP/TP),
# oomptimizer_schema, and the lightning/peft/hydra/metrics imports they required.

import os
import time
from collections import Counter
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .compat import (
    fp32_precision,
    get_pad_id,
    load_checkpoint,
    load_pretrained_hf,
    logging,
    set_model_dict_for_partial_init,
)
from .ear_tts_model import RVQEARTTSModel
from .ear_tts_vae_codec import RVQVAEModel
from .tokenizer import AutoTokenizer


class DuplexEARTTS(nn.Module):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexEARTTS as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.trainer_config = cfg.get("trainer", None)
        self.data_cfg = cfg.data
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.normalize_text = cfg.data.get("normalize_text", False)

        # exp_manager is training-only; keep validation_save_path optional for inference configs.
        if cfg.get("exp_manager", None) is not None:
            self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")
        else:
            self.validation_save_path = None

        # move back text channel by x, in inference it advance the text channel prediction by x frames
        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)

        # Load ForCausalLM
        if self.cfg.tts_config.context_hidden_size is not None:
            self.language_model = self._load_language_model(self.cfg)
            self.embed_tokens = self._load_embed_tokens(self.cfg)
            # delete llm because we use it only to get the  embedding tokens
            del self.language_model

        # get codec run precision
        self.audio_codec_run_dtype = getattr(torch, self.cfg.get("audio_codec_run_dtype", "float32"), torch.float32)

        # Load tokenizer
        self.tokenizer = AutoTokenizer(
            self.cfg.pretrained_lm_name,
            use_fast=True,
            trust_remote_code=True,
            bos_token=self.cfg.get("bos_token", None),
            eos_token=self.cfg.get("eos_token", None),
            pad_token=self.cfg.get("pad_token", None),
        )  # Note that we are using fast tokenizer

        # Instantiate TTS model
        self.tts_model = RVQEARTTSModel(DictConfig(self.cfg.tts_config), tokenizer=self.tokenizer)
        # Load and initialize audio codec, and bind RVQ embeddings to the TTS model
        setup_audio_codec(self)

        self._codebook_size = self.tts_model.config.codebook_size

        # compute samples per frame
        self.source_samples_per_frame = int(self.source_sample_rate * cfg.data.frame_length)

        # get codec silence tokens
        codec_silence_tokens = self.get_codec_silence_frame()
        self.register_buffer("codec_silence_tokens", codec_silence_tokens)

        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_pad_id], device=self.device),
        )

        self._use_fsdp = False
        self._use_tp = False
        self.audio_prompt_latents = nn.ParameterDict()

    def get_codec_silence_frame(self):
        # Generate long zero waveform (silence)
        audio = torch.zeros(1, 10 * self.target_sample_rate).float().to(self.device)
        audio_len = torch.tensor([audio.size(-1)]).long()
        audio, audio_len = self.pad_audio_to_factor(audio, audio_len, self.target_samples_per_frame)

        with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
            sil_codes, _ = self.audio_codec.encode(audio.unsqueeze(1), audio_len)  # [1, T, C]
            sil_codes = sil_codes[0]  # [T, C]

        # Convert each frame (C tokens) into a tuple
        combos = [tuple(row.tolist()) for row in sil_codes]

        # Count frequencies
        counter = Counter(combos)

        # Pick the most common combination
        most_common_combo, freq = counter.most_common(1)[0]

        # Return as tensor [C]
        return torch.tensor(most_common_combo, device=self.device, dtype=torch.long)

    def _load_embed_tokens(self, cfg) -> nn.Embedding:
        """Load token embedding layer for RVQ-EAR-TTS."""
        if self.language_model:
            assert callable(self.language_model.get_input_embeddings)
            embed_tokens: nn.Embedding = self.language_model.get_input_embeddings()
        else:
            embed_tokens_state_dict = torch.load(
                cfg.pretrained_lm_embedding_path, map_location="cpu", weights_only=True
            )

            # Create token embedding layer
            vocab_size, hidden_size = embed_tokens_state_dict["weight"].size()
            embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16)
            embed_tokens.load_state_dict(embed_tokens_state_dict)
        return embed_tokens

    def _load_language_model(self, cfg):
        """Load language model for RVQ-EAR-TTS."""
        if cfg.pretrained_lm_name:
            language_model = load_pretrained_hf(
                self.cfg.pretrained_lm_name, pretrained_weights=True, trust_remote_code=True
            ).eval()
        else:
            language_model = None
        return language_model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        if self.use_local_transformer and self.local_transformer_type == "nar":  # add extra token for mask
            return self._codebook_size + 4
        return self._codebook_size + 3

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        if self.cfg.get("custom_speech_bos_id", None):
            return self.cfg.get("custom_speech_bos_id")
        return self._codebook_size + 2

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        if self.cfg.get("custom_speech_eos_id", None):
            return self.cfg.get("custom_speech_eos_id")
        return self._codebook_size + 1

    @property
    def speech_pad_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        if self.cfg.get("custom_speech_pad_id", None):
            return self.cfg.get("custom_speech_pad_id")
        return self._codebook_size

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def pad_audio_to_factor(self, audio, audio_len, samples_per_frame, downsampling_factor: int = 1):
        """
        Zero pad the end of the audio so that we do not have a partial end frame.
        The output will be zero-padded to have an integer number of frames of
        length `samples_per_frame * downsampling_factor`.

        Args:
            audio: input time-domain signal (B, T)
            audio_len: valid length for each example in the batch (B,)
            samples_per_frame: number of samples per frame
            downsampling_factor: how much each frame is downsampled in later processing

        Returns:
            padded_audio: Padded time-domain signal (B, T')
            padded_len: Adjusted valid lengths (B,)
        """
        with fp32_precision():
            total_factor = samples_per_frame * downsampling_factor
            padded_len = total_factor * torch.ceil(audio_len / total_factor).int()
            max_len = padded_len.max().int().item()
            num_padding = max_len - audio.shape[1]
            padded_audio = F.pad(audio, (0, num_padding))
        return padded_audio, padded_len

    def prepare_inputs(self, batch: dict):
        """
        Prepare inputs, extracting audio tokens and padding if needed.
        """
        # check if audios has the same batch size
        assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
        assert batch["audio_prompt"].size(0) == batch["target_audio"].size(0)

        target_audio = batch["target_audio"]
        target_audio_lens = batch["target_audio_lens"]
        target_text_tokens = batch["target_text_tokens"]
        non_prompt_mask = batch["non_prompt_mask"]
        aligned_attention_mask = batch["aligned_attention_mask"]
        aligned_position_ids = batch["aligned_position_ids"]

        if self.training and (self.cfg.get("empty_turn_probability", 0.0) > 0):
            # Randomly decide whether this batch gets emptied
            if torch.rand(1).item() < self.cfg.empty_turn_probability:
                # Zero out audio
                target_audio = torch.zeros_like(target_audio)

                # Create mask for tokens we want to drop
                # Keep BOS and EOS, drop the rest.
                keep_mask = (target_text_tokens == self.text_bos_id) | (target_text_tokens == self.text_eos_id)
                full_dropout_mask = ~keep_mask  # True = positions to replace with PAD

                # Replace all non-BOS/EOS with PAD
                target_text_tokens = torch.where(
                    full_dropout_mask, torch.full_like(target_text_tokens, self.text_pad_id), target_text_tokens
                )

        # extract target audio codes
        target_audio, target_audio_lens = self.pad_audio_to_factor(
            target_audio, target_audio_lens, self.target_samples_per_frame, 1
        )
        with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
            target_codes, target_codes_lens = self.audio_codec.encode(target_audio.unsqueeze(1), target_audio_lens)

        with fp32_precision():
            target_len = target_codes.shape[1]

            # Pad or truncate sequence variables
            def pad_or_truncate(x, pad_value=0):
                if x.dim() == 2:  # [B, T]
                    L = x.shape[1]
                    if L < target_len:
                        return F.pad(x, (0, target_len - L), value=pad_value)
                    else:
                        return x[:, :target_len]
                return x  # leave others for now

            target_text_tokens = pad_or_truncate(target_text_tokens, pad_value=self.text_pad_id)
            non_prompt_mask = pad_or_truncate(non_prompt_mask, pad_value=0)
            aligned_position_ids = pad_or_truncate(aligned_position_ids, pad_value=0)

            # Correct attention mask padding/truncation
            B, H, L1, L2 = aligned_attention_mask.shape
            new_len = target_len
            if L1 < new_len or L2 < new_len:
                pad_rows = new_len - L1
                pad_cols = new_len - L2
                aligned_attention_mask = F.pad(aligned_attention_mask, (0, pad_cols, 0, pad_rows))
            elif L1 > new_len or L2 > new_len:
                aligned_attention_mask = aligned_attention_mask[:, :, :new_len, :new_len]

        # set the pad token for the first BOS frame
        target_codes_aligned = target_codes.clone()
        target_codes_aligned[:, 0] = self.speech_pad_id

        # set special token in the last audio prompt (it will works as a BOS token)
        pos = non_prompt_mask.float().argmax(dim=1)  # shape: [B]
        row_idx = torch.arange(B, device=self.device)
        # set the extra self.speech_pad_id at first 1 position in non_prompt_mask
        target_codes_aligned[row_idx, pos] = self.speech_pad_id

        # EOS dropout to make the model more robust
        if self.training and self.cfg.get("text_eos_dropout_prob", 0.0) > 0:
            # Mask EOS positions
            eos_mask = target_text_tokens == self.text_eos_id

            # Random dropout only on EOS positions
            dropout_mask = torch.rand(eos_mask.sum(), device=target_text_tokens.device) < self.cfg.text_eos_dropout_prob

            # Scatter dropout decisions into [B, T]
            full_dropout_mask = torch.zeros_like(target_text_tokens, dtype=torch.bool)
            full_dropout_mask[eos_mask] = dropout_mask

            # Replace dropped EOS with PAD
            target_text_tokens = torch.where(
                full_dropout_mask, torch.full_like(target_text_tokens, self.text_pad_id), target_text_tokens
            )

        # BOS dropout to make the model more robust
        if self.training and self.cfg.get("text_bos_dropout_prob", 0.0) > 0:
            # Mask BOS positions
            bos_mask = target_text_tokens == self.text_bos_id

            # Random dropout only on BOS positions
            dropout_mask = torch.rand(bos_mask.sum(), device=target_text_tokens.device) < self.cfg.text_bos_dropout_prob

            # Scatter dropout decisions into [B, T]
            full_dropout_mask = torch.zeros_like(target_text_tokens, dtype=torch.bool)
            full_dropout_mask[bos_mask] = dropout_mask

            # Replace dropped BOS with PAD
            target_text_tokens = torch.where(
                full_dropout_mask,
                torch.full_like(target_text_tokens, self.text_pad_id),
                target_text_tokens,
            )

        # BOS dropout to make the model more robust
        if self.training and self.cfg.get("text_bos_dropout_prob", 0.0) > 0:
            prob = self.cfg.text_bos_dropout_prob  # e.g., 0.5

            # Identify all BOS positions [B, T]
            bos_mask = target_text_tokens == self.text_bos_id

            # Get indices of sequences that actually have a BOS token
            # We need to know *where* the BOS tokens are to drop them.
            # tensor of coordinates: [[batch_idx, seq_idx], ...]
            bos_indices = torch.nonzero(bos_mask)
            num_bos = bos_indices.shape[0]

            if num_bos > 0:
                # Create a random dropout decision for each BOS instance
                drop_decisions = torch.rand(num_bos, device=target_text_tokens.device) < prob

                # Ensure at least one is dropped
                if drop_decisions.sum() == 0:
                    # Pick one random index from the available BOS locations to drop
                    force_idx = torch.randint(0, num_bos, (1,), device=target_text_tokens.device)
                    drop_decisions[force_idx] = True

                # 5. Apply the dropout
                # We need to map the decisions back to the full tensor
                # Create a mask of the same shape as target_text_tokens
                full_dropout_mask = torch.zeros_like(target_text_tokens, dtype=torch.bool)

                # Set True only at the specific (batch, seq) coordinates we chose to drop
                # bos_indices[:, 0] are batch indices, bos_indices[:, 1] are seq indices
                full_dropout_mask[bos_indices[:, 0], bos_indices[:, 1]] = drop_decisions

                # 6. Replace dropped BOS with PAD
                target_text_tokens = torch.where(
                    full_dropout_mask,
                    torch.full_like(target_text_tokens, self.text_pad_id),
                    target_text_tokens,
                )

        # shift text tokens
        subword_ids = F.pad(target_text_tokens[:, 1:], [0, 1])
        # note that we are using a text mask where we are ignoring the desc + audio prompt but we are keeping 1 until the audio ends to support duplex
        subword_mask = F.pad(non_prompt_mask[:, 1:], [0, 1])

        # detach embedding as in eartts
        if self.cfg.tts_config.context_hidden_size is not None:
            context_hidden_state = self.embed_tokens(target_text_tokens).detach()
        else:
            context_hidden_state = None

        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (target_text_tokens.shape[1] - 1) % tp_world_size) != 0:
                target_text_tokens = target_text_tokens[:, :-remainder]
                target_codes_aligned = target_codes_aligned[:, :-remainder]
                target_codes_aligned = target_codes_aligned[:, :-remainder]
                subword_ids = subword_ids[:, :-remainder]
                subword_mask = subword_mask[:, :-remainder]

        return {
            "code": target_codes_aligned,
            "audio_mask": non_prompt_mask,  # set audio_mask as non_prompt_mask to avoid the audio prompt in loss computation
            "attention_mask": aligned_attention_mask,
            "position_ids": aligned_position_ids,
            "subword_ids": subword_ids,
            "subword_mask": subword_mask,
            "context_hidden_state": context_hidden_state,
            "output_lens": target_codes_lens,
            "non_prompt_mask": non_prompt_mask,
            "target_text_tokens": target_text_tokens,
        }

    def _get_generation_config(self, guidance_enabled: bool = False):
        """Get default generation config for EAR-TTS."""
        return {
            "num_iter": 8,
            "guidance_scale": self.cfg.get("inference_guidance_scale", 0.5) if guidance_enabled else None,
            "top_p_or_k": self.cfg.get("inference_top_p_or_k", 0.8),
            "noise_scale": self.cfg.get("inference_noise_scale", 0.8),
            "eos_threshold": -3.0,
        }

    def get_audio_prompt_latent(self, name, B):
        """
        Retrieve a cached audio prompt latent and adapt it to the requested batch size.

        This fetches a latent previously cached via `set_audio_prompt_latent()` and
        ensures the returned tensor has batch size `B` by:
        - returning as-is when the batch already matches,
        - truncating when the cached batch is larger than `B`,
        - expanding when the cached batch is smaller (commonly batch=1 warmup).

        Args:
            name (str):
                Key of the cached audio prompt latent to retrieve.
            B (int):
                Desired batch size.

        Returns:
            Tensor:
                Audio prompt latent with batch dimension equal to `B`, moved to `self.device`.
                Shape: [B, ..., D]

        Raises:
            KeyError:
                If `name` does not exist in `self.audio_prompt_latents`.
        """
        if not hasattr(self, "audio_prompt_latents") or name not in self.audio_prompt_latents:
            raise KeyError(f"Unknown audio prompt latent '{name}'. Call set_audio_prompt_latent(...) first.")

        audio_prompt_latent = self.audio_prompt_latents[name]  # cached on CPU

        if audio_prompt_latent.shape[0] == B:
            out = audio_prompt_latent
        elif audio_prompt_latent.shape[0] >= B:
            out = audio_prompt_latent[:B]
        else:
            out = audio_prompt_latent[:1].expand(B, *audio_prompt_latent.shape[1:])

        return out.to(self.device)

    def set_init_inputs(self, speaker_audio=None, speaker_audio_lens=None, system_prompt=None, speaker_name=None):
        """
        Registers and prepares initial input buffers for text/audio prompt and context, to warm up AR inference.

        Args:
            speaker_audio (torch.Tensor): Batch of prompt audio, (B, T).
            speaker_audio_lens (torch.Tensor): Lengths for each sample in speaker_audio, (B,).
            system_prompt (str, optional): System prompt for context.
            speaker_name (str, optional): Name of a pre-baked speaker latent in audio_prompt_latents.
                When provided, speaker_audio/speaker_audio_lens are ignored and silent audio is used
                as the prompt carrier; the actual voice identity comes from the cached latent.

        Returns:
            dict: Dictionary of input tensors to be passed to inference, with registered buffers.
        """
        # compute prompt audio size and slice it
        with fp32_precision():
            # compute the exact number of samples for the prompt duration
            prompt_audio_size = int(
                ((self.data_cfg.audio_prompt_duration * self.target_sample_rate) // self.target_samples_per_frame)
                * self.target_samples_per_frame
            )

            if speaker_name is not None:
                logging.info(
                    f"set_init_inputs: using pre-baked latent for speaker '{speaker_name}' (silent carrier audio)"
                )
                speaker_audio = torch.zeros((1, prompt_audio_size), device=self.device, dtype=torch.float32)
                speaker_audio_lens = torch.LongTensor([speaker_audio.shape[1]]).to(self.device)

            B, T = speaker_audio.shape
            device = speaker_audio.device
            dtype = speaker_audio.dtype

            # allocate result
            prompt_audio = torch.zeros(B, prompt_audio_size, device=device, dtype=dtype)

            # process each example independently
            for b in range(B):
                valid_len = min(speaker_audio_lens[b].item(), T)

                # handle empty
                if valid_len <= 0:
                    continue

                # valid (non-padded) segment
                valid_segment = speaker_audio[b, :valid_len]

                if valid_len >= prompt_audio_size:
                    # enough valid audio → crop from start (no silence)
                    prompt_audio[b] = valid_segment[:prompt_audio_size]
                else:
                    # too short → repeat and crop
                    repeat_factor = (prompt_audio_size + valid_len - 1) // valid_len  # ceil division
                    expanded = valid_segment.repeat(repeat_factor)
                    prompt_audio[b] = expanded[:prompt_audio_size]

        # add a silence in the end to smooth the transition between prompt and audio tokens
        prompt_audio[:, -int(self.target_samples_per_frame * 2) :] = 0

        # get prompt audio size
        with fp32_precision():
            prompt_audio_text_pad_size = int(prompt_audio_size // self.target_samples_per_frame)

        # create a eos token id
        if system_prompt is not None and self.cfg.get("use_system_prompt", None) and system_prompt != "":
            text_prompt = torch.as_tensor(
                [self.tokenizer.bos] + self.tokenizer.text_to_ids(system_prompt) + [self.tokenizer.eos],
                dtype=torch.long,
                device=self.device,
            )
        else:
            text_prompt = torch.tensor([self.tokenizer.eos], dtype=torch.long, device=self.device)

        # create a padding tensor
        prompt_audio_text_pad = (
            torch.ones(prompt_audio_text_pad_size, device=self.device, dtype=text_prompt.dtype) * self.text_pad_id
        )
        prompt_audio_text_pad[-1] = self.tokenizer.eos

        # Prepend an initial text EOS token followed by padding tokens that match
        # the number of audio-prompt frames (in text-token units).
        target_text_tokens = torch.cat([text_prompt, prompt_audio_text_pad.to(text_prompt.dtype)])

        # create pad audio for the description
        pad_size = text_prompt.size(-1) * self.target_samples_per_frame
        pad_audio = (
            torch.zeros(pad_size, device=prompt_audio.device, dtype=prompt_audio.dtype)
            .unsqueeze(0)
            .repeat(prompt_audio.size(0), 1)
        )

        # repeat to reaches the batch size
        target_text_tokens = target_text_tokens.unsqueeze(0).repeat(prompt_audio.size(0), 1)
        target_audio = torch.cat([pad_audio, prompt_audio], dim=1)

        # extract code codes
        target_audio_len = torch.tensor(
            [target_audio.size(-1)] * target_audio.size(0), dtype=torch.long, device=self.device
        )
        with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
            code, _ = self.audio_codec.encode(target_audio.unsqueeze(1), target_audio_len)

        # get context hidden
        if self.cfg.tts_config.context_hidden_size is not None:
            context_hidden_state = self.embed_tokens(target_text_tokens)
        else:
            context_hidden_state = None

        # create masks
        # non_prompt_mask is all zeros, because all processed is prompt
        non_prompt_mask = torch.zeros_like(target_text_tokens)
        non_prompt_mask[:, -2:] = 1  # set last valid prompt frame as 1 to allow the addition of BOS in the right place
        subword_mask = torch.zeros_like(
            target_text_tokens
        )  # subword_mask is almost all zeros because on the warmup there is only the prompt
        subword_mask[:, -3:] = (
            1  # -3 because of the it start right after the first valid prompt token and it is shifted by 1
        )

        # set the pad token for the first BOS frame
        code[:, 0] = self.speech_pad_id

        # shift subword_ids
        subword_ids = F.pad(target_text_tokens[:, 1:], [0, 1], value=0.0)

        # set special token in the last audio prompt (it will works as a BOS token)
        pos = non_prompt_mask.float().argmax(dim=1)  # shape: [B]
        row_idx = torch.arange(B, device=self.device)
        # set the extra self.speech_pad_id at first 1 position in non_prompt_mask
        code[row_idx, pos] = self.speech_pad_id

        init_inputs = {
            "code": code[:, :-1],
            "audio_mask": non_prompt_mask.bool()[
                :, :-1
            ],  # set audio_mask as non_prompt_mask to avoid the audio prompt in loss computation
            "context_hidden_state": context_hidden_state[:, :-1] if context_hidden_state is not None else None,
            "subword_ids": subword_ids[:, :-1],
            "subword_mask": subword_mask.bool()[:, :-1],
            "non_prompt_mask": non_prompt_mask.bool()[:, :-1],
        }

        if speaker_name is not None:
            init_inputs["audio_prompt_latent"] = self.get_audio_prompt_latent(speaker_name, B=1)

        self._init_input_cache = {}
        for k, v in init_inputs.items():
            if v is None:
                self._init_input_cache[k] = None
            else:
                # clone → removes inference tensor flag
                # detach → ensures no graph refs
                self._init_input_cache[k] = v.detach().clone()

        return init_inputs

    def get_init_inputs(
        self,
        B: int,
        init_inputs_names=[
            "code",
            "audio_mask",
            "context_hidden_state",
            "subword_ids",
            "subword_mask",
            "non_prompt_mask",
        ],
    ):
        """
        Returns a dictionary of initial inputs for inference, using registered buffers.

        Args:
            B (int): Required batch size.
            init_inputs_names (List[str], optional): Names of input buffers to fetch.

        Returns:
            dict: Each key is name from init_inputs_names, and value is tensor of appropriate shape (B, ...).

        Notes:
            Expands batch-1 buffers to B if necessary.
        """
        if init_inputs_names is None:
            init_inputs_names = [
                "code",
                "audio_mask",
                "context_hidden_state",
                "subword_ids",
                "subword_mask",
                "non_prompt_mask",
            ]

        if self._init_input_cache.get("audio_prompt_latent", None) is not None:
            init_inputs_names = list(init_inputs_names) + ["audio_prompt_latent"]

        init_inputs = {}
        for name in init_inputs_names:
            buf = self._init_input_cache.get(name, None)

            if buf is None:
                init_inputs[name] = None
                continue

            # Batch already matches
            if buf.shape[0] == B:
                init_inputs[name] = buf
            elif buf.shape[0] >= B:
                init_inputs[name] = buf[:B]
            else:
                # assume batch=1 warmup → expand
                init_inputs[name] = buf[:1].expand(B, *buf.shape[1:])

        return init_inputs

    @torch.inference_mode()
    def infer_codes_one_step(
        self,
        current_subword_id,
        prev_subword_id,
        current_subword_mask,
        prev_audio_tokens,
        past_key_values,
        guidance_enabled=True,
        generation_config=None,
        ignore_eos_flag_stop=True,
        request_id=None,
    ):
        """
        Runs a single autoregressive prediction step to infer audio codec codes.

        Args:
            current_subword_id (torch.Tensor): Current text token IDs, shape (B, 1).
            prev_subword_id (torch.Tensor): Previous text token IDs, shape (B, 1).
            current_subword_mask (torch.Tensor): Current mask, shape (B, 1).
            prev_audio_tokens (torch.Tensor): Previously generated audio tokens, shape (B, 1, C).
            past_key_values: Key-value cache for transformer decoder state.
            guidance_enabled (bool, optional): Enables classifier-free guidance.
            generation_config (dict, optional): Generation hyperparameters.
            ignore_eos_flag_stop (bool): If True, ignore EOS flag for stopping.

        Returns:
            Tuple[torch.Tensor, Any]:
                - Predicted audio codec token(s), shape (B, 1, C)
                - Updated past_key_values for the next step.
        """

        if self.cfg.tts_config.context_hidden_size is not None:
            # get context_hidden_state it is always one step behind current_subword_id
            # for the first step uses the last step from warmup
            context_hidden_state = self.embed_tokens(prev_subword_id)
        else:
            context_hidden_state = None

        # force silence as next token
        if self.cfg.get("inference_force_speech_silence_on_eos", True):
            silence_codes = self.codec_silence_tokens.view(1, 1, -1).expand(prev_audio_tokens.shape)
            prev_audio_tokens = torch.where(
                current_subword_id.unsqueeze(-1) == self.text_eos_id,
                silence_codes,  # silence
                prev_audio_tokens,  # keep original
            )

        # get subword_ids
        inputs = {
            "code": prev_audio_tokens,
            "context_hidden_state": context_hidden_state,
            "subword_ids": current_subword_id,
            "subword_mask": current_subword_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "guidance_enabled": guidance_enabled,
            "generation_config": generation_config,
            "ignore_eos_flag_stop": ignore_eos_flag_stop,
        }
        # request_id is only used by vLLM backend — skip for native PyTorch inference
        if request_id is not None:
            inputs["request_id"] = request_id

        outputs = self.tts_model(**inputs)

        return outputs["codes"], outputs["past_key_values"]

    @torch.inference_mode()
    def decode_one_audio_step(self, gen_audio_codes_history, number_prev_tokens=None):
        """
        Decodes one step of generated audio codec tokens to raw waveform.

        Args:
            gen_audio_codes_history (torch.Tensor): Audio tokens history, shape (B, T, C).
            number_prev_tokens (int, optional): Number of previous tokens to decode, for incremental decoding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - audio_pred_cur_step: Latest decoded waveform chunk, shape (B, wav_to_token_ratio).
                - audio_len: Lengths (number of samples), shape (B,).
        """
        with fp32_precision(), torch.no_grad():
            if number_prev_tokens:
                gen_audio_codes_history = gen_audio_codes_history[:, -number_prev_tokens:]

            gen_audio_codes_history = replace_control_speech_codes(
                gen_audio_codes_history, self._control_codes, self.codec_silence_tokens
            )
            gen_audio_codes_lens = torch.tensor(
                [gen_audio_codes_history.size(1)] * gen_audio_codes_history.size(0), device=self.device
            )
            audio_pred, audio_len = self.audio_codec.decode(gen_audio_codes_history, gen_audio_codes_lens)

        # return only the current/lastest audio chunk
        audio_pred_cur_step = audio_pred.squeeze(1)[:, -self.audio_codec.config.wav_to_token_ratio :]
        audio_len[:] = self.audio_codec.config.wav_to_token_ratio
        return audio_pred_cur_step, audio_len

    @torch.inference_mode()
    def offline_inference(
        self,
        next_subword_ids: torch.Tensor,
        init_inputs: dict,
        formatter: str = "",
        guidance_enabled: bool = True,
        generation_config: dict = None,
        incremental_audio_decoding: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Runs offline autoregressive inference for the Duplex EAR-TTS speech decoder.

        This method performs **text-to-speech (TTS)** generation: given subword/text
        tokens and prompt-initialization states, it autoregressively generates
        audio codec tokens and decodes them into a waveform.

        Args:
            next_subword_ids (torch.Tensor):
                Conditioning subword/text token IDs for the speech decoder.
                Shape: (B, T_text).

            init_inputs (dict):
                Dictionary of prompt-dependent initial states produced by
                ``get_init_inputs()``. May include:

                    • "code"                 — initial audio tokens (e.g., prompt audio)
                    • "audio_mask"           — mask for prompt audio positions
                    • "context_hidden_state" — decoder hidden state at t = 0
                    • "subword_ids"          — prompt text tokens
                    • "subword_mask"         — mask for prompt text
                    • "non_prompt_mask"      — mask marking positions to be generated

                ``get_init_inputs()`` automatically expands batch-1 buffers to
                batch size B.

            formatter (str, optional):
                Optional formatter identifier used to customize the prompt structure.

            guidance_enabled (bool, optional):
                Whether classifier-free guidance (CFG) is enabled.
                If enabled and ``generation_config`` is ``None``, guidance parameters
                are taken from ``_get_generation_config()``.

            generation_config (dict, optional):
                Settings controlling autoregressive generation, including sampling
                strategy, noise scale, refinement iterations, and EOS rules.
                If ``None``, defaults are taken from
                ``_get_generation_config(guidance_enabled)``.

            incremental_audio_decoding (bool, optional):
                If True, codec-to-waveform decoding is performed incrementally during
                autoregressive generation.
                If False, waveform decoding occurs only after all audio tokens are produced.

        Returns:
            dict[str, torch.Tensor]:
                Contains:

                • **"audio"**:
                Generated waveform of shape ``(B, T_audio)``, obtained via
                ``audio_pred.squeeze(1)``.

                • **"audio_len"**:
                Length of each generated waveform in samples, shape ``(B,)``.
        """
        B = next_subword_ids.size(0)

        if generation_config is None:
            generation_config = self._get_generation_config(guidance_enabled)
            logging.info(f"Doing inference using the following config: {generation_config} !")

        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": guidance_enabled})

        # warmup the model and generate the very first audio token
        outputs = self.tts_model(**init_inputs)

        if self.cfg.get("inference_skip_first_code_prediction_on_init", True):
            # use the last token on init, because we are shifthing it in the model forward, so we dont really need to compute it
            code = init_inputs["code"][:, -1:]
        else:
            code, _, _ = self.tts_model.generate_step(outputs.hidden_states[:, -1:], **generation_config)

        past_key_values = outputs["past_key_values"]

        # use the text tokens to stop generation
        max_steps = next_subword_ids.size(-1)
        # create variable to store the audios
        gen_audio_codes = torch.zeros(
            B, max_steps, self.tts_model.config.num_quantizers, device=self.device, dtype=torch.long
        )

        # init subwork as all ones
        subword_mask = torch.ones(B, max_steps, device=self.device, dtype=torch.bool)
        # get first context subword_id, that is the last subword_ids from the warmup
        first_context_subword_id = init_inputs["subword_ids"][:, -1].unsqueeze(-1)

        # initialize variables used to save the output audio
        audio_pred = None
        audio_pred_len = torch.zeros(B, device=self.device, dtype=torch.long)

        for i in range(max_steps):
            step_start = time.time()
            # current subword id is always seem
            current_subword_id = next_subword_ids[:, i].unsqueeze(-1)

            if i == 0:
                prev_subword_id = first_context_subword_id
            else:
                prev_subword_id = next_subword_ids[:, i - 1].unsqueeze(-1)

            # create subword_mask
            current_subword_mask = subword_mask[:, i].unsqueeze(-1)

            code, past_key_values = self.infer_codes_one_step(
                current_subword_id=current_subword_id,
                prev_subword_id=prev_subword_id,
                current_subword_mask=current_subword_mask,
                prev_audio_tokens=code,
                past_key_values=past_key_values,
                guidance_enabled=guidance_enabled,
                generation_config=generation_config,
                ignore_eos_flag_stop=True,
            )

            # cache audio tokens
            gen_audio_codes[:, i] = code.squeeze(1)

            if incremental_audio_decoding:
                audio_pred_i, audio_pred_i_len = self.decode_one_audio_step(
                    gen_audio_codes[:, : i + 1],
                    number_prev_tokens=self.cfg.get("inference_codec_decoding_prev_tokens_number", None),
                )
                if audio_pred is None:
                    audio_pred = audio_pred_i
                else:
                    audio_pred = torch.cat([audio_pred, audio_pred_i], dim=1)
                audio_pred_len += audio_pred_i_len

            step_time = time.time() - step_start
            logging.info(f"Autoregressive inference step: {i} of {max_steps} take around {step_time}s")

        if not incremental_audio_decoding:
            gen_audio_codes_lens = torch.tensor([gen_audio_codes.shape[1]] * gen_audio_codes.shape[0]).to(self.device)
            # decode audio. Note that it is not necessary because the prompt is removed, so no special token should be on the output, but lets do it for safety
            gen_audio_codes = replace_control_speech_codes(
                gen_audio_codes, self._control_codes, self.codec_silence_tokens
            )
            with ensures_target_precision(self.audio_codec_run_dtype), torch.no_grad():
                audio_pred, audio_pred_len = self.audio_codec.decode(gen_audio_codes, gen_audio_codes_lens)

        return audio_pred.squeeze(1), audio_pred_len

    def maybe_recreate_cached_audio_prompt_latents_structure(self, state_dict):
        """
        Recreate audio_prompt_latents ParameterDict structure
        from keys present in the provided state_dict.
        """
        for k, tensor in state_dict.items():
            if "audio_prompt_latents." in k:
                name = k.split("audio_prompt_latents.")[1]
                if name not in self.audio_prompt_latents:
                    self.audio_prompt_latents[name] = nn.Parameter(
                        torch.zeros_like(tensor),
                        requires_grad=False,
                    )

    def load_state_dict(self, state_dict, strict: bool = True):
        self.maybe_recreate_cached_audio_prompt_latents_structure(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict)
        except RuntimeError:
            logging.info("Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            return super().load_state_dict(model_dict, strict=False)


@contextmanager
def ensures_target_precision(target_dtype):
    """
    Workaround for precision related issues when training with bf16-true PyTorch Lightning precision setting.
    In bf16-true, PTL changes PyTorch's default dtype, which may break implicit assumptions for some models.
    This context manager restores default float32 precision and runs the computation in float32 autocast context.
    """
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=target_dtype):
            yield
    finally:
        torch.set_default_dtype(default_dtype)


def replace_control_speech_codes(
    speech_codes: torch.Tensor, control_codes: torch.Tensor, silence_tokens: torch.Tensor = None
) -> torch.Tensor:
    """
    Replaces control codes (speech BOS, EOS, etc) in `speech_codes` with the first frame which is
    assumed to consist of 'valid' codes representing silence.
    """
    if silence_tokens is not None:
        # Expand to [B, 1, 74]
        silence_tokens_expanded = silence_tokens.unsqueeze(0).unsqueeze(1).expand(speech_codes.shape[0], 1, -1)
        return torch.where(torch.isin(speech_codes, control_codes), silence_tokens_expanded, speech_codes)

    if torch.isin(speech_codes[:, :1], control_codes).any():
        return torch.where(torch.isin(speech_codes, control_codes), torch.zeros_like(speech_codes[:, :1]), speech_codes)
    else:
        return torch.where(torch.isin(speech_codes, control_codes), speech_codes[:, :1], speech_codes)


def setup_audio_codec(model):
    """
    Instantiates the RVQ audio codec and injects codec embeddings into the TTS model.

    This function is responsible only for:
      - Instantiating the codec model (`RVQVAEModel`).
      - Loading pretrained codec weights (if configured).
      - Freezing codec parameters.
      - Registering RVQ embeddings inside the TTS model via `set_rvq_embs`.

    Args:
        model: Model instance of DuplexEARTTS
    """
    with ensures_target_precision(model.audio_codec_run_dtype):
        model.audio_codec = RVQVAEModel(DictConfig(model.cfg.codec_config))
        # load pretrained codec checkpoint
        if model.cfg.get("pretrained_codec_model", None):
            checkpoint_state = load_checkpoint(model.cfg.pretrained_codec_model)
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, model.audio_codec.state_dict())
            model.audio_codec.load_state_dict(checkpoint_state, strict=True)

    for p in model.audio_codec.parameters():
        p.requires_grad = False

    model.audio_codec.eval()

    assert callable(model.tts_model.set_rvq_embs)

    model.tts_model.set_rvq_embs(torch.stack([x.detach() for x in model.audio_codec.prvq.mus_list], 0))
    model.tts_model.rvq_embs = model.tts_model.rvq_embs.to(next(model.tts_model.parameters()).dtype)
    # compute target fps
    model.target_fps = model.target_sample_rate / model.audio_codec.config.wav_to_token_ratio
    model.target_samples_per_frame = model.audio_codec.config.wav_to_token_ratio
