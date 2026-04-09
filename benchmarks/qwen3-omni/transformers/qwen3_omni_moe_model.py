import time

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration


class Qwen3OmniMoeForConditionalGenerationWithLogging(Qwen3OmniMoeForConditionalGeneration):
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor | None = None,
        speaker: str = "Ethan",
        use_audio_in_video: bool = False,
        return_audio: bool | None = None,
        thinker_max_new_tokens: int = 1024,
        thinker_eos_token_id: int = 151645,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 50,
        talker_top_p: float = 1.0,
        talker_temperature: float = 0.9,
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ):
        total_t0 = time.time()
        perf_stats = {
            "thinker_tokens": 0,
            "thinker_time_s": 0.0,
            "thinker_tps": 0.0,
            "talker_tokens": 0,
            "talker_time_s": 0.0,
            "talker_tps": 0.0,
            "code2wav_tokens": 0,
            "code2wav_time_s": 0.0,
            "code2wav_tps": 0.0,
            "total_tokens": 0,
            "total_time_s": 0.0,
            "total_tps": 0.0,
        }
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initialized. "
                "Use `enable_talker` method or set enable_talker in config "
                "to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker

        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
            "eos_token_id": thinker_eos_token_id,
        }

        talker_kwargs = {}
        token2wav_kwargs = {}
        if return_audio:
            speaker_id = self.config.talker_config.speaker_id.get(speaker.lower())
            if speaker_id is None:
                raise NotImplementedError(f"Speaker {speaker} not implemented")
            if input_ids.shape[0] != 1:
                raise NotImplementedError("Qwen3-Omni currently does not support batched inference with audio output")
            talker_suppressed_tokens = [
                i
                for i in range(
                    self.config.talker_config.text_config.vocab_size - 1024,
                    self.config.talker_config.text_config.vocab_size,
                )
                if i != self.config.talker_config.codec_eos_token_id
            ]  # Suppress additional special tokens, should not be predicted
            talker_kwargs = {
                "max_new_tokens": talker_max_new_tokens,
                "do_sample": talker_do_sample,
                "top_k": talker_top_k,
                "top_p": talker_top_p,
                "temperature": talker_temperature,
                "eos_token_id": self.config.talker_config.codec_eos_token_id,
                "repetition_penalty": talker_repetition_penalty,
                "suppress_tokens": talker_suppressed_tokens,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
            }
            token2wav_kwargs = {}

        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            # Process special input values
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key in ("input_features", "attention_mask"):
                thinker_kwargs[key] = value
            # Put other key to shared kwargs
            else:
                shared_kwargs[key] = value

        # Merge kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs and key in ["image_grid_thw", "video_grid_thw", "video_second_per_grid"]:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value

        # 1. Generate from thinker module
        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True

        t0 = time.time()
        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)
        t1 = time.time()
        perf_stats["thinker_time_s"] = max(0.0, t1 - t0)
        try:
            prompt_len = int(input_ids.shape[1]) if input_ids is not None else 0
            total_len = int(thinker_result.sequences.shape[-1])
            thinker_out_len = max(0, total_len - prompt_len)
        except Exception:
            thinker_out_len = 0
        perf_stats["thinker_tokens"] = thinker_out_len
        perf_stats["thinker_tps"] = (
            (thinker_out_len / perf_stats["thinker_time_s"]) if perf_stats["thinker_time_s"] > 0 else 0.0
        )

        if not generate_audio:
            perf_stats["total_tokens"] = perf_stats["thinker_tokens"]
            perf_stats["total_time_s"] = time.time() - total_t0
            perf_stats["total_tps"] = (
                (perf_stats["total_tokens"] / perf_stats["total_time_s"]) if perf_stats["total_time_s"] > 0 else 0.0
            )
            # attach stats to self
            setattr(self, "_perf_stats_last", perf_stats)
            if not hasattr(self, "_perf_stats_history"):
                setattr(self, "_perf_stats_history", [])
            self._perf_stats_history.append(perf_stats)
            return thinker_result, None

        # 2. Prepare talker input
        thinker_embed = torch.cat([hidden_states[0] for hidden_states in thinker_result.hidden_states], dim=1).to(
            self.talker.device
        )  # [1 t d]
        thinker_hidden = torch.cat(
            [
                hidden_states[self.config.talker_config.accept_hidden_layer]
                for hidden_states in thinker_result.hidden_states
            ],
            dim=1,
        ).to(self.talker.device)  # [1 t d]

        im_start_indexes = torch.cat(
            (
                torch.nonzero(input_ids[0] == self.config.im_start_token_id).squeeze(),
                torch.tensor([thinker_result.sequences.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
            ),
            dim=-1,
        ).to(self.talker.device)  # Shape [n_starts + 1]; Take batch 0 since batched inference is not supported here.
        multimodal_mask = (
            (thinker_result.sequences == self.config.thinker_config.audio_token_id) |
            (thinker_result.sequences == self.config.thinker_config.image_token_id) |
            (thinker_result.sequences == self.config.thinker_config.video_token_id)
        ).to(self.talker.device)  # [1 t] # fmt: skip

        talker_special_tokens = torch.tensor(
            [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
            device=self.thinker.device,
            dtype=input_ids.dtype,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = (
            self.talker.text_projection(self.thinker.get_input_embeddings()(talker_special_tokens))
            .to(self.talker.device)
            .chunk(3, dim=1)
        )  # 3 * [1 1 d]

        talker_input_embeds = []  # [1 t d]
        talker_input_ids = []
        # For every chatml parts
        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i]
            segment_end_index = im_start_indexes[i + 1]
            role_token = input_ids[0][im_start_index + 1]
            # Talker should ignore thinker system prompt
            if role_token == self.config.system_token_id:
                continue
            # Talker takes word embeddings for tokens and hidden state from `accept_hidden_layer` for multimodal inputs
            elif role_token == self.config.user_token_id:
                talker_user_part = self._get_talker_user_parts(
                    im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed
                )
                talker_input_embeds.append(talker_user_part)
                talker_input_ids.append(thinker_result.sequences[:, im_start_index:segment_end_index])
            # Take assistant output (for now)
            elif role_token == self.config.assistant_token_id and i == len(im_start_indexes) - 2:
                talker_assistant_embeds, talker_assistant_ids, trailing_text_hidden = self._get_talker_assistant_parts(
                    im_start_index,
                    segment_end_index,
                    speaker_id,
                    thinker_embed,
                    tts_pad_embed,
                    tts_bos_embed,
                    tts_eos_embed,
                )
                talker_input_embeds.append(talker_assistant_embeds)
                talker_input_ids.append(talker_assistant_ids)
            # History assistant output (ignore for now)
            elif role_token == self.config.assistant_token_id and i != len(im_start_indexes) - 2:
                continue
            else:
                raise AssertionError("Expect role id after <|im_start|> (assistant, user, system)")
        talker_input_embed = torch.cat([embed.to(self.talker.device) for embed in talker_input_embeds], dim=1)
        talker_input_id = torch.cat([embed.to(self.talker.device) for embed in talker_input_ids], dim=1)
        t2 = time.time()
        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            talker_input_ids=talker_input_id,  # Not use input_ids to prevent repetition penalty out of bound
            **talker_kwargs,
        )
        t3 = time.time()
        perf_stats["talker_time_s"] = max(0.0, t3 - t2)
        talker_codes = (
            torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)
            .transpose(1, 2)
            .to(self.code2wav.device)
        )
        try:
            # codes shape: (B, num_quantizers, T). We log T as token length.
            perf_stats["talker_tokens"] = int(talker_codes.shape[-1])
        except Exception:
            perf_stats["talker_tokens"] = 0
        perf_stats["talker_tps"] = (
            (perf_stats["talker_tokens"] / perf_stats["talker_time_s"]) if perf_stats["talker_time_s"] > 0 else 0.0
        )
        t4 = time.time()
        talker_wavs = self.code2wav.chunked_decode(talker_codes, chunk_size=300, left_context_size=25).float()
        t5 = time.time()
        perf_stats["code2wav_time_s"] = max(0.0, t5 - t4)
        perf_stats["code2wav_tokens"] = perf_stats["talker_tokens"]  # same T, not times 16
        perf_stats["code2wav_tps"] = (
            (perf_stats["code2wav_tokens"] / perf_stats["code2wav_time_s"])
            if perf_stats["code2wav_time_s"] > 0
            else 0.0
        )
        perf_stats["total_tokens"] = perf_stats["thinker_tokens"] + perf_stats["talker_tokens"]
        perf_stats["total_time_s"] = time.time() - total_t0
        perf_stats["total_tps"] = (
            (perf_stats["total_tokens"] / perf_stats["total_time_s"]) if perf_stats["total_time_s"] > 0 else 0.0
        )
        setattr(self, "_perf_stats_last", perf_stats)
        if not hasattr(self, "_perf_stats_history"):
            setattr(self, "_perf_stats_history", [])
        self._perf_stats_history.append(perf_stats)
        return thinker_result, talker_wavs.float()


__all__ = [
    "Qwen3OmniMoeForConditionalGenerationWithLogging",
]
