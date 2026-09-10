# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from MoonshotAI/Kimi-Audio (MIT), revision
# 349251e1d8f4f98d58fda59246381faecd7392e0, kimia_infer/models/detokenizer/.
# See vllm_omni/model_executor/models/kimi_audio/NOTICE for the upstream license.

import torch

from .bigvgan_wrapper import BigVGANWrapper
from .semantic_fm_prefix_streaming import StreamingSemanticFMWrapper


class PrefixStreamingFlowMatchingDetokenizer:
    def __init__(
        self,
        vocoder: BigVGANWrapper,
        fm: StreamingSemanticFMWrapper,
        look_ahead_tokens: int = 0,
    ) -> None:
        self.dtype = torch.bfloat16

        self.vocoder = vocoder
        self.vocoder.to_dtype(self.dtype)

        self.semantic_fm = fm

        # initialize mel_spec
        self.max_pos_size = 4096
        self.pre_mel = None
        self.frame_size = 480  # how many samples in a frame
        self.pre_wav = None
        self.hamming_window_cache = {}
        self.previous_chunk_left = None
        self.look_ahead_tokens = look_ahead_tokens

        self.clear_states()

    @classmethod
    def from_pretrained(
        cls,
        vocoder_config,
        vocoder_ckpt,
        fm_config,
        fm_ckpt,
        device,
        look_ahead_tokens=0,
        max_kv_cache_tokens=900,
        use_cfg=False,
    ):
        bigvgan = BigVGANWrapper.from_pretrained(vocoder_config, vocoder_ckpt, device)
        semantic_fm = StreamingSemanticFMWrapper.from_pretrained(
            fm_config,
            fm_ckpt,
            device,
            max_kv_cache_tokens=max_kv_cache_tokens,
            use_cfg=use_cfg,
        )
        return cls(bigvgan, semantic_fm, look_ahead_tokens=look_ahead_tokens)

    @torch.inference_mode()
    def detokenize_streaming(
        self,
        semantic_token,
        ode_step=30,
        verbose=False,
        ode_solver="neural_ode_euler",
        is_final=False,
        upsample_factor=1,
    ):
        assert len(semantic_token.shape) == 2 and ode_step > 0
        assert semantic_token.shape[0] == 1

        semantic_token = semantic_token.repeat_interleave(upsample_factor, dim=1)

        semantic_token = semantic_token.squeeze(0)

        if self.look_ahead_tokens != 0 and self.previous_chunk_left is not None:
            semantic_token_previous = self.previous_chunk_left["semantic_token"]
            semantic_token = torch.cat([semantic_token_previous, semantic_token], dim=-1)

        x_t_chunk = torch.randn(semantic_token.shape[0], 80).to(semantic_token.device).to(self.dtype)

        if self.look_ahead_tokens != 0 and self.previous_chunk_left is None:
            self.previous_chunk_left = {"semantic_token": None}

        speech_mel = self.semantic_fm.infer_chunk(
            xt_chunk=x_t_chunk,
            semantic_tokens_chunk=semantic_token,
            start_position_id=self.semantic_fm.start_position_id,
            ode_steps=ode_step,
            verbose=verbose,
            look_ahead_tokens=(self.look_ahead_tokens * upsample_factor if not is_final else 0),
            cache=self.previous_chunk_left,
            ode_solver=ode_solver,
        )

        chunk_size = speech_mel.shape[0]
        length = speech_mel.shape[0]
        self.semantic_fm.start_position_id += length
        self.semantic_fm.update_incremental_state()

        # Retain the trailing half of the first block; subsequent blocks use
        # that history for the official waveform overlap and smoothing.

        if self.pre_mel is None:  # first chunk, related to TTFB
            concat_mel = speech_mel
            concat_reconstructed_wav = self.vocoder.decode_mel(concat_mel)
            if is_final:
                self.clear_states()
                ret_wav = concat_reconstructed_wav.float()
            else:
                reconstructed_wav = concat_reconstructed_wav[
                    :, : int(self.frame_size * chunk_size // 2)
                ]  # return the first half chunk

                self.pre_wav = concat_reconstructed_wav[
                    :, -int(self.frame_size * chunk_size // 2) :
                ]  # log the last half chunk for next generation step
                self.pre_mel = speech_mel[-chunk_size // 2 :, :]

                ret_wav = reconstructed_wav.float()
        else:
            concat_mel = torch.cat([self.pre_mel, speech_mel], dim=0)
            concat_reconstructed_wav = self.vocoder.decode_mel(concat_mel)

            if is_final:
                self.clear_states()
                ret_wav = concat_reconstructed_wav.float()
            else:
                # fetch history
                prev_speech_len = self.pre_wav.shape[1]

                if concat_reconstructed_wav.shape[1] > prev_speech_len * 2:
                    gen_speech_len = prev_speech_len * 2
                else:
                    gen_speech_len = concat_reconstructed_wav.shape[1] // 2

                reconstructed_wav = concat_reconstructed_wav[:, :gen_speech_len]  # return the first half chunk

                if gen_speech_len not in self.hamming_window_cache:
                    self.hamming_window_cache[gen_speech_len] = (
                        torch.hamming_window(gen_speech_len).to(self.dtype).to(semantic_token.device).unsqueeze(0)
                    )

                hamming_window = self.hamming_window_cache[gen_speech_len]

                # apply smoothing of the first half chunk
                reconstructed_wav[:, : int(gen_speech_len // 2)] = (
                    self.pre_wav[:, : int(gen_speech_len // 2)] * hamming_window[:, -int(gen_speech_len // 2) :]
                    + reconstructed_wav[:, : int(gen_speech_len // 2)] * hamming_window[:, : int(gen_speech_len // 2)]
                )

                res_speech_len = concat_reconstructed_wav.shape[1] - gen_speech_len
                res_mel_len = res_speech_len // self.frame_size

                self.pre_wav = concat_reconstructed_wav[:, -res_speech_len:]
                self.pre_mel = speech_mel[-res_mel_len:, :]
                ret_wav = reconstructed_wav.float()

        if not is_final and self.semantic_fm.start_position_id + 2 * chunk_size > self.max_pos_size:
            # Restart acoustic positions without a reference-voice prefix.
            self.semantic_fm.clear_all_states()

        return ret_wav

    def clear_states(self):
        self.semantic_fm.clear_all_states()
        self.previous_chunk_left = None
        self.pre_mel = None
        self.pre_wav = None
