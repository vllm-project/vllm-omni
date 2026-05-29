import os

import torch

from .flow_matching import DiTPrefix
from .modeling_bigvgan import KimiBigVGAN


class PrefixStreamingFlowMatchingDetokenizer:
    def __init__(
        self,
        vocoder: KimiBigVGAN,
        fm: DiTPrefix,
        look_ahead_tokens: int = 0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.dtype = dtype

        self.vocoder = vocoder.to(self.dtype)

        # Attribute name kept for continuity with upstream Moonshot.
        self.semantic_fm = fm

        self.max_pos_size = 4096
        self.pre_mel = None
        self.frame_size = 480
        self.pre_wav = None
        self.hamming_window_cache = {}
        self.previous_chunk_left = None
        self.look_ahead_tokens = look_ahead_tokens

        self.clear_states()

    @classmethod
    def from_pretrained(
        cls,
        fm_config,
        fm_ckpt,
        device,
        look_ahead_tokens=0,
        max_prompt_chunk=2,
        max_kv_cache_tokens=900,
        dtype: torch.dtype = torch.bfloat16,
    ):
        bigvgan = KimiBigVGAN.load_from_hf(device)
        semantic_fm = DiTPrefix.from_pretrained(
            fm_config,
            fm_ckpt,
            device,
            max_prompt_chunk=max_prompt_chunk,
            max_kv_cache_tokens=max_kv_cache_tokens,
            dtype=dtype,
        )
        return cls(bigvgan, semantic_fm, look_ahead_tokens=look_ahead_tokens, dtype=dtype)

    @torch.inference_mode()
    def detokenize_streaming(
        self,
        semantic_token,
        ode_step=30,
        verbose=False,
        is_final=False,
        upsample_factor=1,
    ):
        if semantic_token.dim() != 2 or semantic_token.shape[0] != 1 or ode_step <= 0:
            raise ValueError(
                f"detokenize_streaming expects semantic_token of shape "
                f"[1, T] and ode_step > 0; got shape={tuple(semantic_token.shape)}, "
                f"ode_step={ode_step}"
            )

        semantic_token = semantic_token.repeat_interleave(upsample_factor, dim=1)

        semantic_token = semantic_token.squeeze(0)

        if self.look_ahead_tokens != 0 and self.previous_chunk_left is not None:
            semantic_token_previous = self.previous_chunk_left["semantic_token"]
            if semantic_token_previous is not None:
                semantic_token = torch.cat(
                    [semantic_token_previous.to(semantic_token.device), semantic_token],
                    dim=-1,
                )

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
        )

        chunk_size = speech_mel.shape[0]
        length = speech_mel.shape[0]
        self.semantic_fm.start_position_id += length
        self.semantic_fm.update_incremental_state()
        self.semantic_fm.reserve_kv_cache_tokens += self.semantic_fm.kv_cache_tokens

        # Cross-faded streaming: first chunk returns a half-chunk of audio;
        # later chunks concat with the saved tail and Hamming-window the seam.
        if self.pre_mel is None:
            concat_mel = speech_mel
            concat_reconstructed_wav = self.vocoder.decode_mel(concat_mel)
            if is_final:
                self.clear_states()
                ret_wav = concat_reconstructed_wav.float()
            else:
                reconstructed_wav = concat_reconstructed_wav[:, : int(self.frame_size * chunk_size // 2)]
                self.pre_wav = concat_reconstructed_wav[:, -int(self.frame_size * chunk_size // 2) :]
                self.pre_mel = speech_mel[-chunk_size // 2 :, :]

                ret_wav = reconstructed_wav.float()
        else:
            concat_mel = torch.cat([self.pre_mel, speech_mel], dim=0)
            concat_reconstructed_wav = self.vocoder.decode_mel(concat_mel)

            if is_final:
                self.clear_states()
                ret_wav = concat_reconstructed_wav.float()
            else:
                prev_speech_len = self.pre_wav.shape[1]

                if concat_reconstructed_wav.shape[1] > prev_speech_len * 2:
                    gen_speech_len = prev_speech_len * 2
                else:
                    gen_speech_len = concat_reconstructed_wav.shape[1] // 2

                reconstructed_wav = concat_reconstructed_wav[:, :gen_speech_len]

                if gen_speech_len not in self.hamming_window_cache:
                    self.hamming_window_cache[gen_speech_len] = (
                        torch.hamming_window(gen_speech_len).to(self.dtype).to(semantic_token.device).unsqueeze(0)
                    )

                hamming_window = self.hamming_window_cache[gen_speech_len]

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
            # Out of position id; reset back to a clean state.
            self.semantic_fm.clear_all_states()

        return ret_wav

    def clear_states(self):
        self.semantic_fm.clear_all_states()
        self.previous_chunk_left = None
        self.pre_mel = None
        self.pre_wav = None


def get_audio_detokenizer(model_path, dtype: torch.dtype = torch.bfloat16):
    if not os.path.exists(model_path):
        # HF Hub repo id — fetch the audio_detokenizer subdir into the cache.
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(
            repo_id=model_path,
            allow_patterns=["audio_detokenizer/*"],
        )
    fm_model_config = os.path.join(model_path, "audio_detokenizer", "config.yaml")
    fm_ckpt_path = os.path.join(model_path, "audio_detokenizer", "model.pt")

    device = torch.accelerator.current_device_index()
    detokenizer = PrefixStreamingFlowMatchingDetokenizer.from_pretrained(
        max_prompt_chunk=10,  # 10 * 3 = 30s
        fm_config=fm_model_config,
        fm_ckpt=fm_ckpt_path,
        device=device,
        look_ahead_tokens=12,
        dtype=dtype,
    )

    return detokenizer


def detokenize_noref(detokenizer, tokens):
    """Non-streaming detokenize used by the sync code2wav stage. Mirrors
    upstream ``KimiAudio.detokenize_audio`` (``chunk_size=30``,
    ``first_chunk_size=30``, ``upsample_factor=4``) — NOT the unrelated
    ``detokenize`` function in ``models/detokenizer/__init__.py`` which
    uses 150/100 for the *streaming* path with reference audio."""
    with torch.no_grad():
        detokenizer.clear_states()
        cache_speech_collection = []
        chunk_size = 30
        first_chunk_size = 30
        first_chunk_tokens = tokens[:, :first_chunk_size]
        gen_speech = detokenizer.detokenize_streaming(
            first_chunk_tokens,
            is_final=tokens.size(1) <= first_chunk_size,
            upsample_factor=4,
        )
        cache_speech_collection.append(gen_speech)
        res_tokens = tokens[:, first_chunk_size:]
        for i in range(0, res_tokens.size(1), chunk_size):
            chunk_tokens = res_tokens[:, i : i + chunk_size]
            gen_speech = detokenizer.detokenize_streaming(
                chunk_tokens,
                is_final=(i + chunk_size >= res_tokens.size(1)),
                upsample_factor=4,
            )
            cache_speech_collection.append(gen_speech)

        gen_speech_all = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech_all
