# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Qwen3-TTS for the 310P NPU path."""

from __future__ import annotations

import numpy as np
import torch
from transformers.models.mimi import modeling_mimi
from vllm.multimodal.audio import AudioResampler

from vllm_omni.model_executor.models.common import qwen3_code_predictor
from vllm_omni.model_executor.models.qwen3_tts import prompt_embeds_builder, qwen3_tts_talker

_RUNTIME_DTYPE = torch.float16
_CPU_DEVICE = torch.device("cpu")


class _MimiEuclideanCodebook310P(modeling_mimi.MimiEuclideanCodebook):
    def quantize(self, hidden_states):
        # 310P does not support torch.cdist on NPU.
        device = hidden_states.device
        dists = torch.cdist(
            hidden_states[None].to(_CPU_DEVICE, torch.float32), self.embed[None].to(_CPU_DEVICE, torch.float32), p=2
        )[0]
        return dists.argmin(dim=-1).to(device=device)


class _Qwen3TTSTalker310P(qwen3_tts_talker.Qwen3TTSTalkerForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._embedding_dtype = _RUNTIME_DTYPE
        self._prompt_builder._embedding_dtype = _RUNTIME_DTYPE

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        self.encoder.to(dtype=_RUNTIME_DTYPE)
        return loaded

    def _encode_ref_audio_batch(
        self,
        wavs: list[np.ndarray],
        sr: int,
        *,
        device: torch.device,
    ) -> list[torch.Tensor]:
        fe = self._encoder_feature_extractor
        target_sr = int(fe.sampling_rate)
        if int(sr) != target_sr:
            resampler = AudioResampler(target_sr=target_sr)
            wavs = [resampler.resample(w.astype(np.float32), orig_sr=int(sr)) for w in wavs]

        inputs = fe(raw_audio=wavs, sampling_rate=target_sr, return_tensors="pt").to(device).to(_RUNTIME_DTYPE)

        with torch.inference_mode():
            encoded = self.encoder.encode(
                input_values=inputs["input_values"].squeeze(1).unsqueeze(1),
                return_dict=True,
            )

        audio_codes = encoded.audio_codes[:, : self._encoder_valid_num_quantizers]
        padding_mask = inputs["padding_mask"].squeeze(1)
        downsample = self._encoder_downsample_rate
        return [
            code[..., : -(-mask.sum() // downsample)].transpose(0, 1).to(device=device, dtype=torch.long)
            for code, mask in zip(audio_codes, padding_mask)
        ]


class _Qwen3TTSPromptEmbedsBuilder310P(prompt_embeds_builder.Qwen3TTSPromptEmbedsBuilder):
    def extract_speaker_embedding(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        dev = self._device()
        dtype = self._embedding_dtype
        try:
            spk_param = next(self._speaker_encoder.parameters())
            if spk_param.device != dev or spk_param.dtype != dtype:
                self._speaker_encoder.to(device=dev, dtype=dtype)
        except StopIteration:
            pass

        target_sr = int(getattr(self._config.speaker_encoder_config, "sample_rate", 24000))
        if sr != target_sr:
            resampler = self._get_resampler(int(sr), target_sr)
            wav = resampler.resample(wav.astype(np.float32), orig_sr=int(sr))

        # 310P does not support torch.stft on NPU.
        wav_tensor = torch.from_numpy(wav).to(device=_CPU_DEVICE, dtype=torch.float32).unsqueeze(0)
        mels = prompt_embeds_builder.mel_spectrogram(
            wav_tensor,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        spk = self._speaker_encoder(mels.to(device=dev, dtype=dtype))[0]
        return spk.to(dtype=dtype)


class _Qwen3CodePredictorAttention310P(qwen3_code_predictor.CodePredictorAttention):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._buffers.pop("_fusion_causal_mask", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.device.type != "npu" or attention_mask is None:
            return super().forward(hidden_states, position_embeddings)

        bsz, seq_len, _ = hidden_states.shape
        hidden_shape_q = (bsz, seq_len, self.num_heads, self.head_dim)
        hidden_shape_kv = (bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape_q)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape_kv)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape_kv).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = (q * cos) + (qwen3_code_predictor._rotate_half(q) * sin)
        k = (k * cos) + (qwen3_code_predictor._rotate_half(k) * sin)

        real_tokens = int(bsz) * int(seq_len)
        output_dtype = q.dtype

        import torch_npu
        from vllm_ascend.utils import aligned_16

        q_f = aligned_16(q.to(torch.float16).transpose(1, 2).reshape(real_tokens, self.num_heads, self.head_dim))
        k_f = aligned_16(k.to(torch.float16).transpose(1, 2).reshape(real_tokens, self.num_kv_heads, self.head_dim))
        v_f = aligned_16(v.to(torch.float16).transpose(1, 2).reshape(real_tokens, self.num_kv_heads, self.head_dim))

        aligned_tokens = int(q_f.shape[0])
        seq_lens = torch.full((int(bsz),), int(seq_len), dtype=torch.int32, device="cpu")
        if aligned_tokens > real_tokens:
            seq_lens[-1] += aligned_tokens - real_tokens

        out = torch.empty((aligned_tokens, self.num_heads, self.head_dim), dtype=torch.float16, device=q.device)
        torch_npu._npu_flash_attention(
            query=q_f.contiguous(),
            key=k_f.contiguous(),
            value=v_f.contiguous(),
            mask=attention_mask,
            seq_len=seq_lens,
            scale_value=float(self.scaling),
            num_heads=int(self.num_heads),
            num_kv_heads=int(self.num_kv_heads),
            out=out,
        )
        attn_out = out[:real_tokens].reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return self.o_proj(attn_out.to(output_dtype).transpose(1, 2).reshape(bsz, seq_len, -1))


class _Qwen3CodePredictorDecoderLayer310P(qwen3_code_predictor.CodePredictorDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class _Qwen3CodePredictorBaseModel310P(qwen3_code_predictor.CodePredictorBaseModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._attention_mask_310p = None
        self._attention_mask_310p_device = None
        self._attention_mask_310p_max_seq = ((int(self.config.num_code_groups) + 16) // 16) * 16

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if inputs_embeds.device.type != "npu":
            return super().forward(inputs_embeds, position_ids)

        if self._attention_mask_310p is None or self._attention_mask_310p_device != inputs_embeds.device:
            import torch_npu
            from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, nd_to_nz_2d

            mask = AttentionMaskBuilder310.gen_causal_additive_mask(
                self._attention_mask_310p_max_seq,
                inputs_embeds.device,
            )
            self._attention_mask_310p = torch_npu.npu_format_cast(nd_to_nz_2d(mask), ACL_FORMAT_FRACTAL_NZ)
            self._attention_mask_310p_device = inputs_embeds.device

        input_dtype = inputs_embeds.dtype
        hidden_states = inputs_embeds
        with torch.amp.autocast(inputs_embeds.device.type, enabled=False, dtype=torch.float32):
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    position_embeddings,
                    attention_mask=self._attention_mask_310p,
                )
            hidden_states = self.norm(hidden_states)
        return hidden_states.to(input_dtype)


def apply_talker_patches() -> None:
    modeling_mimi.MimiEuclideanCodebook = _MimiEuclideanCodebook310P
    qwen3_tts_talker.Qwen3TTSTalkerForConditionalGeneration = _Qwen3TTSTalker310P
    qwen3_tts_talker.Qwen3TTSPromptEmbedsBuilder = _Qwen3TTSPromptEmbedsBuilder310P
    prompt_embeds_builder.Qwen3TTSPromptEmbedsBuilder = _Qwen3TTSPromptEmbedsBuilder310P
    qwen3_code_predictor.CodePredictorAttention = _Qwen3CodePredictorAttention310P
    qwen3_code_predictor.CodePredictorDecoderLayer = _Qwen3CodePredictorDecoderLayer310P
    qwen3_code_predictor.CodePredictorBaseModel = _Qwen3CodePredictorBaseModel310P
