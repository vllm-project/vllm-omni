# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Qwen3-TTS for the 310P NPU path."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from vllm.multimodal.audio import AudioResampler
from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
from vllm_ascend.sample.sampler import apply_top_k_top_p, random_sample
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, aligned_16, maybe_trans_nz, nd_to_nz_2d

from vllm_omni.model_executor.models.common import qwen3_code_predictor
from vllm_omni.model_executor.models.qwen3_tts import (
    prompt_embeds_builder,
    qwen3_tts_code2wav,
    qwen3_tts_code_predictor_vllm,
    qwen3_tts_talker,
)

_RUNTIME_DTYPE = torch.float16
_CPU_DEVICE = torch.device("cpu")
_PATCHED = False
_CODE2WAV_PATCHED = False


class _Qwen3TTSTalker310P(qwen3_tts_talker.Qwen3TTSTalkerForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._embedding_dtype = _RUNTIME_DTYPE
        self._prompt_builder._embedding_dtype = _RUNTIME_DTYPE
        # 310P random sampling operators cannot be captured by ACL graphs.
        # Keep Talker-MTP eager so CodePredictor can replay its own NPU graphs
        # without changing sampling semantics.
        self.talker_mtp_graph_safe = False
        self.talker_mtp_accepts_per_row_generators = True

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # The Mimi tokenizer encoder is used only while building ref_audio
        # prompts.  Keep this preprocessing-only module on CPU because the 310P
        # NPU path is sensitive to changing reference-audio shapes.
        self.encoder.to(device=_CPU_DEVICE, dtype=torch.float32)
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

        inputs = fe(raw_audio=wavs, sampling_rate=target_sr, return_tensors="pt").to(torch.float32)

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


# ===================================================================
#  Code2Wav layer patches
# ===================================================================
#
# Code2Wav runs under the 310P graph path after the stage-0 Talker has
# produced codec tokens.  Common NPU weight packing and fused tokenizer ops
# live in the shared model code; the 310P patch only selects the runtime dtype.


class _Qwen3TTSCode2Wav310P(qwen3_tts_code2wav.Qwen3TTSCode2Wav):
    """Qwen3-TTS Code2Wav specialized for the 310P NPU path."""

    def _npu_decoder_runtime_dtype(self, device: torch.device) -> torch.dtype:
        return _RUNTIME_DTYPE


# ===================================================================
#  CodePredictor layer patches
# ===================================================================
#
# Keep the portable implementation in common/qwen3_code_predictor.py.
# The overrides below are installed only by the 310P platform patch because
# the short CodePredictor loop is graph-captured on 310P and needs the 310P
# flash-attention mask layout plus loop-local projection and sampling changes.


class _Qwen3CodePredictorAttention310P(qwen3_code_predictor.CodePredictorAttention):
    """Attention override using 310P RoPE and flash-attention kernels.

    The shared attention path is written in portable PyTorch.  This override
    keeps the 310P-specific RoPE op, token alignment, and FRACTAL_NZ mask path
    scoped to the platform patch.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._buffers.pop("_fusion_causal_mask", None)
        self._q_size = self.num_heads * self.head_dim
        self._kv_size = self.num_kv_heads * self.head_dim
        self._fused_qkv_weight = None
        self._fused_qkv_bias = None

    def prepare_qkv_weights(self) -> None:
        # Pack QKV once so each graph replay uses one matmul and consumes the
        # weight directly in the 310P matmul layout.
        self._fused_qkv_weight = maybe_trans_nz(
            torch.cat((self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0).contiguous()
        )
        if self.q_proj.bias is not None:
            self._fused_qkv_bias = torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias), dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = F.linear(hidden_states, self._fused_qkv_weight, self._fused_qkv_bias)
        q, k, v = qkv.split((self._q_size, self._kv_size, self._kv_size), dim=-1)
        q = self.q_norm(q.view(bsz, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(bsz, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        # Use the fused Ascend RoPE op instead of expanding RoPE into
        # elementwise mul/add/rotate-half kernels.
        q = torch_npu.npu_rotary_mul(q, cos, sin)
        k = torch_npu.npu_rotary_mul(k, cos, sin)

        real_tokens = int(bsz) * int(seq_len)
        output_dtype = q.dtype

        # 310P flash attention consumes token-major fp16 inputs with 16-token
        # alignment; seq_lens carries the padding information.
        q_f = aligned_16(q.transpose(1, 2).reshape(real_tokens, self.num_heads, self.head_dim))
        k_f = aligned_16(k.transpose(1, 2).reshape(real_tokens, self.num_kv_heads, self.head_dim))
        v_f = aligned_16(v.transpose(1, 2).reshape(real_tokens, self.num_kv_heads, self.head_dim))

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
    """Decoder layer override that passes the 310P attention mask."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, attention_mask=attention_mask)
        hidden_states, _, residual = torch_npu.npu_add_rms_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )
        return residual + self.mlp(hidden_states)


class _Qwen3CodePredictorBaseModel310P(qwen3_code_predictor.CodePredictorBaseModel):
    """Base model override with a cached 310P causal mask.

    The 310P flash-attention path consumes the additive causal mask in
    FRACTAL_NZ format.  The CodePredictor sequence length is fixed and short,
    so the mask is built once and reused across graph replays.
    """

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
        if self._attention_mask_310p is None or self._attention_mask_310p_device != inputs_embeds.device:
            # Store the additive causal mask in the format consumed by the
            # 310P flash-attention kernel.
            mask = AttentionMaskBuilder310.gen_causal_additive_mask(
                self._attention_mask_310p_max_seq,
                inputs_embeds.device,
            )
            self._attention_mask_310p = torch_npu.npu_format_cast(nd_to_nz_2d(mask), ACL_FORMAT_FRACTAL_NZ)
            self._attention_mask_310p_device = inputs_embeds.device

        input_dtype = inputs_embeds.dtype
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                attention_mask=self._attention_mask_310p,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states.to(input_dtype)


class _Qwen3TTSTalkerCodePredictor310P(
    qwen3_tts_code_predictor_vllm.Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM
):
    """Qwen3-TTS code predictor specialized for the 310P NPU path."""

    def __init__(
        self,
        *,
        vllm_config,
        config,
        talker_config,
        prefix: str = "code_predictor",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            talker_config=talker_config,
            prefix=prefix,
        )
        self._projected_codec_embed_weight = None

    def _prepare_npu_weights(self) -> None:
        qkv_projections = set()
        with torch.no_grad():
            for layer in self.model.layers:
                attention = layer.self_attn
                attention.prepare_qkv_weights()
                qkv_projections.update((attention.q_proj, attention.k_proj, attention.v_proj))

            for module in self.modules():
                if isinstance(module, nn.Linear) and module not in qkv_projections:
                    module.weight.data = maybe_trans_nz(module.weight.data)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        with torch.no_grad():
            self._projected_codec_embed_weight = torch.stack(
                [self.small_to_mtp_projection(embed.weight).detach() for embed in self.model.codec_embedding],
                dim=0,
            ).contiguous()
        return loaded

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
        generators: Sequence[torch.Generator | None] | None = None,
    ) -> torch.Tensor:
        bsz = int(layer0_code.shape[0])
        if generators is not None and len(generators) != bsz:
            raise ValueError(f"generators must have one entry per row: got {len(generators)} for batch {bsz}")
        num_groups = self._num_groups
        device = layer0_code.device

        self._setup_compile()
        dtype = self._model_dtype

        padded_bsz = self._padded_bsz(bsz)
        self._ensure_buffers(device, dtype, padded_bsz)

        proj_buf = self._proj_buf
        max_seq = num_groups + 1
        projection = self.small_to_mtp_projection
        model_fwd = self._compiled_model_fwd
        lm_heads = self._lm_heads_list
        if generators is not None:
            npu_generators = {i: row_generator for i, row_generator in enumerate(generators) if row_generator}
        elif generator is not None:
            npu_generators = {i: generator for i in range(bsz)}
        else:
            npu_generators = {}

        proj_buf[:padded_bsz].zero_()
        initial_embeds = torch.cat(
            (
                last_talker_hidden.reshape(bsz, 1, -1),
                layer0_embed.reshape(bsz, 1, -1),
            ),
            dim=1,
        )
        if initial_embeds.dtype != dtype:
            initial_embeds = initial_embeds.to(dtype)
        proj_buf[:bsz, :2, :].copy_(projection(initial_embeds))

        stored_mode = self._wrapper_config.sampling_mode == "stored"
        if stored_mode:
            s_top_k = self._top_k
            s_top_p = self._top_p
        else:
            use_sampling = do_sample and temperature > 0
            inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0
            if use_sampling and top_p != 1.0:
                raise NotImplementedError(
                    "top_p sampling is not implemented for the vLLM-native code predictor; please set top_p=1.0."
                )

        top_k_tensor = None
        top_p_tensor = None
        if stored_mode:
            top_k_hint = s_top_k if s_top_k > 0 else None
            if s_top_k > 0:
                top_k_tensor = torch.full((bsz,), s_top_k, dtype=torch.int32, device=device)
            if s_top_p < 1.0:
                top_p_tensor = torch.full((bsz,), s_top_p, dtype=dtype, device=device)
        elif use_sampling:
            top_k_hint = top_k if top_k > 0 else None
            if top_k > 0:
                top_k_tensor = torch.full((bsz,), top_k, dtype=torch.int32, device=device)
        else:
            top_k_hint = None

        all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
        all_codes[:, 0] = layer0_code.reshape(bsz)

        for step in range(1, num_groups):
            graph_key: int | tuple[int, int] = padded_bsz
            seq_len = max_seq
            if self._prefix_graphs_enabled:
                prefix_key = (padded_bsz, step + 1)
                if prefix_key in self._device_graphs:
                    graph_key = prefix_key
                    seq_len = step + 1
            pos_ids = self._bucket_pos_ids.get(graph_key)
            if pos_ids is None:
                pos_ids = (
                    torch.arange(seq_len, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(padded_bsz, -1)
                    .contiguous()
                )

            device_graph_entry = self._device_graphs.get(graph_key)
            if device_graph_entry is not None:
                device_graph_entry[0].replay()
                hidden_out = device_graph_entry[1]
            else:
                hidden_out = model_fwd(proj_buf[:padded_bsz, :seq_len, :], pos_ids)

            logits = lm_heads[step - 1](hidden_out[:bsz, step, :])

            if stored_mode:
                if top_k_tensor is not None or top_p_tensor is not None:
                    logits = apply_top_k_top_p(logits, p=top_p_tensor, k=top_k_tensor, top_k=top_k_hint)
                candidate_indices = None
                if isinstance(logits, tuple):
                    logits, candidate_indices = logits
                probs = F.softmax(logits, dim=-1, dtype=torch.float32)
                code = random_sample(probs, npu_generators)
                if candidate_indices is not None:
                    code = candidate_indices.gather(1, code.unsqueeze(1)).squeeze(1)
            else:
                if use_sampling:
                    scaled = logits * inv_temperature
                    if top_k_tensor is not None:
                        scaled = apply_top_k_top_p(scaled, p=None, k=top_k_tensor, top_k=top_k_hint)
                    candidate_indices = None
                    if isinstance(scaled, tuple):
                        scaled, candidate_indices = scaled
                    probs = F.softmax(scaled, dim=-1, dtype=torch.float32)
                    code = random_sample(probs, npu_generators)
                    if candidate_indices is not None:
                        code = candidate_indices.gather(1, code.unsqueeze(1)).squeeze(1)
                else:
                    code = logits.argmax(dim=-1, keepdim=True)

            all_codes[:, step] = code.reshape(bsz)
            if step < num_groups - 1:
                proj_buf[:bsz, step + 1, :].copy_(
                    F.embedding(code.reshape(-1), self._projected_codec_embed_weight[step - 1])
                )

        return all_codes


# ===================================================================
#  Patch registration
# ===================================================================


def apply_talker_patches() -> None:
    """Install Qwen3-TTS Talker and CodePredictor 310P patches.

    The generic model modules stay unchanged.  Patch registration swaps in the
    310P-specialized CodePredictor classes and wrapper methods only when the
    310P platform applies the Talker patch.
    """

    global _PATCHED

    if _PATCHED:
        return

    qwen3_tts_talker.Qwen3TTSTalkerForConditionalGeneration = _Qwen3TTSTalker310P
    qwen3_tts_talker.Qwen3TTSPromptEmbedsBuilder = _Qwen3TTSPromptEmbedsBuilder310P
    qwen3_tts_talker.Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM = _Qwen3TTSTalkerCodePredictor310P
    qwen3_tts_code_predictor_vllm.Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM = (
        _Qwen3TTSTalkerCodePredictor310P
    )
    qwen3_tts_code_predictor_vllm.CodePredictorBaseModel = _Qwen3CodePredictorBaseModel310P
    qwen3_tts_code_predictor_vllm.Qwen3TTSTalkerCodePredictorModelVLLM = _Qwen3CodePredictorBaseModel310P
    prompt_embeds_builder.Qwen3TTSPromptEmbedsBuilder = _Qwen3TTSPromptEmbedsBuilder310P
    qwen3_code_predictor.CodePredictorAttention = _Qwen3CodePredictorAttention310P
    qwen3_code_predictor.CodePredictorDecoderLayer = _Qwen3CodePredictorDecoderLayer310P
    qwen3_code_predictor.CodePredictorBaseModel = _Qwen3CodePredictorBaseModel310P

    _PATCHED = True


def apply_code2wav_patches() -> None:
    """Install the 310P Code2Wav runtime patch."""
    global _CODE2WAV_PATCHED

    if _CODE2WAV_PATCHED:
        return

    qwen3_tts_code2wav.Qwen3TTSCode2Wav = _Qwen3TTSCode2Wav310P

    _CODE2WAV_PATCHED = True
