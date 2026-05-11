# SPDX-License-Identifier: Apache-2.0

"""vLLM-native code predictor: re-prefill with torch.compile and no KV cache."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
)
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import is_pp_missing_parameter

from vllm_omni.transformers_utils.configs.raon import ENV

try:
    from vllm_omni.platforms import current_omni_platform
except ImportError:
    current_omni_platform = None

logger = init_logger(__name__)


class _CodePredictorAttention(nn.Module):
    """Multi-head SDPA attention with fused QKV, RoPE, q/k norm, and GQA."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self._use_gqa = self.num_kv_heads != self.num_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            prefix=f"{prefix}.qkv_proj",
            disable_tp=True,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            prefix=f"{prefix}.o_proj",
            disable_tp=True,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=getattr(config, "rope_parameters", None),
            dual_chunk_attention_config=None,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        qkv, _ = self.qkv_proj(hidden_states.reshape(bsz * seq_len, -1))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)

        q, k = self.rotary_emb(position_ids, q, k)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scaling,
            is_causal=True,
            enable_gqa=self._use_gqa,
        )

        attn_out = attn_out.transpose(1, 2).reshape(bsz * seq_len, -1)
        output, _ = self.o_proj(attn_out)
        return output.view(bsz, seq_len, -1)


class _CodePredictorMLP(nn.Module):
    """SiLU-gated MLP matching Qwen3MLP structure."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=True,
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
            disable_tp=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up
        x, _ = self.down_proj(x)
        return x


class _CodePredictorDecoderLayer(nn.Module):
    """Transformer decoder layer (SDPA, no KV cache)."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = _CodePredictorAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = _CodePredictorMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CodePredictorModel(nn.Module):
    """Inner transformer for the code predictor."""

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        *,
        embedding_dim: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        emb_dim = embedding_dim or int(config.hidden_size)
        self.layers = nn.ModuleList(
            [_CodePredictorDecoderLayer(config, prefix=f"{prefix}.layers.{i}") for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Index 0 stores layer0 embedding (group G-1); 1..G-1 store groups 0..G-2.
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups)]
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                if mapped.endswith("scale"):
                    mapped = maybe_remap_kv_scale_name(mapped, params_dict)
                    if mapped is None:
                        continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                mapped = maybe_remap_kv_scale_name(name, params_dict)
                if mapped is None:
                    continue
                if name.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(mapped)
        return loaded_params


class RaonCodePredictor(nn.Module):
    """vLLM-native code predictor with re-prefill and torch.compile."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig | None = None,
        config: Qwen3OmniMoeTalkerCodePredictorConfig,
        prefix: str = "code_predictor",
    ) -> None:
        super().__init__()

        # Config.
        self._vllm_config = vllm_config
        self.config = config
        self._num_groups = int(config.num_code_groups)
        self._cp_hidden = int(config.hidden_size)
        self.vocab_size = int(config.vocab_size)

        # Model components.
        self.model = CodePredictorModel(
            config,
            prefix=f"{prefix}.model",
        )
        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )
        self.small_to_mtp_projection = nn.Identity()

        # Runtime state.
        self._proj_buf: torch.Tensor | None = None
        self._compiled_model_fwd: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        self._bucket_sizes: list[int] = []
        self._bucket_pos_ids: dict[int, torch.Tensor] = {}
        self._lm_heads_list: list[nn.Module] | None = None
        self._codec_embeds_list: list[nn.Module] | None = None

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_hidden: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> torch.Tensor:
        """Re-prefill all codebook groups; returns [B, num_code_groups]."""
        bsz = int(layer0_code.shape[0])
        num_groups = self._num_groups
        device = layer0_code.device
        dtype = layer0_embed.dtype

        use_sampling = do_sample and temperature > 0
        inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0

        all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
        all_codes[:, 0] = layer0_code.reshape(bsz)

        self._ensure_buffers(device, dtype)
        self._setup_compile()

        proj_buf = self._proj_buf
        max_seq = self._num_groups + 1

        projection = self.small_to_mtp_projection
        model_fwd = self._compiled_model_fwd
        lm_heads = self._lm_heads_list
        codec_embeds = self._codec_embeds_list
        if model_fwd is None or lm_heads is None or codec_embeds is None:
            raise RuntimeError("RaonCodePredictor compile/runtime buffers were not initialized.")

        padded_bsz = self._padded_bsz(bsz)
        proj_buf[:padded_bsz].zero_()

        proj_buf[:bsz, 0, :] = projection(last_hidden.reshape(bsz, 1, -1)).reshape(bsz, -1)
        proj_buf[:bsz, 1, :] = projection(layer0_embed.reshape(bsz, 1, -1)).reshape(bsz, -1)
        full_pos_ids = self._bucket_pos_ids.get(padded_bsz)
        if full_pos_ids is None:
            full_pos_ids = self._make_pos_ids(
                batch_size=padded_bsz,
                seq_len=max_seq,
                device=device,
            )

        for step in range(1, num_groups):
            projected = proj_buf[:padded_bsz, :max_seq, :]

            hidden_out = model_fwd(projected, full_pos_ids)
            logits = lm_heads[step - 1](hidden_out[:bsz, step, :])

            if use_sampling:
                scaled = logits.float() * inv_temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(min(top_k, scaled.shape[-1]), dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[..., -1:], float("-inf"))
                probs = F.softmax(scaled, dim=-1, dtype=torch.float32)
                probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = logits.argmax(dim=-1, keepdim=True)

            all_codes[:, step] = next_ids.reshape(bsz)

            if step < num_groups - 1:
                new_embed = codec_embeds[step](next_ids)
                proj_buf[:bsz, step + 1, :] = projection(new_embed.reshape(bsz, 1, -1)).reshape(bsz, -1)

        return all_codes

    @torch.inference_mode()
    def predict_codes(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_hidden: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> torch.Tensor:
        """Predict residual codebooks; returns [B, num_code_groups]."""
        return self.forward(
            layer0_code,
            layer0_embed,
            last_hidden,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )

    def _ensure_buffers(self, device: torch.device, dtype: torch.dtype) -> None:
        max_seq = self._num_groups + 1
        if self._proj_buf is not None and self._proj_buf.device == device and self._proj_buf.dtype == dtype:
            return
        max_bsz = 8
        if self._vllm_config is not None:
            max_bsz = max(max_bsz, self._vllm_config.scheduler_config.max_num_seqs)
        self._proj_buf = torch.zeros(max_bsz, max_seq, self._cp_hidden, dtype=dtype, device=device)

    def _setup_compile(self) -> None:
        if self._compiled_model_fwd is not None:
            return

        self._lm_heads_list = list(self.lm_head)
        self._codec_embeds_list = list(self.model.codec_embedding)

        supports_compile = True
        if current_omni_platform is not None and hasattr(current_omni_platform, "supports_torch_inductor"):
            supports_compile = current_omni_platform.supports_torch_inductor()

        if not supports_compile:
            logger.warning_once("code_predictor: torch.compile disabled (platform)")
            self._compiled_model_fwd = self.model.forward
            return

        compile_mode = ENV.cp_compile_mode
        try:
            self._compiled_model_fwd = torch.compile(
                self.model.forward,
                mode=compile_mode,
                dynamic=False,
            )
            logger.info("code_predictor: torch.compile enabled (mode=%s)", compile_mode)
            self._warmup_compile()
        except Exception as exc:
            logger.warning("code_predictor: torch.compile failed (%s), using eager", exc)
            self._compiled_model_fwd = self.model.forward

    def _padded_bsz(self, bsz: int) -> int:
        for bucket in self._bucket_sizes:
            if bsz <= bucket:
                return bucket
        return bsz

    def _warmup_compile(self) -> None:
        max_bsz = 8
        if self._vllm_config is not None:
            max_bsz = self._vllm_config.scheduler_config.max_num_seqs

        bucket_sizes = [1 << i for i in range(max_bsz.bit_length()) if (1 << i) <= max_bsz]
        if max_bsz not in bucket_sizes:
            bucket_sizes.append(max_bsz)
        self._bucket_sizes = sorted(bucket_sizes)

        max_seq = self._num_groups + 1
        device = next(self.model.parameters()).device
        proj_buf = self._proj_buf
        model_fwd = self._compiled_model_fwd
        if proj_buf is None or model_fwd is None:
            raise RuntimeError("RaonCodePredictor warmup called before compile buffers were initialized.")

        for bsz in self._bucket_sizes:
            pos_ids = self._make_pos_ids(batch_size=bsz, seq_len=max_seq, device=device)
            self._bucket_pos_ids[bsz] = pos_ids
            for _ in range(3):
                model_fwd(proj_buf[:bsz, :max_seq, :], pos_ids)
        logger.info("code_predictor: warmup done for buckets %s", self._bucket_sizes)

    @staticmethod
    def _split_checkpoint_weights(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        model_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if name.startswith("model."):
                model_weights.append((name[len("model.") :], tensor))
            else:
                other_weights.append((name, tensor))
        return model_weights, other_weights

    @staticmethod
    def _make_pos_ids(
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        base_pos = torch.arange(seq_len, device=device, dtype=torch.long)
        if batch_size == 1:
            return base_pos
        return base_pos.repeat(batch_size)

    def _load_legacy_fused_lm_head(
        self,
        name: str,
        tensor: torch.Tensor,
        params: dict[str, torch.nn.Parameter],
        loaded: set[str],
    ) -> bool:
        if name != "fused_lm_head" or tensor.ndim != 3:
            return False

        for i in range(self._num_groups - 1):
            target_name = f"lm_head.{i}.weight"
            target_param = params.get(target_name)
            if target_param is None:
                continue
            default_weight_loader(target_param, tensor[i])
            loaded.add(target_name)
        loaded.add("fused_lm_head")
        return True

    def _load_legacy_codec_embedding(
        self,
        name: str,
        tensor: torch.Tensor,
        params: dict[str, torch.nn.Parameter],
        loaded: set[str],
    ) -> bool:
        if name != "codec_embedding.weight" or tensor.shape[0] != self._num_groups * self.vocab_size:
            return False

        for i in range(self._num_groups):
            target_name = f"model.codec_embedding.{i}.weight"
            target_param = params.get(target_name)
            if target_param is None:
                continue
            start = i * self.vocab_size
            end = start + self.vocab_size
            default_weight_loader(target_param, tensor[start:end])
            loaded.add(target_name)
        loaded.add("codec_embedding.weight")
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ctx = set_current_vllm_config(self._vllm_config) if self._vllm_config else None
        try:
            if ctx is not None:
                ctx.__enter__()

            model_weights, other_weights = self._split_checkpoint_weights(weights)
            loaded = {f"model.{name}" for name in self.model.load_weights(model_weights)}
            params = dict(self.named_parameters(remove_duplicate=False))

            for name, tensor in other_weights:
                if self._load_legacy_fused_lm_head(name, tensor, params, loaded):
                    continue
                if self._load_legacy_codec_embedding(name, tensor, params, loaded):
                    continue

                param = params.get(name)
                if param is None:
                    continue
                default_weight_loader(param, tensor)
                loaded.add(name)
            return loaded
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)


class RepetitionAwareSampler:
    """Stateless RAS — history is passed in at call time, no per-request state."""

    def maybe_resample(
        self,
        recent_codes: list[int],
        first_code: int,
        audio_logits: torch.Tensor,
        codebook_size: int,
    ) -> int:
        if not ENV.ras_enabled:
            return first_code
        if first_code >= codebook_size:
            return first_code
        if not recent_codes:
            return first_code

        window = recent_codes[-ENV.ras_window_size :]
        if not window:
            return first_code

        ratio = window.count(first_code) / len(window)
        if ratio <= ENV.ras_repetition_threshold:
            return first_code

        resample_logits = audio_logits.float().clone()
        resample_logits[first_code] = float("-inf")
        if torch.isinf(resample_logits).all():
            return first_code

        old_code = first_code
        probs = torch.softmax(resample_logits, dim=-1, dtype=torch.float32)
        probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
        first_code = int(torch.multinomial(probs, num_samples=1).item())
        logger.debug(
            "[RAS] resampled code %d -> %d (ratio=%.3f window=%d)",
            old_code,
            first_code,
            ratio,
            len(window),
        )
        return first_code
