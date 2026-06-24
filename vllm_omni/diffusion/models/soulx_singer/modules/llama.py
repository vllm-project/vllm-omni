import math

import torch
import torch.nn as nn
from cache_dit import ForwardPattern
from transformers import LlamaConfig, LlamaModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from vllm_omni.diffusion.cache.cache_dit_backend import CacheDiTAdapterConfig
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput, SequenceParallelOutput


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        dtype = x.dtype
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LlamaAdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-6, dim_cond=1024):
        super().__init__()
        self.to_weight = nn.Linear(dim_cond, hidden_size)
        nn.init.zeros_(self.to_weight.weight)
        nn.init.ones_(self.to_weight.bias)
        self.variance_epsilon = eps
        self._is_hf_initialized = True

    def forward(self, hidden_states, cond_embedding):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        weight = self.to_weight(cond_embedding)
        if len(weight.shape) == 2:
            weight = weight.unsqueeze(1)

        return (weight * hidden_states).to(input_dtype)


class LlamaNARDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        self.input_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dim_cond=config.hidden_size,
        )
        self.post_attention_layernorm = LlamaAdaptiveRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dim_cond=config.hidden_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_embedding: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, cond_embedding=cond_embedding)

        # RoPE is computed by the parent model and passed in as (cos, sin).
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            cache_position=cache_position,
            output_attentions=output_attentions,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states,
            cond_embedding=cond_embedding,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class DiffLlama(LlamaModel):
    # Regional torch.compile in DiffusionModelRunner targets these blocks.
    _repeated_blocks = ["LlamaNARDecoderLayer"]
    _layerwise_offload_blocks_attrs = ["layers"]

    # cache-dit: Llama-style backbone with static condition embeddings injected
    # before the decoder loop (like Stable Audio Open). Pattern_3 handles the
    # cross-attention-style conditioning on the ``layers`` ModuleList.
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={
            "layers": ForwardPattern.Pattern_3,
        },
    )

    # Sequence parallelism: shard mel sequence before the first decoder layer and
    # gather after the output projection. Tensor parallelism still requires
    # migrating attention to vLLM parallel linear layers.
    _sp_plan = {
        "layers.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        "mel_out_mlp": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    @staticmethod
    def _is_transformer_block(name: str, module: nn.Module) -> bool:
        del module
        return name.startswith("layers.") and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        mel_dim=100,
        hidden_size=1024,
        num_heads=16,
        num_layers=16,
        dropout=0.1,
        ffn_dropout=0.1,
        attention_dropout=0.0,
        config: LlamaConfig | None = None,
    ):
        if config is None:
            config = LlamaConfig(
                vocab_size=32000,
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 4,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                num_key_value_heads=num_heads,
                max_position_embeddings=4096,
                rms_norm_eps=1e-6,
                attention_dropout=attention_dropout,
                use_cache=False,
                attn_implementation="eager",
            )

        config.use_cache = False

        super().__init__(config)

        # Keep custom decoder layers on the same attention backend.
        layer_config = LlamaConfig(
            vocab_size=config.vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            max_position_embeddings=4096,
            rms_norm_eps=config.rms_norm_eps,
            attention_dropout=attention_dropout,
            use_cache=False,
            attn_implementation="eager",
            bos_token_id=None,
            eos_token_id=None,
        )

        self.layers = nn.ModuleList(
            [
                LlamaNARDecoderLayer(
                    layer_config,
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )

        self.norm = LlamaAdaptiveRMSNorm(hidden_size, dim_cond=hidden_size)

        self.diff_step_embedding = SinusoidalPosEmb(hidden_size)
        self.diff_step_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.mel_mlp = nn.Sequential(
            nn.Linear(mel_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        self.mel_out_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, mel_dim),
        )

        self.embed_tokens = None

        self.post_init()

    def _prepare_decoder_attention_mask(
        self,
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length=0,
    ):
        """
        Build a 4D additive padding mask.

        This model is non-autoregressive, so no causal mask is added here.
        """

        combined_attention_mask = None

        def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None):
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len

            expanded_mask = (
                mask[:, None, None, :]
                .expand(
                    bsz,
                    1,
                    tgt_len,
                    src_len,
                )
                .to(dtype)
            )

            inverted_mask = 1.0 - expanded_mask

            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool),
                torch.finfo(dtype).min,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask,
                inputs_embeds.dtype,
                tgt_len=input_shape[-1],
            ).to(inputs_embeds.device)

            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        x,
        diffusion_step,
        cond,
        x_mask,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = False,
    ) -> BaseModelOutputWithPast | torch.Tensor | dict:
        batch_size, seq_length, _ = x.shape

        cond_embedding = self.cond_mlp(cond)
        x = self.mel_mlp(x)

        diffusion_step = self.diff_step_embedding(diffusion_step).to(device=x.device, dtype=x.dtype)
        diffusion_step = self.diff_step_mlp(diffusion_step)

        x = x + cond_embedding

        inputs_embeds = x
        attention_mask = x_mask

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # KV cache is disabled because this diffusion decoder processes full sequences.

        if past_key_values is not None:
            raise NotImplementedError("KV cache is disabled for this non-autoregressive DiffLlama implementation.")

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                0,
                seq_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.view(batch_size, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length=0,
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_layer_hidden_states = []

        cache_position = torch.arange(
            0,
            seq_length,
            device=hidden_states.device,
            dtype=torch.long,
        )

        # LlamaAttention expects precomputed rotary embeddings.
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError

            layer_outputs = decoder_layer(
                hidden_states,
                cond_embedding=diffusion_step,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]
            all_layer_hidden_states.append(hidden_states.clone())

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states, cond_embedding=diffusion_step)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = self.mel_out_mlp(hidden_states)

        if return_dict:
            return {
                "output": hidden_states,
                "hidden_states": all_layer_hidden_states,
                "attentions": all_self_attns,
            }

        return hidden_states
