"""Kimi-Audio flow-matching DiT: 12.5 Hz semantic tokens -> 50 Hz mel.
Followed by BigVGAN to produce a 24 kHz waveform. See Kimi-Audio paper
Section 2.4 for the architecture."""

import logging
import time
from functools import lru_cache

import torch
import torch.nn as nn
import yaml
from diffusers.models.embeddings import (
    Timesteps,
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from vllm.vllm_flash_attn import flash_attn_varlen_func

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cached_zeros(numel, device="cpu", dtype=torch.float32):
    return torch.zeros(numel, device=device, dtype=dtype)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens,
        max_seqlen,
        cu_seqlens_k,
        max_seqlen_k,
        rotary_pos_emb=None,
        incremental_state=None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        dropout_p = self.attn_drop.p if self.training else 0.0

        qkv = self.qkv(x)
        qkv = qkv.view(B * N, self.num_heads * 3, self.head_dim)
        q, k, v = qkv.split([self.num_heads] * 3, dim=1)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        if rotary_pos_emb is not None:
            q = apply_rotary_emb(q, rotary_pos_emb, use_real=False).type_as(q)
            k = apply_rotary_emb(k, rotary_pos_emb, use_real=False).type_as(k)

        if incremental_state is not None:
            if "prev_k" in incremental_state:
                k = torch.cat([incremental_state["prev_k"], k], dim=1)
            if "prev_v" in incremental_state:
                v = torch.cat([incremental_state["prev_v"], v], dim=1)
            incremental_state["cur_k"] = k
            incremental_state["cur_v"] = v

        q = q.view(B * N, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        x = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
        )

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class _DiTMlp(nn.Module):
    """Attribute names ``fc1``/``fc2`` match timm's ``Mlp`` so existing
    detokenizer checkpoints load unchanged."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = _DiTMlp(in_features=hidden_size, hidden_features=mlp_hidden_dim)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(
        self,
        x,
        c,
        cu_seqlens,
        cu_maxlen,
        cu_seqlens_k,
        cu_maxlen_k,
        rotary_pos_emb=None,
        incremental_state=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)

        x_ = modulate(self.norm1(x), shift_msa, scale_msa)

        if incremental_state is not None:
            if "attn_kvcache" not in incremental_state:
                incremental_state["attn_kvcache"] = {}
            inc_attn = incremental_state["attn_kvcache"]
        else:
            inc_attn = None

        x_ = self.attn(
            x_,
            cu_seqlens=cu_seqlens,
            max_seqlen=cu_maxlen,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=cu_maxlen_k,
            rotary_pos_emb=rotary_pos_emb,
            incremental_state=inc_attn,
        )

        x = x + gate_msa * x_

        x_ = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_ = self.mlp(x_)
        x = x + gate_mlp * x_
        return x


class TimestepEmbedder(nn.Module):
    """``nn.Sequential`` so state-dict keys ``mlp.0.*`` / ``mlp.2.*`` match
    Moonshot checkpoints."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = self.time_proj(t)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class DiTPrefix(nn.Module):
    """Flow-matching DiT with streaming KV cache and reference-audio prefill."""

    def __init__(
        self,
        input_size,
        output_size,
        semantic_vocab_size,
        hidden_size=1024,
        depth=12,
        num_heads=4,
        mlp_ratio=4.0,
        use_rope=False,
        rope_params={
            "max_position_embeddings": 4096,
            "rope_base": 10000.0,
            "rope_interpolation_factor": 1.0,
        },
        max_seq_len=4096,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.semantic_token_embedding = nn.Embedding(semantic_vocab_size, hidden_size)

        self.input_linear = nn.Linear(input_size, hidden_size)

        # No additive sincos PE: Kimi's audio_detokenizer ships
        # ``position_embedding_type: skip`` and relies solely on RoPE in
        # attention. Adding sincos here roughly doubles output amplitude.
        self.use_rope = use_rope

        if self.use_rope:
            assert hidden_size % num_heads == 0
            rope_dim = hidden_size // num_heads
            self.rotary_pos_emb = get_1d_rotary_pos_embed(
                dim=rope_dim,
                pos=rope_params["max_position_embeddings"],
                theta=rope_params["rope_base"],
                use_real=False,
                linear_factor=float(rope_params["rope_interpolation_factor"]),
            )
            if max_seq_len < rope_params["max_position_embeddings"]:
                self.rotary_pos_emb = self.rotary_pos_emb[:max_seq_len].clone()

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, output_size)
        self.initialize_weights()

        self.sigma_min = 1e-4
        self.t_min = 0.0
        self.t_max = 1.0 - self.sigma_min

        self.incremental_state: dict = {}
        self.kv_cache_tokens = 0
        self.position_ids = None
        self.x_cond = None
        self.cu_seqlens = None
        self.cu_maxlen = None
        self.cu_seqlens_k = None
        self.cu_maxlen_k = None
        self.previous_seqlen = None

        self.max_kv_cache_tokens = 900
        self.max_prompt_chunk = 2
        self.reserve_kv_cache_tokens = 0
        self.start_position_id = 0
        self.condition_cache: dict = {"previous_seqlen": 0}

        self.normalize_mel = False
        self.mel_mean = None
        self.mel_std = None

        self.dtype = torch.bfloat16

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # DiT zero-init: adaLN gates + final linear start as identity so
        # untrained blocks don't perturb the input.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _predict_velocity(
        self,
        x,
        position_ids,
        t,
        condition,
        cu_seqlens,
        cu_maxlen,
        cu_seqlens_k,
        cu_maxlen_k,
        incremental_state=None,
    ):
        condition = self.semantic_token_embedding(condition)
        x = self.input_linear(x)

        if self.use_rope:
            bsz, seqlen = position_ids.shape
            if self.rotary_pos_emb.device != position_ids.device:
                self.rotary_pos_emb = self.rotary_pos_emb.to(position_ids.device)
            rotary_pos_emb = torch.zeros(
                (bsz, seqlen, self.rotary_pos_emb.shape[1]),
                dtype=self.rotary_pos_emb.dtype,
                device=self.rotary_pos_emb.device,
            )
            for b in range(bsz):
                rotary_pos_emb[b][:] = self.rotary_pos_emb[position_ids[b]]
        else:
            rotary_pos_emb = None

        t = self.t_embedder(t)
        c = t.unsqueeze(1) + condition

        for block_idx, block in enumerate(self.blocks):
            if incremental_state is not None:
                if block_idx not in incremental_state:
                    incremental_state[block_idx] = {}
                incr = incremental_state[block_idx]
            else:
                incr = None

            x = block(
                x=x,
                c=c,
                cu_seqlens=cu_seqlens,
                cu_maxlen=cu_maxlen,
                cu_seqlens_k=cu_seqlens_k,
                cu_maxlen_k=cu_maxlen_k,
                rotary_pos_emb=rotary_pos_emb,
                incremental_state=incr,
            )

        return self.final_layer(x, c)

    forward = _predict_velocity

    def set_conditions(self, x_cond, start_position_id, cache=None):
        if cache is None:
            cache = {}

        self.x_cond = x_cond

        position_ids_cur = list(range(start_position_id, x_cond.shape[1] + start_position_id))
        position_ids = torch.tensor([position_ids_cur])
        self.position_ids = position_ids.to(x_cond.device).long()

        seq_len = torch.Tensor([position_ids.shape[1]]).to(x_cond.device).long()
        cu_seqlens = torch.cumsum(seq_len, dim=0)
        self.cu_seqlens = torch.cat([torch.Tensor([0]).to(cu_seqlens.device), cu_seqlens], dim=0).int()
        self.cu_maxlen = seq_len.cpu().max()

        if self.cu_seqlens_k is None:
            self.cu_seqlens_k = self.cu_seqlens
            self.cu_maxlen_k = self.cu_maxlen
            previous_seqlen = seq_len
        else:
            previous_seqlen = cache["previous_seqlen"] + seq_len
            cu_seqlens_k = torch.cumsum(previous_seqlen, dim=0)
            self.cu_seqlens_k = torch.cat([torch.Tensor([0]).to(cu_seqlens_k.device), cu_seqlens_k], dim=0).int()
            self.cu_maxlen_k = previous_seqlen.cpu().max()
        self.previous_seqlen = previous_seqlen
        return {"previous_seqlen": previous_seqlen}

    def update_incremental_state(self, condition_cache=None):
        if condition_cache is None:
            condition_cache = self.condition_cache

        # Sliding window: drop oldest entries when cache exceeds
        # ``max_kv_cache_tokens``. Mirrors upstream Moonshot's rotation,
        # which always passes ``reserve_kv_cache_tokens=0``.
        for layer_cache in self.incremental_state.values():
            kv = layer_cache["attn_kvcache"]
            kv["prev_k"] = kv["cur_k"]
            kv["prev_v"] = kv["cur_v"]

            self.kv_cache_tokens = kv["prev_k"].shape[1]

            if self.kv_cache_tokens > self.max_kv_cache_tokens:
                kv["prev_k"] = kv["prev_k"][:, -self.max_kv_cache_tokens :]
                kv["prev_v"] = kv["prev_v"][:, -self.max_kv_cache_tokens :]

                bsz = kv["prev_k"].shape[0]
                self.previous_seqlen = (
                    torch.Tensor([kv["prev_k"].shape[1] for _ in range(bsz)]).to(kv["prev_k"].device).long()
                )
                condition_cache["previous_seqlen"] = self.previous_seqlen
                self.kv_cache_tokens = kv["prev_k"].shape[1]

            kv.pop("cur_k")
            kv.pop("cur_v")

    def clear_all_states(self):
        self.incremental_state = {}
        self.kv_cache_tokens = 0
        self.reserve_kv_cache_tokens = 0
        self.cu_seqlens = None
        self.cu_maxlen = None
        self.cu_seqlens_k = None
        self.cu_maxlen_k = None
        self.previous_seqlen = None
        self.start_position_id = 0
        self.condition_cache = {"previous_seqlen": 0}

    def _ode_step(self, t, x):
        # ``t * 1000`` matches upstream: the DiT was trained on integer
        # timestep IDs, not [0, 1].
        t_long = _cached_zeros(x.shape[0], device=x.device, dtype=torch.long) + (t * 1000).long()
        return self._predict_velocity(
            x=x,
            condition=self.x_cond,
            t=t_long,
            position_ids=self.position_ids,
            cu_seqlens=self.cu_seqlens,
            cu_maxlen=self.cu_maxlen,
            cu_seqlens_k=self.cu_seqlens_k,
            cu_maxlen_k=self.cu_maxlen_k,
            incremental_state=self.incremental_state,
        )

    def _solve_euler(self, time_steps, xt):
        h = (self.t_max - self.t_min) / len(time_steps)
        h = h * torch.ones(xt.shape[0], dtype=xt.dtype, device=xt.device)
        for t in time_steps:
            xt = xt + h * self._ode_step(t, xt)
        return xt

    @torch.inference_mode()
    def infer_chunk(
        self,
        xt_chunk,
        semantic_tokens_chunk,
        start_position_id,
        cache=None,
        look_ahead_tokens=0,
        ode_steps=15,
        verbose=False,
    ):
        device = next(self.parameters()).device

        semantic_tokens_chunk = semantic_tokens_chunk.unsqueeze(0).to(device)
        xt_chunk = xt_chunk.unsqueeze(0).to(device).to(self.dtype)

        t_span = torch.linspace(0, 1, ode_steps)

        cache_ret = self.set_conditions(
            x_cond=semantic_tokens_chunk,
            start_position_id=start_position_id,
            cache=self.condition_cache,
        )

        if verbose:
            t_start = time.time()
        x_t = self._solve_euler(t_span, xt_chunk)

        if look_ahead_tokens > 0:
            cache["semantic_token"] = semantic_tokens_chunk.view(-1)[-look_ahead_tokens:]
            x_t_ret = x_t[:, :-look_ahead_tokens, :]
            self.condition_cache = self.set_conditions(
                x_cond=semantic_tokens_chunk[:, :-look_ahead_tokens],
                start_position_id=start_position_id,
                cache=self.condition_cache,
            )
            # Prime the truncated KV cache with a fake near-final timestep.
            self._ode_step(torch.Tensor([0.999]).to(x_t_ret.device), x_t_ret)
        else:
            x_t_ret = x_t
            self.condition_cache = cache_ret

        if verbose:
            logger.info(f"[ODE Chunk] Time cost: {time.time() - t_start}")

        if self.normalize_mel:
            x_t_ret = x_t_ret * self.mel_std + self.mel_mean
        return x_t_ret.squeeze(0)

    @torch.inference_mode()
    def prefill(self, mel, semantic_token, chunk_size=150, verbose=False):
        """Fill the KV cache from a reference audio prompt."""
        if mel.dim() != 2 or semantic_token.dim() != 1:
            raise ValueError(
                f"prefill expects mel [T, n_mels] and semantic_token [T]; "
                f"got mel={tuple(mel.shape)}, semantic_token={tuple(semantic_token.shape)}"
            )
        if semantic_token.shape[0] != mel.shape[0]:
            raise ValueError(
                f"semantic_token and mel must share T dim; got "
                f"semantic_token={semantic_token.shape[0]}, mel={mel.shape[0]}"
            )
        seq_len = mel.shape[0]
        num_chunks = min(seq_len // chunk_size, self.max_prompt_chunk)
        start_pos = seq_len - num_chunks * chunk_size

        self.prefill_chunk(mel[:start_pos, :], semantic_token[:start_pos], start_position_id=self.start_position_id)
        self.start_position_id += start_pos
        self.update_incremental_state()
        self.reserve_kv_cache_tokens += self.kv_cache_tokens

        if verbose:
            logger.info(f"Prefilling prompt with {num_chunks} chunks")
            t_start = time.time()

        for chunk_id in range(num_chunks):
            start = start_pos + chunk_id * chunk_size
            end = start + chunk_size
            self.prefill_chunk(
                mel[start:end, :],
                semantic_token[start:end],
                start_position_id=self.start_position_id,
            )
            self.start_position_id += end - start
            self.update_incremental_state()
            self.reserve_kv_cache_tokens += self.kv_cache_tokens

        if verbose:
            logger.info(f"Prefilling done in {time.time() - t_start:.2f} seconds")

    def prefill_chunk(self, mel_chunk, semantic_tokens_chunk, start_position_id=0):
        device = next(self.parameters()).device

        semantic_tokens_chunk = semantic_tokens_chunk.unsqueeze(0).to(device)
        mel_chunk = mel_chunk.unsqueeze(0).to(device).to(self.dtype)

        if self.normalize_mel:
            mel_chunk = (mel_chunk - self.mel_mean) / self.mel_std

        self.condition_cache = self.set_conditions(
            x_cond=semantic_tokens_chunk,
            start_position_id=start_position_id,
            cache=self.condition_cache,
        )

        self._ode_step(torch.Tensor([0.999]).to(device), mel_chunk)

    @classmethod
    def from_pretrained(
        cls,
        model_config,
        ckpt_path,
        device,
        max_prompt_chunk=2,
        max_kv_cache_tokens=900,
        dtype: torch.dtype = torch.bfloat16,
    ):
        with open(model_config) as f:
            config = yaml.safe_load(f)
        dit_cfg = config["model"]["dit"]
        dit = cls(
            input_size=dit_cfg["input_size"],
            semantic_vocab_size=dit_cfg["semantic_vocab_size"] + 1,
            hidden_size=dit_cfg["hidden_size"],
            depth=dit_cfg["depth"],
            num_heads=dit_cfg["num_heads"],
            mlp_ratio=dit_cfg["mlp_ratio"],
            use_rope=dit_cfg.get("use_rope", False),
            rope_params=dit_cfg.get(
                "rope_params",
                {
                    "max_position_embeddings": 4096,
                    "rope_base": 10000,
                    "rope_interpolation_factor": 1,
                },
            ),
            max_seq_len=dit_cfg["max_seq_len"],
            output_size=dit_cfg["input_size"],
        )

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
        speech_model_params = {k.replace("speech_model.", ""): v for k, v in state_dict.items() if "speech_model" in k}
        speech_model_params.pop("position_embedding._float_tensor", None)
        dit.load_state_dict(speech_model_params, strict=True)
        logger.info(">>> Loaded checkpoint from %s", ckpt_path)

        dit.max_prompt_chunk = max_prompt_chunk
        dit.max_kv_cache_tokens = max_kv_cache_tokens
        dit.normalize_mel = config.get("normalize_mel", False)
        dit.mel_mean = config.get("mel_mean")
        dit.mel_std = config.get("mel_std")
        dit.dtype = dtype

        dit.to(device).to(dit.dtype).eval()
        return dit
