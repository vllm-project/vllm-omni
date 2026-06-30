# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Miso TTS neural network (MisoLabs/MisoTTS).

This is a direct port of the official Miso TTS implementation from
https://github.com/MisoLabsAI/MisoTTS to ensure identical behavior.

Upstream architecture (``models.py`` in the Miso repo):
  * Llama 3.2 **8B** temporal backbone over frame embeddings (text + 32 RVQ slots)
  * Llama 3.2 **300M** decoder for codebooks 1..31 after C0 from ``codebook0_head``
  * ``projection`` 4096→1536, shared ``audio_embeddings`` / ``text_embeddings``
  * ``generate_frame()`` — one Mimi frame per call (~80 ms)
"""

from __future__ import annotations

import contextlib
import io
import os
import re
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from vllm.logger import init_logger

from vllm_omni.model_executor.models.miso_tts.miso_compat import patch_bitsandbytes_import_for_unquantized_layers

# Import torchtune models like the official implementation
with contextlib.redirect_stdout(io.StringIO()) as _torchtune_stdout:
    from torchtune.models import llama3_2

_torchtune_import_output = _torchtune_stdout.getvalue()
if _torchtune_import_output.strip() != "import error: No module named 'triton'":
    print(_torchtune_import_output, end="")

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

logger = init_logger(__name__)

DEFAULT_MISO_TTS_REPO_ID = "MisoLabs/MisoTTS"
MISO_MS_PER_FRAME = 80.0
MISO_NUM_CODEBOOKS = 32


# ---------------------------------------------------------------------------
# Torchtune model flavors (from official Miso TTS)
# ---------------------------------------------------------------------------


def llama3_2_8B():
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=2048,
        intermediate_dim=14_336,
        attn_dropout=0.1,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_300M():
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=8,
        num_heads=24,
        num_kv_heads=6,
        embed_dim=1536,
        max_seq_len=2048,
        intermediate_dim=6912,
        attn_dropout=0.1,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS = {
    "llama-8B": llama3_2_8B,
    "llama-300M": llama3_2_300M,
}


# ---------------------------------------------------------------------------
# Sampling (upstream ``sample_topk``)
# ---------------------------------------------------------------------------


def _multinomial_sample_one_no_sync(probs: torch.Tensor) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


# ---------------------------------------------------------------------------
# Checkpoint key remap (torchtune → module names)
# ---------------------------------------------------------------------------


def _map_torchtune_layer_key(key: str, *, stack_prefix: str) -> str | None:
    if not key.startswith(f"{stack_prefix}.layers."):
        return None
    rest = key[len(f"{stack_prefix}.layers.") :]
    m = re.match(r"(\d+)\.(.*)", rest)
    if not m:
        return None
    layer_idx, tail = m.group(1), m.group(2)
    prefix = f"{stack_prefix}.layers.{layer_idx}"
    if tail.startswith("attn."):
        sub = tail[len("attn.") :]
        mapping = {
            "q_proj.weight": "self_attn.q_proj.weight",
            "k_proj.weight": "self_attn.k_proj.weight",
            "v_proj.weight": "self_attn.v_proj.weight",
            "output_proj.weight": "self_attn.o_proj.weight",
            "q_norm.scale": "self_attn.q_norm.weight",
            "q_norm.weight": "self_attn.q_norm.weight",
            "k_norm.scale": "self_attn.k_norm.weight",
            "k_norm.weight": "self_attn.k_norm.weight",
        }
        if sub in mapping:
            return f"{prefix}.{mapping[sub]}"
    if tail.startswith("mlp."):
        sub = tail[len("mlp.") :]
        mapping = {
            "w1.weight": "mlp.gate_proj.weight",
            "w2.weight": "mlp.down_proj.weight",
            "w3.weight": "mlp.up_proj.weight",
        }
        if sub in mapping:
            return f"{prefix}.{mapping[sub]}"
    if tail in ("sa_norm.scale", "sa_norm.weight"):
        return f"{prefix}.input_layernorm.weight"
    if tail in ("mlp_norm.scale", "mlp_norm.weight"):
        return f"{prefix}.post_attention_layernorm.weight"
    return None


def remap_miso_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        k = key.removeprefix("module.")
        if k.startswith("backbone."):
            out[_map_torchtune_layer_key(k, stack_prefix="backbone") or k] = tensor
            continue
        if k.startswith("decoder."):
            out[_map_torchtune_layer_key(k, stack_prefix="decoder") or k] = tensor
            continue
        out[k] = tensor
    return out


# ---------------------------------------------------------------------------
# Helper functions from official Miso TTS
# ---------------------------------------------------------------------------


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r




# ---------------------------------------------------------------------------
# Miso ``Model`` (upstream)
# ---------------------------------------------------------------------------


@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int


MISO_TTS_8B_CONFIG = ModelArgs(
    backbone_flavor="llama-8B",
    decoder_flavor="llama-300M",
    text_vocab_size=128_256,
    audio_vocab_size=2051,
    audio_num_codebooks=32,
)


class MisoTTSModel(nn.Module):
    def __init__(self, config: ModelArgs | None = None) -> None:
        super().__init__()
        self.config = config or MISO_TTS_8B_CONFIG

        self.backbone, backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())
        self.decoder, decoder_dim = _prepare_transformer(FLAVORS[config.decoder_flavor]())

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size))

    def setup_caches(self, max_batch_size: int, dtype: torch.dtype | None = None) -> None:
        """Setup KV caches and return a causal mask."""
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        # torchtune's KVCache builds its k/v/cache_pos buffers with bare
        # torch.zeros/torch.arange (no device arg), so they default to CPU.
        # setup_caches runs after the model is already on `device`, and nothing
        # moves these new buffers afterward, which would leave the caches on CPU
        # while activations are on CUDA. Create them under the device context so
        # the factory calls inside torchtune land on the model's device.
        with device:
            self.backbone.setup_caches(max_batch_size, dtype)
            self.decoder.setup_caches(max_batch_size, dtype, decoder_max_seq_len=self.config.audio_num_codebooks)

        self.register_buffer("backbone_causal_mask", _create_causal_mask(self.backbone.max_seq_len, device))
        self.register_buffer("decoder_causal_mask", _create_causal_mask(self.config.audio_num_codebooks, device))

    def reset_caches(self) -> None:
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        dtype = next(self.parameters()).dtype
        b, s, _ = tokens.size()

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        curr_backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        embeds = self._embed_tokens(tokens)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2)
        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask).to(dtype=dtype)

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed = self._embed_audio(0, c0_sample)

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1)
        curr_sample = c0_sample.clone()
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # Decoder caches must be reset every frame.
        self.decoder.reset_caches()
        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self.decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask).to(
                dtype=dtype
            )
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1

        return curr_sample

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)


# ---------------------------------------------------------------------------
# Load checkpoint + Mimi (for talker / mimi stage)
# ---------------------------------------------------------------------------


def _state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}")
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    sd = {k.removeprefix("module."): v for k, v in checkpoint.items() if torch.is_tensor(v)}
    if not sd:
        raise ValueError("Checkpoint did not contain tensor weights")
    return sd


def load_miso_model_weights(path_or_repo: str, device: torch.device, dtype: torch.dtype) -> MisoTTSModel:
    if os.path.isfile(path_or_repo):
        model_file = path_or_repo
    elif os.path.isdir(path_or_repo):
        model_file = os.path.join(path_or_repo, "model.safetensors")
    else:
        model_file = hf_hub_download(repo_id=path_or_repo, filename="model.safetensors")

    model = MisoTTSModel(MISO_TTS_8B_CONFIG)
    if model_file.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to load .safetensors checkpoint files") from exc

        state_dict = load_file(model_file, device="cpu")
    else:
        checkpoint = torch.load(model_file, map_location="cpu")
        state_dict = _state_dict_from_checkpoint(checkpoint)
    
    # The checkpoint uses mlp_norm.scale but torchtune expects sa_norm.scale
    # Copy mlp_norm.scale to sa_norm.scale for compatibility
    for key in list(state_dict.keys()):
        if key.endswith(".mlp_norm.scale"):
            sa_key = key.replace(".mlp_norm.scale", ".sa_norm.scale")
            state_dict[sa_key] = state_dict[key]
    
    model.load_state_dict(state_dict)
    
    # Move to device and cast to target dtype (official approach: load first, then convert)
    model.to(device=device, dtype=dtype)
    model.eval()
    
    return model


def load_mimi_codec(device: torch.device, num_codebooks: int):
    import os
    import sys
    
    patch_bitsandbytes_import_for_unquantized_layers()
    
    # Disable torch._dynamo globally to prevent any compilation
    import torch._dynamo
    torch._dynamo.disable()
    
    from moshi.models import loaders

    w = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(w, device=device)
    mimi.set_num_codebooks(num_codebooks)
    
    return mimi
