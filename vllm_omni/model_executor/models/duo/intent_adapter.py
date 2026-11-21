# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fail-closed fixed Intent adapter inventory and TP-local token replacement.

Inspection is not adapter activation. This module never silently discards PEFT
keys, merges LoRA into BF16, or treats trainable_tokens_delta as an addition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from vllm_omni.diffusion.models.duo.kernels import lora_epilogue, packed_lora_epilogue
from vllm_omni.diffusion.models.duo.weights import FixedRealizationLoRALinear, file_sha256

TOKEN_KEY = "base_model.model.model.language_model.embed_tokens.token_adapter.trainable_tokens_delta"


@dataclass(frozen=True)
class IntentAdapter:
    token_ids: tuple[int, ...]
    token_rows: torch.Tensor
    # HF source module names, without the PEFT base_model.model prefix.
    projections: dict[str, tuple[torch.Tensor, torch.Tensor]]
    scale: float
    effective_vocab_size: int
    config_sha256: str
    weights_sha256: str


def load_intent_adapter(directory: Path, text_config: dict) -> IntentAdapter:
    """Validate exact supported checkpoint contract and consume every tensor."""
    directory = Path(directory)
    config_path = directory / "adapter_config.json"
    weights_path = directory / "adapter_model.safetensors"
    config = json.loads(config_path.read_text())
    required = dict(peft_type="LORA", task_type="CAUSAL_LM", bias="none")
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported Intent adapter {key}={config.get(key)!r}")
    for key in (
        "use_dora",
        "use_rslora",
        "use_qalora",
        "lora_bias",
        "fan_in_fan_out",
        "layer_replication",
        "modules_to_save",
        "rank_pattern",
        "alpha_pattern",
        "target_parameters",
        "layers_to_transform",
        "layers_pattern",
        "exclude_modules",
        "alora_invocation_tokens",
        "ensure_weight_tying",
    ):
        if config.get(key):
            raise ValueError(f"Unsupported Intent adapter option {key}")
    if set(config.get("target_modules", ())) != {"q_proj", "k_proj", "v_proj", "o_proj", "lm_head"}:
        raise ValueError("Intent adapter target modules differ from the supported contract")
    if config.get("r") != 8 or config.get("lora_alpha") != 16:
        raise ValueError("Intent adapter requires rank8/alpha16")
    ids = config.get("trainable_token_indices")
    if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
        raise ValueError("Expected explicit trainable token ID list")
    if len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= text_config["vocab_size"]:
        raise ValueError("Trainable token indices overlap or exceed base vocabulary")
    # Torchtitan resizes embeddings AND untied LM head to len(tokenizer).
    # Check the local tokenizer ID set rather than assuming base config size.
    tokenizer = json.loads((directory / "tokenizer.json").read_text())
    vocab_ids = list(tokenizer["model"]["vocab"].values())
    vocab_ids.extend(token["id"] for token in tokenizer.get("added_tokens", ()))
    unique_ids = set(vocab_ids)
    effective_vocab = max(unique_ids) + 1
    if unique_ids != set(range(effective_vocab)):
        raise ValueError("Tokenizer vocabulary has ID holes")
    if effective_vocab > text_config["vocab_size"]:
        raise ValueError("Tokenizer expansion beyond base requires explicit initialization parity")
    if max(ids) >= effective_vocab:
        raise ValueError("Trained token exceeds effective tokenizer vocabulary")
    hidden = text_config["hidden_size"]
    head = text_config["head_dim"]
    qsize = text_config["num_attention_heads"] * head
    kvsize = text_config["num_key_value_heads"] * head
    expected = {"lm_head": (hidden, effective_vocab)}
    for layer in range(text_config["num_hidden_layers"]):
        prefix = f"model.language_model.layers.{layer}.self_attn"
        expected.update(
            {
                f"{prefix}.q_proj": (hidden, qsize),
                f"{prefix}.k_proj": (hidden, kvsize),
                f"{prefix}.v_proj": (hidden, kvsize),
                f"{prefix}.o_proj": (qsize, hidden),
            }
        )
    keys = {TOKEN_KEY}
    for name in expected:
        keys.update(f"base_model.model.{name}.lora_{suffix}.weight" for suffix in ("A", "B"))
    projections = {}
    with safe_open(str(weights_path), framework="pt", device="cpu") as source:
        actual = set(source.keys())
        if actual != keys:
            raise ValueError(
                f"Intent adapter coverage mismatch: missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
            )
        rows = source.get_tensor(TOKEN_KEY)
        if rows.shape != (len(ids), hidden) or rows.dtype != torch.float32 or not torch.isfinite(rows).all():
            raise ValueError("Invalid FP32 trainable token replacement rows")
        for name, (in_features, out_features) in expected.items():
            a = source.get_tensor(f"base_model.model.{name}.lora_A.weight")
            b = source.get_tensor(f"base_model.model.{name}.lora_B.weight")
            if a.shape != (8, in_features) or b.shape != (out_features, 8):
                raise ValueError(f"Invalid LoRA geometry for {name}: {a.shape}, {b.shape}")
            if a.dtype != torch.float32 or b.dtype != torch.float32:
                raise ValueError(f"Expected FP32 LoRA branch for {name}")
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError(f"Nonfinite Intent LoRA tensors for {name}")
            projections[name] = (a, b)
    return IntentAdapter(
        tuple(ids), rows, projections, 2.0, effective_vocab, file_sha256(config_path), file_sha256(weights_path)
    )


def replace_local_token_rows(embedding: object, token_ids: tuple[int, ...], rows: torch.Tensor) -> int:
    """Replace owned rows in a native VocabParallelEmbedding at load time.

    Supports the original and added-vocab shard layout including padding.
    Does not replace LM-head rows (this checkpoint has untied embeddings).
    Must run after base loading and before graphs/cache publication. Caller must
    synchronize all ranks and bind the effective adapter hash to cache identity.
    """
    weight = embedding.weight
    shard = embedding.shard_indices
    if rows.ndim != 2 or rows.shape != (len(token_ids), weight.shape[1]):
        raise ValueError("Trainable token replacement geometry mismatch")
    if len(set(token_ids)) != len(token_ids) or not torch.isfinite(rows).all():
        raise ValueError("Invalid trainable token replacement values or duplicate IDs")
    if any(i < 0 or i >= embedding.num_embeddings for i in token_ids):
        raise ValueError("Trainable token exceeds embedding vocabulary")
    local_ids, source_rows = [], []
    for source_row, token_id in enumerate(token_ids):
        if shard.org_vocab_start_index <= token_id < shard.org_vocab_end_index:
            local = token_id - shard.org_vocab_start_index
        elif shard.added_vocab_start_index <= token_id < shard.added_vocab_end_index:
            local = token_id - shard.added_vocab_start_index + shard.num_org_elements_padded
        else:
            continue
        local_ids.append(local)
        source_rows.append(source_row)
    if local_ids:
        with torch.no_grad():
            weight.index_copy_(
                0,
                torch.tensor(local_ids, device=weight.device),
                rows[source_rows].to(device=weight.device, dtype=weight.dtype),
            )
    return len(local_ids)


class FixedFP32RowLoRA(torch.nn.Module):
    """Apply the FP32 LoRA delta after the native BF16 base reduction."""

    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base_layer = base
        self.row_parallel = base.tp_size > 1
        if self.row_parallel:
            a = a.chunk(base.tp_size, 1)[base.tp_rank].contiguous()
        self.register_buffer("lora_a", a.to(device=base.weight.device).clone())
        self.register_buffer("lora_b", b.to(device=base.weight.device).clone())
        self.scale = scale

    @property
    def weight(self):
        return self.base_layer.weight

    forward = FixedRealizationLoRALinear.forward


class NativeWideRow(torch.nn.Module):
    """Round base row projections after summing all TP shards, as in Duo."""

    def __init__(self, base, lora=None):
        super().__init__()
        self.base, self.lora = base, lora

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        from vllm.distributed import tensor_model_parallel_all_reduce

        v = x.reshape(-1, x.shape[-1])
        base = torch.mm(v, self.weight.T, out_dtype=torch.float32)
        if self.lora is None:
            return tensor_model_parallel_all_reduce(base).to(x.dtype), None
        low = torch.nn.functional.linear(v.float(), self.lora.lora_a)
        base, low = tensor_model_parallel_all_reduce(torch.cat((base, low), -1)).split(
            (base.shape[-1], low.shape[-1]), dim=-1
        )
        delta = torch.nn.functional.linear(low, self.lora.lora_b) * self.lora.scale
        return (base.to(x.dtype) + delta).to(x.dtype), None


class SplitSiLU(torch.nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class FixedFP32LoRAMethod:
    """Unquantized local projection preserving PEFT's cast/add order.

    Factors are registered on the owning native layer, not held in a closure;
    model device movement and graph capture see stable buffers. This is a fixed
    adapter, not dynamic vLLM LoRA selection. No new attention implementation.
    """

    def __init__(self, base_method: object, parts: tuple[tuple[str, int], ...], scale: float):
        self.base_method = base_method
        self.parts = parts
        self.scale = scale

    def __getattr__(self, name: str):
        # Preserve native weight postprocessing and method metadata. Factors
        # are separate buffers and must not change base weight preparation.
        return getattr(self.base_method, name)

    def apply(self, layer: object, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        base = self.base_method.apply(layer, x, bias)
        value = x.float()
        deltas = []
        for label, _ in self.parts:
            a = getattr(layer, f"duplex_lora_{label}_a")
            b = getattr(layer, f"duplex_lora_{label}_b")
            deltas.append(torch.nn.functional.linear(torch.nn.functional.linear(value, a), b))
        if len(deltas) == 1:
            return lora_epilogue(base, deltas[0], self.scale)
        return packed_lora_epilogue(base, deltas, self.scale)


def shard_intent_lora(layer: object, label: str, a: torch.Tensor, b: torch.Tensor):
    """Use native TP geometry; return owned FP32 factors, never merged weights."""
    rank, tp = layer.tp_rank, layer.tp_size
    if label in ("q", "k", "v"):
        replicas = 1 if label == "q" else layer.num_kv_head_replicas
        owner = rank // replicas
        width = (
            layer.num_heads * layer.head_size
            if label == "q"
            else layer.num_kv_heads * (layer.v_head_size if label == "v" else layer.head_size)
        )
        b = b.narrow(0, owner * width, width)
    elif label == "o":
        width = layer.weight.shape[1]
        if a.shape[1] != width * tp:
            raise ValueError("Row-parallel LoRA input geometry mismatch")
        a = a.narrow(1, rank * width, width)
    elif label == "head":
        shard = layer.shard_indices
        local = b.new_zeros((layer.weight.shape[0], b.shape[1]))
        for start, end, offset in (
            (shard.org_vocab_start_index, shard.org_vocab_end_index, 0),
            (shard.added_vocab_start_index, shard.added_vocab_end_index, shard.num_org_elements_padded),
        ):
            if end > start:
                local[offset : offset + end - start] = b[start:end]
        b = local
    else:
        raise ValueError(f"Unknown Intent LoRA projection {label}")
    return a.contiguous(), b.contiguous()


def install_fixed_intent_adapter(model: object, adapter: IntentAdapter, *, experimental_tp: bool = False) -> dict:
    """Activate a single audited adapter after native base weight loading.

    TP>1 is explicit experimental opt-in: row-local BF16 rounding differs from
    TP1. Numerical parity and graph-safe native runner acceptance remain separate
    gates. Do not install on a live model with published cache or captured graphs.
    """
    decoder = model.language_model.model
    embedding = decoder.embed_tokens
    if embedding.tp_size != 1 and not experimental_tp:
        raise ValueError("Fixed FP32 Intent adapter TP>1 is not yet validated")
    if embedding.num_embeddings != adapter.effective_vocab_size:
        raise ValueError("Intent vocabulary must match the resized adapter tokenizer")
    if getattr(model, "duplex_adapter_manifest", None) is not None:
        raise ValueError("Fixed Intent adapter may only be installed once")
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    targets = []
    row_targets = []
    for layer_idx, layer in enumerate(decoder.layers):
        prefix = f"model.language_model.layers.{layer_idx}.self_attn"
        attention = layer.self_attn
        packed = tuple((label, adapter.projections[f"{prefix}.{label}_proj"]) for label in ("q", "k", "v"))
        targets.append((attention.qkv_proj, packed))
        row_targets.append((attention, adapter.projections[f"{prefix}.o_proj"]))
    targets.append((model.language_model.lm_head, (("head", adapter.projections["lm_head"]),)))
    targets = [
        (layer, tuple((label, shard_intent_lora(layer, label, a, b)) for label, (a, b) in factors))
        for layer, factors in targets
    ]
    # Validate every target before changing any layer.
    for layer, factors in targets:
        if not isinstance(layer.quant_method, UnquantizedLinearMethod | UnquantizedEmbeddingMethod):
            raise ValueError("Fixed FP32 LoRA requires native unquantized projection methods")
        if layer.weight.dtype != torch.bfloat16:
            raise ValueError("Fixed Intent adapter requires supported BF16 base projections")
        if sum(b.shape[0] for _, (_, b) in factors) != layer.weight.shape[0]:
            raise ValueError("Fixed Intent projection physical geometry mismatch")
        if any(a.shape[1] != layer.weight.shape[1] for _, (a, _) in factors):
            raise ValueError("Fixed Intent projection input geometry mismatch")
    count = replace_local_token_rows(embedding, adapter.token_ids, adapter.token_rows)
    for layer, factors in targets:
        parts = []
        for label, (a, b) in factors:
            a = a.to(device=layer.weight.device, dtype=torch.float32)
            b = b.to(device=layer.weight.device, dtype=torch.float32)
            if layer is model.language_model.lm_head and b.shape[0] != layer.weight.shape[0]:
                b = torch.nn.functional.pad(b, (0, 0, 0, layer.weight.shape[0] - b.shape[0]))
            layer.register_buffer(f"duplex_lora_{label}_a", a)
            layer.register_buffer(f"duplex_lora_{label}_b", b)
            parts.append((label, b.shape[0]))
        layer.quant_method = FixedFP32LoRAMethod(layer.quant_method, tuple(parts), adapter.scale)
    # Reduce the FP32 rank-space update before B and the final output cast,
    # matching PEFT's global delta instead of rounding each rank's delta.
    for attention, (a, b) in row_targets:
        attention.o_proj = FixedFP32RowLoRA(attention.o_proj, a, b, adapter.scale)
    if experimental_tp and embedding.tp_size > 1 and getattr(model, "_duplex_wide_rows", False):
        for layer in decoder.layers:
            proj = layer.self_attn.o_proj
            layer.self_attn.o_proj = NativeWideRow(proj.base_layer, proj)
            layer.mlp.down_proj = NativeWideRow(layer.mlp.down_proj)
            layer.mlp.act_fn = SplitSiLU()
    model.duplex_adapter_manifest = dict(
        weights_sha256=adapter.weights_sha256,
        config_sha256=adapter.config_sha256,
        effective_vocab_size=adapter.effective_vocab_size,
        token_rows_replaced=count,
        projection_count=len(adapter.projections),
        tp_size=embedding.tp_size,
        experimental_tp=embedding.tp_size > 1,
        numerical_parity="deferred_not_passed" if embedding.tp_size > 1 else "existing_tp1_path",
        lora_dtype="float32",
        row_lora_accumulation="fp32_rank_space_before_cast",
        merged=False,
    )
    from vllm.logger import init_logger

    init_logger(__name__).info("Duplex fixed adapter TP rank %d: %s", embedding.tp_rank, model.duplex_adapter_manifest)
    return model.duplex_adapter_manifest
