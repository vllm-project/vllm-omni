# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fail-closed original Realization base and single fixed native-layout LoRA loading.

No adapter merging, framework reference imports, or implicit adapter stacking.
Adapter B uses the trained in-memory [Q;K;V]/[gate;up] layout,
not the raw base checkpoint's per-head QKV interleave.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MINIMAX_H3_FP32_PARAM_NAMES,
    _reorder_grouped_qkv_to_qkv,
)

from .kernels import lora_epilogue

logger = init_logger(__name__)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_weight_shape(module):
    """Checkpoint shape, never the rank-local parameter shape."""
    if isinstance(module, ColumnParallelLinear | RowParallelLinear):
        return (module.output_size, module.input_size)
    return tuple(module.weight.shape)


def local_column_rows(module, value):
    """Shard each packed logical projection independently, not the whole pack."""
    if isinstance(module, QKVParallelLinear):
        if module.total_num_kv_heads != module.total_num_heads:
            raise ValueError("Realization requires equal Q/K/V head counts")
        widths = [module.total_num_heads * module.head_size] * 3
    elif isinstance(module, MergedColumnParallelLinear):
        widths = module.output_sizes
    else:
        widths = [module.output_size]
    return torch.cat([part.chunk(module.tp_size, 0)[module.tp_rank] for part in value.split(widths, 0)], 0)


class FixedRealizationLoRALinear(nn.Module):
    """Preserve PEFT FP32 branch + output cast/add order and native tuple API."""

    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base_layer = base
        self.row_parallel = isinstance(base, RowParallelLinear) and base.tp_size > 1
        if self.row_parallel:
            a = a.chunk(base.tp_size, 1)[base.tp_rank].contiguous()
        elif isinstance(base, ColumnParallelLinear) and base.tp_size > 1 and not base.gather_output:
            b = local_column_rows(base, b)
        # Gather-output small replicated projections retain a full adapter B.
        # This preserves native base gather and PEFT add/cast semantics without
        # changing the shared linear module's collective contract.
        self.lora_a = nn.Parameter(a.to(device=base.weight.device).clone(), requires_grad=False)
        self.lora_b = nn.Parameter(b.to(device=base.weight.device).clone(), requires_grad=False)
        self.scale = scale

    @property
    def weight(self):
        return self.base_layer.weight

    def forward(self, value):
        output, bias = self.base_layer(value)
        adapter_input = value
        if self.row_parallel and not self.base_layer.input_is_parallel:
            adapter_input = value.chunk(self.base_layer.tp_size, -1)[self.base_layer.tp_rank].contiguous()
        rank_hidden = torch.nn.functional.linear(adapter_input.float(), self.lora_a)
        if self.row_parallel:
            # Sum rank-space in FP32 BEFORE B; do not reduce cast local deltas
            # or combine LoRA with the BF16 base partial outputs.
            rank_hidden = tensor_model_parallel_all_reduce(rank_hidden)
        delta = torch.nn.functional.linear(rank_hidden, self.lora_b)
        # PEFT adds in adapter precision, then casts the result back. Casting
        # delta to BF16 BEFORE addition is NOT equivalent.
        return lora_epilogue(output, delta, self.scale), bias


def canonical_rope(arch):
    width = arch.rope_inv_freq_len
    return 1 / (10000 ** (torch.arange(0, 2 * width, 2, dtype=torch.float32, device="cpu") / (2 * width)))


def base_inventory(model):
    if getattr(model, "_realization_adapter_loaded", False):
        raise ValueError("Cannot reload base after adapter activation")
    merged = model.quant_config is not None
    params = {n: p for n, p in model.named_parameters() if merged or n != "role_embeddings.weight"}
    params["rope.inv_freq"] = model.rope.inv_freq
    return params


@torch.no_grad()
def load_original_stream(model, weights):
    """Failure poisons load status; never permits forward with partial weights."""
    params = base_inventory(model)
    model._realization_base_loaded = False
    seen = set()
    identities = {}
    for name, value in weights:
        if name not in params or name in seen:
            raise ValueError(f"Unexpected or duplicate Realization base tensor: {name}")
        target = params[name]
        module_name, _, field = name.rpartition(".")
        module = model.get_submodule(module_name)
        dtype = (
            torch.float32
            if name in MINIMAX_H3_FP32_PARAM_NAMES or name in ("rope.inv_freq", "role_embeddings.weight")
            else torch.bfloat16
        )
        if getattr(model, "quant_config", None) is not None:
            if field == "weight_scale":
                dtype = torch.float32
            elif field == "weight" and target.dtype == torch.float8_e4m3fn:
                dtype = torch.float8_e4m3fn
        shape = logical_weight_shape(module) if field == "weight" and target.ndim == 2 else tuple(target.shape)
        if field == "weight_scale":
            shape = (module.output_size,)
            # ModelOpt versions may retain a singleton weight-reduction axis.
            if value.shape == (*shape, 1):
                value = value.squeeze(-1)
        if field == "bias" and isinstance(module, ColumnParallelLinear):
            shape = (module.output_size,)
        if tuple(value.shape) != tuple(shape) or value.dtype != dtype or target.dtype != dtype:
            raise ValueError(f"Realization base shape/dtype mismatch: {name}")
        if not torch.isfinite(value.float() if dtype == torch.float8_e4m3fn else value).all():
            raise ValueError(f"Nonfinite Realization base tensor: {name}")
        if field == "weight_scale" and not (value > 0).all():
            raise ValueError(f"Nonpositive Realization weight scale: {name}")
        if name == "rope.inv_freq" and not torch.equal(value.cpu(), canonical_rope(model.arch)):
            raise ValueError("Realization base RoPE differs from canonical frequencies")
        raw = value.detach().contiguous().cpu()
        digest = hashlib.sha256(str((name, tuple(raw.shape), raw.dtype)).encode())
        digest.update(raw.view(torch.uint8).numpy().tobytes())
        identities[name] = digest.hexdigest()
        if module_name.endswith(".attn.qkv_proj") and field in ("weight", "weight_scale"):
            value = _reorder_grouped_qkv_to_qkv(
                value,
                num_query_groups=model.arch.num_attention_heads,
                heads_per_group=1,
                head_dim=model.arch.attention_head_dim,
            )
        loader = getattr(target, "weight_loader", None)
        if loader is None:
            target.copy_(value.to(target.device))
        elif module_name.endswith(".mlp.fc1") and field in ("weight", "weight_scale"):
            gate, up = value.chunk(2, 0)
            loader(target, gate, 0)
            loader(target, up, 1)
        else:
            # Full logical tensors enter native loaders exactly once.
            loader(target, value)
        seen.add(name)
    if seen != set(params):
        raise ValueError(f"Missing Realization base tensors: {sorted(set(params) - seen)}")
    model.post_load_weights()
    model._realization_base_identity = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    model._realization_base_loaded = True
    if getattr(model, "quant_config", None) is not None:
        quantized = [
            m
            for m in model.modules()
            if getattr(m, "weight", None) is not None and m.weight.dtype == torch.float8_e4m3fn
        ]
        logger.info(
            "Duo ModelOpt checkpoint loaded: %d FP8 projections; methods=%s; kernels=%s",
            len(quantized),
            sorted({type(m.quant_method).__name__ for m in quantized}),
            sorted({type(m.quant_method.kernel).__name__ for m in quantized}),
        )
    return seen


def load_original_directory(model, directory):
    directory = Path(directory)
    index_path = directory / "model.safetensors.index.json"
    if not index_path.exists():
        index_path = directory / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    if set(index) != set(base_inventory(model)):
        raise ValueError("Realization original index does not exactly cover native base")
    # Config comparison must happen before any parameter writes.
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTArchConfig

    config_path = directory / "config.json"
    if MiniMaxH3DiTArchConfig.from_mapping(json.loads(config_path.read_text())) != model.arch:
        raise ValueError("Realization checkpoint architecture differs from constructed model")
    for shard in set(index.values()):
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("Realization shard must be a local safetensors basename")

    def tensors():
        for shard in sorted(set(index.values())):
            with safe_open(str(directory / shard), framework="pt", device="cpu") as source:
                if set(source.keys()) != {k for k, v in index.items() if v == shard}:
                    raise ValueError("Realization shard/index key mismatch")
                for name in source.keys():
                    yield name, source.get_tensor(name)

    loaded = load_original_stream(model, tensors())
    return dict(
        path=str(directory.resolve()),
        tensor_count=len(loaded),
        index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
    )


@torch.no_grad()
def load_fixed_adapter(model, path, *, rank, alpha, role_embedding=True, qkv_layout="logical"):
    if not getattr(model, "_realization_base_loaded", False) or getattr(model, "_realization_adapter_loaded", False):
        raise ValueError("Fixed Realization adapter requires loaded base and cannot be stacked")
    if type(rank) is not int or rank <= 0 or type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Realization adapter requires explicit positive rank/alpha")
    if type(role_embedding) is not bool:
        raise ValueError("Realization role embedding enablement must be explicit bool")
    if qkv_layout != "logical":
        raise ValueError("Only explicit trained logical QKV adapter layout is supported")
    # Every native linear is a target, matching the audited Realization trainables.
    modules = {
        name: module
        for name, module in model.named_modules()
        if hasattr(module, "weight") and module.weight.ndim == 2 and name != "role_embeddings"
    }
    prefix = "base_model.model.dit."
    special = {prefix + "role_embedding.weight", prefix + "rope.inv_freq"}
    expected = special | {prefix + name + f".lora_{side}.default.weight" for name in modules for side in ("A", "B")}
    replacements = {}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        if set(source.keys()) != expected:
            raise ValueError(
                f"Realization adapter key coverage mismatch: missing={sorted(expected - set(source.keys()))}, "
                f"extra={sorted(set(source.keys()) - expected)}"
            )
        for name, module in modules.items():
            a = source.get_tensor(prefix + name + ".lora_A.default.weight")
            b = source.get_tensor(prefix + name + ".lora_B.default.weight")
            rows, columns = logical_weight_shape(module)
            if a.shape != (rank, columns) or b.shape != (rows, rank):
                raise ValueError(f"Realization adapter geometry mismatch: {name}")
            if (
                a.dtype != torch.float32
                or b.dtype != torch.float32
                or not torch.isfinite(a).all()
                or not torch.isfinite(b).all()
            ):
                raise ValueError(f"Realization adapter requires finite FP32 branches: {name}")
            replacements[name] = FixedRealizationLoRALinear(module, a, b, alpha / rank)
        role = source.get_tensor(prefix + "role_embedding.weight")
        rope = source.get_tensor(prefix + "rope.inv_freq")
        if (
            role.shape != model.role_embeddings.weight.shape
            or role.dtype != torch.float32
            or not torch.isfinite(role).all()
        ):
            raise ValueError("Invalid Realization trained role embedding")
        if rope.dtype != torch.float32 or not torch.equal(rope, canonical_rope(model.arch)):
            raise ValueError("Realization adapter RoPE differs from canonical base")
    # Publish only after complete inventory validation; no live session/cache
    # may exist during this load-time operation.
    for name, replacement in replacements.items():
        model.set_submodule(name, replacement)
    model.role_embeddings.weight.copy_(role.to(model.role_embeddings.weight.device))
    model.role_embedding_enabled = role_embedding
    model._realization_adapter_loaded = True
    return dict(
        path=str(Path(path).resolve()),
        weights_sha256=file_sha256(path),
        tensor_count=len(expected),
        projections=len(modules),
        rank=rank,
        alpha=alpha,
        qkv_layout=qkv_layout,
        role_embedding=role_embedding,
    )


@torch.no_grad()
def merge_fixed_projections(model, *, chunk_rows=4096):
    """Opt-in fixed-adapter merge before any execution or cache publication.

    Keep each base projection's dtype. BF16 merged GEMM is numerically different
    from the original separately rounded BF16 base and FP32 adapter branch.
    This is a performance option requiring separate trajectory/quality checks.
    """
    from .checkpoint import SplitQKVRealizationLoRALinear

    if not getattr(model, "_realization_adapter_loaded", False) or getattr(model, "_realization_adapter_merged", False):
        raise ValueError("Realization merge requires one loaded, unmerged fixed adapter")
    if getattr(model, "_realization_has_forwarded", False):
        raise ValueError("Merge Realization projections before any forward or cache creation")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("Realization merge chunk_rows must be positive")
    replacements = []
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        for name, module in list(model.named_modules()):
            if not isinstance(module, FixedRealizationLoRALinear | SplitQKVRealizationLoRALinear):
                continue
            base = module.base_layer
            if isinstance(module, SplitQKVRealizationLoRALinear):
                pairs = list(zip(module.lora_a, module.lora_b, strict=True))
            else:
                b = module.lora_b
                if b.shape[0] != base.weight.shape[0]:
                    b = local_column_rows(base, b)
                pairs = [(module.lora_a, b)]
            cursor = 0
            for a, b in pairs:
                if a.shape[1] != base.weight.shape[1]:
                    raise ValueError("Realization merged local LoRA input geometry differs")
                for start in range(0, b.shape[0], chunk_rows):
                    stop = min(start + chunk_rows, b.shape[0])
                    delta = b[start:stop].float() @ a.float()
                    target = base.weight[cursor + start : cursor + stop]
                    merged = target.float() + delta * module.scale
                    if not torch.isfinite(merged).all():
                        raise FloatingPointError("Nonfinite Realization merged projection")
                    target.copy_(merged.to(target.dtype))
                cursor += b.shape[0]
            if cursor != base.weight.shape[0]:
                raise ValueError("Realization merged local LoRA output geometry differs")
            replacements.append((name, base))
        for name, base in replacements:
            parent, _, leaf = name.rpartition(".")
            setattr(model.get_submodule(parent), leaf, base)
        model._realization_adapter_merged = True
    except Exception:
        # A partial merge cannot be used, even if the caller catches an error.
        model._realization_base_loaded = False
        raise
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    return dict(
        mode="fixed_adapter_merge_v1",
        projections=len(replacements),
        base_dtype_preserved=True,
        merge_accumulation="fp32_ieee",
        quality_validated=False,
    )
