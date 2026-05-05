from collections.abc import Callable

import torch
from vllm.model_executor.models.utils import maybe_prefix


def add_prefix_to_loaded_weights(weights: set[str], prefix: str) -> set[str]:
    """
    Add a prefix to the names of the loaded weights.
    """
    return {maybe_prefix(prefix, name) for name in weights}


def split_list_into_ranges(lst: torch.Tensor, interval: int) -> list[list[int]]:
    if lst.numel() == 0:
        return []

    # Move to CPU and convert to list once (High Speedup)
    # using .item() inside a loop is very slow.
    data_list = lst.detach().cpu().tolist()

    # Calculate max on the list or tensor (Tensor max is fast enough)
    max_val = int(torch.max(lst).item())

    # Pre-allocate buckets
    ranges: list[list[int]] = [[] for _ in range((max_val // interval) + 1)]

    for num in data_list:
        index = int(num // interval)
        ranges[index].append(num)

    return ranges


def safe_tensor_reshape(tensor: torch.Tensor, shape: tuple) -> torch.Tensor:
    """
    Reshape a tensor safely.
    """
    if tensor is None:
        return None
    return tensor.reshape(shape)


def reinit_rotary_inv_freq(
    model: torch.nn.Module,
    base: float = 10000.0,
    match: Callable[[str, torch.nn.Module], bool] | None = None,
) -> int:
    """Recompute ``inv_freq`` buffers on RoPE modules in-place.

    Custom RoPE classes loaded via ``trust_remote_code`` that register
    ``inv_freq`` with ``persistent=False`` and are not in
    ``ROPE_INIT_FUNCTIONS`` come out of ``from_pretrained`` with garbage
    buffer values (shape and dtype correct, contents not). ``cos()`` /
    ``sin()`` of those values produce NaN, so the first forward emits
    NaN logits. Mainstream HF RoPE classes avoid this via
    ``_rope_init_function`` framework integration.

    Recomputes ``1.0 / base^(arange(0, head_dim, 2) / head_dim)``.
    ``head_dim`` is inferred from ``2 * inv_freq.numel()``. Pass
    ``match`` to override the default selector (modules whose
    qualified name ends in ``"rotary_emb"`` and that expose a 1-D
    float ``inv_freq`` tensor). Returns the number of buffers
    re-initialised.
    """
    n_fixed = 0
    for name, module in model.named_modules():
        if match is not None:
            if not match(name, module):
                continue
        elif not name.endswith("rotary_emb"):
            continue
        inv_freq = getattr(module, "inv_freq", None)
        if not isinstance(inv_freq, torch.Tensor) or inv_freq.ndim != 1:
            continue
        head_dim = inv_freq.numel() * 2
        new_inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=inv_freq.device) / head_dim)
        )
        with torch.no_grad():
            inv_freq.copy_(new_inv_freq.to(dtype=inv_freq.dtype))
        n_fixed += 1
    return n_fixed
