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
