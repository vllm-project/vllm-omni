import torch
from vllm.model_executor.models.utils import maybe_prefix


def add_prefix_to_loaded_weights(weights: set[str], prefix: str) -> set[str]:
    """
    Add a prefix to the names of the loaded weights.
    """
    return {maybe_prefix(prefix, name) for name in weights}


def split_list_into_ranges(lst: torch.Tensor, interval: int) -> list[list[int]]:
    ranges: list[list[int]] = [[] for _ in range((max(lst) // interval) + 1)]
    for num in lst:
        index = num // interval
        ranges[index].append(num)
    return ranges
