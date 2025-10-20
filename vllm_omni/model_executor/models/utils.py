from vllm.model_executor.models.utils import maybe_prefix


def add_prefix_to_loaded_weights(weights: set[str], prefix: str) -> set[str]:
    """
    Add a prefix to the names of the loaded weights.
    """
    return {maybe_prefix(prefix, name) for name in weights}