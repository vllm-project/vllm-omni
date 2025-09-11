from dataclasses import dataclass


@dataclass
class DiTCacheTensor:
    """
    A class for specifying how the workers should initialize the DiT cache.
    """
    size: int # the size of the cache tensor in bytes


@dataclass
class DiTCacheConfig:
    """
    The DiT cache configuration of a model.
    """
    """How should model runner initialize the KV cache tensors for each layer"""
    dit_cache_tensors: list[DiTCacheTensor]
    """
    The DiT cache groups of the model.
    For models with only one type of DiT, there is only one group that
    contains all layers.
    """
    kv_cache_groups: list[DiTCacheTensor]