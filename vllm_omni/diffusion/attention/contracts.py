# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup contracts, independent of attention assignments and model names."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodCapabilities:
    local_execution: bool
    uses_request_context: bool = False

    def validate_strategy(self):
        if not self.local_execution or self.uses_request_context:
            raise ValueError("Attention method does not declare immutable local execution")


def validate_strategy_attention(spec, provider):
    """Validate the selected executor, without inheriting a dense provider's sparse support."""
    from vllm_omni.diffusion.data import BlockSparseAttentionSpec

    if isinstance(spec, BlockSparseAttentionSpec):
        from vllm_omni.diffusion.attention.block_sparse import BlockSparseBackend

        backend = BlockSparseBackend
    else:
        backend = provider
    capabilities = backend.strategy_capabilities
    if capabilities is None:
        raise ValueError(f"{backend.__name__} has no attention strategy execution contract")
    capabilities.validate_strategy()


@dataclass(frozen=True)
class StrategyModelSupport:
    """Declared by a transformer implementing explicit operation/layout dispatch."""

    prepares_inputs: bool = False
    ulysses: bool = False


@dataclass(frozen=True)
class AttentionExecutionEnvironment:
    """Bound once from execution configuration, not inferred from an active layout."""

    strategy_enabled: bool = False
    host_prepared: bool = False
    ulysses_degree: int = 1

    @property
    def local_tensor_forward(self):
        return self.strategy_enabled

    @property
    def single_device_strategy(self) -> bool:
        """Whether strategy execution bypasses sequence-parallel coordination."""
        return self.strategy_enabled and self.ulysses_degree == 1

    @classmethod
    def for_model(cls, config, support):
        validate_strategy_parallel(config)
        degree = getattr(getattr(config, "parallel_config", None), "ulysses_degree", 1)
        if degree > 1 and not support.ulysses:
            raise ValueError("This model does not declare Ulysses attention strategy support")
        return cls(strategy_enabled=True, host_prepared=support.prepares_inputs, ulysses_degree=degree)


def validate_strategy_parallel(config):
    """Admit single-device or pure strict Ulysses execution for the PoC."""
    parallel = getattr(config, "parallel_config", None)
    degree = getattr(parallel, "ulysses_degree", 1) or 1
    if (
        getattr(config, "num_gpus", degree) != degree
        or (getattr(parallel, "world_size", degree) or degree) != degree
        or (getattr(parallel, "sequence_parallel_size", degree) or degree) != degree
        or any(
            (getattr(parallel, name, 1) or 1) != 1
            for name in (
                "ring_degree",
                "allgather_degree",
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "cfg_parallel_size",
            )
        )
        or getattr(parallel, "use_hsdp", False)
        or getattr(config, "enable_distributed_layerwise_offload", False)
    ):
        raise ValueError("Attention strategies require single-device or pure Ulysses execution")
    if degree > 1:
        if getattr(parallel, "ulysses_mode", "strict") != "strict":
            raise ValueError("Attention strategies currently require strict Ulysses mode")
        if (
            not getattr(config, "enforce_eager", False)
            and getattr(config, "diffusion_compile_granularity", "full") != "regional"
        ):
            raise ValueError("Ulysses attention strategies require regional compilation or eager execution")


LEGACY_EXECUTION = AttentionExecutionEnvironment()


def bind_attention_execution(model, environment):
    """Bind one immutable environment to participating modules before compilation."""
    for module in model.modules():
        if module is model or hasattr(module, "attention_execution"):
            module.attention_execution = environment
