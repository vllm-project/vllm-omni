from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from vllm.config import VllmConfig

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig


@dataclass
class ForwardContext:
    """
    set forward context for diffusion models
    """

    vllm_config: VllmConfig | None = None
    omni_diffusion_config: OmniDiffusionConfig | None = None
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None = None
    split_text_embed_in_sp: bool = False
    # whether to split the text embed in sequence parallel, if True, the text embed will be split in sequence parallel

    # Sequence Parallel padding support
    # When sequence length is not divisible by SP world size, padding is added
    # These values are used by SequenceParallelGatherHook to remove padding,
    # and by attention layers to create attention masks dynamically
    sp_padding_size: int = 0
    # Original sequence length before padding (for removing padding in gather)
    sp_original_seq_len: int | None = None

    # Set by registry when _sp_plan hooks are applied.
    # When True, sp_active is determined by _sp_shard_depth (for _sp_plan hooks)
    # When False, sp_active defaults to True when sequence_parallel_size > 1 (for manual SP, standalone tests, etc.)
    sp_plan_hooks_applied: bool = False
    # SP active scope tracking within the _sp_plan hook mechanism.
    # Tracks the depth of SP sharding - incremented on shard, decremented on gather
    # Used by attention layers to determine if SP communication should be enabled
    _sp_shard_depth: int = 0

    @property
    def sp_active(self) -> bool:
        """Returns True when SP attention parallelism should be enabled.

        - If _sp_plan hooks are applied: use _sp_shard_depth (0 = outside sharded region).
        - If _sp_plan hooks are NOT applied: default to True when sequence_parallel_size > 1,
          since _sp_shard_depth is only meaningful within the _sp_plan hook mechanism.
        """
        if self.sp_plan_hooks_applied:
            return self._sp_shard_depth > 0
        # No _sp_plan: assume SP active when configured (manual SP, standalone tests)
        if self.omni_diffusion_config is None:
            raise ValueError(
                "omni_diffusion_config is not set when checking sp_active! "
                "This usually means set_forward_context() was not called. "
                "Please call with set_forward_context(omni_diffusion_config=...)."
            )

        sp_size = self.omni_diffusion_config.parallel_config.sequence_parallel_size
        return sp_size is not None and sp_size > 1

    def __post_init__(self):
        pass


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def is_forward_context_available() -> bool:
    return _forward_context is not None


def create_forward_context(
    vllm_config: VllmConfig | None = None,
    omni_diffusion_config: OmniDiffusionConfig | None = None,
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None = None,
    split_text_embed_in_sp: bool = False,
):
    return ForwardContext(
        vllm_config=vllm_config,
        omni_diffusion_config=omni_diffusion_config,
        attn_metadata=attn_metadata,
        split_text_embed_in_sp=split_text_embed_in_sp,
    )


@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager
def set_forward_context(
    vllm_config: VllmConfig | None = None,
    omni_diffusion_config: OmniDiffusionConfig | None = None,
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None = None,
    split_text_embed_in_sp: bool = False,
):
    """A context manager that stores the current forward context,
    can be attention metadata, split_text_embed_in_sp, etc.
    Here we can inject common logic for every model forward pass.
    """
    forward_context = create_forward_context(
        vllm_config=vllm_config,
        omni_diffusion_config=omni_diffusion_config,
        attn_metadata=attn_metadata,
        split_text_embed_in_sp=split_text_embed_in_sp,
    )
    # vLLM CustomOp dispatch (e.g. QKVParallelLinear) requires a global
    # vLLM config set via set_current_vllm_config().
    with override_forward_context(forward_context):
        if vllm_config is None:
            yield
        else:
            # Local import to avoid importing vllm.config.vllm at module import time.
            from vllm.config.vllm import set_current_vllm_config

            with set_current_vllm_config(vllm_config):
                yield
