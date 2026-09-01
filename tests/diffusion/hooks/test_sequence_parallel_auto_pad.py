# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The auto-pad backend-compatibility guard in ``SequenceParallelSplitHook``.

When an SP shard's sequence length is not divisible by the world size, the hook pads it.
Whether the padded positions are then *masked* is up to ``parallel_config.mask_sp_padding``,
which defaults to False: models deliberately leave them unmasked and ``warning_once``
about it, so no mask is ever built and the backend's mask capability is irrelevant.

The guard used to read ``if not attn_backend.supports_attention_mask:`` -- a bound method,
always truthy -- so it never fired. Adding the call parens alone would turn that documented
default into a hard failure, hence the ``mask_sp_padding`` gate. These tests pin both halves:
the guard fires when a mask really will be built, and stays out of the way when it will not.
"""

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
from vllm_omni.diffusion.forward_context import ForwardContext
from vllm_omni.diffusion.hooks.sequence_parallel import SequenceParallelSplitHook

# No ``pytest.mark.sp``: that marker is registered as "(multi-GPU)" in pyproject.toml, and
# these cases fake the SP world size on CPU. ``test_attention_sp.py``, which does test SP for
# real, likewise marks itself only core_model/diffusion/cpu and gates its multi-card case
# with ``hardware_test``.
pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

WORLD_SIZE = 2
INDIVISIBLE_SEQ_LEN = 5  # 5 % 2 != 0, so the auto-pad branch runs


class _MaskCapableBackend:
    @classmethod
    def supports_attention_mask(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "TORCH_SDPA"


class _MaskBlindBackend:
    """Stands in for SAGE_ATTN / RAINFUSION_ATTN / TRTLLM_ATTN."""

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return False

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN"


@pytest.fixture
def auto_pad(monkeypatch):
    """Run ``_shard_with_auto_pad`` with SP state and backend selection faked out.

    ``_shard_with_auto_pad`` imports its dependencies inside the function body, so each
    one is patched at its defining module rather than on the hook's module.
    """

    def _run(*, backend, mask_sp_padding, ring_world_size=1, seq_len=INDIVISIBLE_SEQ_LEN):
        selector_calls = []
        # The real dataclasses rather than SimpleNamespace fakes: both construct with no
        # arguments, so the test also pins the field names and their production defaults
        # (``sp_padding_size`` starts at 0, and the hook keys "already set" off
        # ``sp_original_seq_len is None``).
        ctx = ForwardContext(
            omni_diffusion_config=OmniDiffusionConfig(
                parallel_config=DiffusionParallelConfig(mask_sp_padding=mask_sp_padding)
            )
        )

        monkeypatch.setattr(
            "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_world_size",
            lambda: WORLD_SIZE,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_rank",
            lambda: 0,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.distributed.parallel_state.get_ring_parallel_world_size",
            lambda: ring_world_size,
        )
        monkeypatch.setattr(
            "vllm_omni.diffusion.forward_context.is_forward_context_available",
            lambda: True,
        )
        monkeypatch.setattr("vllm_omni.diffusion.forward_context.get_forward_context", lambda: ctx)

        def _selector(**kwargs):
            """Record that the selector ran, and with which arguments."""
            selector_calls.append(kwargs)
            return backend, None

        monkeypatch.setattr(
            "vllm_omni.diffusion.attention.selector.get_attn_backend_for_role",
            _selector,
        )
        # Taken by the already-divisible early return; the real one needs a live SP group.
        monkeypatch.setattr(
            "vllm_omni.diffusion.hooks.sequence_parallel.sp_shard",
            lambda t, d, validate=True: t.chunk(WORLD_SIZE, dim=d)[0],
        )

        hook = SequenceParallelSplitHook(metadata={}, config=SequenceParallelConfig(ulysses_degree=WORLD_SIZE))
        x = torch.arange(seq_len, dtype=torch.float32).reshape(1, seq_len, 1)
        return hook._shard_with_auto_pad(x, dim=1), ctx, selector_calls

    return _run


def test_default_pads_without_masking_on_a_mask_blind_backend(auto_pad):
    """The regression this gate exists for: mask_sp_padding defaults to False.

    Without the gate, Wan2.2-style models with SP=2, an indivisible latent seq_len and a
    mask-blind backend would start raising, even though nothing builds a mask.
    """
    shard, ctx, selector_calls = auto_pad(backend=_MaskBlindBackend, mask_sp_padding=False)

    assert shard.shape == (1, 3, 1)  # 5 -> padded to 6, split across 2 ranks
    assert ctx.sp_original_seq_len == INDIVISIBLE_SEQ_LEN
    assert ctx.sp_padding_size == 1
    # The selector is not consulted at all on the default path.
    assert selector_calls == []


def test_mask_sp_padding_rejects_a_backend_that_cannot_mask(auto_pad):
    with pytest.raises(ValueError, match="does not support attention_mask"):
        auto_pad(backend=_MaskBlindBackend, mask_sp_padding=True)


def test_mask_sp_padding_error_names_the_escape_hatch(auto_pad):
    """The message has to say how to proceed, since padding-without-masking is valid."""
    with pytest.raises(ValueError, match="mask_sp_padding=False to pad without masking"):
        auto_pad(backend=_MaskBlindBackend, mask_sp_padding=True)


def test_mask_sp_padding_allows_a_mask_capable_backend(auto_pad):
    shard, ctx, selector_calls = auto_pad(backend=_MaskCapableBackend, mask_sp_padding=True)

    assert shard.shape == (1, 3, 1)
    assert ctx.sp_padding_size == 1
    assert len(selector_calls) == 1
    assert selector_calls[0]["role"] == "self"


@pytest.mark.parametrize("mask_sp_padding", [False, True])
def test_divisible_seq_len_never_reaches_the_guard(auto_pad, mask_sp_padding):
    """No padding means no mask and no capability question, whatever the config says."""
    shard, ctx, selector_calls = auto_pad(
        backend=_MaskBlindBackend,
        mask_sp_padding=mask_sp_padding,
        seq_len=WORLD_SIZE * 3,
    )

    assert ctx.sp_padding_size == 0  # untouched: the hook returns before the padding branch
    assert ctx.sp_original_seq_len is None
    assert selector_calls == []
    assert shard.shape[1] == 3


@pytest.mark.parametrize("mask_sp_padding", [False, True])
def test_ring_attention_still_rejected_regardless_of_the_gate(auto_pad, mask_sp_padding):
    """Ring cannot express a padding mask at all, so it is refused either way.

    On the ``mask_sp_padding=True`` path a mask-blind backend raises first; this uses a
    mask-capable one so the ring check is what actually fires.
    """
    with pytest.raises(ValueError, match="Cannot use Ring attention"):
        auto_pad(backend=_MaskCapableBackend, mask_sp_padding=mask_sp_padding, ring_world_size=2)


def test_guard_is_called_not_merely_referenced(auto_pad):
    """A bound-method truthiness check would let this backend through silently."""

    class _NeverCalled(_MaskBlindBackend):
        @classmethod
        def supports_attention_mask(cls):  # noqa: ANN206 - deliberately not a bool subclass
            raise AssertionError("supports_attention_mask must be called, not inspected")

    with pytest.raises(AssertionError, match="must be called"):
        auto_pad(backend=_NeverCalled, mask_sp_padding=True)
