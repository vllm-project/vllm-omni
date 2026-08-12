import pytest
import torch

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache

# `cpu` is the lane selector, not a hardware claim: the only Buildkite steps that
# reach tests/ select `core_model and cpu`, and they run on GPU hardware (l4_1).
# Marking this `cuda` collects it nowhere. Real hardware need is the skipif below.
pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]

NUM_BLOCKS = 10
BLOCK_SIZE = 4
HIDDEN_SIZE = 2
DTYPE = torch.float32


def get_omni_pcache() -> OmniTensorPrefixCache:
    """Build an OmniTensorPrefixCache, but don't init mm tensors."""
    return OmniTensorPrefixCache(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        hs_dtype=DTYPE,
    )


def test_drain_ready_async_writes_is_a_noop_before_the_event_is_ready(monkeypatch):
    """drain_ready_async_writes() must not touch the cache while the
    scheduled D2H's event has not completed, and must consume it exactly
    once after it has.

    Real GPU timing (e.g. via torch.cuda._sleep to stall the copy stream)
    was tried and found unreliable on this box: cross-stream wait_stream/
    wait_event did not reliably delay event.query() from reporting ready
    immediately, likely a virtualization/driver quirk of this cloud GPU.
    Forcing the event's query() result directly tests the branching logic
    in drain_ready_async_writes/_consume_pending_write deterministically,
    independent of real CUDA scheduling behavior.
    """
    cache = get_omni_pcache()
    mm_key = "codes.audio"
    cache.maybe_init_missing_mm_cache_keys({mm_key: torch.zeros(4, HIDDEN_SIZE)}, seq_len=4)

    num_tokens = 4
    slot_mapping_gpu = torch.arange(8, 8 + num_tokens, dtype=torch.int64, device="cuda")
    mm_gpu = {
        mm_key: torch.arange(num_tokens * HIDDEN_SIZE, dtype=torch.float32, device="cuda").reshape(
            num_tokens, HIDDEN_SIZE
        )
    }

    cache.schedule_async_write(
        hidden_states_gpu=None,
        multimodal_outputs_gpu=mm_gpu,
        slot_mapping_gpu=slot_mapping_gpu,
        num_tokens_unpadded=num_tokens,
        num_tokens_padded=num_tokens,
    )

    pending_event = cache._pending_write.event
    rows = cache.mm_outputs_cache[mm_key].view(-1, HIDDEN_SIZE)

    monkeypatch.setattr(pending_event, "query", lambda: False)
    assert cache.drain_ready_async_writes() == 0
    assert torch.all(rows[8 : 8 + num_tokens] == 0)
    assert cache._pending_write is not None

    monkeypatch.setattr(pending_event, "query", lambda: True)
    assert cache.drain_ready_async_writes() == 1
    assert torch.equal(rows[8 : 8 + num_tokens], mm_gpu[mm_key].cpu())
    assert cache._pending_write is None

    # Nothing pending anymore: a second drain is a no-op, not a re-scatter.
    assert cache.drain_ready_async_writes() == 0


def test_schedule_async_write_consumes_previous_pending_write_first():
    """Scheduling step N+1's write must first consume (scatter) step N's
    pending write, even if the caller never explicitly called
    drain_ready_async_writes() in between -- otherwise step N's data would
    be silently dropped when overwritten by step N+1's pending write."""
    cache = get_omni_pcache()
    mm_key = "codes.audio"
    cache.maybe_init_missing_mm_cache_keys({mm_key: torch.zeros(4, HIDDEN_SIZE)}, seq_len=4)

    num_tokens = 4
    first_slots = torch.arange(8, 8 + num_tokens, dtype=torch.int64, device="cuda")
    first_gpu = {mm_key: torch.full((num_tokens, HIDDEN_SIZE), 1.0, device="cuda")}
    second_slots = torch.arange(12, 12 + num_tokens, dtype=torch.int64, device="cuda")
    second_gpu = {mm_key: torch.full((num_tokens, HIDDEN_SIZE), 2.0, device="cuda")}

    cache.schedule_async_write(
        hidden_states_gpu=None,
        multimodal_outputs_gpu=first_gpu,
        slot_mapping_gpu=first_slots,
        num_tokens_unpadded=num_tokens,
        num_tokens_padded=num_tokens,
    )
    torch.accelerator.synchronize()

    # This call's internal _consume_pending_write() must flush the FIRST
    # write before the second one is scheduled; the caller never drained.
    cache.schedule_async_write(
        hidden_states_gpu=None,
        multimodal_outputs_gpu=second_gpu,
        slot_mapping_gpu=second_slots,
        num_tokens_unpadded=num_tokens,
        num_tokens_padded=num_tokens,
    )
    torch.accelerator.synchronize()
    cache.drain_ready_async_writes()

    rows = cache.mm_outputs_cache[mm_key].view(-1, HIDDEN_SIZE)
    assert torch.all(rows[8:12] == 1.0)
    assert torch.all(rows[12:16] == 2.0)
