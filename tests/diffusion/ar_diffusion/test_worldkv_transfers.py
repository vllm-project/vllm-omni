# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA bank transfer ordering, retention rollback and CPU numerical oracle."""

import pytest
import torch

from tests.diffusion.ar_diffusion.test_worldkv import raw, worldkv
from tests.diffusion.ar_diffusion.test_worldkv_retrieval import pose
from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pinned async transfers"),
]


@pytest.mark.parametrize("ratio", [0.0, 0.5, 1.0])
def test_async_bank_matches_cpu_through_retrieval_eviction_and_rollback(monkeypatch, ratio):
    config = worldkv.WorldKVConfig(
        window_frames=8, sink_frames=2, retrieval_frames=2, bank_bytes=9000, world_keep_ratio=ratio
    )
    cpu = worldkv.WorldKV(config, num_layers=2, tokens_per_frame=2)
    gpu = worldkv.WorldKV(config, num_layers=2, tokens_per_frame=2)

    def run(memory, start, commit, device):
        outputs = []
        # Alternate nearest-history hits and older host-bank hits.
        memory.set_pose(pose(start - 4 if start % 4 else 2))
        with memory.forward(start, 2, commit=commit):
            for layer in range(2):
                x = (raw(start) + layer).to(device)
                outputs.append(memory.assemble(layer, x, x, x * 2))
        return outputs

    try:
        # Exercise producer/consumer ordering on a non-default compute stream.
        with torch.Stream(device="cuda"):
            for start in range(0, 24, 2):
                for commit in (False, True):
                    expected = run(cpu, start, commit, "cpu")
                    actual = run(gpu, start, commit, "cuda")
                    for lhs, rhs in zip(actual, expected):
                        for tensor, reference in zip(lhs, rhs):
                            assert torch.equal(tensor.cpu(), reference)
                assert list(cpu.bank) == list(gpu.bank)
                assert gpu.bank_nbytes <= config.bank_bytes
            assert gpu.evicted_chunks > 0
            for chunk in gpu.bank.values():
                assert chunk.ready is not None
                chunk.ready.synchronize()
                for actual, expected in zip(chunk.layers, cpu.bank[chunk.start].layers):
                    assert actual.key.is_pinned() and actual.value.is_pinned()
                    for name in ("key", "value", "positions", "token_ids"):
                        assert torch.equal(getattr(actual, name), getattr(expected, name))

            old_bank, old_history = gpu.bank, gpu.history
            old_keys = {start: chunk.layers[0].key.clone() for start, chunk in old_bank.items()}
            retain = gpu._retained

            def fail(layer, current):
                if layer == 1:
                    raise RuntimeError("late retention failure")
                return retain(layer, current)

            monkeypatch.setattr(gpu, "_retained", fail)
            with pytest.raises(RuntimeError, match="late retention failure"):
                run(gpu, 24, True, "cuda")
            assert gpu.bank is old_bank and gpu.history is old_history
            assert gpu.next_frame == 24 and gpu._publication is None
            for start, chunk in gpu.bank.items():
                assert torch.equal(chunk.layers[0].key, old_keys[start])
            monkeypatch.setattr(gpu, "_retained", retain)
            run(gpu, 24, True, "cuda")

            old_bank, old_history = gpu.bank, gpu.history
            nbytes = worldkv.BankChunk.nbytes.fget

            def fail_after_copy(chunk):
                if chunk.start == 26:
                    raise RuntimeError("publication failed after copy enqueue")
                return nbytes(chunk)

            with monkeypatch.context() as patch:
                patch.setattr(worldkv.BankChunk, "nbytes", property(fail_after_copy))
                with pytest.raises(RuntimeError, match="after copy enqueue"):
                    run(gpu, 26, True, "cuda")
            assert gpu.bank is old_bank and gpu.history is old_history
            assert gpu.next_frame == 26 and gpu._publication is None
            gpu.close()
            assert gpu._publication is None and not gpu.bank
    finally:
        gpu.close()
        cpu.close()
