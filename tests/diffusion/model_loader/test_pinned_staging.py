# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import time

import pytest
import torch

from vllm_omni.diffusion.model_loader import pinned_staging
from vllm_omni.diffusion.model_loader.pinned_staging import pinned_staging_weights_iterator

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


# 32768 fp32 = 128 KiB: above _MIN_STAGE_BYTES so the tensors actually stage
# (sub-64 KiB tensors intentionally pass through, covered below).
def _weights(n=8, numel=32768):
    for i in range(n):
        yield f"layer.{i}.weight", torch.full((numel,), float(i))


@pytest.mark.cpu
def test_stream_contract_and_producer_lifecycle():
    """The iterator is transparent to the consumer and its producers respect
    the read-ahead budget: single-thread runs preserve source order and
    values; multi-thread runs may reorder but yield every item exactly once;
    num_staging_threads=0 resolves to auto instead of stranding the consumer
    with zero producers; producers never run more than max_inflight_bytes
    ahead; a tensor larger than the whole budget is still admitted (no
    deadlock); and early generator close joins the producer threads."""
    # single thread: order + values (checked while iterating, per the
    # recycling contract - a yielded tensor is valid until the next item)
    seen = []
    for name, tensor in pinned_staging_weights_iterator(_weights(), num_staging_threads=1):
        i = int(name.split(".")[1])
        assert torch.equal(tensor, torch.full((32768,), float(i)))
        seen.append(name)
    assert seen == [f"layer.{i}.weight" for i in range(8)]

    # multi thread: any order, complete, no duplicates; 0 = auto must not hang
    seen = []
    for name, tensor in pinned_staging_weights_iterator(_weights(n=32), num_staging_threads=4):
        i = int(name.split(".")[1])
        assert torch.equal(tensor, torch.full((32768,), float(i)))
        seen.append(name)
    assert sorted(seen) == sorted(f"layer.{i}.weight" for i in range(32))
    assert len(list(pinned_staging_weights_iterator(_weights(), num_staging_threads=0))) == 8

    # backpressure: producers stay within the in-flight byte budget
    numel = 1024
    budget = numel * 4 * 2  # ~2 tensors in flight
    produced = []

    def _tracked():
        for i in range(16):
            produced.append(i)
            yield f"w.{i}", torch.zeros(numel)

    gen = pinned_staging_weights_iterator(_tracked(), max_inflight_bytes=budget, num_staging_threads=2)
    next(gen)  # consume one
    time.sleep(0.2)  # let producers run as far ahead as they can
    # budget=2 in flight + up to num_staging_threads staging + 1 consumed
    assert len(produced) <= 6
    gen.close()

    # oversized tensor admitted when nothing else is in flight
    out = list(pinned_staging_weights_iterator(iter([("big", torch.zeros(1 << 16))]), max_inflight_bytes=16))
    assert len(out) == 1

    # early close leaves no producer threads behind
    before = {t.ident for t in threading.enumerate()}
    gen = pinned_staging_weights_iterator(_weights(n=64), max_inflight_bytes=64)
    next(gen)
    gen.close()
    time.sleep(0.3)
    leaked = [t for t in threading.enumerate() if t.name.startswith("pinned-staging") and t.ident not in before]
    assert not leaked


@pytest.mark.cpu
def test_failure_semantics_and_helpers(monkeypatch):
    """Source-iterator exceptions PROPAGATE (a corrupt checkpoint must abort);
    staging-side failures of ANY exception type degrade to pass-through
    (loading correctness must never depend on staging). Plus the pure
    helpers: pow2 bucketing with a 1 MiB floor, auto thread count
    ~usable-cores/4 clamped to [2, 4], bounded free buffers, and _prefault
    never raising."""

    def _bad():
        yield "a", torch.zeros(32768)
        raise RuntimeError("checkpoint corrupt")

    with pytest.raises(RuntimeError, match="checkpoint corrupt"):
        list(pinned_staging_weights_iterator(_bad()))

    for exc in (RuntimeError("cudaHostAlloc failed"), ValueError("unexpected staging failure")):

        def _boom(nbytes, exc=exc):
            raise exc

        monkeypatch.setattr(pinned_staging, "_alloc_pinned", _boom)
        out = list(pinned_staging_weights_iterator(_weights()))
        assert len(out) == 8
        for name, t in out:
            i = int(name.split(".")[1])
            assert torch.equal(t, torch.full((32768,), float(i)))

    b = pinned_staging._bucket_bytes
    assert b(1) == 1 << 20
    assert b(1 << 20) == 1 << 20
    assert b((1 << 20) + 1) == 1 << 21
    assert b(3 << 20) == 4 << 20

    for cores, want in ((16, 4), (8, 2), (224, 4)):
        monkeypatch.setattr(pinned_staging, "_usable_cpu_count", lambda c=cores: c)
        assert pinned_staging._auto_staging_threads() == want

    # Free size classes cannot accumulate beyond the internal cache cap.
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", lambda n: torch.empty(n, dtype=torch.uint8))
    pool = pinned_staging._PinnedBufferPool(2 << 20)
    small = pool.get(1 << 20)
    large = pool.get(2 << 20)
    pool.put(small)
    pool.put(large)
    assert pool.cached_bytes <= 2 << 20
    assert pool.drops == 1 and pool.peak_reserved_bytes == 3 << 20
    reused = pool.get(1 << 20)
    assert reused.data_ptr() == small.data_ptr() and pool.reuses == 1

    pinned_staging._prefault(torch.zeros(4096))
    pinned_staging._prefault(torch.zeros(0))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned allocation requires CUDA")
def test_pool_recycling_and_fused_cast(monkeypatch):
    """CUDA behaviors in one pass. Pooling economics: same-shape tensors
    reuse pooled buffers instead of re-allocating (cudaHostAlloc during
    in-flight H2D stalls >100ms) and recycle their backing storage; sub-64
    KiB tensors bypass the pool entirely (no DMA benefit, and the 1 MiB
    floor would over-allocate up to 256x). Fused-cast three-tier policy,
    applied during the staging copy so H2D stays on the same-dtype pinned
    DMA fast path: exact param-name matches win (protecting intentionally-
    fp32 params); unmapped float names fall back to default_dtype (renamed/
    fused weights, q/k/v -> qkv: 76% of Wan2.2's bytes); no default ->
    unchanged; non-float / quantized dtypes are never cast (bit-exact for
    dequant)."""
    calls = {"n": 0}
    real_alloc = pinned_staging._alloc_pinned

    def _counted(nbytes):
        calls["n"] += 1
        return real_alloc(nbytes)

    monkeypatch.setattr(pinned_staging, "_alloc_pinned", _counted)
    nbytes = 32768 * 4
    ptrs = []
    for name, tensor in pinned_staging_weights_iterator(
        _weights(n=32), max_inflight_bytes=2 * nbytes, num_staging_threads=1
    ):
        i = int(name.split(".")[1])
        assert torch.equal(tensor, torch.full((32768,), float(i)))
        ptrs.append(tensor.data_ptr())
    assert calls["n"] <= 4  # 32 identical shapes served almost entirely from reuse
    assert len(set(ptrs)) < len(ptrs)  # same storage observed more than once

    # tiny tensors: same object out, unpinned; big tensors: staged pinned copy
    monkeypatch.setattr(pinned_staging, "_alloc_pinned", real_alloc)
    tiny = torch.randn(64, dtype=torch.float32)  # 256 B
    big = torch.randn(32768, dtype=torch.float32)  # 128 KiB
    out = {}
    for name, t in pinned_staging_weights_iterator(iter([("tiny", tiny), ("big", big)]), num_staging_threads=1):
        out[name] = (t.is_pinned(), t.data_ptr() == (tiny if name == "tiny" else big).data_ptr(), t.clone())
    assert out["tiny"][1] and not out["tiny"][0]
    assert out["big"][0]
    assert torch.equal(out["tiny"][2], tiny) and torch.equal(out["big"][2], big)

    # fused cast policy tiers
    src = torch.randn(32768, dtype=torch.float32)
    weights = [
        ("mapped", src.clone()),  # exact match -> bf16
        ("norm.weight", src.clone()),  # exact match says KEEP fp32; default must not override
        ("blocks.0.attn.q.weight", src.clone()),  # renamed: falls back to default bf16
        ("ints", torch.arange(32768, dtype=torch.int64)),  # non-float: never cast
        ("quant.scale", src.to(torch.float8_e4m3fn)),  # quantized: never cast
    ]
    out = {}
    for name, t in pinned_staging_weights_iterator(
        iter(weights),
        num_staging_threads=1,
        target_dtypes={"mapped": torch.bfloat16, "norm.weight": torch.float32},
        default_dtype=torch.bfloat16,
    ):
        out[name] = (t.dtype, t.clone())
    assert out["mapped"][0] is torch.bfloat16 and torch.equal(out["mapped"][1], src.to(torch.bfloat16))
    assert out["norm.weight"][0] is torch.float32
    assert out["blocks.0.attn.q.weight"][0] is torch.bfloat16
    assert out["ints"][0] is torch.int64
    assert out["quant.scale"][0] is torch.float8_e4m3fn

    # without a default, unmapped floats stage dtype-unchanged
    out = dict(
        pinned_staging_weights_iterator(
            iter([("unmapped", src.clone())]), num_staging_threads=1, target_dtypes={}, default_dtype=None
        )
    )
    assert out["unmapped"].dtype is torch.float32


@pytest.mark.cpu
def test_loader_gate_and_dtype_wiring(monkeypatch):
    """The loader engages staging automatically for safetensors loads,
    passing the runtime CUDA/pinnable checks INTO the iterator as
    `local_eligible` instead of gating the call (a rank-divergent gate would
    strand TP peers in the pre-flight collective); it silently skips
    incompatible destinations (cpu offload here), builds the cast map from
    the model's params with od_config.dtype as the fallback, and disables
    the blanket fallback for quantized sources (their fp32 scales travel
    under pre-fusion names; a default cast would corrupt dequantization --
    exact-name casts still apply)."""
    from torch import nn

    from vllm_omni.diffusion.model_loader import diffusers_loader as dl

    captured = {}

    def _fake_staging(
        inner,
        target_dtypes=None,
        default_dtype=None,
        local_eligible=True,
        max_inflight_bytes=None,
        context=None,
    ):
        captured["called"] = True
        captured["local_eligible"] = local_eligible
        captured["target_dtypes"] = target_dtypes
        captured["default_dtype"] = default_dtype
        captured["max_inflight_bytes"] = max_inflight_bytes
        captured["context"] = context
        return inner

    monkeypatch.setattr(dl, "cooperative_staging_weights_iterator", _fake_staging)
    monkeypatch.setattr(dl, "safetensors_weights_iterator", lambda *a, **k: iter([]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dl, "is_pin_memory_available", lambda: True)
    monkeypatch.setattr(
        dl.DiffusersPipelineLoader,
        "_prepare_weights",
        lambda self, *a, **k: ("/tmp", ["x.safetensors"], True),
    )
    monkeypatch.setattr(dl.DiffusersPipelineLoader, "_get_checkpoint_adapter", lambda self, *a, **k: None)

    loader = dl.DiffusersPipelineLoader.__new__(dl.DiffusersPipelineLoader)
    loader.quant_config = None
    loader.parallel_config = type("PC", (), {"use_hsdp": False})()
    loader.load_config = type("LC", (), {"safetensors_load_strategy": "eager", "use_tqdm_on_load": False})()
    loader.od_config = type(
        "OD",
        (),
        {
            "enable_multithread_weight_load": False,
            "enable_cpu_offload": False,
            "enable_layerwise_offload": False,
            "enable_distributed_layerwise_offload": False,
            "dtype": torch.bfloat16,
        },
    )()
    src = dl.DiffusersPipelineLoader.ComponentSource(model_or_path="m", subfolder=None, revision=None)

    # default: demand loading stops speculative prewarm, then staging engages.
    handoff = type(
        "Prewarm",
        (),
        {
            "stopped": False,
            "joined": False,
            "stop": lambda self: setattr(self, "stopped", True),
            "join": lambda self, timeout=None: setattr(self, "joined", True),
        },
    )()
    loader._weights_prewarm_handle = handoff
    list(loader._get_weights_iterator(src))
    assert handoff.stopped and handoff.joined and loader._weights_prewarm_handle is None
    assert captured.get("called") is True
    assert captured["local_eligible"] is True
    assert captured["target_dtypes"] is None and captured["default_dtype"] is None

    # runtime ineligibility (no CUDA) still CALLS the iterator -- it must
    # join the group vote -- but with local_eligible=False and no cast map
    captured.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model_probe = nn.Linear(2, 2)
    list(loader._get_weights_iterator(src, model=model_probe))
    assert captured.get("called") is True
    assert captured["local_eligible"] is False
    assert captured["target_dtypes"] is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # Incompatible destinations skip staging without disabling the existing
    # multithread loader, even under TP.
    import vllm.distributed.parallel_state as parallel_state

    captured.clear()
    multithread = {"called": False}
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        dl,
        "multi_thread_safetensors_weights_iterator",
        lambda *a, **k: multithread.update(called=True) or iter([]),
    )
    loader.od_config.enable_multithread_weight_load = True

    # A group preflight veto (including pinned allocation failure) preserves
    # the configured multithread source before entering local fallback.
    vetoed_context = type("Context", (), {"comm": None})()
    monkeypatch.setattr(dl, "prepare_cooperative_staging", lambda eligible: vetoed_context)
    list(loader._get_weights_iterator(src))
    assert captured.get("called") is True
    assert captured["context"] is vetoed_context
    assert multithread["called"] is True

    captured.clear()
    multithread["called"] = False
    loader.od_config.enable_cpu_offload = True
    list(loader._get_weights_iterator(src))
    assert captured.get("called") is None
    assert multithread["called"] is True
    loader.od_config.enable_cpu_offload = False

    captured.clear()
    multithread["called"] = False
    loader.od_config.enable_distributed_layerwise_offload = True
    list(loader._get_weights_iterator(src))
    assert captured.get("called") is None
    assert multithread["called"] is True
    loader.od_config.enable_distributed_layerwise_offload = False

    loader.od_config.enable_multithread_weight_load = False
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    # unquantized + model: exact-name map + od_config.dtype fallback
    model = nn.Linear(2, 2)
    captured.clear()
    list(loader._get_weights_iterator(src, model=model))
    assert "weight" in captured["target_dtypes"]
    assert captured["default_dtype"] is torch.bfloat16

    # quantized source: exact-name casts only, no blanket fallback
    captured.clear()
    loader.quant_config = object()
    list(loader._get_weights_iterator(src, model=model))
    assert "weight" in captured["target_dtypes"]
    assert captured["default_dtype"] is None
