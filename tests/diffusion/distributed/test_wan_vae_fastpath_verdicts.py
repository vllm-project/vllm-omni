# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for bounded per-convolution fast path verdict caches."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import WanRMS_norm
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import forwards as fp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(params=["_SPATIAL_PAD_VERDICTS", "_CONV_OUT_LAYOUT_VERDICTS"])
def verdict_cache(request, monkeypatch):
    cache = weakref.WeakKeyDictionary()
    monkeypatch.setattr(fp, request.param, cache)
    return request.param, cache


@pytest.fixture(autouse=True)
def integer_probe_operands(monkeypatch):
    # CPU convolution formulations are only bitwise interchangeable on integer
    # data; the Gaussian probe is meant to tell cuDNN kernels apart.
    monkeypatch.setattr(
        fp,
        "_random_like",
        lambda t, generator, std=1.0: torch.empty_strided(t.shape, t.stride(), dtype=t.dtype).random_(
            -2, 3, generator=generator
        ),
    )


@pytest.fixture
def cpu_assembly(monkeypatch):
    """Run the verdict paths on CPU using PyTorch in place of Triton assembly."""
    monkeypatch.setattr(fp, "_kernels_allowed", lambda x: True)

    def cat_time(x, cache_x, pad_front, keep_cache_frames=0):
        history = 0 if cache_x is None else cache_x.shape[2]
        zeros = x.new_zeros((*x.shape[:2], pad_front - history, *x.shape[3:]))
        assembled = torch.cat([zeros, *([] if cache_x is None else [cache_x]), x], dim=2)
        keep = min(keep_cache_frames, history + x.shape[2])  # never the zero front padding
        return (assembled, assembled[:, :, -keep:].clone()) if keep_cache_frames else assembled

    def cat_pad(x, cache_x, padding, keep_cache_frames=0, channels_last_output=False):
        in_time = cat_time(x, cache_x, padding[4])
        assembled = torch.nn.functional.pad(in_time, (*padding[:4], 0, 0))
        if channels_last_output:
            assembled = assembled.contiguous(memory_format=torch.channels_last_3d)
        keep = min(keep_cache_frames, (0 if cache_x is None else cache_x.shape[2]) + x.shape[2])
        return (assembled, in_time[:, :, -keep:].clone()) if keep_cache_frames else assembled

    monkeypatch.setattr(fp.dm, "cat_time_5d", cat_time)
    monkeypatch.setattr(fp.dm, "cat_pad_5d", cat_pad)


def _is_candidate_call(spatial_padding: bool, x: torch.Tensor, padding) -> bool:
    """Whether a ``F.conv3d`` call is the fast formulation under verification."""
    return padding == (0, 1, 1) if spatial_padding else x.is_contiguous(memory_format=torch.channels_last_3d)


def test_verdict_cache_bounds_each_convolution_and_releases_modules(verdict_cache) -> None:
    _, cache = verdict_cache
    conv = nn.Conv3d(2, 2, 1)
    other_conv = nn.Conv3d(2, 2, 1)
    verdicts = cache.setdefault(conv, {})
    other_verdicts = cache.setdefault(other_conv, {})
    fp._record_verdict(other_verdicts, (0,), False)

    limit = fp._MAX_VERDICTS_PER_CONV
    for width in range(limit + 1):
        fp._record_verdict(verdicts, (width,), width % 2 == 0)
        assert len(verdicts) <= limit
    assert list(verdicts) == [(width,) for width in range(1, limit + 1)]
    assert set(verdicts.values()) == {True, False}
    assert other_verdicts == {(0,): False}

    # Updating an existing entry at capacity must not evict another shape.
    keys = list(verdicts)
    fp._record_verdict(verdicts, (1,), True)
    assert list(verdicts) == keys
    assert verdicts[(1,)] is True

    conv_ref = weakref.ref(conv)
    del conv
    gc.collect()
    assert conv_ref() is None
    assert list(cache) == [other_conv]


@torch.no_grad()
@pytest.mark.parametrize("matches", [True, False])
def test_evicted_verdict_is_revalidated(verdict_cache, cpu_assembly, monkeypatch, matches: bool) -> None:
    cache_name, cache = verdict_cache
    spatial_padding = cache_name == "_SPATIAL_PAD_VERDICTS"
    monkeypatch.setattr(fp, "_MAX_VERDICTS_PER_CONV", 2)
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    conv.weight.fill_(1)
    conv.bias.zero_()
    conv3d = torch.nn.functional.conv3d

    def checked_conv3d(x, weight, bias, stride, padding, dilation, groups):
        output = conv3d(x, weight, bias, stride, padding, dilation, groups)
        is_fast = _is_candidate_call(spatial_padding, x, padding)
        # Force a mismatch on the candidate path to verify reference selection.
        return output + 1 if is_fast and not matches else output

    monkeypatch.setattr(fp.F, "conv3d", checked_conv3d)
    comparisons = []
    bitwise_equal = fp._bitwise_equal

    def compare(a, b):
        result = bitwise_equal(a, b)
        comparisons.append(result)
        return result

    monkeypatch.setattr(fp, "_bitwise_equal", compare)

    def run(width):
        # Integer-valued inputs/weights keep both CPU convolution layouts exact.
        x = torch.full((1, 2, 1, 3, width), float(width))
        expected = conv3d(torch.nn.functional.pad(x, conv._padding), conv.weight, conv.bias)
        if spatial_padding:
            output = fp._run_cached_causal_conv(conv, x, [None], 0)
        else:
            output = fp._run_conv_out_channels_last(conv, x, [None], 0)
        if output is None:
            assert not spatial_padding and not matches  # Cached rejection declines.
        else:
            assert torch.equal(output, expected)
        assert len(cache[conv]) <= 2

    # One comparison on random operands decides each new shape.
    check = [matches]
    run(4)
    run(5)
    assert comparisons == check * 2
    run(4)
    assert comparisons == check * 2  # Both positive and negative hits are reused.
    run(6)
    assert {key[0][-1] for key in cache[conv]} == {5, 6}
    run(4)
    assert comparisons == check * 4  # The evicted shape is checked again.
    assert {key[0][-1] for key in cache[conv]} == {4, 6}
    assert all(value is matches for value in cache[conv].values())


@torch.no_grad()
@pytest.mark.parametrize("degenerate", ["input", "weight"])
def test_verdict_is_not_taken_from_degenerate_operands(
    verdict_cache, cpu_assembly, monkeypatch, degenerate: str
) -> None:
    """A formulation that only differs on non-zero products must be rejected on its first call.

    All-zero activations (e.g. black frames) or weights make any two kernels
    agree, so a verdict taken from them alone would later return different
    bytes for real inputs, or after an in-place weight update.
    """
    cache_name, cache = verdict_cache
    spatial_padding = cache_name == "_SPATIAL_PAD_VERDICTS"
    conv3d = torch.nn.functional.conv3d

    def other_kernel(x, weight, bias, stride, padding, dilation, groups):
        output = conv3d(x, weight, bias, stride, padding, dilation, groups)
        if _is_candidate_call(spatial_padding, x, padding):
            # Differs from the reference wherever the convolution itself is non-zero.
            output = output + conv3d(x, weight, None, stride, padding, dilation, groups)
        return output

    monkeypatch.setattr(fp.F, "conv3d", other_kernel)
    generator = torch.Generator().manual_seed(0)
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    conv.weight.copy_(torch.randint(-2, 3, conv.weight.shape, generator=generator))
    conv.bias.copy_(torch.randint(-2, 3, conv.bias.shape, generator=generator))
    shape = (1, 2, 1, 3, 4)
    real = torch.randint(-2, 3, shape, generator=generator).float()
    trained = conv.weight.clone()
    if degenerate == "weight":
        conv.weight.zero_()
    first = torch.zeros(shape) if degenerate == "input" else real

    def run(x):
        # Integer-valued operands keep both CPU convolution layouts exact.
        expected = conv3d(torch.nn.functional.pad(x, conv._padding), conv.weight, conv.bias)
        if spatial_padding:
            output = fp._run_cached_causal_conv(conv, x, [None], 0)
        else:
            # Mirror decoder_forward: a declined conv_out takes the standard path.
            output = fp._run_conv_out_channels_last(conv, x, [None], 0)
            if output is None:
                output = fp._run_cached_causal_conv(conv, x, [None], 0)
        assert torch.equal(output, expected)

    run(first)
    assert list(cache[conv].values()) == [False]
    conv.weight.copy_(trained)  # An in-place update keeps the verdict key.
    run(real)
    assert list(cache[conv].values()) == [False]


@torch.no_grad()
@pytest.mark.parametrize("matches", [True, False])
@pytest.mark.parametrize("history", [None, 1, 2])
def test_causal_conv_forward_shares_spatial_padding_verdicts(cpu_assembly, monkeypatch, matches: bool, history) -> None:
    """A normally called (e.g. hooked) convolution keeps the verified cuDNN padding path."""
    cache = weakref.WeakKeyDictionary()
    monkeypatch.setattr(fp, "_SPATIAL_PAD_VERDICTS", cache)
    generator = torch.Generator().manual_seed(0)
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    conv.weight.copy_(torch.randint(-2, 3, conv.weight.shape, generator=generator))
    conv.bias.copy_(torch.randint(-2, 3, conv.bias.shape, generator=generator))
    x = torch.randint(-2, 3, (1, 2, 2, 3, 4), generator=generator).float()
    cache_x = None if history is None else torch.randint(-2, 3, (1, 2, history, 3, 4), generator=generator).float()
    # Integer-valued operands keep both CPU convolution formulations exact.
    expected = fp.WanCausalConv3d.forward(conv, x, cache_x)
    conv3d = torch.nn.functional.conv3d

    def checked_conv3d(x, weight, bias, stride, padding, dilation, groups):
        output = conv3d(x, weight, bias, stride, padding, dilation, groups)
        # Force a mismatch on the candidate path to verify reference selection.
        return output + 1 if _is_candidate_call(True, x, padding) and not matches else output

    monkeypatch.setattr(fp.F, "conv3d", checked_conv3d)
    comparisons = []
    bitwise_equal = fp._bitwise_equal

    def compare(a, b):
        comparisons.append(bitwise_equal(a, b))
        return comparisons[-1]

    monkeypatch.setattr(fp, "_bitwise_equal", compare)
    for _ in range(2):
        assert torch.equal(fp.causal_conv_forward(conv, x, cache_x), expected)
    check = [matches]
    assert comparisons == check  # Verified once, then reused.
    assert list(cache[conv].values()) == [matches]
    # The cached call site shares the verdict instead of probing again.
    assert torch.equal(fp._run_cached_causal_conv(conv, x, [cache_x], 0), expected)
    assert comparisons == check


@torch.no_grad()
@pytest.mark.parametrize("prepad", [False, True])
def test_replaced_cache_is_released_before_convolution(cpu_assembly, monkeypatch, prepad: bool) -> None:
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    x = torch.ones(1, 2, 2, 3, 4)
    slots = [torch.ones_like(x)]
    old_cache = weakref.ref(slots[0])
    key = fp._conv_verdict_key(conv, x, slots[0].shape[2])
    monkeypatch.setattr(fp, "_SPATIAL_PAD_VERDICTS", weakref.WeakKeyDictionary({conv: {key: not prepad}}))
    conv3d = fp.F.conv3d

    def checked_conv(*args, **kwargs):
        assert old_cache() is None, "replaced history must not overlap the convolution output"
        assert slots[0].shape == x.shape
        return conv3d(*args, **kwargs)

    monkeypatch.setattr(fp.F, "conv3d", checked_conv)
    output = fp._run_cached_causal_conv(conv, x, slots, 0)
    assert output.shape == x.shape


@torch.no_grad()
@pytest.mark.parametrize("shortage", ["no_room", "out_of_memory", "level_zero_out_of_resources"])
def test_probe_without_memory_runs_reference_and_retries(verdict_cache, cpu_assembly, monkeypatch, shortage) -> None:
    """A probe that does not fit leaves the shape unverified: reference formulation now, a probe once memory allows."""
    cache_name, cache = verdict_cache
    spatial_padding = cache_name == "_SPATIAL_PAD_VERDICTS"
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    conv.weight.fill_(1)
    conv.bias.zero_()
    conv3d = torch.nn.functional.conv3d
    fast_calls = []

    def counting_conv3d(x, weight, bias, stride, padding, dilation, groups):
        fast_calls.append(_is_candidate_call(spatial_padding, x, padding))
        return conv3d(x, weight, bias, stride, padding, dilation, groups)

    monkeypatch.setattr(fp.F, "conv3d", counting_conv3d)
    random_like = fp._random_like
    room = False

    fit_checks = []

    def probe_fits(x, weight):
        fit_checks.append((x.device.type, x.shape[2]))
        return room or shortage != "no_room"

    def maybe_oom(t, generator, std=1.0):
        if not room and shortage == "out_of_memory":
            raise torch.OutOfMemoryError("probe")
        if not room and shortage == "level_zero_out_of_resources":  # how XPU reports some device OOMs
            raise RuntimeError("level_zero backend failed with error: 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES)")
        return random_like(t, generator, std)

    monkeypatch.setattr(fp, "_probe_fits", probe_fits)
    monkeypatch.setattr(fp, "_random_like", maybe_oom)
    cat_time = fp.dm.cat_time_5d
    time_assemblies = []

    def counting_cat_time(*args, **kwargs):
        time_assemblies.append(True)
        return cat_time(*args, **kwargs)

    monkeypatch.setattr(fp.dm, "cat_time_5d", counting_cat_time)
    fp.reset_probe_stats()

    def run():
        x = torch.full((1, 2, 1, 3, 4), 2.0)
        expected = conv3d(torch.nn.functional.pad(x, conv._padding), conv.weight, conv.bias)
        if spatial_padding:
            output = fp._run_cached_causal_conv(conv, x, [None], 0)
        else:
            output = fp._run_conv_out_channels_last(conv, x, [None], 0)
        return output, expected

    output, expected = run()
    if output is None:
        assert not spatial_padding  # conv_out declines to the standard path before assembling anything
    else:
        assert torch.equal(output, expected)
    assert not cache.get(conv)  # nothing recorded: the shape stays unverified
    assert not any(fast_calls)  # the real input only went through the reference formulation
    if spatial_padding:
        assert not time_assemblies  # probed on a template, so nothing was time-assembled
    declined = "declined_memory" if shortage == "no_room" else "declined_oom"
    assert fp.probe_stats() == {"equal": 0, "not_equal": 0, "declined_memory": 0, "declined_oom": 0, declined: 1}

    room = True
    fast_calls.clear()
    fit_checks.clear()
    output, expected = run()
    # One memory check, on a template of the assembled input: probes run before anything is assembled.
    assert fit_checks == [("meta", conv._padding[4] + 1)]
    assert torch.equal(output, expected)
    assert list(cache[conv].values()) == [True]
    assert fast_calls[-1]  # verified, so the fast formulation ran
    assert fp.probe_stats()["equal"] == 1


def test_probe_fits_counts_free_and_reserved_unallocated_memory(monkeypatch) -> None:
    x = torch.empty(1, 4, 3, 8, 8)  # 3 KiB
    weight = torch.empty(8, 4, 3, 3, 3)  # twice the channels out: 6 KiB of output
    need = weight.nbytes + int(1.1 * (2 * x.nbytes + 2 * 2 * x.nbytes))
    need *= fp._PROBE_HEADROOM  # no probe (and so no cached verdict) near the memory limit
    memory = {"free": int(need), "reserved": 0, "allocated": 0}
    device_module = type(
        "DeviceModule",
        (),
        {
            "mem_get_info": staticmethod(lambda device: (memory["free"], 1 << 40)),
            "memory_reserved": staticmethod(lambda device: memory["reserved"]),
            "memory_allocated": staticmethod(lambda device: memory["allocated"]),
        },
    )
    monkeypatch.setattr(fp.torch, "get_device_module", lambda device: device_module)
    assert fp._probe_fits(x, weight)
    memory["free"] = int(need) - 1
    assert not fp._probe_fits(x, weight)
    memory.update(reserved=10, allocated=9)  # cached allocator blocks not handed out count as free
    assert fp._probe_fits(x, weight)
    # A template of an input not built yet is sized the same, and memory is queried on the weight's device.
    assert fp._probe_fits(torch.empty(x.shape, device="meta"), weight)
    monkeypatch.setattr(fp.torch, "get_device_module", lambda device: object())  # no memory queries
    assert fp._probe_fits(x, weight)


@pytest.mark.parametrize("channels_last,verify", [(False, True), (True, True), (True, False)])
def test_spatial_probe_policy_preserves_exact_paths(monkeypatch, channels_last: bool, verify: bool) -> None:
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    setattr(conv, fp.CFG_ATTR, fp.FastPathConfig(channels_last=channels_last, verify_spatial_padding=verify))
    assembled = torch.ones(1, 2, 4, 3, 4)
    probes = []

    def reject(*args, **kwargs):
        probes.append(True)
        return False

    monkeypatch.setattr(fp, "_equal_on_random_operands", reject)
    verdicts: dict[tuple[str], bool] = {}
    output = fp._conv_with_spatial_padding(conv, assembled, conv.bias, verdicts, ("shape",))
    if channels_last and not verify:
        assert not probes
        assert not verdicts
        expected = fp.F.conv3d(assembled, conv.weight, conv.bias, padding=(0, 1, 1))
    else:
        assert probes == [True]
        assert verdicts == {("shape",): False}
        expected = fp.F.conv3d(fp.F.pad(assembled, (1, 1, 1, 1, 0, 0)), conv.weight, conv.bias)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("path", ["norm_act", "norm_fallback", "conv_out"])
@pytest.mark.parametrize("probe_matches", [False, True])
def test_fused_paths_probe_before_assembly_and_release_replaced_history(
    cpu_assembly, monkeypatch, path: str, probe_matches: bool
) -> None:
    """Probes run on a template before anything is assembled; old history must not overlap the convolution."""
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    conv.weight.fill_(1)
    conv.bias.zero_()
    x = torch.ones(1, 2, 2, 3, 4)
    slots = [torch.ones_like(x)]
    old_cache = weakref.ref(slots[0])
    expected = conv(x, slots[0])
    probes = []

    def probe(template, *args, **kwargs):
        # Nothing is assembled yet: the slot still holds the old history and the operand is only a template.
        assert template.device.type == "meta"
        assert slots[0] is old_cache(), "the cache slot must not be replaced before the probe decides"
        probes.append(True)
        return probe_matches

    monkeypatch.setattr(fp, "_equal_on_random_operands", probe)
    monkeypatch.setattr(fp, "_SPATIAL_PAD_VERDICTS", weakref.WeakKeyDictionary())
    monkeypatch.setattr(fp, "_CONV_OUT_LAYOUT_VERDICTS", weakref.WeakKeyDictionary())
    conv3d = fp.F.conv3d

    def checked_conv(*args, **kwargs):
        assert old_cache() is None, "replaced history must not overlap the convolution output"
        return conv3d(*args, **kwargs)

    monkeypatch.setattr(fp.F, "conv3d", checked_conv)
    if path == "conv_out":
        output = fp._run_conv_out_channels_last(conv, x, slots, 0)
        if not probe_matches:
            # A mismatch declines before assembling: the caller's standard path gets the untouched history.
            assert output is None and probes == [True] and slots[0] is old_cache()
            return
    else:
        norm = WanRMS_norm(2, images=False).eval()
        setattr(norm, fp.CFG_ATTR, fp.FastPathConfig(fuse_norm_cache=True, channels_last=True))

        def assemble(source, gamma, scale, previous, pad, **kwargs):
            # The fixture's real copy preserves ownership while standing in for
            # the CUDA-only normalization/assembly kernel.
            if path == "norm_fallback":
                return None
            return fp.dm.cat_time_5d(source, previous, pad, keep_cache_frames=2)

        monkeypatch.setattr(fp.nc, "norm_act_cat_time", assemble)
        monkeypatch.setattr(fp, "_norm_act", lambda *args, **kwargs: x)
        output = fp._run_norm_act_cached_conv(norm, nn.SiLU(), conv, x, slots, 0)
    assert probes == [True]
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(slots[0], x, rtol=0, atol=0)


def test_other_runtime_errors_in_a_probe_are_not_swallowed(monkeypatch) -> None:
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    x = torch.ones(1, 2, 3, 3, 4)

    def broken(t, generator, std=1.0):
        raise RuntimeError("some other failure")

    monkeypatch.setattr(fp, "_random_like", broken)
    with pytest.raises(RuntimeError, match="some other failure"):
        fp._equal_on_random_operands(x, conv.weight, lambda a, w: a, lambda a, w: a, lambda a: a)
