# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""V1 categorical draws with Torch-compatible Philox streams.

Both nucleus sampling and compacted RAS replacements use this primitive.
Nucleus probabilities stay in ATen to preserve filtering boundaries. RNG
reservation normally stays on the host because V1 owns torch.Generators.
The runner can defer replacement reservations until token feedback.
"""

from threading import Lock

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops import topk_topp_sampler

from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.seeded_exponential import torch_exponential_noise, torch_exponential_policy


@triton.jit
def _categorical_kernel(
    probs,
    state,
    out,
    cumulative,
    indices,
    top_p,
    top_k,
    history,
    rejected,
    threshold,
    replacement_plan,
    active,
    window: tl.constexpr,
    history_block: tl.constexpr,
    nucleus: tl.constexpr,
    vocab: tl.constexpr,
    stride: tl.constexpr,
    row_threads,
    batch_threads,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    if active is not None:
        if not tl.load(active + row):
            return
    cols = tl.arange(0, block)
    if replacement_plan is not None:
        seed = tl.load(replacement_plan + row * 4).to(tl.uint64)
        offset = tl.load(replacement_plan + row * 4 + 1).to(tl.uint64)
        threads = tl.load(replacement_plan + row * 4 + 2)
        start = tl.load(replacement_plan + row * 4 + 3)
        noise = torch_exponential_noise(seed, offset, start + cols, threads)
    else:
        seed = tl.load(state + row * 3).to(tl.uint64)
        offset = tl.load(state + row * 3 + 1).to(tl.uint64)
        if tl.load(state + row * 3 + 2):
            noise = torch_exponential_noise(seed, offset, cols, row_threads)
        else:
            noise = torch_exponential_noise(seed, offset, row * vocab + cols, batch_threads)
    weights = tl.load(probs + row * stride + cols, cols < vocab, other=0.0)
    if nucleus:
        cdf = tl.load(cumulative + row * vocab + cols, cols < vocab, other=0.0)
        p = tl.load(top_p + row)
        k = tl.load(top_k + row)
        keep = ((cdf - weights) < p) & ((k <= 0) | (cols < k))
        weights = weights * keep.to(tl.float32)
    # Match ATen float32 division, including ties at the last bit.
    scores = tl.where(cols < vocab, tl.div_rn(weights, noise), -float("inf"))
    selected = tl.argmax(scores, axis=0, tie_break_left=True)
    if nucleus:
        selected = tl.load(indices + row * vocab + selected)
    tl.store(out + row, selected)
    if window > 0:
        positions = tl.arange(0, history_block)
        tokens = tl.load(history + row * window + positions, positions < window, other=-1)
        repeats = tl.sum((tokens == selected).to(tl.int32), axis=0)
        has_history = tl.sum((tokens >= 0).to(tl.int32), axis=0) > 0
        tl.store(rejected + row, (repeats >= threshold) & has_history)


def _cuda_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
    *,
    nucleus: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    history: torch.Tensor | None = None,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reserve exactly the draws that Torch would consume, then fuse selection.

    A partially seeded draw reserves one default-generator *matrix* first,
    including overwritten seeded rows, followed by request draws in dict order.
    """
    rows, vocab = probs.shape
    window = history.shape[1] if history is not None else 0
    out = torch.empty(rows, dtype=torch.long, device=probs.device)
    rejected = torch.empty(rows, dtype=torch.bool, device=probs.device) if history is not None else None
    if rejected is not None and window == 0:
        rejected.zero_()
    if rows == 0:
        return out, rejected
    if vocab == 0 or probs.stride(1) != 1:
        raise ValueError("sampling requires nonempty contiguous vocabulary rows")
    if any(row < 0 or row >= rows for row in generators):
        raise ValueError("generator row outside sampling batch")

    row_threads, row_increment = torch_exponential_policy(vocab, probs.device)
    batch_threads, batch_increment = torch_exponential_policy(rows * vocab, probs.device)

    def reserve(generator: torch.Generator, increment: int) -> tuple[int, int]:
        seed, offset = generator.initial_seed(), generator.get_offset()
        generator.set_offset(offset + increment)
        return seed if seed < 1 << 63 else seed - (1 << 64), offset

    # Per-row (seed, offset, row-draw flag). Default draws index the full matrix.
    states = [(0, 0, 0)] * rows
    if len(generators) != rows:
        default = torch.cuda.default_generators[probs.device.index or 0]
        states = [(*reserve(default, batch_increment), 0)] * rows
    for row, generator in generators.items():
        states[row] = (*reserve(generator, row_increment), 1)
    # Neither changing offsets nor seeded-row membership specialize the kernel.
    state = torch.tensor(states, dtype=torch.int64, pin_memory=True).to(probs.device, non_blocking=True)
    cumulative, indices, top_p, top_k = nucleus if nucleus is not None else (None, None, None, None)
    _categorical_kernel[(rows,)](
        probs,
        state,
        out,
        cumulative,
        indices,
        top_p,
        top_k,
        history,
        rejected,
        threshold,
        None,
        None,
        window=window,
        history_block=triton.next_power_of_2(max(window, 1)),
        nucleus=nucleus is not None,
        vocab=vocab,
        stride=probs.stride(0),
        row_threads=row_threads,
        batch_threads=batch_threads,
        block=triton.next_power_of_2(vocab),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out, rejected


def _use_triton(probs: torch.Tensor) -> bool:
    # Bound supported batch sizes and per-program vocabulary storage.
    return (
        current_omni_platform.is_cuda()
        and probs.is_cuda
        and probs.dtype == torch.float32
        and probs.shape[0] <= 64
        and probs.shape[1] <= 65536
    )


def categorical_sample(probs: torch.Tensor, generators: dict[int, torch.Generator]) -> torch.Tensor:
    """Return an int64 token vector; replacements supply only rejected rows."""
    if not _use_triton(probs):
        return topk_topp_sampler.random_sample(probs, generators).reshape(-1).long()
    return _cuda_sample(probs, generators)[0]


def nucleus_sample(
    probs: torch.Tensor,
    indices: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    history: torch.Tensor | None,
    threshold: float,
    generators: dict[int, torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Filter sorted probabilities, draw, map token ids and check RAS history.

    ATen owns the stable sort, softmax and cumulative sum. Fusing pointwise
    masks and reductions does not change the full-distribution top-p boundary.
    """
    cumulative = probs.cumsum(dim=1)
    if _use_triton(probs):
        return _cuda_sample(
            probs, generators, nucleus=(cumulative, indices, top_p, top_k), history=history, threshold=threshold
        )
    keep = (cumulative - probs) < top_p.unsqueeze(1)
    ranks = torch.arange(probs.shape[1], device=probs.device)
    keep &= (top_k.unsqueeze(1) <= 0) | (ranks.unsqueeze(0) < top_k.unsqueeze(1))
    draws = topk_topp_sampler.random_sample(probs * keep.to(probs.dtype), generators).reshape(-1, 1).long()
    sampled = indices.gather(1, draws).squeeze(1)
    repeated = None
    if history is not None:
        repeated = ((history == sampled.unsqueeze(1)).sum(dim=1) >= threshold) & (history >= 0).any(dim=1)
    return sampled, repeated


@triton.jit(do_not_specialize=["rows"])
def _prepare_replacement_kernel(
    rejected,
    snapshots,
    plan,
    scores,
    sampled,
    prepare_probs: tl.constexpr,
    vocab_block: tl.constexpr,
    rows,
    vocab: tl.constexpr,
    block: tl.constexpr,
    row_threads: tl.constexpr,
    row_increment: tl.constexpr,
    max_threads: tl.constexpr,
):
    row = tl.program_id(0)
    if not tl.load(rejected + row):
        return
    positions = tl.arange(0, block)
    valid = positions < rows
    active = tl.load(rejected + positions, valid, other=0)
    groups = tl.load(snapshots + positions * 4 + 2, valid, other=-1)
    seeded = tl.load(snapshots + positions * 4 + 3, valid, other=0)
    count = tl.sum(active.to(tl.int32), 0)
    rank = tl.sum((active & (positions < row)).to(tl.int32), 0)
    # Torch reserves a matrix for *all* compacted rows if any is unseeded.
    batch_threads = tl.minimum(max_threads, tl.cdiv(count * vocab, 256) * 256)
    batch_increment = tl.cdiv(count * vocab, batch_threads * 4) * 4
    uses_default = tl.sum((active & (seeded == 0)).to(tl.int32), 0) > 0
    group = tl.load(snapshots + row * 4 + 2)
    is_seeded = tl.load(snapshots + row * 4 + 3) != 0
    offset = tl.load(snapshots + row * 4 + 1)
    if is_seeded:
        preceding = tl.sum((active & (seeded != 0) & (groups == group) & (positions < row)).to(tl.int32), 0)
        offset += preceding * row_increment + tl.where((group == 0) & uses_default, batch_increment, 0)
    seed = tl.load(snapshots + row * 4).to(tl.uint64)
    threads = tl.where(is_seeded, row_threads, batch_threads)
    start = tl.where(is_seeded, 0, rank * vocab)

    if prepare_probs:
        cols = tl.arange(0, vocab_block)
        values = tl.load(scores + row * vocab + cols, cols < vocab, other=-float("inf"))
        selected = tl.load(sampled + row)
        has_other = tl.sum(((cols < vocab) & (cols != selected) & (values > -float("inf"))).to(tl.int32), 0) > 0
        values = tl.where((cols == selected) & has_other, -float("inf"), values)
        maximum = tl.max(values, 0)
        # ATen cunn_SoftMaxForwardReg: 1024 lanes, sequential register sums,
        # then warp and block reductions. Keep normalization and expf/div.rn.
        lanes = tl.arange(0, 1024)
        partial = tl.full((1024,), 0.0, tl.float32)
        for register in tl.static_range(triton.cdiv(vocab, 1024)):
            positions = lanes + register * 1024
            value = tl.load(scores + row * vocab + positions, positions < vocab, other=-float("inf"))
            value = tl.where((positions == selected) & has_other, -float("inf"), value)
            term = tl.extra.cuda.libdevice.exp(value - maximum)
            partial = partial + tl.where(positions < vocab, term, 0.0)
        denominator = tl.sum(partial, 0)
        weights = tl.div_rn(tl.extra.cuda.libdevice.exp(values - maximum), denominator)
        noise = torch_exponential_noise(seed, offset.to(tl.uint64), start + cols, threads)
        draws = tl.where(cols < vocab, tl.div_rn(weights, noise), -float("inf"))
        tl.store(sampled + row, tl.argmax(draws, axis=0, tie_break_left=True))
    else:
        tl.store(plan + row * 4, seed)
        tl.store(plan + row * 4 + 1, offset)
        tl.store(plan + row * 4 + 2, threads)
        tl.store(plan + row * 4 + 3, start)


def conditional_sample(
    scores: torch.Tensor,
    sampled: torch.Tensor,
    rejected: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Sample rejected rows without advancing host generators until feedback.

    Snapshots follow the initial draw. Group zero is the default generator;
    compacted request draws follow ascending row order, including aliases.
    """
    scores = scores.contiguous()
    rows, vocab = scores.shape
    default = torch.cuda.default_generators[scores.device.index or 0]
    groups = {id(default): 0}
    snapshots = []
    for row in range(rows):
        generator = generators.get(row, default)
        group = groups.setdefault(id(generator), len(groups))
        seed = generator.initial_seed()
        snapshots.append(
            (seed if seed < 1 << 63 else seed - (1 << 64), generator.get_offset(), group, row in generators)
        )
    state = torch.tensor(snapshots, dtype=torch.int64, pin_memory=True).to(scores.device, non_blocking=True)
    row_threads, row_increment = torch_exponential_policy(vocab, scores.device)
    props = torch.cuda.get_device_properties(scores.device)
    # Match ATen's register softmax branch. Other vocabulary sizes retain ATen.
    prepare_probs = 2048 < vocab <= 9216
    plan = None if prepare_probs else torch.empty((rows, 4), dtype=torch.int64, device=scores.device)
    _prepare_replacement_kernel[(rows,)](
        rejected,
        state,
        plan,
        scores,
        sampled,
        prepare_probs,
        triton.next_power_of_2(vocab),
        rows,
        vocab,
        64,
        row_threads,
        row_increment,
        props.multi_processor_count * (props.max_threads_per_multi_processor // 256) * 256,
        num_warps=32 if prepare_probs else 4,
        enable_fp_fusion=False,
    )
    if prepare_probs:
        return sampled
    # Preserve ATen outside the verified register-softmax branch.
    scores = scores.clone()
    ids = sampled.unsqueeze(1)
    selected = scores.gather(1, ids)
    scores.scatter_(1, ids, float("-inf"))
    scores.scatter_(1, ids, torch.where(torch.isfinite(scores).any(1, keepdim=True), float("-inf"), selected))
    probs = scores.softmax(1)
    _categorical_kernel[(rows,)](
        probs,
        None,
        sampled,
        None,
        None,
        None,
        None,
        None,
        None,
        0.0,
        plan,
        rejected,
        window=0,
        history_block=1,
        nucleus=False,
        vocab=vocab,
        stride=probs.stride(0),
        row_threads=0,
        batch_threads=0,
        block=triton.next_power_of_2(vocab),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return sampled


def commit_rejection_rng(
    rejected: list[bool],
    generators: dict[int, torch.Generator],
    vocab: int,
    device: torch.device,
) -> None:
    """Commit the same reservations as a native compacted replacement draw."""
    rows = [row for row, active in enumerate(rejected) if active]
    if not rows:
        return
    if any(row not in generators for row in rows):
        default = torch.cuda.default_generators[device.index or 0]
        increment = torch_exponential_policy(len(rows) * vocab, device)[1]
        default.set_offset(default.get_offset() + increment)
    increment = torch_exponential_policy(vocab, device)[1]
    for row in rows:
        if row in generators:
            generator = generators[row]
            generator.set_offset(generator.get_offset() + increment)


class DeferredRejection:
    """One step's RNG commit, shared by token consumption and the next sample.

    This owns generator references across batch compaction/removal. The output
    consumer and runner may race; the lock makes the commit exactly once.
    """

    def __init__(self, rejected: torch.Tensor, generators: dict[int, torch.Generator], vocab: int):
        self.rejected = rejected
        self.generators = generators.copy()
        self.vocab = vocab
        self.host: torch.Tensor | None = None
        self.ready: torch.Event | None = None
        self._lock = Lock()
        self._committed = False
        self.invalid_logits: torch.Tensor | None = None

    def flags(self) -> torch.Tensor:
        if self.invalid_logits is None:
            return self.rejected
        return torch.cat((self.rejected, self.invalid_logits))

    def invalid_rows(self) -> list[int]:
        """Read only after the shared token completion event has finished."""
        if self.invalid_logits is None:
            return []
        assert self.host is not None
        return [row for row, invalid in enumerate(self.host[self.rejected.numel() :].tolist()) if invalid]

    def stage(self, stream: torch.cuda.Stream, ready: torch.Event) -> None:
        # Token copies are already queued on this stream. Extend their existing
        # event to cover flags, without waiting or changing token allocation.
        flags = self.flags()
        stream.wait_stream(torch.cuda.current_stream())
        self.host = torch.empty(flags.shape, dtype=flags.dtype, pin_memory=True)
        self.ready = ready
        with torch.cuda.stream(stream):
            self.host.copy_(flags, non_blocking=True)
            flags.record_stream(stream)
            ready.record()

    def commit(self, *, ready: bool = False) -> None:
        with self._lock:
            if self._committed:
                return
            assert self.host is not None and self.ready is not None
            # History/output consumption normally already waited. Requests with
            # no token-history dependency still must commit before reusing RNG.
            if not ready and not self.ready.query():
                self.ready.synchronize()
            commit_rejection_rng(
                self.host[: self.rejected.numel()].tolist(), self.generators, self.vocab, self.rejected.device
            )
            self._committed = True
