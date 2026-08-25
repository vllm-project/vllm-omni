# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Recommend a sequence-parallel attention strategy from the attention shape.

Which of Ulysses / Ring / AllGather-KV is cheapest follows from the model's
head configuration, so it can be computed instead of swept. This script prints
the legal set, the per-rank communication volume of each legal strategy, and
the recommendation, so the numbers behind the choice stay visible.

The reasoning, its derivation and a hardware check are in
``.claude/skills/diffusion-perf-opt/references/sp-strategy-selection.md``.

Usage takes the three numbers the decision actually depends on::

    python sp_strategy_advisor.py NUM_HEADS NUM_KV_HEADS SP_DEGREE
    python sp_strategy_advisor.py 32 4 4
    python sp_strategy_advisor.py 32 4 4 --causal
    python sp_strategy_advisor.py 32 4 4 --seq-len 4095   # check divisibility

``attention_mask`` and the KV-replicating Ulysses variant are parameters of
:func:`recommend` rather than flags, since they describe an attention
implementation rather than a deployment choice.
"""

from __future__ import annotations

import argparse


def legality(
    *,
    num_heads: int,
    num_kv_heads: int,
    seq_len: int | None,
    sp_degree: int,
    causal: bool,
    attention_mask: bool,
    ulysses_replicates_kv: bool,
) -> dict[str, str | None]:
    """Return ``{strategy: reason_it_cannot_run_or_None}``.

    ``seq_len`` may be ``None`` when the sequence is known to shard evenly; the
    divisibility checks are then skipped rather than guessed at.
    """
    indivisible = seq_len is not None and seq_len % sp_degree
    reasons: dict[str, str | None] = {}

    if num_heads % sp_degree:
        reasons["ulysses"] = f"num_heads {num_heads} is not divisible by sp_degree {sp_degree}"
    elif num_kv_heads % sp_degree and not ulysses_replicates_kv:
        reasons["ulysses"] = (
            f"num_kv_heads {num_kv_heads} is not divisible by sp_degree {sp_degree}; "
            "strict Ulysses cannot split the KV heads"
        )
    else:
        reasons["ulysses"] = None

    if causal:
        reasons["allgather_kv"] = "AllGather-KV does not support causal attention"
    elif indivisible:
        reasons["allgather_kv"] = f"seq_len {seq_len} is not divisible by sp_degree {sp_degree}"
    else:
        reasons["allgather_kv"] = None

    if indivisible:
        reasons["ring"] = f"seq_len {seq_len} is not divisible by sp_degree {sp_degree}"
    elif attention_mask:
        reasons["ring"] = "Ring does not support attention masks"
    else:
        reasons["ring"] = None

    return reasons


def comm_volume(*, num_heads: int, num_kv_heads: int, sp_degree: int, seq_len: int = 1) -> dict[str, float]:
    """Per-rank bytes moved off-rank, in units of ``batch * head_dim * dtype_bytes``.

    Only the ratios between strategies matter, and those are independent of
    ``seq_len``, so it defaults to 1 and is present for absolute figures.

    Ring moves the same K/V bytes as AllGather-KV but splits them into
    ``sp_degree - 1`` sequential hops, so equal volume does not mean equal time.
    """
    shard = seq_len / sp_degree
    fraction = (sp_degree - 1) / sp_degree
    return {
        # Four all-to-all exchanges: Q, K, V and the attention output.
        "ulysses": (2 * num_heads + 2 * num_kv_heads) * shard * fraction,
        "allgather_kv": 2 * num_kv_heads * seq_len * fraction,
        "ring": 2 * num_kv_heads * shard * (sp_degree - 1),
    }


def recommend(
    *,
    num_heads: int,
    num_kv_heads: int,
    sp_degree: int,
    seq_len: int | None = None,
    causal: bool = False,
    attention_mask: bool = False,
    ulysses_replicates_kv: bool = False,
) -> str:
    if num_heads % num_kv_heads:
        raise ValueError(f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}")
    if min(num_heads, num_kv_heads, sp_degree) <= 0 or (seq_len is not None and seq_len <= 0):
        raise ValueError("all dimensions must be positive")

    reasons = legality(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        seq_len=seq_len,
        sp_degree=sp_degree,
        causal=causal,
        attention_mask=attention_mask,
        ulysses_replicates_kv=ulysses_replicates_kv,
    )
    volumes = comm_volume(num_heads=num_heads, num_kv_heads=num_kv_heads, sp_degree=sp_degree)
    legal = [name for name, reason in reasons.items() if reason is None]

    shape = f"shape: heads={num_heads} kv_heads={num_kv_heads} sp_degree={sp_degree}"
    lines = [shape if seq_len is None else f"{shape} seq_len={seq_len}"]
    lines.append(f"group size num_heads/num_kv_heads = {num_heads / num_kv_heads:g}")
    lines.append("")
    lines.append(f"{'strategy':<14}{'rel. volume':>12}  status")
    baseline = volumes["ulysses"]
    for name in ("ulysses", "allgather_kv", "ring"):
        rel = volumes[name] / baseline if baseline else float("nan")
        status = "legal" if reasons[name] is None else f"ILLEGAL: {reasons[name]}"
        lines.append(f"{name:<14}{rel:>12.3f}  {status}")
    lines.append("")

    if not legal:
        lines.append("No legal strategy. Reduce sp_degree, or pad the sequence length.")
        return "\n".join(lines)

    # Ring never beats AllGather-KV on volume and pays sp_degree-1 sequential
    # hops, so it is only recommended when nothing else is available.
    preference = [name for name in ("allgather_kv", "ulysses") if name in legal]
    if preference:
        best = min(preference, key=lambda name: volumes[name])
    else:
        best = "ring"

    lines.append(f"recommended: {best}")
    if best == "allgather_kv":
        lines.append(
            f"  num_heads/num_kv_heads = {num_heads / num_kv_heads:g} > sp_degree-1 = {sp_degree - 1}, "
            "so gathering KV moves less than Ulysses' all-to-all."
        )
    elif best == "ulysses":
        if "allgather_kv" in legal:
            lines.append(
                f"  num_heads/num_kv_heads = {num_heads / num_kv_heads:g} <= sp_degree-1 = {sp_degree - 1}, "
                "so the all-to-all moves less than a full KV gather."
            )
        else:
            lines.append(f"  AllGather-KV is unavailable here: {reasons['allgather_kv']}")
    else:
        lines.append("  Only Ring is legal. It moves the same bytes as AllGather-KV but in")
        lines.append(f"  {sp_degree - 1} sequential hops, so expect a latency penalty of that order.")

    lines.append("")
    lines.append("This is a volume model. Measure when two strategies are within ~20%, when the")
    lines.append("interconnect is not uniform, or when communication is a small share of step time.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("num_heads", type=int)
    parser.add_argument("num_kv_heads", type=int)
    parser.add_argument("sp_degree", type=int)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Global, pre-sharding sequence length. Only affects the divisibility "
        "checks; omit it if the sequence is known to shard evenly.",
    )
    parser.add_argument("--causal", action="store_true", help="Rules out AllGather-KV.")
    args = parser.parse_args()

    print(
        recommend(
            seq_len=args.seq_len,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            sp_degree=args.sp_degree,
            causal=args.causal,
        )
    )


if __name__ == "__main__":
    main()
