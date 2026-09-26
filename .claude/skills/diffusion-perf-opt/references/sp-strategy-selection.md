# Choosing between Ulysses, Ring and AllGather-KV

The three sequence-parallel attention strategies in vLLM-Omni move different
tensors across ranks. Which one is cheapest is decided by the model's
attention shape, not by the workload, so it can be computed from the model
config rather than measured. Compute it first; measure only to confirm.

## Step 1 — eliminate the illegal strategies

These are hard constraints, not preferences. Check them before any cost work.

| strategy | requires |
|---|---|
| Ulysses (strict) | `num_heads % sp_degree == 0` **and** `num_kv_heads % sp_degree == 0` |
| AllGather-KV | non-causal attention, and `seq_len % sp_degree == 0` |
| Ring | `seq_len % sp_degree == 0`, and no attention mask |

`num_kv_heads % sp_degree` is the one that bites: a GQA model with 4 KV heads
cannot use strict Ulysses at `sp_degree=8`. Ulysses variants that replicate KV
heads ("advanced UAA") lift that constraint at the cost of extra KV traffic.

## Step 2 — compute the per-rank communication volume

Let `S` = global sequence length, `N` = sp_degree, `H` = num_heads,
`H_kv` = num_kv_heads, `D` = head_dim, `B` = batch, `b` = dtype bytes.
Drop the common factor `B·D·b`:

```
Ulysses       4 all-to-all (Q, K, V, O), each moving (N-1)/N of a local shard
              U = (2H + 2H_kv) · (S/N) · (N-1)/N

AllGather-KV  every rank gathers the global K and V
              A = 2H_kv · S · (N-1)/N

Ring          same K/V bytes as AllGather, but split into N-1 sequential hops
              R = 2H_kv · (S/N) · (N-1)      ( = A )
```

Ulysses volume is dominated by `H`; the other two by `H_kv`. So the group size
`H / H_kv` is what decides, and the crossover is exact:

```
A / U  =  N · H_kv / (H + H_kv)

A < U   <=>   H / H_kv  >  N - 1
```

**Decision rule: prefer AllGather-KV when `num_heads / num_kv_heads > sp_degree - 1`.**

For MHA (`H_kv == H`) that reduces to `1 > N-1`, i.e. only true at `N = 1`:
plain MHA always favours Ulysses. Wider GQA groups and MQA favour AllGather-KV,
and the advantage grows with `N`.

## Step 3 — treat Ring as a fallback, not a peer

Ring moves exactly the same bytes as AllGather-KV, so it never wins on volume.
It pays `N-1` sequential hops instead of one bandwidth-optimal collective, so
it loses on latency by roughly that factor unless the implementation overlaps
communication with compute well. Reach for Ring when it is the only legal
option — typically causal attention, which AllGather-KV does not support.

Ring also has a quality caveat that the volume model cannot see. On the BAGEL
run below the raw Ring path failed the quality gate outright (14.6% mean
DINOv2 drop against an 8% threshold, 17.9% worst frame against 15%), while a
guarded Ring variant passed at a further latency cost. So when Ring is the only
legal option, validate output quality rather than assuming the strategies are
numerically interchangeable.

## Worked example, checked against hardware

BAGEL GQA, `S=4096, H=32, H_kv=4, D=128`, `N=4`, 4x A800-SXM4 80 GB:

- Legality: all three legal (`32 % 4 == 0`, `4 % 4 == 0`, `4096 % 4 == 0`, non-causal).
- Rule: `H/H_kv = 8 > N-1 = 3` -> **AllGather-KV**.
- Volume: `A/U = 4·4/(32+4) = 0.444`.

Measured mean steady-state step latency:

Two seeds per strategy; quality gate is mean DINOv2 drop <= 8% and worst
single frame <= 15%.

| strategy | mean step | peak mem | DINOv2 mean | worst frame | comm implied by the model |
|---|---:|---:|---:|---:|---:|
| **AllGather-KV** | **169.57 ms** | 30,587 MiB | 5.8% | 9.2% | 30.2 ms |
| Ulysses | 207.29 ms | 31,205 MiB | 5.3% | 9.0% | 67.9 ms |
| Hybrid (Ulysses+Ring) | 231.69 ms | 31,119 MiB | 5.7% | 9.0% | — |
| Ring (guarded) | 236.57 ms | 30,633 MiB | 5.7% | 8.8% | — |
| Ring (raw) | 244.77 ms | 30,645 MiB | **14.6%** | **17.9%** | 105.4 ms |

Raw Ring is the only row that fails the gate, and it fails on both metrics.

Solving the two Ulysses/AllGather points for a common compute time gives
139.4 ms of compute and a 33% communication share for Ulysses. Ring then falls
out at 105.4 ms of communication for the same bytes AllGather moves in
30.2 ms — a 3.5x penalty against `N-1 = 3` sequential hops. The closed form
reproduces all three measurements, which is why the table above is a sanity
anchor rather than a calibration input.

`examples/offline_inference/diffusion/sp_strategy_advisor.py` applies the rules
above and prints the legal set with relative volumes, if you would rather run
the arithmetic than do it by hand.

## When to stop computing and measure

The volume model ignores overlap quality, kernel efficiency and topology. Measure when:

- two strategies come out within ~20% of each other,
- the interconnect is not uniform (PCIe hosts, multi-node, mixed NVLink),
- communication is a small share of step time anyway — then this choice is not
  the bottleneck and the effort belongs elsewhere.

Report the shape, `sp_degree`, the rule's verdict, and the measured latencies
together, so the next reader can tell a prediction from an observation.
