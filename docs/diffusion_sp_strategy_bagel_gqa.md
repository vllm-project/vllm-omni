# BAGEL GQA: measured comparison of the three SP attention strategies

Measured on a real BAGEL GQA workload with 4x A800-SXM4 80 GB GPUs and two
seeds per strategy. Timings are steady-state denoising step latency.

Shape: `seq_len=4096`, `num_heads=32`, `num_kv_heads=4`, `head_dim=128`,
`sp_degree=4`, non-causal.

| strategy | mean step (ms) | peak memory (MiB) | DINOv2 mean drop | max frame drop |
| --- | ---: | ---: | ---: | ---: |
| Ulysses | 207.29 | 31,205 | 5.3% | 9.0% |
| Hybrid (Ulysses + Ring) | 231.69 | 31,119 | 5.7% | 9.0% |
| Ring (raw) | 244.77 | 30,645 | **14.6%** | **17.9%** |
| Ring (guard4) | 236.57 | 30,633 | 5.7% | 8.8% |
| AllGather-KV | **169.57** | 30,587 | 5.8% | 9.2% |

The quality gate is mean DINOv2 drop <= 8% and maximum single-frame drop <=
15%. **Raw Ring fails that gate**; the guarded Ring path passes. AllGather-KV
is the fastest quality-compliant configuration in this run.

This is one shape on one interconnect, not a claim that one strategy is
universally optimal.

## What the numbers confirm

The choice is computable from the attention shape. With `num_heads/num_kv_heads
= 8` and `sp_degree = 4`, the rule `num_heads / num_kv_heads > sp_degree - 1`
picks AllGather-KV, which is what measured fastest. The per-rank communication
volume model predicts `AllGather / Ulysses = 4*4/(32+4) = 0.444`.

Solving the Ulysses and AllGather-KV points for a common compute time gives
139.4 ms of compute, 67.9 ms of Ulysses communication and 30.2 ms for
AllGather-KV -- a 33% communication share. Ring then falls out at 105.4 ms of
communication for the same bytes AllGather-KV moves in 30.2 ms, a 3.5x penalty
against its `sp_degree - 1 = 3` sequential hops.

The closed form reproduces all three measurements, so this table is a sanity
anchor for the calculation rather than a calibration input to it.

## Reproducing the recommendation

```console
$ python examples/offline_inference/diffusion/sp_strategy_advisor.py \
      --seq-len 4096 --num-heads 32 --num-kv-heads 4 --sp-degree 4
...
recommended: allgather_kv
  num_heads/num_kv_heads = 8 > sp_degree-1 = 3, so gathering KV moves less than
  Ulysses' all-to-all.
```

The derivation, the legality constraints and the guidance on when to stop
computing and measure are in
`.claude/skills/diffusion-perf-opt/references/sp-strategy-selection.md`.
