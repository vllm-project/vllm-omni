# Mooncake KV Transfer

## PR summary

This change adds an opt-in AR-to-DiT KV handoff using vLLM's native
`KVTransferConfig`, Scheduler-owned diffusion KV pages, and
`MooncakeConnector`. The existing `OmniKVTransferManager` path remains the
fallback when `kv_transfer_config` is absent.

The orchestrator creates one transfer ticket per request and resolves the
bound AR replica's Mooncake endpoint after generation. DiT allocates the
destination pages, waits for every worker rank to finish loading, and releases
the source and destination pages only after the connector lifecycle completes.
Connector failures fail-close the diffusion engine before page teardown.

## Reference configuration

| Stage | Replicas | Devices | Parallelism | KV cache |
| --- | ---: | --- | --- | ---: |
| AR | 2 | 0-3 | TP=2 per replica | 2 GiB per worker |
| DiT | 2 | 4-7 | TP=2 per replica, EP enabled | 1 GiB per worker |

The acceptance workload uses 512x512 images, two denoising steps, guidance
scale 2.0, and `max_num_seqs=1`.

For a four-GPU debug run, set both stages to one replica and assign AR TP=2 to
devices `0,1` and DiT TP=2 to devices `2,3`.

## Experiment

The final comparison used one node with eight NVIDIA RTX PRO 6000 Blackwell
Server Edition GPUs, TCP Mooncake transport, the same 50-prompt/seed T2I
dataset, concurrency 8, and two independent 100-request runs per
implementation. Each side completed 200/200 requests without an OOM or request
failure.

| Implementation | Run 1 (req/s) | Run 2 (req/s) | Mean (req/s) |
| --- | ---: | ---: | ---: |
| `OmniKVTransferManager` | 4.4103 | 4.4497 | 4.4300 |
| Native `MooncakeConnector` | 4.4457 | 4.4699 | 4.4578 |

Native throughput is **0.63% higher** than the legacy mean. The absolute
performance difference is below the 1% acceptance threshold.

For accuracy, a non-zero transferred prefix and the resulting attention output
were compared bit-for-bit with the legacy manager path:

```text
prefix_bits_equal    True
attention_bits_equal True
max_abs_diff         0.0
```

The one-AR/one-DiT TP=2 debug topology also completed 20/20 sequential
requests. A separate real-TCP transport check transferred both CFG rows from a
TP=1 producer to a strict-Ulysses SP=2 consumer with exact values on both
target ranks.

## Scope and limitations

- The fully validated end-to-end topology is 2x AR TP=2 plus 2x DiT TP=2.
- Heterogeneous TP=1 to SP=2 reshaping is transport-tested, but arbitrary mixed
  TP/CFG/SP end-to-end combinations have not been exhaustively validated.
- CFG-parallel end-to-end deployment is not part of this acceptance run.
- Native AR-to-DiT transfer currently requires `async_chunk: false`.
- TCP is validated. RDMA and multi-node deployment are not covered here.
