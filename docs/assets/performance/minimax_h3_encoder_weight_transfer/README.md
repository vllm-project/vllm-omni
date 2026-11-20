# MiniMax-H3 encoder paired performance evidence

Measured checkout: `7602d7980d8add4277353e8bc162ac1f98d98c19`. [Source equivalence](source-equivalence.json) records the reviewed PR head and its docs-only difference. Hardware/workload: 8 × Ascend 910B4, BF16, native layerwise offload, encoder TP8, DiT USP8, replicated vision, FLASH_ATTN, VAE tile8, case01, 480×864, 294 frames, 24 fps, seed42 and eight denoiser calls. Only encoder weight packing changes.

Three complete off/packed/off contrasts share one exclusive lease. Each arm has exactly two complete validated warmups and five formal requests. The primary metric is per-request rank-max `_encode_text_hidden` wall time: prepare/load, encoder compute, and offload including production's existing synchronization. No extra formal device synchronization or per-copy callback was added. `encode_prompt` and HTTP E2E are secondary; H2D submission times are host times, not pure device DMA duration.

| Round | Completed encoder control / packed | Encoder saving | HTTP E2E control / packed |
| --- | ---: | ---: | ---: |
| 1 | 0.6741 / 0.6781 s | -3.97 ms | 61.652 / 61.928 s |
| 2 | 0.6847 / 0.6822 s | 2.54 ms | 61.700 / 61.875 s |
| 3 | 0.6955 / 0.6682 s | 27.27 ms | 61.879 / 61.814 s |

Completed encoder: **0.6848 → 0.6762 s**, no distinguishable benefit established. Paired-round t95 saving interval: [-32.32, 49.56] ms; hierarchical block-bootstrap: [-6.47, 26.43] ms; 2/3 positive rounds.

HTTP E2E: **61.744 → 61.873 s**, no distinguishable benefit established. Paired-round t95 saving interval: [-0.563, 0.305] s; hierarchical block-bootstrap: [-0.339, 0.084] s.

Packing covers **7.51%** of staged encoder host storage (rank0: 562,078,464/7,480,909,280 bytes; all ranks are archived). Non-block H2D calls: 28 → 5; block H2D calls remain 79. Transferred bytes are essentially unchanged; grouping reduces API calls. Actual copies are traced and checked in warmups. Formal counting follows successful load calls and the verified immutable source-loop group sizes, without intercepting each copy.

All 63 requests pass HTTP/media/full-decode checks and output-byte equality. All warmup encoder activations match. See [raw requests and statistics](paired-results.json), [manifest](run-manifest.json), [dependencies](dependencies.json), [numeric server excerpts](server.log.txt) and [prespecified analysis contract](analysis-contract.json).

The confidence analysis treats the three round-level contrasts as the replication units. It reports a paired t interval (df2) and a hierarchical circular length-two block bootstrap (10,000 samples, seed42), retaining adjacent-request correlation. Both lower bounds must be positive and every round must save time to confirm a gain. Single-request SD is descriptive, not a minimum effect threshold. Three rounds and linear drift correction cannot exclude every systematic bias.

Recompute the public results and warmup copy-count checks using Python standard library only:

```bash
python reproduce_paired.py.txt paired-results.json
```

This is an evidence archive. Original measurement sources (`observer.py.txt`, `observer_base.py.txt`, `control.py.txt`, `server.py.txt`, `stage.sh.txt`, `lifecycle.py.txt`) retain host-specific dependencies and private inputs are represented by hashes. The [previous three-request-per-arm evidence](earlier-three-request-results.json) is retained separately; it is not mixed into these estimates. No untraceable historical 3.37% timing is reused.

The earlier measurement wrappers and manifest remain pinned at [their original docs commit](https://github.com/vllm-project/vllm-omni/blob/0683e378ba85ac8c752844d0a6ce6978bccf8081/docs/assets/performance/minimax_h3_encoder_weight_transfer/README.md). The `parameter_precision` field describes currently registered/materialized tensors and is not a full model-size count; staging coverage is calculated from actual host masters instead.
