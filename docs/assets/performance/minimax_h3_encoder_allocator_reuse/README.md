# MiniMax-H3 encoder allocator reuse repair

Actual execution snapshots: baseline `7602d7980d8add4277353e8bc162ac1f98d98c19`, local candidate `b8918a7465491d3a2fb1c00dafc6ae18a9157a20`. The baseline PR head is `4f68b8047569ae1100e525c326d1fd09312edfa9` and differs from the measured baseline only in documentation. Candidate source is archived in [encoder-fixed.py.txt](encoder-fixed.py.txt), [code-fix.patch.txt](code-fix.patch.txt), [actual commit object](measured-candidate-commit.txt) and [source identity](source-identity.json). Delivery must match this production-file fingerprint; a later delivery commit must not be substituted for the actual measured snapshot.

Hardware/workload: 8 × Ascend 910B4, native layerwise offload, BF16, encoder TP8, DiT USP8, replicated vision, FLASH_ATTN, VAE tile8, case01, 480×864, 294 frames, 24 FPS, seed42 and NFE8. The matched A/B/A compares **unfixed packed / fixed packed / unfixed packed**; only allocator retention changes. Each arm has two complete validated warmups and five formal requests. This is a cache-repair comparison, not a combined off-versus-packed benchmark.

| Arm | Completed encoder mean ± SD | HTTP E2E mean ± SD |
| --- | ---: | ---: |
| Unfixed packed before | 0.7328 ± 0.0285 s | 62.158 ± 0.291 s |
| Fixed packed | 0.6008 ± 0.0060 s | 62.191 ± 0.261 s |
| Unfixed packed after | 0.7345 ± 0.0194 s | 62.347 ± 0.224 s |

Completed encoder: **0.7336 → 0.6008 s**, saving **132.81 ms (18.10%)**. The 95% circular length-two block-bootstrap saving interval is **122.96–142.26 ms** (10,000 samples, seed42). The primary boundary includes existing production synchronization; no new device synchronization was added by measurement.

HTTP E2E: **62.253 → 62.191 s**, saving 61.22 ms. Its 95% saving interval is **-0.195–0.313 s**, so no distinguishable E2E gain is claimed. One matched window cannot establish a universal gain or reproduce historical 1.5-second/3.37% claims.

All 21 MP4 outputs are byte-identical and fully decoded. Warmup encoder activations and H2D copy traces pass exact checks. Live GPU weight tensors are released in every arm; this fix retains **free allocator blocks**, not live weights. H2D weight copies still occur.

In the unfixed controls, encoder offload calls `empty_cache` once per request; fixed packed calls it zero times in this window. Rank0 mean cache-release host cost: 100.28 ms before versus 1.67 ms fixed. Rank0 mean cached allocator bytes after release: 61919744 before versus 1487983104 fixed. All ranks and before/after allocated, reserved and physical-free bytes are in [raw results](cache-results.json).

The existing `BoundedAllocatorCache` is component-owned. It retains cache only while cached-but-unallocated bytes do not exceed 25% of device capacity and at least 5% is physically free. Memory pressure, unavailable telemetry and forced failure cleanup still empty the cache. Existing DLO policy is preserved; default `off` is unchanged. Executor shutdown cleanup is unchanged. CPU regression: 62 passed, including the new policy-selection test that fails on the unfixed code. CPU tests bypass framework package/logger/platform bootstrap and execute the existing CPU fixtures; they are not a device substitute.

[Manifest and commands](run-manifest.json), [numeric logs](server.log.txt), and original measurement wrappers are an evidence archive with host-specific dependencies. Private prompt/media contents are represented by hashes. Memory telemetry adds equal observation points in all arms; cache-release timings are host wall times, not DMA timings.

Recalculate public metrics and memory-policy gates using Python standard library only:

```bash
python reproduce_cache.py.txt cache-results.json
```

Apply the archived production patch to the recorded baseline with `git apply --unidiff-zero code-fix.patch.txt`. Its zero-context format preserves the exact code change without whitespace-only context lines.
