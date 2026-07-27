# 20260727 Fused Gated Residual

- Decision: keep.
- Base Git SHA: `8dbb503e98119ea55d9cdfd5fcb1a817737d5fdd`.
- Scope: local proxy, not an official score.
- Hardware: one physical Ascend 910C card; Stage 2 uses logical chip 1.
- Change: replace the three `x + gate * branch` pairs in each NPU CosyVoice2
  DiT streaming block with `torch.addcmul`.
- Rollback: set `VLLM_OMNI_MINICPMO_FUSED_GATED_RESIDUAL=off`.

## Unprofiled A/B/A

The fixed workload used text input, text+audio output, seed 42, 128
Thinker/Talker tokens, two warmups, and 20 measured requests at C1 and C4. Each
phase used a clean server restart; all 120 requests passed.

| Concurrency | Metric | A1 | Candidate | A2 | Candidate vs A mean |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | first audio p50 (s) | 1.1404 | 1.1417 | 1.1771 | +1.49% |
| 1 | E2E p50 (s) | 2.2243 | 2.2371 | 2.3864 | +3.05% |
| 1 | audio seconds/s | 4.5899 | 4.4917 | 4.2399 | +1.74% |
| 4 | first audio p50 (s) | 2.0108 | 2.0175 | 2.0548 | +0.76% |
| 4 | E2E p50 (s) | 4.0877 | 4.0388 | 4.0894 | +1.23% |
| 4 | audio seconds/s | 9.8283 | 9.9297 | 9.8563 | +0.89% |

C4 is the strongest attribution: baseline throughput differs by 0.28% and E2E
p50 by 0.04%, while the candidate improves their mean by 0.89% and 1.23%.

## Effect Proxy

Every output remained valid non-empty 24 kHz PCM. At C4, candidate/baseline
waveform correlation has median 1.0 and minimum 0.99999983, matching the two
baseline runs; 15 of 20 files are byte-identical in both comparisons. The
official effect threshold is still `UNRESOLVED`.

## Profile Evidence

Matching Stage-2-only captures used one request after two unprofiled warmups.
The comparison signature passed.

| Diagnostic | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Kernel/operator calls | 97,957 | 94,597 | -3,360 (-3.43%) |
| Add calls | 11,968 | 8,608 | -3,360 |
| Mul calls | 9,126 | 5,766 | -3,360 |
| AxpyV2 calls | 0 | 3,360 | +3,360 |
| Aggregate kernel time (ms) | 1,009.3 | 1,015.9 | +0.66% |

The exact call-count change confirms the intended mechanism: each of the 1,120
DiT block executions replaces three Mul+Add pairs with three fused operations.
Profile time is diagnostic and not used as score evidence; the device-time
increase reinforces that the unprofiled gain comes from lower launch/dispatch
work rather than faster aggregate kernels.

## Validation

- Fresh-service smoke: text-only, text+audio, image+audio, audio+audio, and
  video+audio all passed.
- Correctness gate: passed with no failures.
- Focused tests: 36 passed.
- Ruff check/format and `git diff --check`: passed.
- Service cleanup: no residual NPU process.

Artifact SHA256 values:

- Baseline profile analysis: `a4d7fc4fe3be430e667a34c4941dd54d1dcdf068e382fc3f279b4753ad766df0`
- Candidate profile analysis: `ff6144ea4dd636c8d620bd5a40b539062439a37754963a3cdca04935c1537f83`
- Profile comparison: `3abd205d2646e88afedb46468a1269b9d9e9c5c7850c377976c1919eeec36901`
- Candidate benchmark: `fab1508ce33c19be564a5471ec714ba1d28773b87b8fb12966c8552d64c5564b`
- Full smoke: `4078465d52db8ff86e3158531c20be89cef9caaa3ebc599c7433f4790f1478d9`
- Correctness gate: `aacff4a0992bc7f57634477ca6d2ec7754bd78aafce3e6e6836165abe9f097cf`
