# MiniMax-H3 RGB reuse performance evidence

Measured code: `cef8d680fccbc38ce3901ec051a3f91ea1d2ecfa`. This is CPU input preparation for ref2va-v13-avatar12s/case01, not a whole-model E2E measurement. Each A/B/A arm has two complete warmups and three formal calls. Only `reference_video_decode_reuse` changes. All 15 VAE inputs, Qwen samples, audio arrays and timestamps match byte-for-byte.

| CPU preparation arm | Mean ± SD |
| --- | ---: |
| off before | 2.965 ± 0.102 s |
| on | 2.554 ± 0.039 s |
| off after | 3.175 ± 0.294 s |

At the on arm's mean timestamp, interpolated control is 3.070 s versus 2.554 s, 16.81% lower. The 95% bootstrap saving interval is 0.379–0.669 s (10,000 samples, seed42). Saving exceeds the largest within-arm SD and the lower interval bound is positive.

[Raw results and dependencies](cpu-results.json), [input hashes and command](cpu-manifest.json), [original driver](cpu_driver.py.txt), [preparation code](cpu_prepare.py.txt) and [compatibility helper](compat_bootstrap.py.txt). This is an evidence archive: original wrappers retain host-specific paths and private inputs are represented by hashes.

Recompute the statistics and all input signatures using only the public records:

```bash
python reproduce_cpu_statistics.py.txt cpu-results.json
```

No accepted E2E timing is available. The [service attempt](failed-attempt.json) failed the same-configuration media byte gate with reuse disabled in both warmups. CPU inputs and encoder activations matched, but decoded video/audio did not; no formal timing samples were collected. Its cause is unresolved and cannot be attributed to toggling RGB reuse.
