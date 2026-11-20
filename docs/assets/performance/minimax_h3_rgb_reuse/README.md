# MiniMax-H3 performance evidence

Measured source: `cef8d680fccbc38ce3901ec051a3f91ea1d2ecfa`.

## Matched service-window results

Each arm has two complete validated warmups and three formal requests. These CPU preparation calls were measured alongside the service window; they are separate from HTTP time and the earlier independent CPU benchmark.

| Arm | HTTP E2E mean ± SD | CPU preparation mean ± SD |
| --- | ---: | ---: |
| RGB reuse off before | 61.239 ± 0.244 s | 3.304 ± 0.030 s |
| RGB reuse on | 61.508 ± 0.097 s | 2.941 ± 0.011 s |
| RGB reuse off after | 61.991 ± 0.050 s | 3.301 ± 0.014 s |

Timestamp-interpolated HTTP control: **61.613 s** versus **61.508 s** with reuse on. Saving: 0.106 s (0.17%). The 95% bootstrap saving interval is **-0.044–0.248 s**; **no distinguishable E2E gain** was measured.

CPU preparation in the same window: **3.303 → 2.941 s (10.95% lower)**. Its 95% bootstrap saving interval is 0.344–0.381 s and exceeds the observed within-arm noise.

All 15 requests passed HTTP/media checks, full ffmpeg decoding and exact output-byte equality; warmup encoder activations also matched. These results do not erase the earlier cold-start failure documented below.

[Results and commands](results.json) include dependency versions, actual per-request HTTP status, output byte counts and SHA256, stage timers, executed-path counters, source revision and launch/request arguments. [Server excerpts](server.log.txt) preserve the numeric instrumentation rows and HTTP 200 access-log lines. Private prompt/media contents are represented by hashes; no credentials are exported.

Client E2E runs from sending the completed multipart HTTP request to reading the full MP4 response. Service startup, model loading and warmups are excluded. Stage values are auxiliary unsynchronized host wall times; the complete HTTP request is the performance authority. Tensor hashing runs only during warmups. All formal artifacts are byte-identical and completely decoded by ffmpeg.

A/B/A controls are interpolated at the optimized arm's mean request timestamp. Bootstrap intervals are recomputed from the raw request values (10,000 samples, seed42). An E2E gain is claimed only when its lower interval bound is positive and its magnitude exceeds the observed within-arm SD.

Launch commands in results.json replace only model and run-directory paths with $MODEL and $RUN. The run-local wrapper sets torch deterministic algorithms and disables encoder cuDNN SDP consistently; it adds passive counters/host timers without changing model source or adding device synchronization. The installed vLLM 0.33 dependency lacks an unrelated MiniCPM-o ngram blacklist API, so the wrapper skips that unrelated AR patch before importing the unchanged H3 diffusion pipeline.

Exact measurement wrapper sources: [server.py](server.py.txt), [observer.py](observer.py.txt) and [compatibility helper](compat_bootstrap.py.txt).

Request execution and validation: [controller](control.py.txt), [sender](request_sender.py.txt), [CPU preparation](cpu_prepare.py.txt), [environment setup](stage.sh.txt), [lifecycle](lifecycle.py.txt) and [statistics](analysis.py.txt).

This directory is an evidence archive. The original wrappers retain host-specific paths and require the private case01 inputs and scheduler job layout; they are not a portable benchmark entry point. The [measurement manifest](run-manifest.json) replaces model, data and run-directory paths with placeholders.

To recompute the published means, drift correction and bootstrap intervals directly from the archived public requests:

```bash
python reproduce_statistics.py.txt results.json
```

A [previous attempt](failed-attempt.json) stopped at the same-configuration media byte gate during the second warmup with RGB reuse disabled in both requests. CPU inputs and encoder activations matched, but decoded video/audio did not. It contributed no formal timing samples. Any passing retry is evidence for that measured window and does not explain or erase the earlier cold-start difference.

The independent CPU preparation A/B/A is retained separately in [CPU results](cpu-results.json) and [CPU manifest](cpu-manifest.json); it must not be mixed with the service-window CPU values above. `target_dimensions` describes the generated output target, while `prepared_reference_shapes` describes the measured reference input: 52 frames at height 1280 and width 704. The RGB sidecar is 140,574,720 bytes (134.1 MiB); 360 frames at the same canvas would require 928.1 MiB per reference.

The exact independent CPU driver is [cpu_only.py](cpu_only.py.txt). It reads the archived path-redacted [run manifest](run-manifest.json), with the same schema as the executed driver. Private inputs and local paths need to be supplied to rerun preparation.
