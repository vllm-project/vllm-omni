# MiniMax-H3 performance evidence

Measured source: `7602d7980d8add4277353e8bc162ac1f98d98c19`.

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
