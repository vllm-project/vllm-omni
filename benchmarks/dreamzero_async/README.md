# DreamZero Async

Benchmarks for DreamZero W8 asynchronous control.

## Mock Latency Benchmark

`mock_benchmark.py` validates the control-loop metric math without a GPU or model weights. It compares:

- naive synchronous control: wait for one forward before every chunk
- W8-style async control: block for `A1`, then execute chunks while future work runs

```bash
python benchmarks/dreamzero_async/mock_benchmark.py \
    --num-chunks 6 \
    --forward-latency-s 0.9 \
    --action-horizon 24 \
    --control-hz 15 \
    --output-dir outputs/dreamzero_async/mock
```

Outputs:

- `client_events.jsonl`
- `server_events.jsonl`
- `summary.json`
- `result_table.md`

This benchmark is only a CPU smoke test for benchmark logic. Real W8 numbers must come from the websocket replay benchmark on the same model, config, and GPU setup as the synchronous baseline.

## Async Replay Artifacts

The OpenPI sync client and async replay client can write comparable live-server artifacts.

Sync baseline:

```bash
python examples/online_serving/dreamzero/openpi_client.py \
    --host 127.0.0.1 \
    --port 8000 \
    --video-dir outputs/dreamzero/assets \
    --num-chunks 2 \
    --output-dir outputs/dreamzero_async/sync_openpi
```

Async replay:

```bash
python examples/online_serving/dreamzero/async_client.py \
    --host 127.0.0.1 \
    --port 8000 \
    --video-dir outputs/dreamzero/assets \
    --num-chunks 2 \
    --output-dir outputs/dreamzero_async/replay
```

Outputs:

- `client_events.jsonl`
- `summary.json`
- `result_table.md`

Compare the two live replay summaries:

```bash
python benchmarks/dreamzero_async/compare_replays.py \
    --sync-summary outputs/dreamzero_async/sync_openpi/summary.json \
    --async-summary outputs/dreamzero_async/replay/summary.json \
    --output-dir outputs/dreamzero_async/compare
```

For repeated live benchmark pairs, use:

```bash
python benchmarks/dreamzero_async/live_benchmark.py \
    --host 127.0.0.1 \
    --port 8000 \
    --video-dir outputs/dreamzero/assets \
    --num-chunks 2 \
    --warmups 1 \
    --repeats 3 \
    --output-dir outputs/dreamzero_async/live_benchmark
```

`live_benchmark.py` defaults to `--order async-first`. That order intentionally runs the async endpoint before the sync OpenPI baseline, so BDE session cleanup regressions show up as a later sync failure. Use `--order sync-first` when measuring baseline first is more important than cleanup stress.

Use these artifacts for demo smoke data. A sync-vs-async speedup claim must run both paths on the same checkpoint, deploy config, GPU setup, and replay length, with async realtime control timing enabled.

`summary.json` records the async validity evidence:

- `deadline_miss_count`: number of chunk boundaries where the needed action chunk was not ready. In real-behavior replay, the client waits and this becomes measured idle time.
- `non_sim_conditioned_post_bootstrap_chunks`: executed chunks after `A1` that did not depend on the matching simulated observation.
- `server_error_count` and `underruns`.

If the configured `chunk_timeout_s` expires before the required action arrives, the async client raises immediately and the benchmark command fails.

`--require-valid-speedup` treats dropped work, server errors, wrong provenance, or execution-coverage mismatch as a failed benchmark proof, even if raw elapsed time is lower. Deadline misses remain visible in the artifact and table because they explain async idle time, but the eventual executed post-bootstrap action should still be the sim-conditioned action.

## PR-Style Suite Benchmark

`suite_benchmark.py` runs one or more named server variants, waits for `/health`, runs repeated sync-vs-async replay pairs for each variant, and writes a PR-style comparison table. This is the preferred benchmark driver when comparing W8 against server/runtime modes in the style of the DreamZero W1 performance PR.

Start from the example config:

```bash
cp benchmarks/dreamzero_async/suite_config.example.json /tmp/dreamzero_async_suite.json
```

Edit the variant commands and benchmark length in the copied config, then validate the plan:

```bash
.venv/bin/python benchmarks/dreamzero_async/suite_benchmark.py \
    --config /tmp/dreamzero_async_suite.json \
    --output-dir outputs/dreamzero_async/suite \
    --dry-run
```

Run the suite:

```bash
.venv/bin/python benchmarks/dreamzero_async/suite_benchmark.py \
    --config /tmp/dreamzero_async_suite.json \
    --output-dir outputs/dreamzero_async/suite
```

Artifacts:

- `plan.json`: resolved suite config plus environment metadata
- `summary.json`: machine-readable variant comparison
- `result_table.md`: PR-ready comparison table
- `<variant>/benchmark/`: per-run sync/async artifacts from `live_benchmark.py`
- `<variant>/logs/server.log`: server log for that variant

The example config defaults to 15 chunks, 1 warmup, and 3 measured repeats to match the scale of PR-style DreamZero performance reporting. It sets `realtime=true` and `chunk_timeout_s=10` so the async client behaves like a real controller: execute available chunks, send the next observation at the boundary, then wait idle if the next action has not arrived yet. It also sets `repeat_last_observation=true` so the small public DROID sample assets can drive a longer perf/soak run. Turn that off when using a real 15-chunk replay video set.
