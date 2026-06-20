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

Use these artifacts for demo smoke data. A final sync-vs-async benchmark should run both paths on the same checkpoint, deploy config, GPU setup, and replay length.
