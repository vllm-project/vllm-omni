# DreamZero OpenPI Example

This example shows how to serve DreamZero with `vllm serve --omni` and connect a
compatible OpenPI websocket client using bundled real camera videos.

## Files

- `run_server.sh`: launch DreamZero OpenPI serving
- `openpi_client.py`: websocket client that sends real observations
- `assets/`: minimal real camera videos used by the example

## Start the server

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
examples/online_serving/dreamzero/run_server.sh
```

If you only want 1 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
CFG_PARALLEL_SIZE=1 \
examples/online_serving/dreamzero/run_server.sh
```

The websocket endpoint is:

- `ws://127.0.0.1:8000/v1/realtime/robot/openpi`

## Run the client

From the repository root:

```bash
python examples/online_serving/dreamzero/openpi_client.py \
  --host 127.0.0.1 \
  --port 8000
```

The client sends:

- one initial single-frame observation
- one four-frame observation
- one websocket reset
- one post-reset single-frame observation

It validates:

- DreamZero metadata contract
- action tensor shape `(24, 8)`
- finite action values
- reset response

## Optional upstream parity checks

The upstream DreamZero-dependent parity tests are kept under:

- `tests/dreamzero/upstream/`

Those tests require a local upstream DreamZero checkout and are not needed for
the standard vLLM example above.
