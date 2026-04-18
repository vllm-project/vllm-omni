# DreamZero Quick Start

This document is the shortest path to launching the DreamZero service and connecting the compatible client.

The commands below assume you run them from the repository root.

For the self-contained example, use the bundled client and videos under
`examples/online_serving/dreamzero/`.

Upstream DreamZero-dependent parity checks are optional and live under
`tests/dreamzero/upstream/`.

---

## 1. Start the vLLM DreamZero server

Default example: official HF model + `CF_P=2`.

```bash
ATTENTION_BACKEND=torch \
DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA \
CUDA_VISIBLE_DEVICES=0,1 \
MASTER_PORT=29628 \
vllm serve \
  GEAR-Dreams/DreamZero-DROID \
  --omni \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name dreamzero-droid \
  --cfg-parallel-size 2 \
  --enforce-eager
```

If you only have 1 GPU:

- change `CUDA_VISIBLE_DEVICES=0,1` to `CUDA_VISIBLE_DEVICES=0`
- remove `--cfg-parallel-size 2`

OpenPI WebSocket endpoint:

- `ws://127.0.0.1:8000/v1/realtime/robot/openpi`

---

## 2. Connect the client to the vLLM server

Use the self-contained DreamZero example client:

- `examples/online_serving/dreamzero/openpi_client.py`

When connecting to vLLM, the default websocket path already targets OpenPI:

```bash
python examples/online_serving/dreamzero/openpi_client.py \
  --host 127.0.0.1 \
  --port 8000
```

---

## 3. Standard online e2e test

The standard self-contained online serving e2e test is:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/e2e/online_serving/test_dreamzero.py -q
```

This test starts a real DreamZero server, sends bundled real camera videos, and
checks metadata, action output shape, finite values, and reset behavior.

---

## 4. Shared example test

The example test executes the same client script from `examples/`:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/examples/online_serving/test_dreamzero.py -q
```

---

## 5. Optional upstream parity baseline

The currently validated strict-parity baseline is:

- upstream DreamZero in eager mode
- no `torch.compile`
- no DiT cache / skip schedule
- `TP=1`
- `CF_P=1` or `CF_P=2`

Current status:

- `TP=1, CF_P=1`: strict parity
- `TP=1, CF_P=2`: strict parity
- `TP=2, CF_P=1/2`: runs, but strict numerical parity is not guaranteed

---

## 6. Recommended first run

If you want the least surprising setup, start with:

- `GEAR-Dreams/DreamZero-DROID`
- `--enforce-eager`
- `TP=1`
- `CF_P=1`

Then move to `CF_P=2` if you want CFG parallel.

---

## 7. Formal upstream end-to-end parity test

The formal server-vs-server parity test is:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/dreamzero/upstream/test_openpi_e2e_source_parity.py -q
```

Run the same parity test on GPUs `0,1` with `CF_P=2`:

```bash
OPENPI_E2E_GPUS=0,1 \
OPENPI_E2E_CFG_PARALLEL_SIZE=2 \
PYTHONPATH=. .venv/bin/python -m pytest tests/dreamzero/upstream/test_openpi_e2e_source_parity.py -q
```

This test checks:

- upstream DreamZero server
- vLLM DreamZero server
- the same DreamZero-compatible client logic
- strict action-output parity under the non-TP, non-compile baseline

---

## 8. Related docs

- `docs/models/dreamzero/README.md`: DreamZero documentation index
- `examples/online_serving/dreamzero/README.md`: bundled OpenPI example
