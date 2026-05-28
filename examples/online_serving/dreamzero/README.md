# DreamZero

DreamZero is the robot-policy serving example for the OpenPI-compatible `/v1/realtime/robot/openpi` endpoint.

## Supported checkpoints

| Model | HuggingFace repo |
|---|---|
| DreamZero-DROID | `GEAR-Dreams/DreamZero-DROID` |

## Quick start

### Start the server

```bash
bash examples/online_serving/dreamzero/run_server.sh
```

By default this launches:

```bash
vllm serve GEAR-Dreams/DreamZero-DROID --omni --port 8000 \
    --served-model-name dreamzero-droid \
    --deploy-config vllm_omni/deploy/dreamzero_tp1_cfg2.yaml \
    --enforce-eager --disable-log-stats
```

Override `MODEL`, `PORT`, `HOST`, `DEPLOY_CONFIG`, or `SERVED_MODEL_NAME` through the script environment if needed.

### Download example assets

The OpenPI client and DROID sim-eval example expect the three camera MP4 files in `outputs/dreamzero/assets`.

```bash
hf download YangshenDeng/vllm-omni-dreamzero-assets --repo-type dataset --local-dir outputs/dreamzero/assets
```

### Run the OpenPI client

```bash
python examples/online_serving/dreamzero/openpi_client.py \
    --host 127.0.0.1 \
    --port 8000 \
    --video-dir outputs/dreamzero/assets
```

This client uses downloaded example videos and talks to the OpenPI websocket server.

### Run DROID sim eval

```bash
${ISAACLAB_LAUNCHER} -p examples/online_serving/dreamzero/droid_sim_eval_client.py \
    --host 127.0.0.1 \
    --port 8000 \
    --scene 1 \
    --episodes 1 \
    --headless \
    --device cuda:0
```

Set `ISAACLAB_LAUNCHER=path/to/isaaclab.sh` from the vLLM-Omni repository root before running the command.
This launches Isaac Lab / sim-evals and runs the DROID benchmark loop against the same websocket endpoint.

### Export comparison videos

```bash
python examples/online_serving/dreamzero/export_prediction_video.py \
    --deploy-config vllm_omni/deploy/dreamzero_tp1_cfg2.yaml \
    --save-input-video \
    --save-gif
```

The export script writes the input rollout video and the predicted output video artifacts for side-by-side inspection.

### MolmoSpace demo

```bash
python examples/online_serving/dreamzero/molmospace_dreamzero_eval_demo.py \
    --host 127.0.0.1 \
    --port 8000 \
    --benchmark_dir /path/to/benchmark \
    --output_dir /path/to/output
```

This demo adapts DreamZero to the MolmoSpace-style remote policy eval loop.
