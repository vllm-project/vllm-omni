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

### Execution modes

DreamZero can serve the same OpenPI endpoint and request format in three ways.
The client command and request schema are identical across all three — only
`--deploy-config` changes:

* **Monolithic** — encode, denoise, and decode run in one process on one
  device. This is the default and the compatibility path documented above.
* **Disaggregated** — encode, denoise, and decode run as three independent
  stages, each on its own device.
* **Disaggregated with TP=4 denoise** — same three stages, but the denoise
  stage is sharded across 4 devices with tensor parallelism.

See [Run monolithic DreamZero](#run-monolithic-dreamzero),
[Run disaggregated DreamZero](#run-disaggregated-dreamzero), and
[Run disaggregated DreamZero with TP=4 denoise](#run-disaggregated-dreamzero-with-tp4-denoise)
below.

In the disaggregated modes, the final stage runs DreamZero's lightweight
decode/postprocess path (action denormalization + normalized latent
passthrough). It does not currently perform a standalone full VAE-to-RGB
decode.

### Install example dependencies

The core `pip install -e .` setup does not include the extra packages used by the DreamZero example scripts.

- `openpi_client.py`:
  `openpi-client`, `websockets`, `opencv-python`
- `droid_sim_eval_client.py`:
  `mediapy`, `websockets`, `openpi-client`

The DROID sim-eval script also needs an Isaac Lab environment that provides `isaaclab`, `isaaclab_tasks`, `sim_evals`, and `gymnasium`.

If you run the DROID client on Python < 3.12, also install `typing-extensions`.

### Configure TP and CFG parallelism

The bundled DreamZero configs are:

| Config | Purpose |
|---|---|
| `vllm_omni/deploy/dreamzero.yaml` | Monolithic DreamZero, TP=1, CFG parallel disabled |
| `vllm_omni/deploy/dreamzero_tp1_cfg2.yaml` | Monolithic DreamZero, TP=1, CFG parallel size=2 |
| `vllm_omni/deploy/dreamzero_disaggregated.yaml` | Three-stage DreamZero deployment with one device per stage |
| `vllm_omni/deploy/dreamzero_disaggregated_tp4denoise.yaml` | Three-stage deployment with TP=4 on the denoise stage |

For other monolithic topologies, use CLI parallelism flags and update stage 0 `devices` with `--stage-overrides`. The number of listed devices must match `tensor_parallel_size * cfg_parallel_size`.

TP=2 with CFG parallel disabled:

```bash
vllm serve GEAR-Dreams/DreamZero-DROID --omni --port 8000 \
    --served-model-name dreamzero-droid \
    --deploy-config vllm_omni/deploy/dreamzero.yaml \
    --tensor-parallel-size 2 \
    --stage-overrides '{"0": {"devices": "0,1"}}' \
    --enforce-eager --disable-log-stats
```

TP=2 with CFG parallel size=2:

```bash
vllm serve GEAR-Dreams/DreamZero-DROID --omni --port 8000 \
    --served-model-name dreamzero-droid \
    --deploy-config vllm_omni/deploy/dreamzero.yaml \
    --tensor-parallel-size 2 \
    --cfg-parallel-size 2 \
    --stage-overrides '{"0": {"devices": "0,1,2,3"}}' \
    --enforce-eager --disable-log-stats
```

### Download example assets

The OpenPI client and DROID sim-eval example expect the three camera MP4 files in `outputs/dreamzero/assets`.

```bash
hf download YangshenDeng/vllm-omni-dreamzero-assets --repo-type dataset --local-dir outputs/dreamzero/assets
```

### Run monolithic DreamZero

```bash
vllm serve GEAR-Dreams/DreamZero-DROID \
  --omni \
  --port 8000 \
  --served-model-name dreamzero-droid \
  --deploy-config vllm_omni/deploy/dreamzero.yaml \
  --enforce-eager \
  --disable-log-stats
```

This runs the complete DreamZero pipeline in one monolithic stage.

### Run disaggregated DreamZero

Device placement:

```text
device 0: encode
device 1: denoise
device 2: decode/postprocess
```

```bash
vllm serve GEAR-Dreams/DreamZero-DROID \
  --omni \
  --port 8000 \
  --served-model-name dreamzero-droid \
  --deploy-config vllm_omni/deploy/dreamzero_disaggregated.yaml \
  --enforce-eager \
  --disable-log-stats
```

The deployment mode changes, but the OpenPI endpoint and request schema
remain unchanged.

The final stage performs DreamZero's lightweight decode/postprocess path and
returns actions plus normalized latent output. It does not currently perform
a standalone full VAE-to-RGB decode.

### Run disaggregated DreamZero with TP=4 denoise

Device placement (6 visible devices required):

```text
device 0: encode
devices 1-4: denoise with tensor parallel size 4
device 5: decode/postprocess
```

```bash
vllm serve GEAR-Dreams/DreamZero-DROID \
  --omni \
  --port 8000 \
  --served-model-name dreamzero-droid \
  --deploy-config vllm_omni/deploy/dreamzero_disaggregated_tp4denoise.yaml \
  --enforce-eager \
  --disable-log-stats
```

### Run the OpenPI client

```bash
python examples/online_serving/dreamzero/openpi_client.py \
    --host 127.0.0.1 \
    --port 8000 \
    --video-dir outputs/dreamzero/assets
```

This client uses downloaded example videos and talks to the OpenPI websocket
server. Use the same client command for monolithic and disaggregated
deployments — to compare the two modes, stop the current server, restart it
with the other deploy config, and run the same client input again.

### Compare monolithic and disaggregated execution

1. Start the monolithic server with `dreamzero.yaml`.
2. Run the OpenPI client and record latency, output actions, and memory usage.
3. Stop the monolithic server.
4. Start the disaggregated server with `dreamzero_disaggregated.yaml` or
   `dreamzero_disaggregated_tp4denoise.yaml`.
5. Run the same client with the same assets and prompt.
6. Compare the results.

For a valid comparison, keep the following fixed unless it is the variable
being tested: input videos, prompt, session behavior, model checkpoint,
inference-step configuration, and parallelism configuration. Compare
numerical behavior and end-to-end outputs under the same request conditions;
this is not a claim of byte-identical output.

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

### Export comparison videos offline

```bash
python examples/offline_inference/dreamzero/export_prediction_video.py \
    --deploy-config vllm_omni/deploy/dreamzero_tp1_cfg2.yaml \
    --save-input-video \
    --save-gif
```

The export script uses local `Omni` inference, not the websocket server. It writes the input rollout video and the predicted output video artifacts for side-by-side inspection.

### MolmoSpace demo

```bash
python examples/online_serving/dreamzero/molmospace_dreamzero_eval_demo.py \
    --host 127.0.0.1 \
    --port 8000 \
    --benchmark_dir /path/to/benchmark \
    --output_dir /path/to/output
```

This demo adapts DreamZero to the MolmoSpace-style remote policy eval loop.
