# OpenPI Robot Policy WebSocket API

Use `WS /v1/realtime/robot/openpi` for low-latency robot-policy inference with
OpenPI-compatible clients. A persistent WebSocket carries observations to the
loaded policy model and returns action arrays.

Despite the `/realtime` prefix, this endpoint does not use the OpenAI Realtime
event schema. It uses binary MessagePack frames with NumPy extensions.

## Availability

The endpoint is available only when the loaded diffusion policy configuration
contains `policy_server_config`. The server sends that model-specific
configuration to the client as the first binary frame.

For example, DreamZero declares:

```yaml
policy_server_config:
  image_resolution: [180, 320]
  n_external_cameras: 2
  needs_wrist_camera: true
  needs_stereo_camera: false
  needs_session_id: true
  action_space: joint_position
```

Without this configuration, the WebSocket returns `Robot policy not available`
and closes.

## DreamZero Quick Start

Start the bundled DreamZero deployment:

```bash
vllm serve GEAR-Dreams/DreamZero-DROID --omni --port 8091 \
  --served-model-name dreamzero-droid \
  --deploy-config vllm_omni/deploy/dreamzero_tp1_cfg2.yaml \
  --enforce-eager --disable-log-stats
```

Install the optional client dependencies, download the sample camera inputs,
and run the client:

```bash
pip install openpi-client websockets opencv-python

hf download YangshenDeng/vllm-omni-dreamzero-assets \
  --repo-type dataset \
  --local-dir outputs/dreamzero/assets

python examples/online_serving/dreamzero/openpi_client.py \
  --host 127.0.0.1 \
  --port 8091 \
  --video-dir outputs/dreamzero/assets
```

## Protocol

The connection is request-response after an initial server handshake:

```text
connect
  <- msgpack(policy_server_config)

infer
  -> msgpack({"endpoint": "infer", "session_id": "...", ...observation})
  <- msgpack(ndarray | dict[str, ndarray])

reset
  -> msgpack({"endpoint": "reset"})
  <- msgpack({"status": "reset successful"})
```

If `endpoint` is omitted, the server treats the message as `infer`. Each
inference message produces one action response after the engine request
completes; action tokens or intermediate tensors are not streamed.

NumPy arrays use the marker format implemented by `openpi-client`. The server
also accepts the legacy vLLM NumPy marker representation on input. JSON text
frames are not observation messages.

## Sessions and Reset

- The API layer's current-session and first-call counters are scoped to each
  WebSocket connection.
- `session_id` identifies model-side state across observations on that
  connection. If omitted, it defaults to `default`.
- The first inference for a session is sent to the model with `reset=true`.
- Changing `session_id`, or sending the `reset` command, causes the next
  inference to start with `reset=true`.
- The policy pipeline owns observation transforms and persistent model state,
  normally keyed by `session_id`; the API layer forwards the raw observation
  dictionary.

## Limits and Errors

- Maximum inbound payload size is 64 MiB.
- The server closes an idle connection after 30 seconds.
- Invalid binary input returns a MessagePack
  `{"type":"error","message":"Invalid request payload"}` response.
- Inference failures return a generic `Internal inference error` without
  exposing an internal traceback.
- Unsupported NumPy object, structured, and complex dtypes are rejected.

Observation keys, camera layout, state tensors, and action shapes are defined
by the loaded policy rather than by this transport. Read the handshake before
constructing observations. See the [DreamZero example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/dreamzero)
for a complete OpenPI client and DROID simulation loop.
