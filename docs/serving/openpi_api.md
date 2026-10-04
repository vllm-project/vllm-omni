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
  -> msgpack({"endpoint": "infer", "session_id": "...", "seed": <optional int>, ...observation})
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
- An optional integer `seed` in an inference message becomes the engine
  request's `sampling_params.seed`. A policy whose sampling consumes the
  request generator (GR00T-N1.7) returns the same action chunk for the same
  observation and seed. If omitted, the engine assigns a random per-request
  seed, so unseeded inferences are independent of each other. The API layer
  consumes `seed`; it is not forwarded inside the observation.

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

## Robot Policy Contract (Phase 1)

This contract implements the minimum serving boundary proposed in
[RFC #6069](https://github.com/vllm-project/vllm-omni/issues/6069). The optional
capability and output checks below refine that proposal; they do not require
existing policies to advertise new fields.

### Ownership and requests

Shared serving code owns transport, action discovery, and output validation.
Model pipelines own observation processing, prompts, action generation,
decoding, normalization, and session state. No universal VLA model class or
additional robot endpoint is required.

The semantic request consists of an instruction, camera observations, optional
robot state, optional session identity/reset, and model-specific extensions.
These describe the request rather than prescribe literal wire keys: a policy
may use `prompt`, `images`, or named observation fields. Serving continues to
forward the raw observation dictionary after consuming transport fields.
Reset follows the existing command/first-inference behavior described above;
this contract does not introduce a new reset field or session lifecycle.

### Static capabilities and dynamic metadata

`policy_server_config` is the connection-level capability advertisement. It
can declare the output layout and action semantics known for the loaded policy:

| Field | Meaning and validation when supplied |
| --- | --- |
| `action_horizon` | Fixed number of steps in each returned chunk; must equal actual H. Existing pi0 and GR00T declarations retain this meaning. |
| `max_action_horizon` | Optional upper bound on H; actual H may be smaller. A fixed horizon cannot exceed this bound. |
| `default_action_horizon` | Informational default; does not impose equality on an output. |
| `action_dim` | Final dimension D of a dense action array, after model postprocessing. |
| `action_keys` | Exact set of named action components for dictionary output; order is not significant. |
| `action_space` | Model-defined, non-empty semantic label. If repeated in output metadata, the labels must agree. |

`max_action_horizon` and `default_action_horizon` are optional extensions in this
Phase 1 contract, not required fields from the RFC. Missing shape capabilities
do not cause serving to invent a horizon or dimension. Configured capabilities
should agree with the loaded checkpoint; model loading remains responsible for
that check, as GR00T already does. Serving checks actual outputs against the
advertised constraints, but cannot establish that a configuration describes
the checkpoint correctly before inference.

Control interval/frequency may also be advertised as policy-specific
capabilities. Phase 1 does not require them, infer them from H, or define an
execution scheduler. Their units and semantics must be documented by the
policy. A generated chunk's horizon is distinct from how many steps a client
chooses to execute before observing and replanning.

Per-inference metadata describes the actual generated output:

| `metadata.actions` field | Meaning and validation when supplied |
| --- | --- |
| `horizon` | Actual returned H, a positive integer. |
| `action_dim` | Actual dense output D, a positive integer. |
| `valid_steps` | Number of valid steps from the start of the chunk; integer in [0, H]. Does not prescribe how many steps to execute. |
| `action_space` | Optional repeated semantic label, checked against the handshake. |

Scalar `action_dim` is defined only for dense actions. Named components can
have different dimensions; Phase 1 uses `action_keys` and leaves a per-key
dimension schema for future work. It does not interpret a scalar dimension as
the sum of named components. Metadata fields `raw_action_dim`, `action_mode`,
`domain_id`, and unknown model extensions remain preserved by the formatter.
In particular, a raw model dimension need not equal the final action dimension.

### Pipeline output and formatter boundary

A policy postprocessor can produce the existing diffusion envelope:

```python
{
    "payload": {"actions": actions},
    "metadata": {
        "actions": {
            "horizon": 4,
            "action_dim": 3,  # dense actions only
            "valid_steps": 4,
            "raw_action_dim": 32,
            "action_mode": "policy",
        }
    },
}
```

After formatter normalization, serving reads
`multimodal_output["actions"]` and optional
`multimodal_output["metadata"]["actions"]`. Legacy pipeline outputs such as
`DiffusionOutput(output={"actions": actions})` remain supported. Attach new
metadata through the envelope at the postprocess boundary, rather than assuming
a legacy payload dictionary has envelope semantics.

Dense chunks have layout `[H, D]` or `[B, H, D]`. Dictionary values use the same
layouts with independently sized D for each component. All named components
must share leading dimensions, including batch size when present. Serving
preserves batch dimensions: pi0 uses `[H, D]`, while GR00T retains `[B, H, D_k]`.
No automatic squeeze, flatten, or concatenation is performed. Historical dense
vectors remain accepted only when no explicit shape contract is supplied.

Serving converts action values to float32 as before, then rejects empty arrays,
empty dictionaries, non-finite values, unsupported layouts, inconsistent named
chunks, and contradictions with advertised capabilities or output metadata.
Optional integer fields reject booleans and floating-point values. Missing
metadata is accepted; supplied metadata groups must be mappings. Contract
failures use the existing generic inference error response.

### Compatibility and validation limits

The handshake and inference response remain unchanged:
`policy_server_config` first, then an ndarray or dictionary of ndarrays for each
inference. Output metadata is available internally for validation and in the
formatted engine result; the OpenPI response does not transmit it. Consequently,
clients cannot use dynamic `valid_steps` from this response, and serving neither
truncates nor executes a chunk. Policies requiring clients to consume dynamic
metadata need a separately negotiated protocol extension.

Phase 1 validation covers synthetic dense and named policy envelopes through
the formatter and OpenPI extraction, legacy outputs without metadata, and
invalid shape/metadata cases. These checks establish output consistency, not
policy accuracy, physical units, robot safety, closed-loop performance, or
model inference parity. Engine scheduling, offline endpoint integration, and
model-specific realtime backends remain outside this change.
