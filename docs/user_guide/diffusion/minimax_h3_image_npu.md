# MiniMax-H3-Image on Ascend

MiniMax-H3-Image enables Ascend VAE graphs and scoped deterministic execution
by default when the graph runtime is compatible. The default VAE retains
single-tile decoding and the native FP32 encoder; approximate batching and
reduced-precision encoding require explicit configuration. Host-master AdaLN
offload also defaults on for the unquantized image pipeline when the AdaLN
cache is enabled and DiT offload/HSDP is not selected. These defaults apply
only to the NPU image pipeline; other platforms and pipelines are unchanged.
Keep the native six denoising steps, canvas, prompts and output transport when
comparing latency. Graph capture and the first request are warmup costs.

## Four-device serving

Select four exclusively allocated devices in your deployment environment and
source the installed CANN environment before starting the server. The defaults
include the host-master AdaLN stager for resident Ulysses deployment:

```bash
HCCL_DETERMINISTIC=true TASK_QUEUE_ENABLE=1 vllm serve "${MODEL_PATH:?Set the model directory}" \
  --omni --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --num-gpus 4 --tensor-parallel-size 1 --usp 4 \
  --text-encoder-tp-size 4
```

AdaLN offload preserves the existing projection-cache and staging ownership.
Host memory must hold the offloaded weights. This is distinct from approximate
cross-step activation reuse. The deterministic option enables PyTorch's
process-wide deterministic algorithms during each forward and restores the caller's
previous enabled/warn-only settings on return or exception. Set `HCCL_DETERMINISTIC=true` separately
before starting the worker processes.
To keep AdaLN weights resident, pass
`--cache-config '{"minimax_h3_adaln_offload":false}'`. Disabling AdaLN offload
does not disable the projection cache. Explicitly enabling it with quantization,
DiT offload or HSDP remains an error.

## VAE options

Pass these options through `--additional-config`:

| Option | Default | Behavior |
| --- | --- | --- |
| `npu_deterministic` | `true` | Enable deterministic PyTorch algorithms during each forward; restore caller settings afterward. |
| `npu_disable_internal_format` | `false` | Disable NPU internal formats even with VAE graphs off, for matched integration controls. Scoped to forward, including the DiT. |
| `npu_disable_jit_compile` | `false` | Disable NPU JIT compilation even with VAE graphs off, for matched integration controls. Scoped to forward, including the DiT. |
| `npu_vae_tile_graph` | `true` | Replay VAE tile kernels using NPU graphs. Set false for native VAE execution. |
| `npu_vae_tile_streams` | `1` | Independent decoder graph streams, from 1 to 8. |
| `npu_vae_tile_batch` | `1` | Decode tiles in batches of 1, 2, 4, 8, 16 or 32. |
| `npu_vae_encoder_precision` | `fp32` | Encoder graph autocast: `fp16` or `bf16`; `fp32` retains the native encoder. |
| `npu_vae_encoder_batch` | `1` | Encode tiles in batches of 1, 2, 4, 8 or 16; batching requires reduced encoder precision. |

`--enforce-eager` or disabling graph decode disables these VAE graph paths.
Decoder batching and multiple decoder streams cannot be enabled together.
`npu_vae_tile_streams=4` selects single-tile multi-stream decoding. Explicitly requesting both a batch
greater than one and multiple streams is an error. Explicitly requesting a larger FP32 encoder batch
is an error.
Single-tile graph replay retains the eager kernels. Tile batching and reduced
encoder precision are **approximate and opt-in**: outputs can differ
from the previous native VAE defaults. Validate final images for your workload,
including editing references. Component parity
alone is insufficient to establish final-image quality.

To use four decoder streams with the native FP32 encoder:

```json
{
  "npu_vae_tile_streams": 4,
  "npu_vae_encoder_precision": "fp32"
}
```

To opt into the approximate batch32 decoder and FP16 batch8 encoder:

```json
{
  "npu_vae_tile_batch": 32,
  "npu_vae_encoder_precision": "fp16",
  "npu_vae_encoder_batch": 8
}
```

The earlier batch32/FP16 performance observations apply to this explicit profile,
not the current single-tile/FP32 defaults. Broader quality and memory acceptance
is required before promoting the approximate profile to a default.

To restore the previous native VAE, resident AdaLN weights and caller-controlled
PyTorch determinism, append:

```bash
--cache-config '{"minimax_h3_adaln_offload":false}' \
--additional-config '{"npu_vae_tile_graph":false,"npu_deterministic":false}'
```

Graphs retain static buffers and private stream workspaces. Larger batches use
more memory; choose smaller batches if the model and graph pools do not fit.
Weight relocation invalidates cached graphs. Shape caches are bounded; shapes
outside their limits use the corresponding eager computation. A failed
multi-stream capture is fatal to that stage and requires a restart.

Use `TASK_QUEUE_ENABLE=1` (or `0`) with NPU graphs and leave
`ASCEND_LAUNCH_BLOCKING` unset or set to `0`. With incompatible settings or missing
NPUGraph APIs, implicit graph enablement logs a warning and keeps the native VAE.
Explicit `npu_vae_tile_graph=true` rejects these conditions during adapter setup,
before any capture or runtime-setting change. Eager/graph-disable overrides take
precedence. Set environment variables before starting the worker; changing them
after the NPU runtime has initialized is unsupported.

During each forward, the graph adapter disables NPU internal formats and JIT
compilation for the worker, including the DiT. It restores their previous effective
values on return or exception, including when determinism is opted out. Adapter
setup does not change these process-wide settings, so failed initialization and
inline engine teardown leave no adapter-owned global setting behind. A graph-off control must set `npu_disable_internal_format=true`
and `npu_disable_jit_compile=true` to match those runtime settings before
comparing VAE execution; toggling only the graph flag compares coupled runtime changes.
Historical graph-on/off measurements without this match do not isolate the
VAE's contribution.

For a same-layout control, set `npu_vae_tile_graph=false` and
`npu_disable_internal_format=true` and `npu_disable_jit_compile=true` while retaining the
same parallelism, AdaLN offload and deterministic settings. Measure complete
HTTP response receipt, with the same warmup and repetition policy, separately
from profiling. No cross-hardware performance or quality guarantee is implied.
