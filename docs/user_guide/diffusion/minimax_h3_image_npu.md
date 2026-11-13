# MiniMax-H3-Image on Ascend

MiniMax-H3-Image enables Ascend VAE graphs, batch32 decoding, FP16 batch8
encoding and scoped deterministic execution by default. Host-master AdaLN
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
| `npu_disable_internal_format` | `false` | Disable NPU internal formats even with VAE graphs off, for matched integration controls. This is worker-wide. |
| `npu_disable_jit_compile` | `false` | Disable NPU JIT compilation even with VAE graphs off, for matched integration controls. This is worker-wide. |
| `npu_vae_tile_graph` | `true` | Replay VAE tile kernels using NPU graphs. Set false for native VAE execution. |
| `npu_vae_tile_streams` | `1` | Independent decoder graph streams, from 1 to 8. |
| `npu_vae_tile_batch` | `32` (`1` with multiple streams) | Decode tiles in batches of 1, 2, 4, 8, 16 or 32. |
| `npu_vae_encoder_precision` | `fp16` | Encoder graph autocast: `fp16` or `bf16`; `fp32` retains the native encoder. |
| `npu_vae_encoder_batch` | `8` (`1` with FP32) | Encode tiles in batches of 1, 2, 4, 8 or 16; batching requires reduced encoder precision. |

`--enforce-eager` or disabling graph decode disables these VAE graph paths.
Decoder batching and multiple decoder streams cannot be enabled together.
An explicit `npu_vae_tile_streams=4` selects single-tile multi-stream decoding
unless a decoder batch is also specified. Explicitly requesting both a batch
greater than one and multiple streams is an error. Likewise, selecting FP32
defaults the encoder batch to one; explicitly requesting a larger FP32 batch
is an error.
Single-tile graph replay retains the eager kernels. Tile batching and reduced
encoder precision are **approximate and enabled by default**: outputs can differ
from the previous native VAE defaults. Validate final images for your workload,
including editing references. Component parity
alone is insufficient to establish final-image quality.

To use four decoder streams with the native FP32 encoder instead of batching:

```json
{
  "npu_vae_tile_streams": 4,
  "npu_vae_encoder_precision": "fp32"
}
```

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

Use `TASK_QUEUE_ENABLE=1` (or `0`) with NPU graphs. The tested torch-npu
runtime rejects graph capture with `TASK_QUEUE_ENABLE=2`.

The graph adapter also disables NPU internal formats and JIT compilation for
the worker. A graph-off control must set `npu_disable_internal_format=true`
and `npu_disable_jit_compile=true` to match those runtime settings before
comparing VAE execution; toggling only the graph flag compares coupled runtime changes.
Historical graph-on/off measurements without this match do not isolate the
VAE's contribution.

For a same-layout control, set `npu_vae_tile_graph=false` and
`npu_disable_internal_format=true` and `npu_disable_jit_compile=true` while retaining the
same parallelism, AdaLN offload and deterministic settings. Measure complete
HTTP response receipt, with the same warmup and repetition policy, separately
from profiling. No cross-hardware performance or quality guarantee is implied.
