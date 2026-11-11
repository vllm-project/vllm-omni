# MiniMax-H3-Image on Ascend

MiniMax-H3-Image has opt-in Ascend VAE tile graphs and tile batching. These
settings apply only to the image pipeline; they do not change other models.
Keep the native six denoising steps, canvas, prompts and output transport when
comparing latency. Graph capture and the first request are warmup costs.

## Four-device serving

Select four exclusively allocated devices in your deployment environment and
source the installed CANN environment before starting the server. For a
memory-constrained resident Ulysses deployment, the existing host-master AdaLN
stager can also run on NPU:

```bash
HCCL_DETERMINISTIC=true vllm serve "${MODEL_PATH:?Set the model directory}" \
  --omni --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --num-gpus 4 --tensor-parallel-size 1 --usp 4 \
  --text-encoder-tp-size 4 \
  --cache-config '{"minimax_h3_adaln_offload":true}' \
  --additional-config '{"text_encoder_cudnn_sdp":false,"npu_deterministic":true,"npu_vae_tile_graph":true,"npu_vae_tile_streams":4}'
```

AdaLN offload preserves the existing projection-cache and staging ownership.
Host memory must hold the offloaded weights. This is distinct from approximate
cross-step activation reuse. The deterministic option enables PyTorch's
process-wide deterministic algorithms; set `HCCL_DETERMINISTIC=true` separately
before starting the worker processes.

## VAE options

Pass these options through `--additional-config`:

| Option | Default | Behavior |
| --- | --- | --- |
| `npu_deterministic` | `false` | Enable deterministic PyTorch algorithms in the worker. |
| `npu_vae_tile_graph` | `false` | Replay VAE tile kernels using NPU graphs. |
| `npu_vae_tile_streams` | `1` | Independent decoder graph streams, from 1 to 8. |
| `npu_vae_tile_batch` | `1` | Decode tiles in batches of 1, 2, 4, 8, 16 or 32. |
| `npu_vae_encoder_precision` | `fp32` | Optional encoder graph autocast: `fp16` or `bf16`. |
| `npu_vae_encoder_batch` | `1` | Encode tiles in batches of 1, 2, 4, 8 or 16; batching requires reduced encoder precision. |

`--enforce-eager` or disabling graph decode disables these VAE graph paths.
Decoder batching and multiple decoder streams cannot be enabled together.
Single-tile graph replay retains the eager kernels. Tile batching and reduced
encoder precision are **approximate options**: validate final images for your
workload, including editing references, before selecting them. Component parity
alone is insufficient to establish final-image quality.

For example, a separately evaluated approximate configuration is:

```json
{
  "text_encoder_cudnn_sdp": false,
  "npu_deterministic": true,
  "npu_vae_tile_graph": true,
  "npu_vae_tile_batch": 32,
  "npu_vae_encoder_precision": "fp16",
  "npu_vae_encoder_batch": 8
}
```

Graphs retain static buffers and private stream workspaces. Larger batches use
more memory; choose smaller batches if the model and graph pools do not fit.
Weight relocation invalidates cached graphs. Shape caches are bounded; shapes
outside their limits use the corresponding eager computation. A failed
multi-stream capture is fatal to that stage and requires a restart.

For a same-layout control, set `npu_vae_tile_graph=false` while retaining the
same parallelism, AdaLN offload and deterministic settings. Measure complete
HTTP response receipt, with the same warmup and repetition policy, separately
from profiling. No cross-hardware performance or quality guarantee is implied.
