# Diffusion Execution Modes

vLLM-Omni supports complete-request execution and step-wise diffusion
execution. Both modes use the same asynchronous engine output stream; the
configuration controls how work is scheduled and whether intermediate outputs
are exposed.

## Choose a Mode

| Goal | CLI configuration |
|---|---|
| Serial request execution | `--max-num-seqs 1` |
| Fused request-level batching | `--max-num-seqs N` |
| Single-request step execution | `--step-execution --max-num-seqs 1` |
| Step-wise continuous batching | `--step-execution --max-num-seqs N` |
| Chunked diffusion output | `--diffusion-streaming-output` |

`N` must be greater than `1` to allow batching. The selected pipeline must
support request-level batching or batched-step execution as appropriate;
single-request step support alone is not sufficient for step-wise continuous
batching.

## Request Execution

Request execution is the default when `--step-execution` is omitted. The
pipeline performs a complete `forward()` for each scheduler wave.

### Serial Requests

Use the conservative serial path with any request-mode pipeline:

```bash
vllm serve MODEL --omni \
  --port 8091 \
  --max-num-seqs 1
```

### Request-Level Batching

Set `max_num_seqs` above one to combine compatible independent requests into a
single pipeline forward:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --max-num-seqs 4
```

For bursty traffic, add a small admission window:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --max-num-seqs 4 \
  --request-batch-max-wait-ms 20
```

`--request-batch-max-wait-ms 0` is the default. A nonzero value can improve
batch formation but adds up to that much latency before a new scheduler wave.
It has no effect in step mode.

Each prompt remains a separate logical request. Do not submit a top-level list
as one packed prompt. Concurrent serving requests are batched internally when
their shapes, guidance settings, output counts, and LoRA settings are
compatible.

Only pipelines that declare request-batch support accept
`max_num_seqs>1` in request mode. Unsupported pipelines fail during engine
initialization; use `max_num_seqs=1` instead.

## Step Execution

Step execution lets the scheduler advance and abort requests between denoise
steps:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 1
```

Set a larger capacity to allow compatible requests to share denoise waves:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 8
```

Step continuous batching is experimental. Start with `max_num_seqs=1` when
validating a model or debugging correctness, then increase it for
multi-request throughput.

Step execution is capability-based, not a generic switch for every diffusion
model. Qwen-Image supports step-wise continuous batching. HunyuanImage3 also
supports it, but only when its resolved self-attention backend is `TORCH_SDPA`;
set `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` or configure
`diffusion_attention_config.default.backend=TORCH_SDPA` before using
`--max-num-seqs >1`. See the
[HunyuanImage-3.0 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Tencent/HunyuanImage-3.0-Instruct.md)
for its validated configuration. Helios supports single-request step execution only: use
`--step-execution --max-num-seqs 1` for Helios. MiniMax H3 supports step-wise
continuous batching by packing co-batched requests into one sequence that keeps
a separate attention document per request; that layout needs a backend which
honors the packed `cu_seqlens` metadata, so run it with
`--diffusion-attention-backend FLASH_ATTN` (other backends stay correct but
fall back to one transformer forward per request). Batching H3 does not improve
its throughput — see the measured numbers in the
[MiniMax-H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
— so keep `--max-num-seqs 1` unless you need step-level scheduling. Consult the
selected pipeline's documentation and source for the latest support status.

## Streaming Output

Use `--diffusion-streaming-output` for a pipeline that can produce intermediate
diffusion outputs:

```bash
vllm serve BestWishYsh/Helios-Distilled --omni \
  --port 8000 \
  --diffusion-streaming-output
```

Streaming output requires step execution. If
`--diffusion-streaming-output` is set without `--step-execution`, the engine
enables step execution automatically. Model initialization fails if the
pipeline does not implement step execution.

Chunk-capable pipelines emit intermediate and final outputs through the same
request stream. Final-only step pipelines emit only the final result even when
streaming output is enabled.

The normal non-streaming serving path uses the same internal stream but drains
it and returns only the final output. Users do not need to enable
`--diffusion-streaming-output` to benefit from the unified engine lifecycle.

## Send Requests

Execution-mode flags configure the server. They do not change the
OpenAI-compatible client request format. For example, after starting any of
the non-streaming configurations above:

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat sitting on a windowsill",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > cat.png
```

To create a request-level batch, send multiple independent requests
concurrently. Do not put several prompts into one `prompt` field:

```bash
for prompt in "a red fox" "a blue bird" "a green frog"; do
  curl -s -X POST http://localhost:8091/v1/images/generations \
    -H "Content-Type: application/json" \
    -d "{\"prompt\":\"${prompt}\",\"size\":\"1024x1024\",\"seed\":42}" \
    > /dev/null &
done
wait
```

The scheduler may batch these requests when their sampling parameters are
compatible. See the
[Image Generation API](../../serving/image_generation_api.md) for response
formats and additional client examples. Streaming models can use
model-specific streaming endpoints documented by their serving guide.

## Python API

The Python arguments mirror the CLI flags:

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image",
    max_num_seqs=4,
    request_batch_max_wait_ms=20.0,
)

outputs = omni.generate(
    [
        "a cup of coffee on a table",
        "a toy dinosaur on a sandy beach",
        "a fox waking up in bed and yawning",
    ]
)
```

Each list item passed to `Omni.generate()` becomes an independent logical
request that the scheduler may batch with compatible items.

For step execution:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Qwen/Qwen-Image",
    step_execution=True,
    max_num_seqs=1,
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

## Deploy Configuration

Use the equivalent engine arguments in a deployment YAML:

```yaml
stages:
  - stage_id: 0
    step_execution: false
    max_num_seqs: 4
    request_batch_max_wait_ms: 20
```

For step execution, set `step_execution: true` and remove
`request_batch_max_wait_ms`.

## CLI Reference

| Flag | Default | Effect |
|---|---:|---|
| `--step-execution` | disabled | Select step-wise scheduling |
| `--max-num-seqs` | `1` for diffusion stages | Set request- or step-scheduler capacity |
| `--request-batch-max-wait-ms` | `0` | Wait for burst coalescing in request mode |
| `--diffusion-streaming-output` | disabled | Expose supported intermediate diffusion outputs and require step execution |

## Limitations and Troubleshooting

- Requests batch only when their compatibility-sensitive parameters match.
- Different LoRA adapters or scales run in separate batches.
- FIFO scheduling can cause an incompatible request to block later compatible
  requests.
- All diffusion cache backends are unsupported in step mode. KV transfer is
  supported for newly admitted step requests; some other request-mode extras
  remain unsupported.
- If request-mode startup reports that the pipeline does not support batching,
  use `--max-num-seqs 1`.
- If step-mode startup mentions `prepare_encode()`, `denoise_step()`,
  `step_scheduler()`, or `post_decode()`, the pipeline does not implement the
  required step contract.

For implementation details and model-author guidance, see
[Diffusion Continuous Batching](../../design/feature/diffusion_continuous_batching.md).
