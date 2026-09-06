# Scheduler-Managed Paged KV Cache

Scheduler-managed paged KV cache stores diffusion DiT key/value tensors in
Worker-owned pages. It is useful when a denoising request should keep its
stable prompt/reference prefix in device memory while rewriting only the
changing image span on later steps. The current HunyuanImage-3.0 integration
is request-level only; it does not silently change an unsupported request to
dense attention.

Paging changes KV storage and scheduling ownership; it does not change the
model's causal/full attention semantics.

## Choose a mode

| Need | Mode |
| --- | --- |
| Broadest compatibility or step execution | `dense_legacy` (default) |
| HunyuanImage-3.0 request-mode serving with Scheduler-managed KV pages | `paged_scheduler` |

Use `dense_legacy` when the model, attention backend, or offload combination
has not been validated for paged KV. HunyuanImage-3.0 paged KV currently accepts
request-level execution only; set `step_execution: false`.

## Supported scope

| Model | Platform | Execution | Backend | Status |
| --- | --- | --- | --- | --- |
| HunyuanImage-3.0 | NVIDIA CUDA | Request-level | `FLASH_ATTN` (native FlashAttention/FA3) | Implemented and validated |
| HunyuanImage-3.0 | Ascend NPU | Request-level | `FLASH_ATTN` (Ascend FIA) | Implemented and validated |
| Other diffusion models | Any | N/A | Any | Not integrated or tested |

Only the logical `FLASH_ATTN` backend currently advertises paged-KV support.
`TORCH_SDPA`, `CUDNN_ATTN`, `FLASHINFER_ATTN`, SAGE, Hub, and other diffusion
backends do not transparently fall back to dense execution for a paged request.

Strict Ulysses SP and two-branch CFG parallel are supported in HunyuanImage-3.0
request mode when the selected device topology has been validated. Ring SP,
AllGather-KV SP, Hunyuan paged step execution, and independent public-request
batching are not supported by this path. DreamZero and LingBot-World use the
separate `ar_diffusion_kv` imported-AR-KV contract.

## Configure

The diffusion KV fields are stage settings. The two
`diffusion_kv_*` fields are diffusion-specific engine fields and are forwarded
through `engine_extras`; `kv_cache_memory_bytes` is a standard stage field and
is placed directly under the stage:

```yaml
stages:
  - stage_id: 0
    max_num_seqs: 1
    enforce_eager: true
    diffusion_attention_backend: FLASH_ATTN
    step_execution: false
    kv_cache_memory_bytes: 536870912  # 512 MiB per Worker/rank
    engine_extras:
      diffusion_kv_mode: paged_scheduler
      diffusion_kv_max_rows_per_request: 2
```

| Field | Required | Meaning |
| --- | --- | --- |
| `diffusion_kv_mode` | Yes | `paged_scheduler` enables Scheduler-managed pages; `dense_legacy` is the default |
| `diffusion_kv_max_rows_per_request` | Yes for paged | Positive Worker row limit, including all internal CFG branches; use `1` without CFG and `2` for standard CFG |
| `kv_cache_memory_bytes` | Optional | Explicit physical KV-pool budget per Worker/rank; omit it to use automatic sizing |
| `gpu_memory_utilization` | Optional | Automatic pool-sizing fraction when no explicit byte budget is set |
| `max_num_seqs` | Recommended `1` | Public request capacity; it is separate from the internal CFG row limit |
| `step_execution` | Must be `false` | Hunyuan paged KV is request-level only |
| `diffusion_attention_backend` | `FLASH_ATTN` | Required logical backend for the current paged implementation |

`kv_cache_memory_bytes` describes pool capacity, not live payload. Reserved
memory can therefore exceed the memory currently used by active requests.
The Scheduler allocates all rows belonging to one request together: a request
with CFG disabled has one row, while standard two-branch CFG has two rows. CFG
rows are internal branches, not independent public requests.

Paged KV is currently unquantized BF16. `diffusion_kv_cache_dtype` does not
enable quantized paged KV, and model weight quantization has not been validated
in combination with this mode. CPU offload, layerwise offload, and distributed
layerwise offload (DLO) are currently untested with paged KV. Leave these
options disabled for paged KV; use `dense_legacy` when offload is required.

## Hunyuan request-mode example

Create a small deploy overlay (for example,
`hunyuan_image3_paged.yaml`) with the stage settings above, then start the
validated Hunyuan request-mode path:

```bash
vllm serve tencent/HunyuanImage-3.0-Instruct \
  --omni \
  --trust-remote-code \
  --port 8091 \
  --deploy-config /path/to/hunyuan_image3_paged.yaml
```

The base Hunyuan DiT deploy file can be reused and overridden from the command
line when a separate overlay is inconvenient:

```bash
vllm serve tencent/HunyuanImage-3.0-Instruct \
  --omni \
  --trust-remote-code \
  --deploy-config vllm_omni/deploy/hunyuan_image3_dit.yaml \
  --stage-overrides \
  '{"0":{"diffusion_kv_mode":"paged_scheduler","diffusion_kv_max_rows_per_request":2,"kv_cache_memory_bytes":536870912,"step_execution":false,"diffusion_attention_backend":"FLASH_ATTN"}}'
```

After the server is ready, send the normal Hunyuan image-generation request:

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "modalities": ["image"],
    "messages": [{"role": "user", "content": "A red ceramic vase on a wooden table"}],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 20,
      "guidance_scale": 5.0,
      "seed": 1234
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' \
    | cut -d',' -f2- | base64 -d > hunyuan_image3_paged.png
```

The same configuration is shown in the
[HunyuanImage-3.0 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Tencent/HunyuanImage-3.0-Instruct.md).

Confirm from startup and worker logs that the resolved backend is
`FLASH_ATTN` and that the paged native path is active. A paged request must
fail explicitly if its Worker metadata or native backend is unavailable; it
must not silently switch to dense attention.

## Parallelism

Hunyuan request-mode paged KV can be combined with tensor parallelism, strict
Ulysses SP, or two-way CFG parallel when the selected topology has been
validated. Configure the topology in the stage's existing `parallel_config`:

| Setting | User-visible effect | Paged-KV requirement |
| --- | --- | --- |
| `tensor_parallel_size` | Shards model weights across ranks | Device count must match the stage topology |
| `sequence_parallel_size` + strict Ulysses | Shards the sequence across ranks | Ring and AllGather-KV modes are not supported |
| `cfg_parallel_size: 2` | Places the positive and negative branches on separate ranks | Set `diffusion_kv_max_rows_per_request` to at least `2` |

`max_num_seqs` controls public request concurrency and is independent of the
internal CFG row limit. Keep `max_num_seqs: 1` for the current Hunyuan paged
path; increasing it does not enable independent request batching.

## Troubleshooting

- `paged_scheduler requires diffusion_kv_max_rows_per_request`: add a positive
  stage value and include all CFG rows.
- Unsupported backend error: select `FLASH_ATTN`; the other attention backends
  currently do not implement the paged-KV capability.
- Step-mode rejection: remove `step_execution: true` and use request-level
  execution, or select `dense_legacy` for step execution.
- Imported AR-KV rejection: `paged_scheduler` does not consume the
  `ar_diffusion_kv` contract used by DreamZero or LingBot-World.
- Missing image output: include `"modalities": ["image"]` when using the chat
  completions endpoint, and decode the returned data URI as shown above.

For the ownership and lifecycle contract, see the
[Scheduler-managed paged KV design](../../design/feature/diffusion_paged_kv_cache.md).
