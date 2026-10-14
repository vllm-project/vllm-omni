# Qwen-Image-Edit

> Image editing serving

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen-Image-Edit`
- Task: Text-guided single-image editing
- Mode: Online serving with the OpenAI-compatible multimodal endpoint
- Maintainer: Community

## When to use this recipe

Use this recipe as a baseline for serving `Qwen/Qwen-Image-Edit` for single-image editing on H200 141GB. Two configurations are validated below, sharing one software environment:

| Configuration | Parallelism | Notes |
| --- | --- | --- |
| `1x H200` | none | Single-card baseline; ~60 GiB peak VRAM on a 141 GB card |
| `2x H200` | TP=2 | Lower single-request latency (≈ 1.60× speedup) and lower per-GPU VRAM |

## References

- Model: <https://huggingface.co/Qwen/Qwen-Image-Edit>
- Example guide: [`examples/online_serving/image_to_image/README.md`](../../examples/online_serving/image_to_image/README.md)
- User guide: [`docs/user_guide/examples/online_serving/image_to_image.md`](../../docs/user_guide/examples/online_serving/image_to_image.md)

## Hardware Support

### GPU

The two hardware configurations below share the same software environment.

#### Environment

- OS: Linux
- Python: 3.12+
- Driver / runtime: CUDA-capable runtime matching the repository build
- vLLM: use versions from your current checkout, >=0.20.0
- vLLM-Omni: use versions from your current checkout

#### 1x H200 141GB

##### Command

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen-Image-Edit --omni \
  --port 8092
```

The bundled launcher is equivalent:

```bash
bash examples/online_serving/image_to_image/run_server.sh
```

The server is ready when the log shows `INFO:     Application startup complete.` (preceded by `Started server process [PID]` and `Waiting for application startup.`).

##### Verification

Run a single-image edit against the public example asset:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png

PROMPT="Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'"
IMG_B64=$(base64 -w0 qwen-bear.png)

curl -s http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg prompt "$PROMPT" --arg img "$IMG_B64" '{
    messages: [{
      role: "user",
      content: [
        {type: "text", text: $prompt},
        {type: "image_url", image_url: {url: ("data:image/png;base64," + $img)}}
      ]
    }],
    extra_body: {
      height: 1024,
      width: 1024,
      num_inference_steps: 50,
      guidance_scale: 1.0,
      seed: 42
    }
  }')" \
  | jq -r '.choices[0].message.content[0].image_url.url' \
  | cut -d',' -f2- | base64 -d > qwen_image_edit_1card.png
```

The decoded PNG is a 1024×1024 RGB image of the bear mascot dancing under stylized moonlight, surrounded by floating stars and a "Be Kind" text bubble.

You can also run the offline CLI without standing up a server:

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --image qwen-bear.png \
  --prompt "Let this mascot dance under the moon, ..." \
  --output qwen_image_edit_offline.png \
  --num-inference-steps 50 \
  --cfg-scale 4.0
```

Validated reference numbers for 1024×1024, 50 denoising steps, batch size 1, `guidance_scale=1.0`, seed 42 (`Cold` = first request after startup, `Warm` = identical subsequent request):

| Cold latency | Warm latency | Peak GPU memory |
| ---:         | ---:         | ---:            |
| 17.80 s      | 17.78 s      | 61,091 MiB (~59.66 GiB) |

##### Notes

- Generation parameters live under `extra_body` for `/v1/chat/completions`, or as top-level form fields when using `/v1/images/edits`. See [`examples/online_serving/image_to_image/README.md`](../../examples/online_serving/image_to_image/README.md) for the full request / response schema.
- Memory: a 1024×1024 / 50-step edit uses ~60 GiB on a single H200, well within the 141 GB headroom.
- Multi-image variants of Qwen-Image-Edit (e.g. `Qwen-Image-Edit-2509`) are intentionally out of scope for this recipe and are not validated here.

#### 2x H200 141GB (TP=2)

##### Command

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen-Image-Edit --omni \
  --port 8092 \
  --tensor-parallel-size 2
```

In addition to the `Application startup complete.` line, the TP=2 server's `non-default args` log line echoes `'tensor_parallel_size': 2`, and a separate `Worker N: Initialized device and distributed environment` line is emitted for each rank — one per visible GPU.

##### Verification

Run the same `curl` command as in the **1x H200** Verification section above against the TP=2 server (same prompt, same seed, same image). Save the result to `qwen_image_edit_2card_tp2.png` so the two outputs sit side by side for comparison. During TP=2 inference, `nvidia-smi` shows sustained `~99 %` utilization on both GPUs, which is the expected signal that TP wiring is active.

Same workload as the 1x table above (1024×1024, 50 steps, `guidance_scale=1.0`, seed 42):

| Configuration | Cold latency | Warm latency | Peak per-GPU memory |
| ---           | ---:         | ---:         | ---:                |
| 1x H200       | 17.80 s      | 17.78 s      | 61,091 MiB (~59.66 GiB) |
| 2x H200 TP=2  | 11.16 s      | 11.09 s      | 49,377 MiB (~48.22 GiB) |

TP=2 reduces single-request warm latency from `17.78 s` to `11.09 s` (≈ **1.60×**) and drops per-GPU peak VRAM by ~11.4 GiB.

##### Notes

- TP=2 shards model weights across both cards, which is what produces both the latency reduction and the per-GPU VRAM drop above.
- CFG-Parallel (`--cfg-parallel-size 2`) is supported on the Qwen-Image family but only when `cfg_scale > 1`. The verification command above uses `guidance_scale=1.0` to match the public example image, so it does not exercise CFG-parallel; a separate run with `cfg_scale > 1` is needed if you want to benchmark that path.
- For deeper acceleration knobs (Cache-DiT, sequence parallel, HSDP) see [`docs/user_guide/diffusion/parallelism/overview.md`](../../docs/user_guide/diffusion/parallelism/overview.md); this recipe intentionally documents only the validated TP=2 baseline.

#### Profiling

The numbers below are from a warm request at 1024×1024, 50 steps, and `guidance_scale=1.0`, measured with `--enable-diffusion-pipeline-profiler` and a 250 ms `nvidia-smi --query-gpu=memory.used` sampler. For TP=2, latency uses rank max, and VRAM is per GPU.

Per-stage latency:

| Stage | TP=1 | TP=2 | Notes |
| --- | ---: | ---: | --- |
| `text_encoder.forward` | 0.132 s | 0.135 s | ~1.0× |
| `vae.encode` | 0.040 s | 0.041 s | ~1.0× |
| `diffuse` | 17.105 s | 10.382 s | **1.65×** speedup |
| `vae.decode` | 0.064 s | 0.066 s | ~1.0× |
| `forward` | 17.372 s | 10.652 s | 1.63× speedup |

Per-GPU VRAM:

| Snapshot | TP=1 | TP=2 | Notes |
| --- | ---: | ---: | --- |
| Static VRAM | 58,777 MiB | 47,065 MiB | after model load, before request |
| Peak VRAM | 61,091 MiB | 49,377 MiB | during inference |
| Request-time increment | 2,314 MiB | 2,312 MiB | peak minus static |

The latency gain mainly comes from the DiT denoising stage (`diffuse` in the table above). The other measured stages, such as text encoder and VAE encode/decode, stay essentially unchanged between TP=1 and TP=2, which is consistent with them not benefiting meaningfully from TP in this pipeline. The `forward` row's 1.63× speedup tracks `diffuse`, since `diffuse` accounts for about 98% of total pipeline time.

For memory, TP=2 saves about 11.4 GiB per GPU, mostly from static model-state memory, likely dominated by tensor-parallel sharding of the DiT weights. The request-time increment is almost unchanged, so TP=2 does not materially reduce transient inference-time memory for this single-image, batch-1 workload.
