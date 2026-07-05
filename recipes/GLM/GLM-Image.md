# GLM-Image for text-to-image and image editing

## Summary

- Vendor: Z.ai
- Model: `GLM/GLM-Image`
- Task: Text-to-image (T2I) and image-to-image
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`GLM/GLM-Image` with vLLM-Omni on **two 80 GB NVIDIA A800** GPUs (Ampere-class,
same default layout as the upstream **2×A100 80GB** example: Stage 0 AR on GPU 0,
Stage 1 diffusion on GPU 1) and validate the deployment with the existing
`examples/online_serving/glm_image` clients.

## References

- Upstream or canonical docs:
  [`docs/user_guide/examples/online_serving/glm_image.md`](../../docs/user_guide/examples/online_serving/glm_image.md)
- Related issue or discussion:
  [#2888](https://github.com/vllm-project/vllm-omni/pull/2888)

## Hardware Support

This recipe documents **dual-GPU** CUDA layouts on A800 80 GB
for the same software stack. Add more platforms (for example ROCm / NPU) as
community validation lands.

## GPU

### 2× A800 80GB

#### Environment

These versions were taken from a working **editable** install: activate `vllm-omni/.venv` (or your equivalent), then align `pip` / Git with the rows below when reproducing this recipe.

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA CUDA stack with **two** A800 80 GB GPUs visible (set `CUDA_VISIBLE_DEVICES` on your host if needed)
- vLLM: **0.19.0**
- vLLM-Omni: **0.19.0rc2.dev138+g38d5f2d53** (editable install from this repo; Git **`38d5f2d5`**, `git describe` ≈ **`v0.19.0rc1-138-g38d5f2d5`**)
- Transformers: **5.5.4** (same `.venv` as above; required so `glm_image` configs load for Stage 0)

#### Command

Start the server from the repository root:

```bash
vllm serve zai-org/GLM-Image --omni --port 8091
```

To use the bundled stage config explicitly (same default as above):

```bash
vllm serve zai-org/GLM-Image \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/glm_image.yaml
```

#### Verification

Run one of the existing example clients after the server is ready:

```bash
curl -s http://172.18.69.133:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A beautiful landscape painting"}
    ],
    "extra_body": {
      "height": 1920,
      "width": 1920,
      "num_inference_steps": 50,
      "true_cfg_scale": 1.5,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > land.png
```
After the command finishes, check for the output files:

```bash
ls output.png
```

#### Sample end-to-end metrics

One representative **offline** GLM-Image E2E run on this recipe’s **2× A800 80GB**.
Overall summary from the run’s metrics. Rough wall-time split: **Stage 0 (AR)** ~**25 s**,
**Stage 1 (diffusion)** ~**34 s** (see `e2e_stage_*_wall_time_ms` below).

| Field | Value |
| --- | ---: |
| e2e_requests | 1 |
| e2e_wall_time_ms | 61,148.679 |
| e2e_total_tokens | 1,300 |
| e2e_avg_time_per_request_ms | 61,148.679 |
| e2e_avg_tokens_per_s | 21.260 |
| e2e_stage_0_wall_time_ms | 24,708.760 |
| e2e_stage_1_wall_time_ms | 33,787.442 |

#### Notes

- Memory usage: Roughly **~38 GiB + KV** on Stage 0 (AR) and **~20 GiB** on Stage 1 (DiT+VAE) per the user guide; two 80 GB cards match the default split.
- Key flags: `--omni` is required; `--stage-configs-path` is optional unless you use a custom YAML (for example single-GPU).
- Keep **Transformers ≥ 5.5.1** (this recipe used **5.5.4**) so `glm_image` configs resolve; otherwise Stage 0 can fail at `ModelConfig` validation.
- Known limitations: This starter recipe follows the dual-GPU online path documented under `examples/online_serving/glm_image`. The first request may be slower due to warmup.
- Generation time: about **61 s** wall time end-to-end for the sample above (50 inference steps, 1024×1024).
