# MammothModa2-Preview on Ascend NPU

## Summary

- Vendor: ByteDance Research
- Model: `bytedance-research/MammothModa2-Preview`
- Task: Text-to-image (AR to DiT)
- Mode: Offline inference
- Hardware: 1x Ascend 910B3 64GB
- Maintainer: Community

## When to use this recipe

Use the recorded versions below to reproduce the pre-MRv2 MammothModa2-Preview
text-to-image smoke test on one Ascend 910B3. Validation covered one 1024x1024
request with vLLM 0.34.0 at vLLM-Omni revision `0.33.0rc2.dev385+g62d0e880a`.

Validation on the current Model Runner V2 stack remains pending. The
[MRv2 T2I tracking issue](https://github.com/vllm-project/vllm-omni/issues/10752)
tracks missing `generated_len` metadata that can cause the AR stage to skip
grid constraints.

## References

- Related recipe: [MammothModa2](./MammothModa2.md)
- Shared example: [text_to_image.py](../../examples/offline_inference/text_to_image/text_to_image.py)

## Environment and deployment

- Accelerator: one Ascend 910B3 with 64 GB device memory
- Runtime: vLLM-Omni with vllm-ascend and torch-npu; follow the
  [NPU installation guide](../../docs/getting_started/installation/npu.md)
- Deploy config: `vllm_omni/deploy/mammoth_moda2.yaml`, both stages on device 0
- Execution: eager, request mode, one request
- AR/DiT `gpu_memory_utilization`: 0.5/0.3, as committed; these are configuration
  budgets, not measured peak-memory values
- Successful-run vLLM version: `0.34.0`
- Successful-run vLLM-Omni version: `0.33.0rc2.dev385+g62d0e880a`


## Offline command

Use the local Preview checkpoint and run from the repository root:

```bash
python3 examples/offline_inference/text_to_image/text_to_image.py \
  --model ./MammothModa2-Preview \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --seed 42 \
  --extra-body '{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": 50}' \
  --enable-diffusion-pipeline-profiler \
  --log-stats \
  --output mammoth_t2i.png
```

## Verification

The reported scope is 1024x1024, seed 42, 50 diffusion steps, guidance 4.0, and CFG range [0, 1].

Check the saved image in the inference environment (requires Pillow):

```bash
python3 -c 'from PIL import Image; im = Image.open("mammoth_t2i.png"); im.load(); assert im.size == (1024, 1024); assert im.mode == "RGB"; print(im.size, im.mode)'
```
