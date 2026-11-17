# MammothModa2-Preview on Ascend NPU

## Summary

- Vendor: ByteDance Research
- Model: `bytedance-research/MammothModa2-Preview`
- Task: Text-to-image (AR to DiT)
- Mode: Offline inference
- Hardware: 1x Ascend 910B3 64GB
- Maintainer: Community

## When to use this recipe

Use this recipe to reproduce the reported MammothModa2-Preview text-to-image
smoke test on one Ascend 910B3. Qualification covers one 1024x1024 request;
it does not establish support for other tasks or MammothModa2-Dev on NPU.

## References

- Related recipe: [MammothModa2](./MammothModa2.md)
- Successful-run evidence: [PR #8978](https://github.com/vllm-project/vllm-omni/pull/8978)
- Shared example: [text_to_image.py](../../examples/offline_inference/text_to_image/text_to_image.py)

## Environment and deployment

- Accelerator: one Ascend 910B3 with 64 GB device memory
- Runtime: vLLM-Omni with vllm-ascend and torch-npu; follow the
  [NPU installation guide](../../docs/getting_started/installation/npu.md)
- Deploy config: `vllm_omni/deploy/mammoth_moda2.yaml`, both stages on device 0
- Execution: eager, request mode, one request
- AR/DiT `gpu_memory_utilization`: 0.5/0.3, as committed; these are configuration
  budgets, not measured peak-memory values
- Successful-run vLLM version: `0.33.0`
- Successful-run vLLM-Omni version: `0.1.dev1+g08e9267ed.npu`
- Successful-run code revision: `75c26a8ef5746c0f5d93b1c5772b3c5204935f8c`
- Qualification limitations: CANN/driver and torch-npu versions and host
  memory were not recorded

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
