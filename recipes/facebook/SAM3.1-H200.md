# SAM 3.1 promptable image segmentation — H200

## Summary

- Vendor: Meta
- Model: `facebook/sam3.1` (SAM 3.1 detector)
- Task: Promptable concept segmentation of one image (text and/or exemplar boxes)
- Mode: Online serving (`POST /v1/images/segmentations`) and offline `Omni.generate`
- Hardware: 1x NVIDIA H200 141GB
- Maintainer: Community

## When to use this recipe

Serve SAM 3.1 to get every instance of a concept ("yellow car", "person") in an
image as masks, boxes and scores, for example to cut out subjects or build
layers before image or video editing with other vLLM-Omni models.

## Supported model contract

| Item | Contract |
| ---- | -------- |
| Endpoint | `POST /v1/images/segmentations` ([Image Segmentation API](../../docs/serving/image_segmentation_api.md)) |
| Input | One uploaded image per request, any size; resized internally to 1008x1008 |
| Prompts | Text concept, up to 64 exemplar boxes with positive (`1`) / negative (`0`) labels, or both |
| Output | Instances above `score_threshold` (default 0.5): mask (PNG or COCO RLE), `[x0, y0, x1, y1]` box and score, all at the input resolution |
| Checkpoints | Meta's native `sam3*.safetensors` layout and Transformers exports (`Sam3Model`, `Sam3VideoModel`) |
| Not covered | Video tracking with SAM 3.1's multiplex tracker, point prompts and interactive refinement; the tracker weights in the checkpoint are skipped. SAM 3 video tracking is in the [SAM 3 video recipe](SAM3-Video-H200.md) |

## References

- Upstream: <https://github.com/facebookresearch/sam3>, <https://huggingface.co/facebook/sam3.1>
- Reference implementation used for parity: Transformers `Sam3Model`

## Hardware

- Accelerator model and per-device memory: NVIDIA H200, 141 GB
- Number of devices: 1
- Device interconnect: not used
- Qualification scope: single-request online serving of the detector in BF16

## Software environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12.15
- Driver / runtime: NVIDIA 590.48.01, CUDA 13.2, PyTorch 2.13.0
- vLLM version: 0.32.0
- vLLM-Omni version or commit: `9d2241d48` plus this change

## Command

`facebook/sam3.1` is gated: accept its terms on Hugging Face and export
`HF_TOKEN` first. The command was qualified with `AEmotionStudio/sam3.1`, an
ungated copy of the same native checkpoint.

```bash
vllm serve facebook/sam3.1 --omni --port 8091
```

## Verification

```bash
curl -s http://localhost:8091/v1/images/segmentations \
  -F "image=@street.png" -F "prompt=person" -F "mask_format=rle" \
  | jq '[.data[] | {score, box}]'
```

Expected: one entry per person with scores around 0.9 and boxes in input
pixels. On the H200 this request takes about 190 ms end to end (median of 10
warm requests, 2048x824 image, RLE masks).

## Notes

- Memory usage: 1.59 GiB of detector weights in BF16; 2.2 GiB process-scoped
  GPU memory after loading. Startup to ready took 55 s including warmup.
- Accuracy: on 12 text prompts over 3 images, outputs match the Transformers
  reference loaded from the same native checkpoint through its official
  conversion: mask IoU 1.0 for all 27 instances in FP32 and BF16, max score
  difference 1.2e-7 (FP32) and 0 (BF16).
- Known limitations: one image per request; no request batching.

## Supported features

| Feature | Status |
| ------- | ------ |
| BF16 / FP32 | Supported (BF16 default) |
| Native Meta and Transformers checkpoints | Supported |
| [Cache acceleration](../../docs/user_guide/diffusion_features.md) | Not applicable (single forward pass) |
| Tensor / sequence / CFG parallelism, CPU offload, quantization | Not supported |
