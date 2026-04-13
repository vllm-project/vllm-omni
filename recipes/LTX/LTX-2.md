# LTX-2 T2V and I2V

> LTX-2 for text-to-video & image-to-video on 1x H200 141GB

## Summary

- Vendor: Lightricks
- Model: `Lightricks/LTX-2`
- Task: Text-to-video generation and image-to-video generation
- Mode: Online serving with the OpenAI-compatible video API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to deploy the base `Lightricks/LTX-2` model
for both text-to-video and image-to-video generation. The same model ID is used for
both tasks; switch the pipeline class with `--model-class-name` depending on
whether you want T2V or I2V.

## References

- Model: <https://huggingface.co/Lightricks/LTX-2>
- User guide:
  [`docs/user_guide/examples/online_serving/text_to_video.md`](../../docs/user_guide/examples/online_serving/text_to_video.md)
- Example guide:
  [`examples/online_serving/text_to_video/README.md`](../../examples/online_serving/text_to_video/README.md)
  [`examples/online_serving/image_to_video/README.md`](../../examples/online_serving/image_to_video/README.md)

## Hardware Support

## GPU

### 1x H200 141GB (Text-to-Video)

#### Environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12+
- Driver / runtime: NVIDIA CUDA environment with H200 141GB GPU
- vLLM version: Match the repository requirements from your current checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

```bash
vllm serve Lightricks/LTX-2 \
  --omni \
  --model-class-name LTX2Pipeline
```

#### Verification

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cinematic close-up of ocean waves at golden hour." \
  -F "negative_prompt=worst quality, inconsistent motion, blurry, jittery, distorted" \
  -F "width=512" \
  -F "height=768" \
  -F "num_frames=121" \
  -F "fps=24" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -o ltx2_t2v_h200.mp4
```

#### Notes

- Memory usage: Model loads at ~73.5GB, peaks at ~73.5 GiB during inference.
- Key flags:
  - `--model-class-name LTX2Pipeline` selects the text-to-video pipeline explicitly.
- Known limitations:
  - This recipe is tested on 1x H200 validation; add measured latency and VRAM after running it on your target machine.

### 1x H200 141GB (Image-to-Video)

#### Environment

- OS: Ubuntu 22.04.5 LTS
- Python: 3.12+
- Driver / runtime: NVIDIA CUDA environment with H200 141GB GPU
- vLLM version: Match the repository requirements from your current checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

```bash
vllm serve Lightricks/LTX-2 \
  --omni \
  --model-class-name LTX2ImageToVideoPipeline
```

#### Verification

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/videos/sync \
  -F "prompt=A plush toy astronaut gently waving while the camera slowly pushes in." \
  -F "input_reference=@/absolute/path/to/reference.png" \
  -F "width=512" \
  -F "height=768" \
  -F "num_frames=121" \
  -F "fps=24" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -o ltx2_i2v_h200.mp4
```

#### Notes

- Memory usage: Model loads at ~73.5GB, peaks at ~73.5 GiB during inference.
- Key flags:
  - `--model-class-name LTX2ImageToVideoPipeline` switches the same `Lightricks/LTX-2` weights into image-to-video mode.
  - `input_reference=@...` uploads the reference image directly
- Known limitations:
  - Use `image_reference` when you want to pass a URL or JSON-safe image reference
instead of uploading a file.
  - Do not send `input_reference` and `image_reference` in the same request.
  - This recipe is tested on 1x H200 validation; add measured latency and VRAM after running it on your target machine.
