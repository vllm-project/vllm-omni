# HunyuanVideo-1.5 for text-to-video

## Summary

- Vendor: Tencent-Hunyuan
- Model: `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`
- Task: Text-to-video generation
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to generate a 480p, 121-frame HunyuanVideo-1.5 text-to-video
sample on one 80 GB A100.

## References

- Upstream or canonical docs:
  [`docs/user_guide/examples/offline_inference/text_to_video.md`](../../docs/user_guide/examples/offline_inference/text_to_video.md)
- Related example under `examples/`:
  [`examples/offline_inference/text_to_video/text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)
- Related online serving examples:
  [`examples/online_serving/text_to_video/run_server_hunyuan_video_15.sh`](../../examples/online_serving/text_to_video/run_server_hunyuan_video_15.sh)
  and
  [`examples/online_serving/text_to_video/run_curl_hunyuan_video_15.sh`](../../examples/online_serving/text_to_video/run_curl_hunyuan_video_15.sh)
- Related issue or discussion:
  [RFC: add recipes folder](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

This recipe documents one validated CUDA GPU configuration.

## GPU

### 1x A100 80GB

#### Environment

- Python: 3.10.12
- GPU: `NVIDIA A100-SXM4-80GB`, 81920 MiB reported in `nvidia-smi` samples
- Runtime: CUDA `cu130` wheel stack
- vLLM version: `0.19.0+cu130`
- vLLM-Omni version or commit: `0.1.dev1415+g6b52db9e2`
- Related package versions: `torch==2.10.0+cu130`,
  `diffusers==0.37.1`, `transformers==4.57.6`, `accelerate==1.12.0`

#### Command

Run from the repository root. The validated full 480p run used VAE tiling and
slicing.

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --prompt "A cat walks through a sunlit garden, flowers swaying gently in the breeze." \
  --height 480 \
  --width 832 \
  --num-frames 121 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --flow-shift 5.0 \
  --fps 24 \
  --tensor-parallel-size 1 \
  --vae-use-tiling \
  --vae-use-slicing \
  --output hunyuan_video_15_output.mp4
```

#### Verification

Check that the output file exists and is non-empty:

```bash
test -s hunyuan_video_15_output.mp4
```

Expected successful log lines include:

```text
Generation completed successfully.
Saved generated video to hunyuan_video_15_output.mp4
```

#### Notes

- Memory usage: The validated run peaked at `62051 MiB` in `nvidia-smi`
  samples. The vLLM-Omni request log reported `78.54 GB reserved` and
  `57.43 GB allocated` at peak.
- Runtime: The validated generation took `1599.1544` seconds.
- Key flags: `--tensor-parallel-size 1` keeps the model on one GPU.
  `--vae-use-tiling` and `--vae-use-slicing` were used for the validated
  480p command.
- Online serving: HunyuanVideo-1.5 T2V has online serving examples under
  `examples/online_serving/text_to_video/`. This recipe remains scoped to the
  offline command above because that was the path validated for this 1x A100
  run.
- Known limitations: This recipe covers only the 480p command with VAE tiling
  and slicing. The no-tiling/no-slicing and 720p variants are not covered here.
