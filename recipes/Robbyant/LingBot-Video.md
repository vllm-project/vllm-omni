# LingBot-Video

> Native dense text-to-video serving for `robbyant/lingbot-video-dense-1.3b`

## Summary

- Vendor: Robbyant
- Model: `robbyant/lingbot-video-dense-1.3b`
- Task: Text-to-video generation
- Mode: Offline generation and online serving with the OpenAI-compatible `/v1/videos` API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run the dense LingBot-Video checkpoint with
vLLM-Omni's native pipeline. The runtime path does not import the upstream
`lingbot_video` Python package; it loads the checkpoint components directly
with the in-tree `LingBotVideoPipeline`, dense DiT transformer, shared FlowUniPC
scheduler, Qwen3-VL text encoder, and Wan VAE.

This recipe is intentionally text-to-video only. Text-to-image, image-conditioned
video, TI2V, MoE checkpoints, and fused-expert kernels are not covered here.

## References

- Dense checkpoint: <https://huggingface.co/robbyant/lingbot-video-dense-1.3b>
- Upstream project: <https://github.com/Robbyant/lingbot-video>
- Related offline example: [`examples/offline_inference/text_to_video/text_to_video_lingbot.py`](../../examples/offline_inference/text_to_video/text_to_video_lingbot.py)
- Related online video API docs: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)

## Hardware Support

This recipe documents the CUDA single-GPU dense checkpoint path. Multi-GPU
parallelism, Cache-DiT, CPU offload, and the LingBot MoE path are not validated
for this model in this PR.

## GPU

### 1 x NVIDIA L20X / A100 / H100

The dense 1.3B checkpoint has been smoke-tested on a single NVIDIA L20X at a
small validation shape (`192x320`, 9 frames, 2 steps). Use a larger GPU or lower
resolution/frame count if your target workload runs out of memory.

#### Offline T2V

```bash
CUDA_VISIBLE_DEVICES=0 LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa \
python examples/offline_inference/text_to_video/text_to_video_lingbot.py \
  --model robbyant/lingbot-video-dense-1.3b \
  --prompt "a robotic arm picks up a red block" \
  --output lingbot_t2v.mp4 \
  --height 192 \
  --width 320 \
  --num-frames 9 \
  --num-inference-steps 2 \
  --guidance-scale 3.0 \
  --flow-shift 3.0 \
  --seed 42 \
  --fps 24
```

#### Online Serving

```bash
CUDA_VISIBLE_DEVICES=0 LINGBOT_QWEN_ATTN_IMPLEMENTATION=sdpa \
vllm serve robbyant/lingbot-video-dense-1.3b \
  --omni \
  --model-class-name LingBotVideoPipeline \
  --port 8091
```

After the server is ready, submit a text-to-video job:

```bash
create_response=$(curl -s http://localhost:8091/v1/videos \
  -F "model=robbyant/lingbot-video-dense-1.3b" \
  -F "prompt=a robotic arm picks up a red block" \
  -F "width=320" \
  -F "height=192" \
  -F "num_frames=9" \
  -F "fps=24" \
  -F "num_inference_steps=2" \
  -F "guidance_scale=3.0" \
  -F "flow_shift=3.0" \
  -F "seed=42")

video_id=$(echo "${create_response}" | jq -r '.id')
while true; do
  status=$(curl -s "http://localhost:8091/v1/videos/${video_id}" | jq -r '.status')
  if [ "${status}" = "completed" ]; then
    break
  fi
  if [ "${status}" = "failed" ]; then
    curl -s "http://localhost:8091/v1/videos/${video_id}" | jq .
    exit 1
  fi
  sleep 2
done

curl -L "http://localhost:8091/v1/videos/${video_id}/content" -o lingbot_t2v.mp4
```

## Key Parameters

| Parameter | Suggested smoke value | Notes |
|-----------|-----------------------|-------|
| `height` | `192` | Must be a multiple of 16 |
| `width` | `320` | Must be a multiple of 16 |
| `num_frames` | `9` | Must be `1` or `4n + 1`; this PR validates T2V with video outputs |
| `num_inference_steps` | `2` | Use more steps for quality sweeps |
| `guidance_scale` | `3.0` | CFG is active when this is greater than `1.0` |
| `flow_shift` | `3.0` | Scheduler flow-shift; aliases the pipeline's internal `shift` |
| `negative_prompt` | model default | Optional text describing artifacts to avoid |
| `fps` | `24` | Output MP4 frame rate |

## Validation

Local dense smoke run:

- Shape: 9 frames at `192x320`
- Steps: 2
- Request generation time: `0.2923s`
- Peak reserved GPU memory: `14548 MiB`

Local parity harness against the upstream repository:

- Shape: `[9, 192, 320, 3]`
- MAE: `0.0065238`
- MSE: `0.00006650`
- PSNR: `41.77 dB`
- Native request time: `0.2875s`

Do not treat these as production benchmarks; they are draft-PR validation
numbers for the small dense smoke configuration.

## Known Limitations

- T2V only. T2I, I2V, and TI2V are not claimed by this PR.
- Dense checkpoint only. MoE/fused-expert support belongs in a separate PR.
- No Cache-DiT, TeaCache, tensor parallelism, sequence parallelism, CFG
  parallelism, HSDP, CPU offload, or VAE patch parallelism validation yet.
- Only one request per LingBot pipeline batch is currently supported.
