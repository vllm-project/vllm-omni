# LingBot-Video

> Native dense and MoE text-to-video serving for LingBot-Video

## Summary

- Vendor: Robbyant
- Models: `robbyant/lingbot-video-dense-1.3b` and `robbyant/lingbot-video-moe-30b-a3b`
- Task: Text-to-video generation
- Mode: Offline generation and online serving with the OpenAI-compatible `/v1/videos` API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run a dense or MoE LingBot-Video checkpoint
with vLLM-Omni's native pipeline. The runtime path does not import the upstream
`lingbot_video` Python package; it loads checkpoint components directly with
the in-tree `LingBotVideoPipeline`, dense or routed-MoE DiT blocks, shared
FlowUniPC scheduler, Qwen3-VL text encoder, and Wan VAE.

This first MoE integration is intentionally text-to-video and single-GPU only.
It targets the base BF16 checkpoint without its optional refiner or additional
parallel, cache, quantized, or expert-kernel backends.

## References

- Dense checkpoint: <https://huggingface.co/robbyant/lingbot-video-dense-1.3b>
- MoE checkpoint: <https://huggingface.co/robbyant/lingbot-video-moe-30b-a3b>
- Upstream project: <https://github.com/Robbyant/lingbot-video>
- Related offline example: [`examples/offline_inference/text_to_video/text_to_video_lingbot.py`](../../examples/offline_inference/text_to_video/text_to_video_lingbot.py)
- Related online video API docs: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)

## Hardware Support

This recipe documents the CUDA single-GPU dense and BF16 MoE checkpoint paths.
Multi-GPU parallelism, Cache-DiT, quantization, and CPU offload are not
validated for LingBot-Video in this PR.

## GPU

### 1 x NVIDIA L20X

Both checkpoints have been smoke-tested on one NVIDIA L20X at `192x320`,
9 frames, and 2 steps. The MoE smoke reserved approximately `67.70 GiB` of GPU
memory, so use a GPU with at least about 70 GiB of available memory for this
small validation shape. Larger resolutions, frame counts, or concurrent
requests require additional headroom.

The MoE path is validated with BF16 expert weights.

#### Dense offline T2V

```bash
CUDA_VISIBLE_DEVICES=0 \
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

#### MoE offline T2V

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/offline_inference/text_to_video/text_to_video_lingbot.py \
  --model robbyant/lingbot-video-moe-30b-a3b \
  --prompt "a robotic arm picks up a red block" \
  --output lingbot_moe_t2v.mp4 \
  --height 192 \
  --width 320 \
  --num-frames 9 \
  --num-inference-steps 2 \
  --guidance-scale 3.0 \
  --flow-shift 3.0 \
  --seed 42 \
  --fps 24
```

#### Online serving

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve robbyant/lingbot-video-dense-1.3b \
  --omni \
  --model-class-name LingBotVideoPipeline \
  --default-sampling-params \
  '{"0":{"num_frames":81,"num_inference_steps":40,"guidance_scale":6.0}}' \
  --port 8091
```

For the MoE checkpoint, use the same single-GPU pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve robbyant/lingbot-video-moe-30b-a3b \
  --omni \
  --model-class-name LingBotVideoPipeline \
  --default-sampling-params \
  '{"0":{"num_frames":81,"num_inference_steps":40,"guidance_scale":6.0}}' \
  --port 8091
```

These stage defaults match the LingBot reference pipeline. Request-level
values continue to override them, so the smaller smoke request below remains
unchanged.

When serving MoE, replace the request's `model` form value below with
`robbyant/lingbot-video-moe-30b-a3b`.

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

Local dense parity harness against the upstream repository:

- Shape: `[9, 192, 320, 3]`
- MAE: `0.0065238`
- MSE: `0.00006650`
- PSNR: `41.77 dB`
- Native request time: `0.2875s`

Local MoE validation on one NVIDIA L20X used checkpoint revision
`f2e538f64afe00cc4ae674db2aeb52e2945edfd5`:

- Loaded all `977` transformer state keys and `30,084,506,176` parameters.
- Router weights and correction biases stayed in FP32; routed and shared
  expert weights loaded in BF16.
- The complete 48-layer transformer forward returned finite
  `[1, 16, 1, 8, 8]` BF16 output.
- With both implementations using matched math SDPA backends, the complete
  native transformer matched the upstream implementation bitwise
  (`max_abs=0`, `mean_abs=0`).
- An isolated sparse MoE block exercising the router, grouped experts, scatter
  restore, and shared expert also matched the upstream implementation bitwise.
- The native pipeline generated a 9-frame `192x320` MP4 in 2 steps with
  `69326 MiB` (`67.70 GiB`) peak reserved GPU memory.
- The online `/v1/videos` smoke created, polled, and downloaded the generated
  MP4 successfully (`1 passed`).

Do not treat these as production benchmarks; they are functional smoke plus
controlled numerical-parity evidence for small validation inputs.

## Reproducing MoE numerical parity

The in-tree parity harness compares the native MoE implementation with a local
checkout of the upstream LingBot-Video implementation. It uses local files
only and does not download checkpoints.

Set paths for the upstream checkout and the cached MoE checkpoint:

```bash
export LINGBOT_VIDEO_REPO=/path/to/lingbot-video
export LINGBOT_VIDEO_MOE_MODEL=/path/to/lingbot-video-moe-30b-a3b
```

Run the lightweight sparse-block comparison:

```bash
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/lingbot_video/moe_transformer_parity.py \
  --scope block \
  --official-repo "${LINGBOT_VIDEO_REPO}" \
  --output-json /tmp/lingbot_moe_block_parity.json
```

This path covers bias-corrected router selection, group-limited top-k, routed
experts, FP32 scatter-weighted restore, padding masks, and the shared expert.
It exits with a nonzero status unless the upstream and native block outputs are
bitwise equal.

The real-checkpoint transformer comparison requires one GPU with enough memory
for the 30B checkpoint. Models are loaded sequentially, so two copies are not
resident on the GPU at the same time:

```bash
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/lingbot_video/moe_transformer_parity.py \
  --scope transformer \
  --official-repo "${LINGBOT_VIDEO_REPO}" \
  --model "${LINGBOT_VIDEO_MOE_MODEL}" \
  --output-json /tmp/lingbot_moe_transformer_parity.json
```

Transformer parity fixes the official implementation to
`diffusers:_native_math` and the native implementation to
`TORCH_SDPA + SDPBackend.MATH`. The expected result is `exact=true`,
`output.equal=true`, and zero max/mean/RMSE error. Different fused attention
kernels can introduce BF16 rounding differences before MoE routing, so
automatic attention-backend comparisons are diagnostic only and are not the
bitwise correctness oracle.

## Known Limitations

- T2V base-transformer inference only. T2I, I2V, TI2V, and the checkpoint's
  optional `refiner/` transformer are not supported by this PR.
- Only the BF16 MoE checkpoint is validated in this first integration.
- No HSDP, tensor, sequence, expert, or CFG parallelism is claimed.
- No Cache-DiT, TeaCache, CPU offload, VAE patch parallelism, or quantized
  inference is claimed.
- Optional Triton, SGLang, FP8, and alternative fused-expert backends from the
  upstream project are not included.
- Only one request per LingBot pipeline batch is currently supported.
