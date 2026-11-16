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
- Related offline example: [`examples/offline_inference/text_to_video/text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)
- Related online video API docs: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)

## Hardware Support

This recipe documents the CUDA single-GPU dense and BF16 MoE checkpoint paths.
Multi-GPU parallelism and Cache-DiT are not validated for LingBot-Video in this
PR. Quantization is rejected at startup, whether it comes from `--quantization`
or from a quantization config in the checkpoint, because the projections are
plain `nn.Linear` layers that quantization would leave unchanged.

## GPU

### 1 x NVIDIA L20X

Both checkpoints have been smoke-tested on one NVIDIA L20X at `192x320`,
9 frames, and 2 steps. Without offload the MoE smoke reserved approximately
`67.70 GiB` of GPU memory on L20X. The same shape reserved `72314 MiB` on one
H200 and `75224 MiB` on one H100, so plan for an 80 GB class card when serving
MoE without offload. Larger resolutions, frame counts, or concurrent requests
require additional headroom.

The MoE path is validated with BF16 expert weights.

#### Dense offline T2V

```bash
CUDA_VISIBLE_DEVICES=0 \
python examples/offline_inference/text_to_video/text_to_video.py \
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
python examples/offline_inference/text_to_video/text_to_video.py \
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

### 1 x NVIDIA H100 80GB or H200 with offload

Both offload modes are supported. `--enable-cpu-offload` is model-level
offload: the text encoder and the DiT swap places on the GPU, so only one of
them is resident at a time, but the complete DiT is still on the GPU while it
denoises. For the MoE smoke shape it lowers the peak from `72314 MiB` to
`63634 MiB` on one H200 and from `75224 MiB` to `66534 MiB` on one H100.
`--enable-layerwise-offload` streams the DiT blocks instead, peaks at
`13160 MiB` on both, and is the mode to use on smaller cards. If both flags are
set, layerwise offload is used.

With offload enabled the pipeline is built in host memory. With
`--enable-cpu-offload`, the startup warmup pins the DiT and the text encoder in
host memory the first time each one is swapped, so requests swap from pinned
memory instead of pageable memory. Layerwise offload keeps its own pinned
buffers.

The same modes can be selected with `--diffusion-offload-config`.
`{"mode":"module","components":["dit","text_encoder"]}` matches
`--enable-cpu-offload`, and `{"mode":"layer","components":["dit","text_encoder"]}`
matches `--enable-layerwise-offload`. The layer form streams both the DiT
blocks and the Qwen3-VL text encoder blocks. `{"mode":"layer","components":["dit"]}`
streams only the DiT and keeps the text encoder resident.

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve robbyant/lingbot-video-moe-30b-a3b \
  --omni \
  --model-class-name LingBotVideoPipeline \
  --enable-cpu-offload \
  --vae-use-tiling \
  --vae-use-slicing \
  --default-sampling-params \
  '{"0":{"num_frames":81,"num_inference_steps":40,"guidance_scale":6.0}}' \
  --port 8091
```

Submit the same text-to-video job as above, with
`model=robbyant/lingbot-video-moe-30b-a3b`.

### 1 x NVIDIA RTX 5090 32GB

Layerwise offload streams the DiT blocks and the text encoder blocks from host
memory, so the MoE checkpoint also serves on a 32 GiB card.

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with a 32 GiB RTX 5090 GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve robbyant/lingbot-video-moe-30b-a3b \
  --omni \
  --model-class-name LingBotVideoPipeline \
  --enable-layerwise-offload \
  --vae-use-tiling \
  --vae-use-slicing \
  --default-sampling-params \
  '{"0":{"num_frames":81,"num_inference_steps":40,"guidance_scale":6.0}}' \
  --port 8091
```

#### Verification

Submit the same text-to-video job as above, with
`model=robbyant/lingbot-video-moe-30b-a3b`.

#### Notes

- Memory usage: the offline runner reports `10192 MiB` worker peak reserved
  for 9 frames at `192x320` in 2 steps and `9152 MiB` for 25 frames at
  `480x480` in 20 steps. Up to that shape the peak is set by the VAE, the
  non-block modules, and the prefetched blocks; 81 frames at `480x832` in
  4 steps raise it to `16104 MiB`. Selecting only `"dit"` keeps the text
  encoder resident and peaks at `16278 MiB` for 9 frames at `192x320`. The
  40-step serving default has not been measured on this card.
- Key flags: layer-mode offload is required on this card, and
  `--enable-layerwise-offload` is its lowest-memory form.
  `--vae-use-tiling` and `--vae-use-slicing` keep decode memory flat at larger
  shapes. The host holds the offloaded weights, so size host memory for the full
  BF16 checkpoint.
- Known limitations: `--enable-cpu-offload` moves the complete DiT to the GPU for
  the first denoising forward, so it fails on this card with a CUDA
  out-of-memory error. With the default startup warmup that forward runs before
  the server reports ready. Use `--enable-layerwise-offload` instead.

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

Local MoE offload validation on one RTX 5090 (32 GiB) with the offline
`text_to_video.py` runner, `--vae-use-tiling --vae-use-slicing`, seed `42`,
reporting worker peak reserved GPU memory and generation time for one request:

| Offload selector | 9 frames, `192x320`, 2 steps | 25 frames, `480x480`, 20 steps | 81 frames, `480x832`, 4 steps |
|------------------|------------------------------|--------------------------------|-------------------------------|
| `--enable-layerwise-offload` | `10192 MiB`, `6.7s` | `9152 MiB`, `47.4s` | `16104 MiB`, `38.4s` |
| `{"mode":"layer","components":["dit"]}` | `16278 MiB`, `4.8s` | not measured | not measured |
| `--enable-cpu-offload` | CUDA out-of-memory during warmup | not measured | not measured |

- `{"mode":"layer","components":["dit","text_encoder"]}` selects the same
  plan as `--enable-layerwise-offload` and matched its peaks.
- At the small shape, repeated runs of one configuration with the same seed
  alternate between two outputs that differ by up to `45` of `255`, and every
  layerwise selector produced one of those two. At the larger shape, the
  `--enable-layerwise-offload` output from before the text encoder blocks
  could stream matched the current output bitwise.
- `vllm serve` with `--enable-layerwise-offload` reported ready after about
  `70s` and completed 8 of 8 random 25-frame `480x480` 10-step requests at
  concurrency 2, with a mean latency of `43.6s` and `11050 MiB` in use
  afterwards. With only the DiT streamed it completed 8 of 8 with `17212 MiB`
  in use.
- At vLLM-Omni `0fc38ce3b`, which built the pipeline on the GPU, every
  offload selector ran out of CUDA memory while the pipeline loaded on this
  card, both offline and in `vllm serve`.
- `--enable-cpu-offload` moves the complete DiT to the GPU for the first
  denoising forward. With the default startup warmup that forward is the dummy
  run, so the server never reports ready on this card.
- Distributed layerwise offload is rejected at startup for
  `--enable-distributed-layerwise-offload`, `weight_transfer: "allgather"`, and
  `resident_layers`.

Dense checkpoint on the same card at the small shape: no offload peaks at
`14340 MiB` in `0.46s`, `--enable-cpu-offload` at `9408 MiB` in `1.99s`,
`{"mode":"layer","components":["dit"]}` at `12028 MiB` in `0.59s`, and
`--enable-layerwise-offload` at `4796 MiB` in `0.79s`. The no-offload, CPU
offload, and layerwise outputs match each other bitwise.

Local MoE offload validation on one H200 with the same runner, shapes, and
seed, reporting worker peak reserved GPU memory. Generation times are in the
H100 table below:

| Offload selector | 9 frames, `192x320`, 2 steps | 25 frames, `480x480`, 20 steps |
|------------------|------------------------------|--------------------------------|
| none | `72314 MiB` | `72314 MiB` |
| `--enable-cpu-offload` | `63634 MiB` | `63634 MiB` |
| `{"mode":"module","components":["dit"]}` | `72126 MiB` | not measured |
| `{"mode":"module","components":["text_encoder"]}` | `66960 MiB` | not measured |
| `--enable-layerwise-offload` | `13160 MiB` | `13160 MiB` |
| `{"mode":"layer","components":["dit"]}` | `20286 MiB` | not measured |

- Each model-level selector matched the previous implementation bitwise when
  the two ran back to back.
- `vllm serve` with `--enable-cpu-offload` reported ready after about `86s`,
  logged `Model-level offloading enabled`, and `nvidia-smi` showed
  `60811 MiB` in use after one 9-frame request, versus `73087 MiB` without
  offload. With `--enable-layerwise-offload` it showed `14215 MiB`, and with
  only the DiT streamed `21321 MiB`.

Local MoE offload validation on one H100 80GB with the same runner and seed,
reporting worker peak reserved GPU memory and generation time for one request:

| Offload selector | 9 frames, `192x320`, 2 steps | 25 frames, `480x480`, 20 steps | 81 frames, `480x832`, 4 steps |
|------------------|------------------------------|--------------------------------|-------------------------------|
| none | `75224 MiB`, `1.4s` | `75224 MiB`, `10.6s` | not measured |
| `--enable-cpu-offload` | `66534 MiB`, `3.8s` | `66534 MiB`, `13.7s` | `69896 MiB`, `20.6s` |
| `{"mode":"module","components":["dit"]}` | `75026 MiB`, `3.3s` | not measured | not measured |
| `{"mode":"module","components":["text_encoder"]}` | `66940 MiB`, `1.3s` | not measured | not measured |
| `--enable-layerwise-offload` | `13160 MiB`, `5.1s` | `13160 MiB`, `45.4s` | `21396 MiB`, `17.9s` |
| `{"mode":"layer","components":["dit"]}` | `20286 MiB`, `4.9s` | not measured | not measured |

- At vLLM-Omni `0fc38ce3b`, both layerwise selectors kept the whole model on
  the GPU and peaked at `74338 MiB`, while the model-level selectors already
  ran with the same peaks as above.
- `vllm serve` completed 8 of 8 random 25-frame `480x480` 10-step requests at
  concurrency 2 in every mode. `nvidia-smi` showed `61541 MiB` in use
  afterwards with `--enable-cpu-offload` at a mean latency of `16.5s`,
  `14311 MiB` with `--enable-layerwise-offload` at `43.6s`, and `76093 MiB`
  without offload at `10.8s`.

The e2e coverage for both offload flags lives in
[`tests/e2e/online_serving/test_lingbot_video_offload.py`](../../tests/e2e/online_serving/test_lingbot_video_offload.py).

Do not treat these as production benchmarks; they are functional smoke plus
controlled numerical-parity evidence for single-GPU validation runs.

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
- No Cache-DiT, TeaCache, VAE patch parallelism, or quantized inference is
  claimed.
- CPU and layerwise offload are supported. Only layerwise offload brings the 30B
  checkpoint onto a 32 GiB card; model-level CPU offload loads the complete DiT
  for denoising and peaked at `63634 MiB` on one H200 and `66534 MiB` to
  `69896 MiB` on one H100. Offload has not been measured on cards with more
  than 32 GiB and less than 80 GB of memory.
- Quantization is rejected at startup.
- Distributed layerwise offload is rejected at startup. This covers
  `--enable-distributed-layerwise-offload`, `weight_transfer: "allgather"`, and
  `resident_layers` in `--diffusion-offload-config`. Each of these selects the
  distributed layerwise backend, which has not been validated with this
  pipeline, and the model declares no resident DiT paths.
- Optional Triton, SGLang, FP8, and alternative fused-expert backends from the
  upstream project are not included.
- Only one request per LingBot pipeline batch is currently supported.
