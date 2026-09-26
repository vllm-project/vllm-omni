# SANA-Video 2.0 5B

`SanaVideo2Pipeline` implements native text-to-video (T2V) and
text-image-to-video (TI2V) inference for the official **50-step**
`Efficient-Large-Model/SANA-Video_2.0_5B_720p` checkpoint. Providing one image
selects TI2V. The 4-step preview and 14B models are separate variants and are
not supported by this pipeline.

The transformer and denoising loops run in vLLM-Omni. The pipeline reuses Gemma
for text conditioning and the Diffusers LTX 2.3 VAE. T2V uses second-order
multistep flow DPM-Solver++; TI2V uses FlowMatch Euler with a clean, fixed first
latent frame and frame-dependent timesteps.

Sequence parallelism supports SP2 and SP4/SP8 with the configurations below.
SP2 has been measured on two A800 GPUs; SP4/SP8 GPU validation is pending.

## Components

Default Hub revisions are pinned:

| Component | Repository | Revision |
| --- | --- | --- |
| Transformer | `Efficient-Large-Model/SANA-Video_2.0_5B_720p` | `f2d95fa06400f186fd1b077d12c69c8ac58aba48` |
| VAE | `Efficient-Large-Model/LTX-2.3-Diffusers` | `362acdf779d42e785fb26910c32254e9458e78c2` |
| Text encoder/tokenizer | `Efficient-Large-Model/gemma-2-2b-it` | `569d9809d0c8b6722d4d31b5a77a2ec7a400650a` |

A local transformer directory must contain `config.yaml` with
`model.model: SanaVideo2_5B` and
`checkpoints/SANA_Video_2.0_5B_720p.pth`. It is detected without a Diffusers
`model_index.json`. The VAE path must contain its `vae/` subdirectory. Text
encoder and tokenizer overrides point directly to their component directories.

## Native offline inference

```python
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    engine = Omni(
        model="Efficient-Large-Model/SANA-Video_2.0_5B_720p",
        dtype="bfloat16",
        enforce_eager=True,
        # Optional local component overrides:
        # model_config={
        #     "vae_model": "/path/to/LTX-2.3-Diffusers",
        #     "text_encoder_model": "/path/to/gemma/text_encoder",
        #     "tokenizer_model": "/path/to/gemma/tokenizer",
        # },
    )
    try:
        prompt = {"prompt": "A small red boat sailing across a calm lake."}
        # For TI2V, supply exactly one RGB image:
        # prompt["multi_modal_data"] = {"image": Image.open("reference.png").convert("RGB")}
        outputs = engine.generate(
            prompt,
            OmniDiffusionSamplingParams(
                height=64,
                width=96,
                num_frames=9,
                num_inference_steps=50,
                guidance_scale=8.0,
                seed=42,
                extra_args={"motion_score": 10, "flow_shift": 12.0},
            ),
        )
        return outputs
    finally:
        engine.close()


if __name__ == "__main__":
    main()
```

The dimensions above are a small correctness smoke test, not a video quality
preset. The upstream release defaults are height 736, width 1280, 193 frames,
50 steps, CFG 8, flow shift 12, and 24 FPS. Heights and widths must be positive
multiples of 32; frame counts must satisfy `(num_frames - 1) % 8 == 0`.
The release envelope is at most 193 frames, area `736 * 1280`, and 1280 pixels
on either axis. Successful small-shape validation alone does not establish
quality or memory requirements at the release dimensions.

The pipeline adds the release prompt instruction and motion-score suffix.
When the negative prompt is omitted, it uses the release negative prompt;
passing an empty string explicitly requests empty negative conditioning.
CFG is enabled when guidance exceeds 1. Images undergo RGB conversion,
bicubic resize to fill, and center cropping inside the pipeline.

Component overrides use `model_config`, with optional `vae_revision`,
`text_encoder_revision`, and `tokenizer_revision`. `custom_pipeline_args` is a
separate framework mechanism for replacing the pipeline class.

## Current execution scope

The native default uses one NVIDIA GPU with BF16 or FP32 transformer weights.
Experimental pure-Ulysses SP, TP2, and CFG2 paths are described below.
Pipeline parallelism, distributed VAE, HSDP, quantization, caching, and
CPU/layerwise offload are rejected. Custom sigma/timestep schedules and multiple
videos per request are not implemented. Reducing the inference step count does
not turn the formal checkpoint into the 4-step preview.

## Experimental sequence parallelism

The SP-only measurements in this section use vLLM 0.29.0, PyTorch 2.13 and
Diffusers 0.40, with TP=1 and CFG parallel=1. Native batch CFG remains enabled.
The transformer
splits flattened video tokens before allocating Attention Residual buffers.
Linear layers sum their FP32 states over the full SP group; softmax anchors
use the shared Ulysses communication with native FP32 SDPA. Each denoising
prediction is gathered back to all ranks. Text conditioning, patch embedding,
the sampler and VAE remain replicated.

SP2 results below compare the same checkpoint, conditioning, seed and sampler
settings against SP1. Both FP32 and BF16 runs use the native BF16 VAE.

| Configuration | Input contract | Validation status |
| --- | --- | --- |
| SP1 | Native path | Native evidence below; small real-checkpoint BF16 repeats bitwise |
| SP2, `strict` | Token count divisible by 2 | NCCL, full-size video comparisons and repeated performance measurements below |
| SP2, `advanced_uaa` | At least one video token per rank | NCCL and real-checkpoint entrypoint smoke checked; full-size trajectories not run |
| SP4/8, `advanced_uaa` | At least one video token per rank | GPU results unavailable |

The released model has 10 softmax heads. SP4/SP8 require `advanced_uaa`, which
pads heads inside the shared communication strategy. Video tokens are split
unevenly without introducing attention padding. Both RoPE and TI2V frame
conditioning retain their global coordinates. Inputs with fewer tokens than
SP ranks, or uneven token counts in `strict` mode, are rejected before model
collectives. There is no Ring or AllGather-KV integration in this path.

For offline inference, add these arguments to the `Omni` constructor above:

```python
ulysses_degree=2,
ulysses_mode="advanced_uaa",
```

For an experimental two-GPU server:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve Efficient-Large-Model/SANA-Video_2.0_5B_720p \
  --omni --dtype bfloat16 --enforce-eager --usp 2 --ulysses-mode advanced_uaa \
  --host 127.0.0.1 --port 8091
```

The normal startup warmup uses 512x512 and 9 frames (512 latent tokens).
The 64x96, 9-frame example has 12 latent tokens; use `advanced_uaa` for SP8.
With vLLM 0.29.0, both SP1 and SP2 completed standard Omni startup, default
warmup and three consecutive two-step T2V/TI2V requests, including a changed
shape. These two-step requests are an entrypoint smoke test.

### Measured results

The focused tests exercise actual multi-process collectives on a narrow
32-layer transformer with 24 linear layers, eight 10-head softmax anchors,
nonzero learned Attention Residual projections and batch CFG. This preserves
depth and communication structure, but does not replace real 5B checkpoint
or full-size video validation.

CPU FP32 Gloo observations (vLLM 0.29.0, seed 8006, hidden size 120, 32 layers;
all ranks):

| Check | SP2 strict max abs / relative L2 | SP4 advanced max abs / relative L2 |
| --- | --- | --- |
| T2V/TI2V transformer, consecutive shapes | `9.239e-7 / 6.212e-7` | `4.768e-7 / 4.006e-7` |
| Five-step T2V, CFG 8 | `2.146e-6 / 8.697e-7` | `1.729e-6 / 7.941e-7` |
| Five-step TI2V, CFG 8 | `4.835e-6 / 2.545e-6` | `7.488e-6 / 3.277e-6` |

These are maxima across the recorded cases/steps, compared with the same
weights on SP1. SP4 includes 15 tokens split 4/4/4/3 and head padding 10 to 12.
Neither the random narrow model nor CPU Gloo establishes GPU or video quality.

Two A800-SXM4-80GB GPUs with NVLink pass the narrow-model NCCL checks
in both SP2 modes (PyTorch 2.13, vLLM 0.29.0, Diffusers 0.40, seed 8006).
Measurements use highest FP32 matmul precision and disable cuDNN TF32:

| Check | SP2 strict max abs / relative L2 | SP2 advanced max abs / relative L2 |
| --- | --- | --- |
| T2V/TI2V transformer, consecutive shapes | `6.56e-7 / 3.48e-7` | `7.75e-7 / 4.76e-7` |
| Five-step T2V, CFG 8 | `1.91e-6 / 8.10e-7` | `1.91e-6 / 8.10e-7` |
| Five-step TI2V, CFG 8 | `4.14e-6 / 2.03e-6` | `4.14e-6 / 2.03e-6` |

SP2 advanced includes 15 tokens split 8/7. SP4/SP8 NCCL were not run on this
two-GPU host.

```bash
python -m pytest -o addopts= -q tests/diffusion/models/sana_video2
python -m pytest -o addopts= -s -q \
  tests/diffusion/distributed/test_sana_video2_sp_numeric.py -k gloo
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -o addopts= -s -q \
  tests/diffusion/distributed/test_sana_video2_sp_numeric.py -k nccl
```

The NCCL cases skip degrees larger than the visible device count. Tests print
maximum absolute and relative L2 error by rank and compare every recorded
sampler step.

## Experimental TP and CFG parallelism

Set `tensor_parallel_size=2` for TP2 or `cfg_parallel_size=2` for CFG2 in the
offline `Omni` constructor. TP splits self- and cross-attention heads and the
SwiGLU intermediate width. Q/K RMSNorm still computes statistics across the
full checkpoint channel dimension; other transformer state remains replicated.
CFG2 computes the positive branch on CFG rank 0 and the negative branch on CFG
rank 1. With guidance at most 1, both ranks compute the conditional branch. Both
ranks continue the same sampler after the branch exchange.

For TP2+SP2, the released 10 softmax heads become five heads per TP rank.
Select `ulysses_mode="advanced_uaa"`; `strict` cannot divide those five heads
across SP2. The same mode supports uneven video-token shards. A combined
offline configuration is:

```python
engine = Omni(
    model="Efficient-Large-Model/SANA-Video_2.0_5B_720p",
    dtype="bfloat16",
    enforce_eager=True,
    tensor_parallel_size=2,
    cfg_parallel_size=2,
    ulysses_degree=2,
    ulysses_mode="advanced_uaa",
)
```

This combination needs eight GPUs. TP2 alone and CFG2 alone need two GPUs;
TP2+CFG2 and CFG2+SP2 need four. The implementation accepts only TP1/2 and
CFG1/2. Keep `--usp` equal to the Ulysses degree when serving with SP.

The new paths were checked with synthetic same-weight models using real CPU
Gloo process groups: TP2 and TP2+SP2 on 32 layers, and CFG2, TP2+CFG2,
CFG2+SP2, TP2+SP2+CFG2 on T2V/TI2V sampler steps with guidance on/off.
These checks do not establish 5B checkpoint behavior, NCCL correctness,
decoded-video accuracy, or throughput. Multi-GPU NCCL and full-checkpoint runs
remain unverified for TP and CFG.
Run the focused checks with:

```bash
python -m pytest -o addopts= -q tests/diffusion/models/sana_video2
python -m pytest -o addopts= -q \
  tests/diffusion/distributed/test_sana_video2_tp_numeric.py \
  tests/diffusion/distributed/test_sana_video2_tp_cfg_pipeline.py
```

In the direct pipeline, real 5B weights were compared at three small shapes
for 50 steps with seed 42, CFG 8 and flow shift 12. Across 150 FP32 latent
pairs, worst relative L2 was `8.549824e-6` and maximum absolute error was
`1.23977e-4`. Decoding used the BF16 VAE and reached `0.0086742266` relative
L2; the decoded result is therefore not a pure FP32 comparison.
For small BF16 cases, SP1 repeated bitwise and SP2 final latent relative L2
ranged from `0.0531` to `0.0577`.

Full-size BF16 SP1/SP2 T2V and TI2V runs at 736x1280/193 frames completed
50 steps and decoding. Final latent relative L2 was `0.08282` (T2V) and
`0.16842` (TI2V); decoded relative L2 was `0.12059` and `0.20095`, respectively.
All saved tensors were finite and both SP2 ranks agreed. Full-size FP32
results with the same BF16 VAE are:

| Case | Final latent relative L2 / max abs | Decoded relative L2 / max abs |
| --- | --- | --- |
| T2V | `4.253818e-4 / 0.0953167` | `0.00672967 / 0.1796875` |
| TI2V | `5.317898e-5 / 0.00335784` | `0.00592159 / 0.19921875` |

All 100 saved FP32 latent pairs and both decoded pairs were finite; the two
SP2 ranks agreed on final output hashes. A full-size 32-layer FP32 forward
on identical saved inputs measured `5.96490e-7` relative L2 and `7.89762e-6`
maximum absolute error. Decoded video comparisons are reported separately.

### Decoded video comparison

The following SP1/SP2 metrics use all 193 native-postprocessed RGB uint8
frames before MP4 encoding. PSNR uses the mean squared error over all pixels;
SSIM uses the repository accuracy helper's TorchMetrics settings. The prompt
is “A small red boat sailing across a calm lake.” TI2V uses a fixed synthetic
RGB conditioning image. Both cases use seed 42 and 50 denoising steps.

| Transformer dtype | Case | PSNR, dB | Mean SSIM | Worst-frame SSIM |
| --- | --- | ---: | ---: | ---: |
| BF16 | T2V | 30.418 | 0.958446 | 0.952506 |
| BF16 | TI2V | 25.733 | 0.938138 | 0.915454 |
| FP32 | T2V | 53.808 | 0.999215 | 0.999113 |
| FP32 | TI2V | 54.311 | 0.999196 | 0.998331 |

These measurements cover one prompt/seed and one TI2V conditioning image.

### Performance

Measured on 2x A800-SXM4-80GB with NVLink, using BF16 T2V, 736x1280, 193 frames,
10 denoising steps, seed 42, CFG 8 and flow shift 12. Each configuration ran
five measured requests after a two-step warmup; compile, caching and offload
were disabled. End-to-end timing covers `pipeline.generate`, including text
conditioning and VAE decoding, with device synchronization at both boundaries.
Startup and artifact writes are excluded. SP2 reports the slower rank.

| Metric | SP1 | SP2 strict |
| --- | --- | --- |
| End-to-end median (range), seconds | `74.589 (74.450–74.681)` | `39.365 (39.194–39.407)` |
| DiT median (range), seconds | `71.849 (71.717–71.933)` | `36.610 (36.466–36.645)` |
| Denoising median (range), seconds | `71.853 (71.721–71.938)` | `36.614 (36.470–36.650)` |
| Peak allocated per rank, GiB | `32.390` | `32.390` |
| Peak reserved per rank, GiB | `46.342` | `42.941` |

The measured end-to-end speedup is **1.895x** and DiT speedup is **1.963x**
for this ten-step workload. These timings do not measure 50-step latency.
A separate two-step profile recorded the FP32 softmax attention kernel at
`8.476 s` on SP1 and `4.107 / 4.094 s` per SP2 rank. NCCL kernel durations
summed to `0.105 / 0.107 s` per SP2 rank; concurrent streams can overlap.

## Online requests

The official model ID and local release directories resolve to the same native
pipeline. Start a single-GPU server with:

```bash
vllm serve Efficient-Large-Model/SANA-Video_2.0_5B_720p \
  --omni --dtype bfloat16 --enforce-eager --host 127.0.0.1 --port 8091
```

For local component overrides, use the generic diffusion stage override:

```bash
--stage-overrides '{"0":{"model_config":{"vae_model":"/path/to/LTX-2.3-Diffusers","text_encoder_model":"/path/to/gemma/text_encoder","tokenizer_model":"/path/to/gemma/tokenizer"}}}'
```

Create a small T2V request:

```bash
curl --fail-with-body http://127.0.0.1:8091/v1/videos \
  -F 'prompt=A small red boat sailing across a calm lake.' \
  -F height=64 -F width=96 -F num_frames=9 -F fps=24 \
  -F num_inference_steps=50 -F guidance_scale=8 -F seed=42 \
  -F 'extra_params={"motion_score":10,"flow_shift":12}'
```

For TI2V, add `-F 'input_reference=@reference.png'` to the same request. Use
the returned `id` to query `/v1/videos/{id}` and, once completed, download
`/v1/videos/{id}/content`. Omitting `num_frames` selects 193; this default does
not prohibit shorter legal clips. The service preserves the source image's
geometry for the pipeline's own resize and center crop.

## Validation scope

The following results are recorded native SP1 baseline evidence from the
original integration, not new measurements of the SP path. They used one
NVIDIA H20-3e with BF16 weights and the release FP32 self-attention path:

| Check | Scope | Result |
| --- | --- | --- |
| Upstream sampler comparison | T2V/TI2V, steps 2/5/50, shifts 1/12 | Exact trajectories |
| Real-weight upstream comparison | 64x96, 9 frames, 50 steps, CFG 8 | Exact step latents and decoded outputs |
| Native offline and HTTP video API | T2V/TI2V, 64x96, 9 frames | Generation and MP4 export/download pass |
| Full release dimensions | T2V/TI2V, 736x1280, 193 frames, 50 steps, CFG 8 | Finite decoded outputs; 193-frame MP4 files at 24 FPS |

The full-dimension runs used approximately 32.42 GiB peak allocated GPU
memory. Observed wall times were 498 seconds for T2V and 496 seconds for TI2V,
including text/image conditioning, denoising, VAE decoding and saving the raw
output tensor, but excluding MP4 encoding. These are single-run capacity and
correctness observations, not a repeated latency benchmark or a speedup claim.
Full-dimension output was not compared numerically with an upstream full video;
the exact real-weight comparison above used the stated small dimensions.
