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

Use one NVIDIA GPU with BF16 or FP32 transformer weights. TP, SP, CFG
parallelism, pipeline parallelism, distributed VAE, HSDP, quantization,
caching, and CPU/layerwise offload are rejected. Custom sigma/timestep schedules
and multiple videos per request are not implemented. Reducing the inference
step count does not turn the formal checkpoint into the 4-step preview.

Upstream correctness reference:
`NVlabs/Sana@e93c883e10730ee5a4a6edf1cbcf501dc4ef753b`.

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

Validated on one NVIDIA H20-3e with BF16 weights and the release FP32
self-attention path:

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
