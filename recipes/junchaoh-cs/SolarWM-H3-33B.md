# SolarWM-H3-33B

SolarWM-H3 is a camera-controlled autoregressive world model built from
MiniMax-H3. This integration loads the published Stage-2 student and its LoRA
weights, retaining H3 text/image/audio conditioning and Solar's camera ProPE.
The output is video at 24 FPS; the model's audio input is the encoded silence
condition used during training.

The model is published at
[`junchaoh-cs/SolarWM-H3-33B`](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B).
Its `SolarWM-h3-33B-base/` directory contains the Diffusers base components;
choose a published Stage-2 artifact directory for `solarwm_checkpoint_path`.
The original H3 weights have their own access and license requirements.

Install vLLM-Omni with a Diffusers version that provides
`AutoencoderKLMiniMaxH3` and `AutoencoderKLMiniMaxH3Audio`, and install FFmpeg
for MP4 export. Authenticate to Hugging Face if the selected components
require access approval.

The initial execution configuration uses BF16 DiT weights, eager execution,
TP1 and optional strict pure Ulysses sequence parallelism. Tensor parallelism,
Ring/AllGather attention, CFG parallelism, HSDP, quantization and approximate
DiT caching are rejected. Four GPUs with Ulysses degree 4 are a useful starting
configuration for the full 768×1344 workload; memory and speed depend on the
hardware and require workload-specific measurement.

Ulysses replicates the DiT weights. Rank 0 also holds the Qwen encoder during
conditioning, so each device must have sufficient memory independently.
Published-checkpoint validation used NVIDIA L20X devices with approximately
140 GiB of memory; full-resolution memory requirements have not been measured.

Use the shared image-to-video runner with the published model root. It discovers
`SolarWM-h3-33B-sgf-stage2-158f-fix/` alongside the base components. Create
the complete camera inputs shown below, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/offline_inference/image_to_video/image_to_video.py \
  --model junchaoh-cs/SolarWM-H3-33B \
  --model-class-name SolarWMH3Pipeline \
  --image /data/first_frame.png \
  --prompt "A camera moves forward through a sunlit room." \
  --negative-prompt "" --seed 42 \
  --height 768 --width 1344 --num-frames 158 \
  --num-inference-steps 4 --guidance-scale 1 --flow-shift 12 --fps 24 \
  --ulysses-degree 4 --enforce-eager \
  --extra-body "$(cat /data/request.json)" \
  --output solarwm_h3.mp4
```

The camera input contains absolute camera-to-world matrices and pinhole
intrinsics. Supply one pose for every requested pixel frame. This example
creates a complete 158-frame forward trajectory and both files used by the
CLI and Python examples:

```bash
python - <<'PY'
import json
from pathlib import Path

import numpy as np

poses = np.broadcast_to(np.eye(4), (158, 4, 4)).copy()
poses[:, 2, 3] = np.linspace(0.0, 1.57, 158)
camera = {
    "schema": "solarwm.camera_trajectory.v1",
    "c2w": poses.tolist(),
    "intrinsics": [[0.8, 0, 0.5], [0, 0.8, 0.5], [0, 0, 1]],
    "intrinsics_normalized": True,
}
Path("/data/camera.json").write_text(json.dumps(camera))
Path("/data/request.json").write_text(json.dumps({"camera": camera}))
PY
```

Intrinsics may also be supplied
per frame as `[T,3,3]`. Set `intrinsics_normalized: true` for focal lengths and
principal points normalized by image dimensions; omit it for pixel intrinsics.
Output height and width must be multiples of 32.

Camera motion is controlled by poses. Intrinsics are validated and normalized,
but the released checkpoint uses fixed focal values in PRoPE. Changing supplied
focal lengths or principal points does not provide zoom conditioning.

The shared runner accepts camera controls through `--extra-body`. It does not
expose arbitrary engine `model_config` overrides. To use a separate local
Stage-2 checkpoint, use the Omni API and set `solarwm_checkpoint_path` explicitly:

```python
import json
from pathlib import Path

import numpy as np
from diffusers.utils import export_to_video

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    camera = json.loads(Path("/data/camera.json").read_text())
    omni = Omni(
        model="junchaoh-cs/SolarWM-H3-33B",
        model_class_name="SolarWMH3Pipeline",
        dtype="bfloat16",
        enforce_eager=True,
        tensor_parallel_size=1,
        ulysses_degree=4,
        model_config={
            "solarwm_checkpoint_path": "/models/SolarWM-h3-33B-sgf-stage2-158f-fix",
            "ar_diffusion_height": 768,
            "ar_diffusion_width": 1344,
        },
    )
    prompt = {
        "prompt": "A camera moves forward through a sunlit room.",
        "multi_modal_data": {"image": "/data/first_frame.png"},
    }
    sampling = OmniDiffusionSamplingParams(
        height=768,
        width=1344,
        num_frames=158,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=42,
        fps=24,
        output_type="pil",
        extra_args={"camera": camera},
    )
    try:
        output = omni.generate(prompt, sampling, use_tqdm=False)[0]
    finally:
        omni.close()
    frames = np.asarray(output.images[0], dtype=np.float32) / 255.0
    export_to_video(list(frames), "solarwm_h3.mp4", fps=24)


if __name__ == "__main__":
    main()
```

`model_config.solarwm_silence_path` optionally supplies a safetensors artifact
with the official `silence_158f` tensor. When absent, the audio VAE encodes the
fixed stereo zero waveform and applies its trained latent normalization.
Silence conditioning is encoded audio; an all-zero latent tensor changes the
model input.

The ordinary offline request runs a complete rollout with bounded local KV and returns
one complete video. The opt-in deployment file is
`vllm_omni/deploy/solarwm_h3.yaml`; selecting it enables the AR engine and
streamed step execution. Its
`ar_diffusion_height` and `ar_diffusion_width` fix the cache geometry and must
match each request. The AR runtime serializes generation and owns session KV;
`max_num_seqs` remains 1.

Clean-history KV and retained decoder buffers are bounded. To preserve the
released random-number sequence, initialization samples the complete trajectory
noise on GPU before moving it to CPU. That startup allocation and the retained
host noise grow with the requested rollout length.

With the AR deployment, consume `omni.generate(..., py_generator=True)` and
process every returned output. Outputs with an empty `images` list contain
metadata only; concatenate the videos in subsequent `output.images` lists in
arrival order. The shared image-to-video CLI above exports a complete ordinary
request and selects its first output, so use a streaming consumer when selecting
`deploy_config="vllm_omni/deploy/solarwm_h3.yaml"`.

Streaming decode retains session-specific latent lookahead and raw H3 overlap.
Each five-latent block advances 17 frames, and the decoder waits for two future
latents before completing its first clip. The first AR chunk therefore returns
metadata with no video frames; after the second block the first 17 frames are
available. Reset, close and failed generation
release the retained decode buffers. The final flush removes official temporal
padding and trims the result to `num_frames`.

CPU tests cover camera layout, sampling/KV transactions and decoder boundaries,
including a reduced real H3 VAE. The published full H3 VAE was also checked on
an NVIDIA L20X: streamed and whole decoding were bit-identical for 7, 8, 10 and
17 latent frames at 32×32, and for 7 latent frames with spatial tiling enabled
at 288×288. This verifies decoder equivalence; full-model video quality,
multicard speed and end-to-end latency require separate fixed-input runs.

The published 33B base and Stage-2 EMA also completed native pipeline smoke runs
for 22 and 158 frames at 32×32 on one NVIDIA L20X, including real text/image/audio
conditioning, finite video output, exact frame counts and session cleanup.
These synthetic-input runs verify execution rather than visual quality.

A reduced native H3 transformer also passed eight sequential noisy/clean chunk
pairs with strict two-rank Ulysses against one-rank execution. Prediction maximum
absolute error was `1.79e-7`, and the raw K/V histories joined across head shards
were bit-identical. This exercises packed sequence padding and sliding history;
it is independent of full-checkpoint quality and performance validation.

To opt into the published-checkpoint generation regression, provide a local
model root containing both the base components and the Stage-2 artifacts:

```bash
CUDA_VISIBLE_DEVICES=0 \
  VLLM_TEST_SOLARWM_H3_MODEL=/models/SolarWM-H3-33B \
  pytest tests/diffusion/models/solarwm_h3/test_pipeline_device.py -q
```

The test downloads no weights and runs at 32×32 to check the real conditioning,
student, adapter and decoder path, output trimming, and session cleanup. It
requires enough GPU memory for the unquantized 33B model and conditioners.
