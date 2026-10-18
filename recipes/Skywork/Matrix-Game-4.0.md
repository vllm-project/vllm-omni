# Matrix-Game 4.0 for neural game rendering

## Summary

- Vendor: Skywork
- Model: `Skywork/Matrix-Game-4.0`
- Task: Low-poly geometry-to-video rendering
- Mode: Single-GPU inference through the Python `DiffusionEngine`

## When to use this recipe

Use this recipe to run City Racing at **864×480 with one denoising step**, standard
online FP8, CUDA Graph replay, `TORCH_SDPA` and the TAEW2 codec. This is the
recommended configuration for the measured environment below. Castle and Ancient
Egypt use the same pipeline with their own checkpoints and geometry inputs.

The game engine supplies low-poly RGB frames. vLLM-Omni generates the detailed
video; the application handles controls, HUD, audio and browser transport. This
integration exposes a Python chunk API, not an HTTP/WebSocket game server.
No coding-agent API is needed while playing.

## References

- Upstream repository: <https://github.com/SkyworkAI/Matrix-Game/tree/main/Matrix-Game-4>
- Model weights and game assets: <https://huggingface.co/Skywork/Matrix-Game-4.0>
- Native game setup: <https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-4/docs/racing.md>
- Pipeline and framework reuse: [Matrix-Game README](../../vllm_omni/diffusion/models/matrix_game4/README.md)

## Hardware Support

The integration uses shared, device-dispatched vLLM layers without an SM120-only
implementation or architecture gate. The configuration below is measured on
one RTX PRO 5000 72GB Blackwell. RTX 5090 remains a deployment target; its memory
fit, output quality and performance have not been measured here.

## GPU

### 1x NVIDIA RTX PRO 5000 72GB Blackwell

#### Environment

- OS: Linux x86_64
- Python: 3.12.13
- Driver / runtime: NVIDIA driver `580.95.05`, CUDA `13.0`
- PyTorch: `2.13.0+cu130`
- vLLM: `0.31.0`
- Transformers / Diffusers: `5.14.1` / `0.40.0`
- vLLM-Omni inference revision used for the measurement: `4afaeca43a91b0141611d474fd8d2910384f7add`

From a vLLM-Omni checkout containing this integration:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm==0.31.0 --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6
uv pip install -e .
```

The driver must support the selected CUDA build. See the
[GPU installation guide](../../docs/getting_started/installation/gpu.md) for the
framework installation requirements. The model uses Omni's native implementation;
an external Matrix-Game Python inference package is not required.

#### Prepare resources

Run these commands from the vLLM-Omni repository root in a fresh resource
directory. They download the Racing checkpoint, shared text encoder/tokenizer,
TAEW2 codec and native game assets at pinned revisions.

```bash
export MATRIX_RESOURCES="$PWD/matrix-resources"
MATRIX_REV=4419accdb7bb7881234c5914de8cd915266e0325
WAN_REV=b8bc1a65ab71d054ba4636dc0dac104aa4df2686

hf download alibaba-pai/Wan2.2-Fun-5B-Control \
  --revision "$WAN_REV" \
  --include 'models_t5_umt5-xxl-enc-bf16.pth' 'google/umt5-xxl/*' \
  --local-dir "$MATRIX_RESOURCES/base"

hf download Skywork/Matrix-Game-4.0 --revision "$MATRIX_REV" \
  --include 'base/taew2_2.pth' 'LICENSE' 'THIRD_PARTY_NOTICES.md' 'licenses/*' \
    'driving_game/model.safetensors' 'driving_game/sink.png' \
    'driving_game/inference.json' 'racing/assets/*' \
  --local-dir "$MATRIX_RESOURCES"

mkdir -p "$MATRIX_RESOURCES/models"
ln -s ../driving_game "$MATRIX_RESOURCES/models/racing"
```

The required model layout is:

```text
matrix-resources/
  base/
    models_t5_umt5-xxl-enc-bf16.pth
    google/umt5-xxl/
    taew2_2.pth
  models/racing/
    model.safetensors
    sink.png
    inference.json
  racing/assets/                  # External game input generation
```

Each scene checkpoint is approximately 10 GB; the shared text checkpoint is
approximately 11.4 GB. Allow additional download-cache space and host RAM for
checkpoint loading. `inference.json` supplies the released scene caption.
Only one scene is loaded per worker.

| Scene argument | Hub checkpoint directory | Local model directory |
| --- | --- | --- |
| `racing` | `driving_game` | `models/racing` |
| `castle` | `fps-castle` | `models/castle` |
| `ancient-egypt` | `fps-egypt` | `models/ancient-egypt` |

Other scenes require their corresponding checkpoint, sink, inference metadata
and actual game geometry. Do not render Racing geometry with a shooter checkpoint.

#### Generate low-poly input

The released C++ game owns world state, vehicle physics and collisions. Its
OpenGL renderer draws the geometry; the neural model receives the resulting
RGB frames. The browser only sends controls and displays the final video.
For headless live capture, EGL creates the OpenGL context without a display.
This requires working hardware OpenGL/EGL and graphics driver libraries in
addition to CUDA; `CUDA_VISIBLE_DEVICES` alone does not select the OpenGL device.

Use the upstream game as an external input producer. Its C++ renderer is not
bundled into vLLM-Omni, and its Python neural renderer is not used by this recipe.
The following desktop recorder needs a graphical session. Install the
[native dependencies](https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-4/docs/installation.md)
(CMake, C++20, SDL2, OpenGL/EGL, zlib and FFmpeg development libraries, including
`libswscale-dev` on Ubuntu), then build the supplied game:

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git ../matrix-game-source
git -C ../matrix-game-source checkout 5446525c80da2a4a2f87801addbbc7faa8d2486e
export MATRIX_GAME_SOURCE="$(cd ../matrix-game-source/Matrix-Game-4 && pwd)"

(
  cd "$MATRIX_GAME_SOURCE"
  bash scripts/build_racing.sh -DRACING_ASSET_ROOT="$MATRIX_RESOURCES/racing"
  game/racing/bin/CitySampleBBoxGame \
    --data-dir "$MATRIX_RESOURCES/racing" --race --hud \
    --fps 48 --physics-hz 480 --offline-record captures
)
```

Start a race and press **F9** to begin recording; drive for at least 11 seconds,
then press F9 again. W/S accelerate/reverse, A/D steer and Space applies the
handbrake. The first take's `captures/take-001/reference.mkv` contains clean RGB;
`ui.mkv` and audio are separate. Keep the released camera and rendering profile.
Do not feed HUD overlays or generated neural frames back as geometry.

From the vLLM-Omni root, convert the first 481 captured frames without resizing
or changing their cadence. This requires the `ffmpeg` executable:

```python
import os
import subprocess
from pathlib import Path

import numpy as np

source = Path(os.environ["MATRIX_GAME_SOURCE"]) / "captures/take-001/reference.mkv"
pixels = subprocess.check_output([
    "ffmpeg", "-v", "error", "-i", str(source), "-map", "0:v:0",
    "-frames:v", "481", "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
])
expected_bytes = 481 * 480 * 864 * 3
if len(pixels) != expected_bytes:
    raise ValueError("Expected at least 481 clean RGB frames at 864x480")
np.save("racing-geometry.npy", np.frombuffer(pixels, dtype=np.uint8).reshape(481, 480, 864, 3))
```

A live application instead forwards frames from the game's
`CitySampleReferenceServer` IPC stream to the chunk API below. It must handle
game startup, controls and media delivery; this PR does not provide that bridge.
An existing geometry recording can be processed without running OpenGL on the
inference host.

#### Command

Run from the vLLM-Omni repository root:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/diffusion/benchmark_matrix_game4.py \
  --scene racing --resources "$MATRIX_RESOURCES" --input ./racing-geometry.npy \
  --output ./qualification/racing --frames 481 --steps 1 --seed 42 \
  --cuda-graph --fp8 --attention-backend TORCH_SDPA --warmup-chunks 25
```

#### Verification

A successful run writes `qualification/racing.npy` with uint8 RGB shape
`[481,480,864,3]` and `qualification/racing.json` with configuration and timing
metrics. It processes the opening frame and 120 four-frame chunks, crosses the
temporal KV window and closes the session.

One measured Racing run on the environment above produced:

| Metric | Result |
| --- | --- |
| Generated throughput | 76.37 FPS |
| Four-frame chunk latency, p50 / p95 | 52.36 / 52.91 ms |
| Observed allocator peak, allocated / reserved | 8.22 / 8.54 GiB |

The first 25 chunks are excluded, leaving 384 measured frames. Timing covers
CPU uint8 input through CPU uint8 output, including normalization, codec,
denoising, clean-KV update, engine scheduling and postprocessing. Loading,
warmup, session close, game rendering, video encoding and network delivery are
excluded. These are single-run model-pipeline measurements, not browser FPS or
input-to-display latency. Results depend on the captured sequence and system.
Allocator peaks do not establish cold-start or whole-device memory requirements.

#### Important flags and deploy config

- `--steps 1` selects the one-step denoising schedule.
- `--fp8` sets the standard `quantization_config="fp8"`; vLLM owns online
  quantization, weight loading and device-specific kernel selection.
- `--cuda-graph` enables model-owned graph replay for the steady-state DiT and
  continued codec chunks. Keep the outer engine's `enforce_eager=True` so session
  bookkeeping stays outside capture.
- `TORCH_SDPA` selects the existing shared attention backend.
- TAEW2 encodes geometry and decodes generated latents. Embeddings, output head
  and codec remain BF16. The fixed-caption benchmark above releases T5 after
  startup; live dynamic captions keep it resident.

For an application using `DiffusionEngine`, the corresponding configuration is:

```python
from vllm_omni.diffusion.data import OmniDiffusionConfig, VideoOutputTransportConfig

config = OmniDiffusionConfig(
    model="./matrix-resources",
    model_class_name="MatrixGame4Pipeline",
    dtype="bfloat16",
    num_gpus=1,
    max_num_seqs=1,
    distributed_executor_backend="uni",
    enforce_eager=True,
    quantization_config="fp8",
    diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
    enable_prompt_embed_cache=True,
    prompt_embed_cache_size=8,
    model_config={"scene": "racing", "prompt_mode": "dynamic", "cuda_graph": True,
                  "height": 480, "width": 864},
    video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=True),
)
```

| Request field | Contract |
| --- | --- |
| `prompt["multi_modal_data"]["video"]` | CPU `torch.uint8` RGB tensor, `[T,480,864,3]` |
| `prompt["prompt"]` | Engine caption for these exact frames; empty text retains the current caption |
| First request | One frame; `matrix_frame_start=0` |
| Continuations | Four frames; offsets `1,5,9,...` |
| Sampling | Seed 42, one step, guidance scale 1, `output_type="np"` |
| `result.images[0]` | uint8 `[1,T,480,864,3]`, same frame count as input |
| Complete clip | Supply `1+4k` frames without a session ID; closes automatically |

For streaming, set sampling `extra_args` to `matrix_session_id`,
`matrix_frame_start` and `matrix_end_session`. Reuse the same session ID and
serialize requests. Close the owning session with `matrix_end_session=True`
on its last chunk, or send an explicit close request without video in a
`finally` block. There is no idle timeout. The
[benchmark](../../benchmarks/diffusion/benchmark_matrix_game4.py) demonstrates
the complete request and cleanup sequence. Racing returns the decoded sink
image as frame zero, followed by generated four-frame blocks.

For live Racing, forward the caption together with its RGB chunk from
`CitySampleReferenceServer`. The upstream adapter exposes it as
`states[-1]["chunk_prompt"]`; it describes visibility-tested entities, driving
state and events from that chunk. No coding-agent or LLM call is involved.
Changing the caption updates only text conditioning, preserving visual KV and
codec history. The shared prompt cache stores up to eight embeddings, and graph
replay retains the text-KV buffer addresses across updates.

For an offline recording with matching engine captions, add
`--captions ./captions.jsonl` to the benchmark command. Each JSONL row contains
`frame_start`, `frame_end` (inclusive) and `prompt`, covering frame 0, then 1–4,
5–8, etc. Plain recorded RGB does not include these captions; collect them from
the same engine run. Captions are limited to 8192 characters and use the released
512-token truncation. Dynamic T5 residency and caption changes add memory and
latency, so the fixed-caption performance table above is not a dynamic-mode result.

#### Known limitations

- Standard online FP8 changes numerical output. This throughput measurement
  does not establish perceptual equivalence to the original implementation.
  Other scenes have not been requalified with this FP8 configuration.
- One resident session and one GPU per worker; multi-GPU execution, concurrent
  sessions and the generic AR-Diffusion paged-KV/session protocol are unsupported.
- Dynamic captions are supported for Racing; Castle and Ancient Egypt use fixed
  captions. Changing scene requires a new worker.
- LoRA, offload, serialized/per-component quantization and the full Wan VAE are
  not supported by this pipeline.
