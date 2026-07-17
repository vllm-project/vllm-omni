# Cosmos3-Nano

> Text-to-image, text-to-video, image-to-video, and video-to-video serving

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Cosmos3-Nano`
- Task: Text-to-image (T2I), text-to-video (T2V), image-to-video (I2V), and video-to-video (V2V) generation, with optional transfer controls, synchronized audio (video + sound), action policy
- Mode: Online serving with the OpenAI-compatible image/video APIs, plus offline generation via the `Omni` API
- Maintainer: Community

## When to use this recipe

Use this recipe to deploy `nvidia/Cosmos3-Nano` for image and video generation.
A single pipeline class (`Cosmos3OmniDiffusersPipeline`) serves these modes; the
mode is selected per request:

- **T2I** — `POST /v1/images/generations` (or a prompt carrying `modalities=["image"]`).
- **T2V** — `POST /v1/videos/sync` with `num_frames > 1` and no reference image/video.
- **I2V** — `POST /v1/videos/sync` with a reference image (`input_reference` file
  upload, or `image_reference` JSON).
- **V2V** — `POST /v1/videos/sync` with a reference video (`input_reference`
  video upload, or `video_reference` JSON). Cosmos3 conditions on selected
  reference-video latent frames; use `extra_params.condition_frame_indexes_vision`
  and `extra_params.condition_video_keep` to choose which prefix/tail frames guide
  generation.
- **Transfer V2V** — pass one or more transfer hints in `extra_params`
  (`edge`, `blur`, `depth`, `seg`, `wsm`) to guide generation with control
  frames. Transfer mode is video-only and cannot be combined with sound or action.
- **T2VS / I2VS** — add `generate_sound=true` (and optional `sound_duration`) to a
  T2V/I2V `/v1/videos/sync` request to also generate synchronized audio, muxed into
  the mp4 as AAC 48 kHz stereo. See the official model card's "Video + Audio" examples.
- **Action** — pass `extra_params={"action_mode": ...}` to drive Physical-AI tasks:
  - `forward_dynamics` — given a first frame or video **and** an action trajectory,
    roll out the resulting video. Synchronous: `POST /v1/videos/sync`.
  - `policy` — given a first frame or video and a language instruction,
    **predict** the action trajectory (and a rollout video). Use the async
    `POST /v1/videos` endpoint and read the predicted action from the top-level
    `action` field.
  - `inverse_dynamics` — given a video, **recover** the action trajectory. Use
    the async `POST /v1/videos` endpoint and read the recovered action from
    the top-level `action` field
    (`{data, shape, dtype, raw_action_dim, domain_id}`).

  Action requests also take `domain_name` (e.g. `av`, `bridge_orig_lerobot`,
  `droid_lerobot`, `agibotworld`, …; or a numeric `domain_id`), `raw_action_dim`,
  and `action_chunk_size` (must equal `num_frames` or `num_frames - 1`). For
  `forward_dynamics` also pass the `action` array. The dedicated policy checkpoint
  **`nvidia/Cosmos3-Nano-Policy-DROID`** is served the same way
  (`domain_name=droid_lerobot`).

- **DROID OpenPI policy server** — serve `nvidia/Cosmos3-Nano-Policy-DROID` and
  connect an OpenPI-compatible websocket client to `/v1/realtime/robot/openpi`.
  This path returns action chunks directly instead of an mp4.

  Action requests can use `input_reference` or `video_reference` for video input.
  `policy` and `forward_dynamics` can also use an image reference; `inverse_dynamics`
  requires a video reference.

## References

- Model card (authoritative usage + example assets): <https://huggingface.co/nvidia/Cosmos3-Nano>
- Example inputs/outputs live in the repo's `assets/` (`example_t2v_prompt.json`,
  `example_i2v_prompt.json`, `example_i2v_input.jpg`, `negative_prompt.json`;
  audio examples: `example_t2vs_prompt.json`, `example_t2vs_output.mp4`,
  `example_i2vs_output.mp4`).
- Prompt upsampling (recommended for quality): the model expects JSON-upsampled
  structured prompts; see NVIDIA's `cosmos-framework` prompt-upsampling docs.
- Pipeline: [`vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py`](../../vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py)
- Smoke tests (canonical request formats): [`tests/e2e/accuracy/test_cosmos3_similarity.py`](../../tests/e2e/accuracy/test_cosmos3_similarity.py)

## Hardware Support

## GPU

### 1x H200 141GB / B300 (Online serving)

#### Environment

- OS: Ubuntu 22.04+
- Python: 3.12+
- Driver / runtime: NVIDIA CUDA environment
- vLLM version: match the repository requirements from your current checkout
- vLLM-Omni version or commit: use the commit you are deploying from

#### Command

Requires the `vllm-omni` package (or the `vllm/vllm-omni:cosmos3` container),
which provides the `vllm serve … --omni` entrypoint used below.

Safety guardrails are **on by default** (NVIDIA Open Model License). They load
the **gated** `nvidia/Cosmos-1.0-Guardrail` model, so to keep them on you must:

1. `pip install cosmos-guardrail`
2. Accept the license at <https://huggingface.co/nvidia/Cosmos-1.0-Guardrail>
3. Export a token with access: `export HF_TOKEN=hf_...`

Then launch the recommended server:

```bash
vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --init-timeout 1800
```

To run **without** guardrails (you are responsible for license compliance),
add `--no-guardrails` (no token/`cosmos-guardrail` needed). For extra GPUs use
`--ulysses-degree N` (context parallel) or `--tensor-parallel-size N`;
`--enable-layerwise-offload` reduces VRAM on smaller GPUs;
`--quantization fp8` (online, no calibration) cuts peak VRAM for 720p video
generation from ~50 GB to ~36 GB with BF16-level quality (T2V composition can
shift at the same seed). The pipeline
auto-resolves from `model_index.json`; pass
`--model-class-name Cosmos3OmniDiffusersPipeline` to force it explicitly.

#### Verification

Best quality uses the JSON-upsampled prompts from `assets/` (download with
`hf download nvidia/Cosmos3-Nano assets/ --local-dir Cosmos3-Nano`). Minimal
self-contained examples:

```bash
curl http://localhost:8000/v1/models

# Text-to-image -> /v1/images/generations  (1024x1024, 50 steps; base64 PNG)
curl -sS -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Cosmos3-Nano",
    "prompt": "A photorealistic red sports car on a city street at golden hour, cinematic lighting.",
    "negative_prompt": "blurry, distorted, low quality",
    "size": "1024x1024", "n": 1, "response_format": "b64_json",
    "num_inference_steps": 50, "guidance_scale": 7.0, "seed": 42
  }' | python -c "import sys,json,base64; open('cosmos3_t2i.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"

# Text-to-video -> /v1/videos/sync  (720p, 189 frames @ 24fps; official params)
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Nano" \
  -F "prompt=A robot arm is cleaning a plate in the kitchen" \
  -F "negative_prompt=blurry, distorted, low quality, jittery, deformed" \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" \
  -F "num_inference_steps=35" -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}' \
  -F "seed=123" \
  -o cosmos3_t2v.mp4

# Image-to-video -> /v1/videos/sync with an uploaded reference image
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Nano" \
  -F "prompt=The scene comes to life with smooth, natural motion." \
  -F "negative_prompt=blurry, distorted, low quality" \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" \
  -F "num_inference_steps=35" -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}' \
  -F "seed=1111" \
  -F "input_reference=@/path/to/reference.jpg;type=image/jpeg" \
  -o cosmos3_i2v.mp4

# Video-to-video -> /v1/videos/sync with an uploaded reference video.
# By default Cosmos3 conditions on latent indexes [0, 1]. For the default
# temporal VAE stride this decodes only the first 5 input frames.
# The model works best when the prompt describes the actual situation happening in the video.
# Generic prompts may create sub-standard generations.
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Nano" \
  -F "prompt=Continue the same scene with smooth natural motion and consistent subjects." \
  -F "negative_prompt=blurry, distorted, low quality, jittery, deformed" \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" \
  -F "num_inference_steps=35" -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true,"condition_frame_indexes_vision":[0,1],"condition_video_keep":"first"}' \
  -F "seed=2222" \
  -F "input_reference=@/path/to/reference.mp4;type=video/mp4" \
  -o cosmos3_v2v.mp4

# V2V can also use a JSON-safe URL/data-URL video reference. Do not combine
# video_reference with input_reference or image_reference.
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Nano" \
  -F "prompt=Continue the same scene with smooth natural motion and consistent subjects." \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" \
  -F "num_inference_steps=35" -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F 'extra_params={"condition_frame_indexes_vision":[0,1],"condition_video_keep":"last"}' \
  -F 'video_reference={"video_url":"https://example.com/reference.mp4"}' \
  -o cosmos3_v2v_from_url.mp4

# Transfer V2V with a precomputed depth control video. `control_path` can point
# to a local image/video; edge and blur can also be computed from `input_reference`
# by passing `"edge":true` or `"blur":true`.
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Nano" \
  -F "prompt=Generate a realistic scene following the provided control video." \
  -F "size=1280x720" -F "num_frames=121" \
  -F "num_inference_steps=50" -F "seed=125" \
  -F 'extra_params={"depth":{"control_path":"/path/to/depth_control.mp4"},"max_frames":121,"resolution":"720","num_video_frames_per_chunk":121}' \
  -o cosmos3_transfer_depth.mp4

# Text-to-video-with-sound
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Nano" \
  -F "prompt=The video opens with a view of a well-lit indoor fruit display. A robotic arm picks up a pear, an orange, and a carambola one by one, placing each into a plastic bag in a shopping cart with red handles. The video is 7.875 seconds long, 24 FPS, and 1280x720. Audio description: soft servo whirs, gentle fruit thuds, plastic bag rustling, and a faint refrigeration hum." \
  -F "negative_prompt=blurry, distorted, low quality" \
  -F "size=1280x720" \
  -F "num_frames=189" \
  -F "fps=24" \
  -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" \
  -F "flow_shift=10.0" \
  -F "seed=0" \
  -F "generate_sound=true" \
  -F "sound_duration=7.875" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}' \
  -o cosmos3_t2v_with_sound.mp4

# Action — forward dynamics (first frame + action trajectory -> rollout video).
# Synchronous; `action` is a JSON array shaped [action_chunk_size, raw_action_dim].
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  --form-string "model=nvidia/Cosmos3-Nano" \
  --form-string "prompt=You are an autonomous vehicle. This video is captured from a first-person perspective." \
  -F "input_reference=@first_frame.jpg;type=image/jpeg" \
  -F "size=640x480" -F "num_frames=61" -F "fps=10" \
  -F "num_inference_steps=30" -F "guidance_scale=1.0" -F "flow_shift=5.0" \
  --form-string "extra_params={\"action_mode\":\"forward_dynamics\",\"domain_name\":\"av\",\"raw_action_dim\":9,\"action_chunk_size\":60,\"action\":$(cat action.json)}" \
  -F "seed=0" \
  -o cosmos3_forward_dynamics.mp4

# Action — policy (first frame + instruction -> predicted action trajectory + video).
# Asynchronous: POST returns a job id; poll, then read the predicted action from
# the top-level `action` field ({data, shape, dtype, raw_action_dim, domain_id}).
VIDEO_ID=$(curl -sS -X POST http://localhost:8000/v1/videos \
  -H "Accept: application/json" \
  --form-string "model=nvidia/Cosmos3-Nano" \
  --form-string "prompt=Pick up the banana and place it in the bowl." \
  -F "input_reference=@first_frame.jpg;type=image/jpeg" \
  -F "size=640x480" -F "num_frames=17" -F "fps=5" \
  -F "num_inference_steps=30" -F "guidance_scale=1.0" -F "flow_shift=5.0" \
  --form-string 'extra_params={"action_mode":"policy","domain_name":"bridge_orig_lerobot","raw_action_dim":10,"action_chunk_size":16}' \
  -F "seed=0" | jq -r '.id')
# poll until status == completed, then:
curl -sS "http://localhost:8000/v1/videos/$VIDEO_ID" | jq '.action | {shape, dtype, raw_action_dim, domain_id}'
curl -sS -L "http://localhost:8000/v1/videos/$VIDEO_ID/content" -o cosmos3_policy.mp4

# Action — inverse dynamics (video -> recovered action trajectory).
# Asynchronous: use the job metadata to read the recovered action.
VIDEO_ID=$(curl -sS -X POST http://localhost:8000/v1/videos \
  -H "Accept: application/json" \
  --form-string "model=nvidia/Cosmos3-Nano" \
  --form-string "prompt=Recover the robot action trajectory from this clip." \
  -F "input_reference=@motion_clip.mp4;type=video/mp4" \
  -F "size=640x480" -F "num_frames=17" -F "fps=5" \
  -F "num_inference_steps=30" -F "guidance_scale=1.0" -F "flow_shift=5.0" \
  --form-string 'extra_params={"action_mode":"inverse_dynamics","domain_name":"bridge_orig_lerobot","raw_action_dim":10,"action_chunk_size":16}' \
  -F "seed=0" | jq -r '.id')
# poll until status == completed, then:
curl -sS "http://localhost:8000/v1/videos/$VIDEO_ID" | jq '.action | {shape, dtype, raw_action_dim, domain_id}'
curl -sS -L "http://localhost:8000/v1/videos/$VIDEO_ID/content" -o cosmos3_inverse_dynamics.mp4

# DROID OpenPI policy server (websocket action serving).
# Requires cosmos_framework on PYTHONPATH because the pipeline reuses the
# reference RoboLab action transforms. If your checkpoint config already
# includes policy_server_config, omit the stage_overrides file and flag.
cat > cosmos3_droid_openpi_stage_overrides.json <<'JSON'
{
  "0": {
    "model_config": {
      "policy_server_config": {
        "image_resolution": [540, 640],
        "n_external_cameras": 2,
        "needs_wrist_camera": true,
        "needs_stereo_camera": false,
        "needs_session_id": true,
        "action_space": "joint_position"
      }
    }
  }
}
JSON

vllm serve nvidia/Cosmos3-Nano-Policy-DROID \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --no-guardrails \
  --stage-overrides "$(cat cosmos3_droid_openpi_stage_overrides.json)"

# Point an OpenPI websocket client at:
#   ws://localhost:8000/v1/realtime/robot/openpi
# The first server message is policy_server_config. Each infer request sends a
# msgpack-numpy observation dict and receives a writable float32 action array.
```

#### Notes

- **Measured latency (1x B300, bf16, guardrails off):**
  - T2I 1024² — 10 / 25 / 50 steps → ~0.4 / 0.7 / **1.3 s**
  - T2V 1280×720 @ 35 steps — 25 / 49 / 93 / **189** frames → ~7 / 15 / 33 / **~93 s**
  - I2V 1280×720, 189 frames @ 35 steps → ~**99 s**
  - Action 640×480 @ 30 steps — forward-dynamics 61f ~**4 s**, policy 17f ~**1–3 s**.
  - Guardrails-on overhead: ~8% on T2I, negligible on video.
- **Memory:** transformer ~17 GiB (bf16); peak ~46 GiB for 720p video on 1 GPU;
  full repo (transformer + Wan VAE + Qwen3-VL vision encoder + audio tokenizer)
  ~33 GB on disk.
- **Determinism:** identical seed reproduces identical output on the same
  hardware; outputs are not bit-identical across different GPU types.
- **Supported sizes (per model card):** 256p / 480p / 720p at 16:9, 4:3, 1:1,
  3:4, 9:16. Defaults: T2I 1024², 50 steps, guidance 7.0; T2V/I2V/V2V
  1280×720, 189 frames, 35 steps, guidance 6.0, `flow_shift=10.0`.
- **Key flags / params:** `--no-guardrails` (server) or
  `extra_params={"guardrails":false}` (per request) toggles safety. The
  per-request flag only takes effect when the server was launched **with**
  guardrails enabled (it cannot re-enable them on a `--no-guardrails` server).
  `use_resolution_template` / `use_duration_template` are off by default and only
  needed when not using upsampled prompts that already encode resolution/duration.
  For V2V, `condition_frame_indexes_vision` selects the clean conditioned latent
  frame indexes (default `[0, 1]`), and `condition_video_keep` selects whether the
  API decodes the first or last needed reference frames (`"first"` by default).
- **Transfer controls:** `extra_params` may include `edge`, `blur`, `depth`,
  `seg`, or `wsm`. Each hint accepts `true`, a path string, or an object such as
  `{"control_path": "/path/to/control.mp4"}`; `edge` also accepts
  `preset_edge_threshold` and `blur` accepts `preset_blur_strength`.
  Transfer-level options include `control_guidance`,
  `control_guidance_interval`, `num_video_frames_per_chunk` (default `93`,
  `101` for WSM), `num_conditional_frames` (default `1`),
  `num_first_chunk_conditional_frames`, `max_frames`,
  `show_control_condition`, `show_input`, and
  `share_vision_temporal_positions`. Non-WSM transfer preserves the input video
  fps when available; WSM defaults to 10 fps unless `fps` is supplied.
- **DROID OpenPI observations:** include a string `prompt`, either
  `observation/image` or the three-view DROID camera keys
  (`observation/wrist_image_left`, `observation/exterior_image_1_left`,
  `observation/exterior_image_2_left`), plus `observation/gripper_position` and
  `observation/joint_position`. Optional extra params include `history_length`,
  `conditioning_fps`, `action_chunk_size`, `raw_action_dim`, `deterministic_seed`,
  and `session_id`.
- **Known limitations:**
  - Guardrails-on requires `cosmos-guardrail` **and** access to the gated
    `nvidia/Cosmos-1.0-Guardrail` repo (accept license + `HF_TOKEN`); otherwise
    the server fails at pipeline build with a gated-repo / safety-checker error.
  - A guardrail-blocked prompt currently returns HTTP 500
    (`"Guardrail blocked prompt"`).
  - Action `forward_dynamics`, `policy`, and `inverse_dynamics` are supported
    online. Use async `POST /v1/videos` when you need the predicted/recovered
    action payload under the top-level `action` field; sync `/v1/videos/sync`
    returns raw MP4 bytes and does not expose action metadata in the response body.

### 1x GPU (Offline generation)

#### Environment

- OS: Ubuntu 22.04+
- Python: 3.12+
- Driver / runtime: NVIDIA CUDA environment
- vLLM-Omni version or commit: use the commit you are deploying from

#### Command

Cosmos3 runs through the standard task examples; pass model-specific knobs via
`--extra-body`. Guardrails are on by default — pass `"guardrails": false` for a
quick local run (install `cosmos-guardrail` + accept the gated repo to enable them).

```bash
# Text-to-image -> examples/offline_inference/text_to_image
python examples/offline_inference/text_to_image/text_to_image.py \
  --model nvidia/Cosmos3-Nano \
  --prompt "A photorealistic red sports car at golden hour, cinematic lighting." \
  --negative-prompt "blurry, distorted, low quality" \
  --height 1024 --width 1024 --num-inference-steps 50 --guidance-scale 7.0 \
  --extra-body '{"flow_shift": 3.0, "guardrails": false}' \
  --output cosmos3_t2i.png

# Text-to-video -> examples/offline_inference/text_to_video
python examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Nano \
  --prompt "A robot arm is cleaning a plate in the kitchen." \
  --negative-prompt "blurry, distorted, low quality, jittery, deformed" \
  --height 720 --width 1280 --num-frames 189 --fps 24 \
  --num-inference-steps 35 --guidance-scale 6.0 \
  --extra-body '{"flow_shift": 10.0, "max_sequence_length": 4096, "guardrails": false,
                 "use_resolution_template": false, "use_duration_template": false}' \
  --output cosmos3_t2v.mp4

# Image-to-video -> examples/offline_inference/image_to_video
# (Cosmos3 bundles example frames under assets/; any RGB image works too.)
python examples/offline_inference/image_to_video/image_to_video.py \
  --model nvidia/Cosmos3-Nano \
  --image /path/to/Cosmos3-Nano/assets/example_i2v_input.jpg \
  --prompt "The scene comes to life with smooth, natural motion." \
  --height 720 --width 1280 --num-frames 189 --fps 24 \
  --num-inference-steps 35 --guidance-scale 6.0 \
  --extra-body '{"flow_shift": 10.0, "max_sequence_length": 4096, "guardrails": false}' \
  --output cosmos3_i2v.mp4
```

#### Verification

```bash
python -c "from PIL import Image; im=Image.open('cosmos3_t2i.png'); print('image', im.size, im.mode)"
ffprobe -v error -show_entries stream=codec_type,nb_frames,width,height cosmos3_t2v.mp4
```

#### Notes

- A single `Cosmos3OmniDiffusersPipeline` serves every mode; the standard examples
  select it automatically from `model_index.json`. T2I is chosen by the
  `text_to_image` prompt builder (which marks `modalities=["image"]`); `text_to_video`
  defaults to T2V; `image_to_video` adds `multi_modal_data={"image": ...}` (I2V).
  V2V is served online (`/v1/videos/sync`).
- Model-specific knobs (`flow_shift`, `max_sequence_length`, `condition_*`,
  `generate_sound`/`sound_duration`, `guardrails`, `action_*`, ...) are declared
  once in `vllm_omni/model_extras/cosmos3.py` and forwarded through `--extra-body`;
  unknown keys for the model are dropped.

## NPU

### 1x Ascend 910B / 910C (Atlas A2 / A3) — Online serving

#### Environment

- OS: Linux (aarch64)
- Python: 3.12+
- Driver / runtime: CANN 8.5.1 + NNAL + Ascend 910B / 910C
- vLLM version: match the repository requirements from your current checkout
- vLLM-Ascend version: match the repository requirements from your current checkout
- vLLM-Omni version or commit: use the commit you are deploying from

#### Command

Requires the `vllm-omni` package (or the `quay.io/atlas-ci/vllm-ascend` A2 / A3 container),
which provides the `vllm serve … --omni` entrypoint used below.

Safety guardrails are **on by default** (NVIDIA Open Model License). They load
the **gated** `nvidia/Cosmos-1.0-Guardrail` model, so to keep them on you must:

1. `pip install cosmos-guardrail`
2. Accept the license at <https://huggingface.co/nvidia/Cosmos-1.0-Guardrail>
3. Export a token with access: `export HF_TOKEN=hf_...`

Then launch the recommended server:

```bash
vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --init-timeout 1800
```

To run **without** guardrails (you are responsible for license compliance),
add `--no-guardrails` (no token/`cosmos-guardrail` needed). For tensor parallel
add `--tensor-parallel-size 8`. `--quantization fp8` and
`--enable-layerwise-offload` are not supported on NPU.
The pipeline auto-resolves from `model_index.json`; pass
`--model-class-name Cosmos3OmniDiffusersPipeline` to force it explicitly.

#### Verification

Best quality uses the JSON-upsampled prompts from `assets/` (download with
`hf download nvidia/Cosmos3-Nano assets/ --local-dir Cosmos3-Nano`). Minimal
self-contained examples:

```bash
curl http://localhost:8000/v1/models

# Text-to-image -> /v1/images/generations  (1024x1024, 10 steps; base64 PNG)
curl -sS -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Cosmos3-Nano",
    "prompt": "A photorealistic red sports car on a city street at golden hour, cinematic lighting.",
    "negative_prompt": "blurry, distorted, low quality",
    "size": "1024x1024", "n": 1, "response_format": "b64_json",
    "num_inference_steps": 10, "guidance_scale": 7.0, "seed": 42
  }' | python -c "import sys,json,base64; open('cosmos3_t2i.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"

# Text-to-video -> /v1/videos/sync  (720p, 49 frames @ 24fps)
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Nano" \
  -F "prompt=A robot arm is cleaning a plate in the kitchen" \
  -F "negative_prompt=blurry, distorted, low quality, jittery, deformed" \
  -F "size=1280x720" -F "num_frames=49" -F "fps=24" \
  -F "num_inference_steps=20" -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F "seed=123" \
  -o cosmos3_t2v.mp4

# Image-to-video -> /v1/videos/sync with an uploaded reference image
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Nano" \
  -F "prompt=The scene comes to life with smooth, natural motion." \
  -F "size=1280x720" -F "num_frames=25" -F "fps=8" \
  -F "num_inference_steps=10" -F "guidance_scale=6.0" \
  -F "seed=42" \
  -F "input_reference=@reference.jpg;type=image/jpeg" \
  -o cosmos3_i2v.mp4

# Video-to-video -> /v1/videos/sync with an uploaded reference video
curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Nano" \
  -F "prompt=Continue the same scene with smooth natural motion and consistent subjects." \
  -F "size=1280x720" -F "num_frames=17" -F "fps=5" \
  -F "num_inference_steps=10" -F "guidance_scale=6.0" \
  -F "seed=42" \
  -F "input_reference=@reference.mp4;type=video/mp4" \
  -o cosmos3_v2v.mp4
```

#### Notes

- **Measured latency (1x Ascend 910B / 910C, bf16, guardrails off):**
  - T2I 1024² — 10 steps → ~8 s
  - T2V 1280×720 @ 20 steps — 49 frames → ~55 s
  - I2V 1280×720 @ 10 steps — 25 frames → ~25 s
  - V2V 480×320 @ 10 steps — 17 frames → ~12 s
- **Memory:** transformer ~17 GiB (bf16); peak ~46 GiB for 720p video on 1 NPU;
  full repo (transformer + Wan VAE + Qwen3-VL vision encoder + audio tokenizer)
  ~33 GB on disk.
- **Determinism:** identical seed reproduces identical output on the same
  hardware; outputs are not bit-identical across different GPU/NPU types.
- **Supported sizes (per model card):** 256p / 480p / 720p at 16:9, 4:3, 1:1,
  3:4, 9:16. Defaults: T2I 1024², 50 steps, guidance 7.0; T2V/I2V/V2V
  1280×720, 35 steps, guidance 6.0, `flow_shift=10.0`.
- **Key flags / params:** `--no-guardrails` (optional, to disable guardrails), `--init-timeout 1800`
  (for model loading), `--tensor-parallel-size 8` for multi-NPU, and
  `--model-class-name Cosmos3OmniDiffusersPipeline` to force the pipeline class.
- **Known limitations:**
  - Transfer V2V with `extra_params` (`edge`/`blur`/`depth`/`seg`/`wsm`) hits a
    resolution-parsing bug; basic V2V without transfer hints works.
  - FP8 online quantization and layerwise offload are not supported on NPU.

### 1x Ascend 910B / 910C (Atlas A2 / A3) — Offline generation

#### Environment

- OS: Linux (aarch64)
- Python: 3.12+
- Driver / runtime: CANN 8.5.1 + NNAL + Ascend 910B / 910C
- vLLM-Omni version or commit: use the commit you are deploying from

#### Command

The same offline task examples run on NPU; pass model-specific knobs via
`--extra-body`. Guardrails are on by default — pass `"guardrails": false` for a
quick local run (install `cosmos-guardrail` + accept the gated repo to enable them).

```bash
# Text-to-image -> examples/offline_inference/text_to_image
python examples/offline_inference/text_to_image/text_to_image.py \
  --model nvidia/Cosmos3-Nano \
  --prompt "A photorealistic red sports car at golden hour, cinematic lighting." \
  --negative-prompt "blurry, distorted, low quality" \
  --height 1024 --width 1024 --num-inference-steps 50 --guidance-scale 7.0 \
  --extra-body '{"flow_shift": 3.0, "guardrails": false}' \
  --output cosmos3_t2i.png

# Text-to-video -> examples/offline_inference/text_to_video
python examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Nano \
  --prompt "A robot arm is cleaning a plate in the kitchen." \
  --negative-prompt "blurry, distorted, low quality, jittery, deformed" \
  --height 720 --width 1280 --num-frames 189 --fps 24 \
  --num-inference-steps 35 --guidance-scale 6.0 \
  --extra-body '{"flow_shift": 10.0, "max_sequence_length": 4096, "guardrails": false,
                 "use_resolution_template": false, "use_duration_template": false}' \
  --output cosmos3_t2v.mp4

# Image-to-video -> examples/offline_inference/image_to_video
# (Cosmos3 bundles example frames under assets/; any RGB image works too.)
python examples/offline_inference/image_to_video/image_to_video.py \
  --model nvidia/Cosmos3-Nano \
  --image /path/to/Cosmos3-Nano/assets/example_i2v_input.jpg \
  --prompt "The scene comes to life with smooth, natural motion." \
  --height 720 --width 1280 --num-frames 189 --fps 24 \
  --num-inference-steps 35 --guidance-scale 6.0 \
  --extra-body '{"flow_shift": 10.0, "max_sequence_length": 4096, "guardrails": false}' \
  --output cosmos3_i2v.mp4
```

#### Verification

```bash
python -c "from PIL import Image; im=Image.open('cosmos3_t2i.png'); print('image', im.size, im.mode)"
ffprobe -v error -show_entries stream=codec_type,nb_frames,width,height cosmos3_t2v.mp4
```

#### Notes

- Guardrails are on by default on NPU with `cosmos-guardrail` installed. Pass
  `"guardrails": false` in `--extra-body` to disable them (there is no
  `--no-guardrails` flag for the offline scripts).
- Video at the 189-frame default takes ~15 min/clip on 1 NPU; reduce
  `--num-frames` for faster iteration.
