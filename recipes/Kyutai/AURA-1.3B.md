# Kyutai AURA-1.3B

> Speech-driven talking-head video from one portrait, generated chunk by chunk

## Summary

- Vendor: Kyutai
- Model: `kyutai/AURA-1.3B` (with the VAE and umT5 encoder of `Wan-AI/Wan2.1-T2V-1.3B`, and `kyutai/mimi` for speech features)
- Task: one portrait image + a speech recording -> a talking-head video with the speech track, 576x576 or 768x512 (width x height) at 16 fps
- Modes: offline generation, synchronous HTTP serving, and WebSocket video output. The model generates chunks of 16 frames (13 for the first) over a rolling K/V history.
- Hardware validated: NVIDIA H100 80GB and NVIDIA GeForce RTX 3090 24 GB
- Maintainer: Community

The weights are licensed separately under CC BY-NC 4.0 and are restricted to non-commercial use. The vLLM-Omni integration code remains Apache-2.0. Animate only portraits for which you have the subject's consent, and disclose that the videos are AI-generated.

## How it works

AURA ([paper](https://arxiv.org/abs/2611.26540)) is a 4-step causal student of a Wan2.1-T2V-1.3B diffusion transformer with speech cross-attention. Each chunk of 4 latent frames is sampled at timesteps 1000, 745, 498 and 255 without classifier-free guidance. The history keeps the first chunk as an identity anchor plus the most recent chunk, so memory stays constant on long streams; the causal VAE decodes each chunk as soon as it is sampled.

## Offline generation

```python
import numpy as np
import PIL.Image

from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="kyutai/AURA-1.3B", max_num_seqs=1)
output = omni.generate(
    {"prompt": "", "multi_modal_data": {"image": PIL.Image.open("portrait.png"), "audio": "speech.wav"}},
    OmniDiffusionSamplingParams(seed=42),
)[0]

# The speech track, its rate and the fps are in output.multimodal_output.
frames = np.asarray(output.images)
frames = frames.reshape(-1, *frames.shape[-3:])  # [T, H, W, 3]
if frames.dtype != np.uint8:  # float frames in [0, 1]
    frames = (frames * 255).round().astype(np.uint8)
extra = output.multimodal_output
mp4 = mux_video_audio_bytes(
    frames,
    np.asarray(extra["audio"]),
    fps=extra["fps"],
    audio_sample_rate=extra["audio_sample_rate"],
)
with open("aura.mp4", "wb") as f:
    f.write(mp4)
```

- `prompt` may stay empty: a default description of a person speaking to the camera is then used. The checkpoint was distilled with one fixed description of its own, so output without a prompt differs from the reference implementation's default output; pass a description as `prompt` to steer it.
- The video ends with the speech: the last chunk is cropped to the speech's length and no later chunk is generated; `extra_args={"num_chunks": N}` stops it after N chunks.
- The portrait is resized and center-cropped to the closest training resolution: 576x576, or 768x512 (width x height). Portraits where the face fills a good part of the frame work best.
- The first request compiles the transformer blocks (regional compilation); start the engine once and reuse it.

### Text encoder CPU offload

To keep umT5 weights in CPU memory between prompt encodes, select module-level
offload for the text encoder:

```python
omni = Omni(
    model="kyutai/AURA-1.3B",
    max_num_seqs=1,
    diffusion_offload_config={
        "mode": "module",
        "components": ["text_encoder"],
        "pin_memory": True,
    },
)
```

Each request loads umT5, encodes that request's prompt, and releases the GPU
weight storage before chunk generation. The prompt embedding stays on the GPU;
different prompts do not require a shared embedding cache. The immutable CPU
weights are retained, so offload does not copy the weights back from the GPU.
DiT, VAE, and Mimi remain device resident, and their weight precision is unchanged.

## Online serving

```bash
vllm serve kyutai/AURA-1.3B --omni --port 8091

curl -s http://localhost:8091/v1/videos/sync \
  -F "prompt=A person speaking to the camera." \
  -F "input_reference=@portrait.png" \
  --form-string "audio_reference={\"audio_url\": \"data:audio/wav;base64,$(base64 -w0 speech.wav)\"}" \
  -o aura.mp4
```

- The video API requires a non-empty `prompt`; any short description of the speaker works.
- The video plays at the model's 16 fps with the speech track; omit `fps` or pass 16, since any other `fps` returns HTTP 400. `extra_params.num_chunks`, when given, must be a positive integer.
- One request runs at a time.

### RTX 3090 (24 GB)

Serve the full AURA-1.3B pipeline on one RTX 3090 with pinned CPU offload
for umT5 and expandable CUDA allocator segments:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  vllm serve kyutai/AURA-1.3B --omni --port 8091 \
  --diffusion-offload-config '{"mode":"module","components":["text_encoder"],"pin_memory":true}'
```

If you have downloaded only the required AURA checkpoint, pass its local snapshot
directory in place of the model ID. The directory must contain
`stage2_dmd/step-6500_ema_merged.safetensors`. The Wan VAE, umT5 weights and
tokenizer, and Mimi weights and configuration are also required.

On one RTX 3090 with vLLM 0.34.0, the above offload and allocator settings were
tested using a local AURA snapshot. After a short warm-up request, a
`/v1/videos/sync` request with a synthetic 576×576 image and a 10-second
synthetic tone returned HTTP 200. The MP4 contained 160 frames at 16 fps and
an audio track. The pipeline completed 11 chunks, with the first chunk ready
in 4.18 s and the entire HTTP request taking 48.54 s. Peak device memory
observed by `nvidia-smi` at 200 ms intervals was 23,994 MiB. With the default
allocator, a shorter smoke request failed during VAE decoding with CUDA OOM.

### Realtime video output

Use [`WS /v1/realtime/video`](https://github.com/vllm-project/vllm-omni/blob/main/docs/serving/streaming_video_output_api.md) to
receive fragmented MP4 (fMP4) with audio as the model generates it. Send the
portrait and the complete audio clip together in `session.start`. AURA encodes
the full audio clip before generating the first video chunk. AURA requires
streaming output and the multiprocessing (`mp`) diffusion executor for
intermediate video delivery, including on a single GPU.

```bash
vllm serve kyutai/AURA-1.3B --omni --port 8091 \
  --distributed-executor-backend mp \
  --diffusion-streaming-output
```

With `websockets` installed and `portrait.png` and `speech.wav` in the
working directory, set the WebSocket URL to your server, send the request, and
save the binary messages in order.

```python
import base64
import json
from pathlib import Path

from websockets.sync.client import connect


def data_url(path, media_type):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


request = {
    "type": "session.start",
    "prompt": "A person speaking to the camera.",
    "format": "m4s",
    "fps": 16,
    "seed": 42,
    "image_reference": {"image_url": data_url("portrait.png", "image/png")},
    "audio_reference": {"audio_url": data_url("speech.wav", "audio/wav")},
}
with connect("ws://127.0.0.1:8091/v1/realtime/video", max_size=None) as ws:
    ws.send(json.dumps(request))
    with open("aura-stream.mp4", "wb") as output:
        for message in ws:
            if isinstance(message, bytes):
                output.write(message)
                continue
            event = json.loads(message)
            if event["type"] == "error":
                raise RuntimeError(event["message"])
            if event["type"] == "session.done":
                break
        else:
            raise RuntimeError("The WebSocket closed before session.done.")
```

The server sends `video.chunk_metadata` before each binary message. The
messages include the final encoder trailer before `session.done`.

To watch the chunks play as they arrive, use the
[Gradio streaming video demo](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/streaming_video_generation#run-the-gradio-demo).
In the UI, point **Server Host** and **Server Port** at your WebSocket service
and set **Model** to the served model name. Upload a portrait and an audio
clip, enter a prompt, leave **FPS** at 16, and click **Start**. The preview
starts muted for browser autoplay. Use its volume control to hear the audio.
The demo's **Width** and **Height** fields are sent with the WebSocket request,
but AURA does not use them to set its output size. It chooses 576x576 or
768x512 from the uploaded portrait's aspect ratio and resizes/crops the portrait
to that bucket, so changing those fields does not change the generated
resolution.
The shared demo's **Prompt Updates** table schedules mid-generation
`session.interaction` events; AURA does not support changing the prompt during
generation. Turn off **Enable Prompt Updates** when using AURA.

## Validation

Measured on 1x H100 80GB HBM3 against Kyutai's reference implementation ([kyutai-labs/aura](https://github.com/kyutai-labs/aura) @ `c06aa7d`, run in its own environment) on the sample of the model card (frame 0 and the 9.08 s speech of `assets/sample_1.mp4`), seed 42, 10 chunks (157 frames), with the reference's own default description passed as the prompt:

| Check | Result |
| --- | --- |
| Reference against itself (flash vs memory-efficient SDPA): per-frame PSNR floor | min 28.93 dB, mean 40.48 dB |
| vLLM-Omni, SDPA backend, against the reference | min 29.35 dB, mean 42.41 dB |
| vLLM-Omni, FLASH_ATTN backend, against the reference | min 32.83 dB, mean 44.31 dB |
| vLLM-Omni through `Omni` with regional compilation, against the reference | min 30.69 dB, mean 42.38 dB |
| Speed through `Omni`, warm request of 157 frames (9.81 s of video) | 7.08 s with regional compilation (22.2 fps, 0.72x real time, all per-request costs included); 10.07 s with `enforce_eager` (15.6 fps) |

The pass line sits 1 dB under the floor (every frame at or above 27.93 dB, the mean at or above 39.48 dB); every frame of the three vLLM-Omni rows passes it.

## Current limitations

- AURA-5B (wav2vec2 features, Wan2.2 VAE, 720p at 25 fps) is not supported yet.
- The speech is read as a whole file before generation starts; live speech input while the video streams is not supported.
- One request at a time (`max_num_seqs=1`); no tensor or sequence parallelism. CPU offload is limited to module-level `text_encoder` offload; DiT offload and HSDP are not supported.

## References

- Paper: [AURA: AUdio-dRiven streaming Avatar](https://arxiv.org/abs/2611.26540) (BMVC 2026)
- Weights: [kyutai/AURA-1.3B](https://huggingface.co/kyutai/AURA-1.3B)
- Reference code: [kyutai-labs/aura](https://github.com/kyutai-labs/aura)
