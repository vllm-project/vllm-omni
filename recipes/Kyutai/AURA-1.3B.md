# Kyutai AURA-1.3B

> Speech-driven talking-head video from one portrait, generated chunk by chunk

## Summary

- Vendor: Kyutai
- Model: `kyutai/AURA-1.3B` (with the VAE and umT5 encoder of `Wan-AI/Wan2.1-T2V-1.3B`, and `kyutai/mimi` for speech features)
- Task: one portrait image + a speech recording -> a talking-head video with the speech track, 576x576 or 768x512 (width x height) at 16 fps
- Mode: offline generation; the video is generated in chunks of 16 frames (13 for the first) over a rolling K/V history
- Hardware validated: NVIDIA H100 80GB
- Maintainer: Community

The weights are licensed separately under CC BY-NC 4.0 and are restricted to non-commercial use. The vLLM-Omni integration code remains Apache-2.0. Animate only portraits for which you have the subject's consent, and disclose that the videos are AI-generated.

## How it works

AURA ([paper](https://arxiv.org/abs/2611.26540)) is a 4-step causal student of a Wan2.1-T2V-1.3B diffusion transformer with speech cross-attention. Each chunk of 4 latent frames is sampled at timesteps 1000, 745, 498 and 255 without classifier-free guidance. The history keeps the first chunk as an identity anchor plus the most recent chunk, so memory stays constant on long streams; the causal VAE decodes each chunk as soon as it is sampled.

## Offline generation

```python
import PIL.Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="kyutai/AURA-1.3B", max_num_seqs=1)
output = omni.generate(
    {"prompt": "", "multi_modal_data": {"image": PIL.Image.open("portrait.png"), "audio": "speech.wav"}},
    OmniDiffusionSamplingParams(seed=42),
)[0]
frames = output.images  # the video frames; the speech track and its rate are in output.multimodal_output
```

- `prompt` may stay empty: a default description of a person speaking to the camera is then used. The checkpoint was distilled with one fixed description; any description can be passed as `prompt`.
- The video covers the whole speech, rounded up to a whole chunk; `extra_args={"num_chunks": N}` limits it to N chunks.
- The portrait is resized and center-cropped to the closest training resolution: 576x576, or 768x512 (width x height). Portraits where the face fills a good part of the frame work best.
- The first request compiles the transformer blocks (regional compilation); start the engine once and reuse it.

## Validation

Measured on 1x H100 80GB HBM3 against Kyutai's reference implementation ([kyutai-labs/aura](https://github.com/kyutai-labs/aura) @ `c06aa7d`, run in its own environment) on the sample of the model card (frame 0 and the 9.08 s speech of `assets/sample_1.mp4`), seed 42, 10 chunks (157 frames), with the reference's own default description passed as the prompt:

| Check | Result |
|---|---|
| Reference against itself (flash vs memory-efficient SDPA): per-frame PSNR floor | min 28.93 dB, mean 40.48 dB |
| vLLM-Omni, SDPA backend, against the reference | min 29.35 dB, mean 42.41 dB |
| vLLM-Omni, FLASH_ATTN backend, against the reference | min 32.83 dB, mean 44.31 dB |
| vLLM-Omni through `Omni` with regional compilation, against the reference | min 30.69 dB, mean 42.38 dB |
| Speed through `Omni`, warm request of 157 frames (9.81 s of video) | 7.08 s with regional compilation (22.2 fps, 0.72x real time, all per-request costs included); 10.07 s with `enforce_eager` (15.6 fps) |

The pass line sits 1 dB under the floor (every frame at or above 27.93 dB, the mean at or above 39.48 dB); every frame of the three vLLM-Omni rows passes it.

## Current limitations

- AURA-5B (wav2vec2 features, Wan2.2 VAE, 720p at 25 fps) is not supported yet.
- The speech is read as a whole file before generation starts; live speech input while the video streams is not supported.
- One request at a time (`max_num_seqs=1`); no tensor or sequence parallelism, no CPU offload.

## References

- Paper: [AURA: AUdio-dRiven streaming Avatar](https://arxiv.org/abs/2611.26540) (BMVC 2026)
- Weights: [kyutai/AURA-1.3B](https://huggingface.co/kyutai/AURA-1.3B)
- Reference code: [kyutai-labs/aura](https://github.com/kyutai-labs/aura)
