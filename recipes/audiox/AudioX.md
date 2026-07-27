# AudioX

> AudioX MMDiT for unified audio + music generation: t2a / t2m / v2a / v2m / tv2a / tv2m

## Summary

- Vendor: HKUSTAudio (project), `zhangj1an/AudioX` weight bundle
- Model: `zhangj1an/AudioX`
- Task: Text/video → audio or music. Six tasks: `t2a`, `t2m`, `v2a`, `v2m`, `tv2a`, `tv2m`.
- Mode: Offline inference + online serving (pure diffusion)
- Maintainer: Community

## When to use this recipe

Use this recipe to run AudioX for sound-effect (`*2a`) or music (`*2m`) generation
from a text prompt and/or video clip. AudioX is a unified diffusion transformer
that produces stereo 44.1 kHz audio up to ~10 s per call.

## References

- Project page: <https://zeyuet.github.io/AudioX/>
- vLLM-Omni weight bundle: <https://huggingface.co/zhangj1an/AudioX>
- Pipeline: `vllm_omni.diffusion.models.audiox.pipeline_audiox.AudioXPipeline`
- Input transforms: `vllm_omni.transformers_utils.processors.audiox`
- Param contract: `vllm_omni/model_extras/audiox.py` (declared `extra_body` knobs)
- Offline example: [`examples/offline_inference/text_to_audio/text_to_audio.py`](../../examples/offline_inference/text_to_audio/text_to_audio.py)
- Online: standard OpenAI chat-completions endpoint (see commands below)

## Hardware Support

## GPU

### 1x L4 24GB

#### Environment

- OS: Ubuntu 22.04
- Python: 3.10+
- Driver / runtime: CUDA 12.4
- vLLM version: 0.20.0
- vLLM-Omni version: 0.1.x

#### Command

AudioX uses the **standard** `text_to_audio` example offline and the standard
OpenAI chat-completions endpoint online. All model-specific knobs
(`audiox_task`, `seconds_start`, `seconds_total`, `sigma_min`, `sigma_max`,
`cfg_rescale`, `video_path`, `audio_path`) are declared in
`vllm_omni/model_extras/audiox.py` and routed through `extra_body` (online) /
`--extra-body` (offline). The offline example also exposes `--task`, `--video`,
`--audio-start`, and `--audio-length` shortcuts for the most common knobs.

Offline — text tasks (`t2a` / `t2m`):

```bash
huggingface-cli download zhangj1an/AudioX --local-dir ./audiox_weights
python examples/offline_inference/text_to_audio/text_to_audio.py \
  --model ./audiox_weights --task t2a \
  --prompt "Fireworks burst twice, followed by a clock ticking." \
  --num-inference-steps 250 --guidance-scale 6.0 --audio-length 10.0 \
  --extra-body '{"sigma_min": 0.03, "sigma_max": 1000.0}' \
  --output t2a.wav
```

Offline — video-conditioned tasks (`v2a` / `v2m` / `tv2a` / `tv2m`) need `--video`:

```bash
python examples/offline_inference/text_to_audio/text_to_audio.py \
  --model ./audiox_weights --task tv2a \
  --prompt "drum beating sound and human talking" \
  --video https://zeyuet.github.io/AudioX/static/samples/V2M/1XeBotOFqHA.mp4 \
  --num-inference-steps 250 --guidance-scale 6.0 --audio-length 10.0 \
  --output tv2a.wav
```

Online:

```bash
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
  vllm serve zhangj1an/AudioX --omni --model-class-name AudioXPipeline --port 8099

curl -sS -X POST http://localhost:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "zhangj1an/AudioX",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Uplifting ukulele"}]}],
    "extra_body": {
      "num_inference_steps": 250,
      "guidance_scale": 7.0,
      "seed": 42,
      "audiox_task": "t2m",
      "seconds_total": 10.0,
      "sigma_min": 0.3,
      "sigma_max": 500.0
    }
  }' > t2m.json
```

For `v2*` / `tv2*` online, attach the video as a `video_url` content item
(data URI for local files) in the message `content`.

#### Verification

```bash
# Health check
curl http://localhost:8099/health

# Listen to the saved file (stereo, 44.1 kHz, sigma_min=0.03, sigma_max=1000 — upstream defaults)
ffprobe t2a.wav
```

#### Notes

- Memory usage: ~10 GB peak with `num_inference_steps=250`, 10 s of audio.
- Output rate: 44.1 kHz stereo regardless of requested rate.
- Supported tasks: `t2a`, `t2m`, `v2a`, `v2m`, `tv2a`, `tv2m`. Pass via the
  `audiox_task` knob (in `extra_body` online, or `--task` offline).
- Video conditioning: `v2*` and `tv2*` require a video; offline via `--video`
  (→ `video_path` knob), online via a `video_url` content item.
- Cache acceleration is **not** supported (AudioXPipeline is in `_NO_CACHE_ACCELERATION`).
- Tensor parallelism is supported via `--tensor-parallel-size` (DiT QKV is sharded with
  `QKVParallelLinear`); cross-attention K/V is also TP-sharded.

### Known limitations

- Inference uses an inlined DPM-Solver++(3M) SDE sampler (k-diffusion port). Replacing it with
  diffusers' `EDMDPMSolverMultistepScheduler` introduces a fixed ~861 Hz resonance and is not
  recommended.
- Generation is fixed at 10 s (configured by the bundle's `sample_size`); longer outputs require
  a different bundle.
