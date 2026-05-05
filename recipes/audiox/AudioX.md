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
- Offline example: [`examples/offline_inference/audiox/`](../../examples/offline_inference/audiox/)
- Online example: [`examples/online_serving/audiox/`](../../examples/online_serving/audiox/)

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

Offline (text-to-audio):

```bash
huggingface-cli download zhangj1an/AudioX --local-dir ./audiox_weights
python examples/offline_inference/audiox/end2end.py \
  --model ./audiox_weights \
  --tasks t2a \
  --num-inference-steps 250 \
  --seconds-total 10
```

Online:

```bash
bash examples/online_serving/audiox/run_server.sh
python examples/online_serving/audiox/openai_chat_client.py \
  --task t2a \
  --prompt "Fireworks burst twice, followed by a clock ticking." \
  --output t2a.wav
```

#### Verification

```bash
# Health check
curl http://localhost:8099/health

# Listen to the saved file (stereo, 44.1 kHz, sigma_min=0.03, sigma_max=1000 — upstream defaults)
ffprobe t2a.wav
```

#### Notes

- Memory usage: ~10 GB peak with `num_inference_steps=250`, 10 s of audio.
- Output rate: 44.1 kHz stereo, regardless of `--sample-rate` (resampled in the example
  script if requested).
- Supported tasks: `t2a`, `t2m`, `v2a`, `v2m`, `tv2a`, `tv2m`. Pass via
  `extra_args["audiox_task"]` (offline) or the `extra_args` field in the OpenAI
  chat-completions body (online).
- Video conditioning: `v2*` and `tv2*` require a video file; the online client
  attaches it as an OpenAI `video_url` content item (data URI for local files).
- Cache acceleration is **not** supported (AudioXPipeline is in `_NO_CACHE_ACCELERATION`).
- Tensor parallelism is supported via `--tensor-parallel-size` (DiT QKV is sharded with
  `QKVParallelLinear`); cross-attention K/V is also TP-sharded.

### Known limitations

- Inference uses an inlined DPM-Solver++(3M) SDE sampler (k-diffusion port). Replacing it with
  diffusers' `EDMDPMSolverMultistepScheduler` introduces a fixed ~861 Hz resonance and is not
  recommended.
- Generation is fixed at 10 s (configured by the bundle's `sample_size`); longer outputs require
  a different bundle.
