# Stable-Audio-Open Text-To-Audio Generation on 1x GPU

> Text-to-audio recipe for Stable Audio Open with offline inference and
> OpenAI-compatible online serving on 1x RTX 4090 24GB.

## Summary

- Vendor: Stability AI
- Model: `stabilityai/stable-audio-open-1.0`
- Task: Text-to-audio generation
- Mode: Offline inference and online serving
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to run Stable Audio Open on a single RTX 4090
24GB GPU for music or sound-effect generation. The recipe covers a 10-second
offline validation sample with TeaCache and online serving through the
`/v1/audio/generate` endpoint.

## References

- Model: <https://huggingface.co/stabilityai/stable-audio-open-1.0>
- Offline example:
  [`examples/offline_inference/text_to_audio`](../../examples/offline_inference/text_to_audio)
- Online example:
  [`examples/online_serving/stable_audio`](../../examples/online_serving/stable_audio)
- Audio generation API:
  [`docs/serving/audio_generate_api.md`](../../docs/serving/audio_generate_api.md)
- Related issue:
  [#2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

### GPU

#### 1x RTX 4090 24GB

##### Environment

- OS: Ubuntu 22.04.5
- Python: 3.12
- GPU: NVIDIA GeForce RTX 4090, 24564 MiB VRAM
- Driver / runtime: NVIDIA driver 595.80, CUDA-capable runtime matching the
  repository build
- vLLM version: 0.22.0
- vLLM-Omni version: source checkout
- PyTorch: 2.11.0+cu130
- Model path used in the commands below: `/path/to/stable-audio-open-1.0`

Stable Audio Open is a gated Hugging Face model. Accept the model license on
the Hugging Face model card before downloading the checkpoint.

```bash
hf auth login

hf download stabilityai/stable-audio-open-1.0 \
  --local-dir /path/to/stable-audio-open-1.0
```

##### Commands

Run a 10-second offline validation sample from the repository root:

```bash
python examples/offline_inference/text_to_audio/text_to_audio.py \
  --model /path/to/stable-audio-open-1.0 \
  --prompt "A gentle piano melody with soft room ambience" \
  --negative-prompt "Low quality, distorted, noisy" \
  --seed 42 \
  --guidance-scale 7.0 \
  --audio-length 10.0 \
  --num-inference-steps 50 \
  --cache-backend tea_cache \
  --output examples/offline_inference/text_to_audio/stable_audio_10s.wav
```

Start the online serving endpoint:

```bash
vllm-omni serve /path/to/stable-audio-open-1.0 \
  --host 0.0.0.0 \
  --port 8091 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --enforce-eager \
  --omni
```

Generate a 10-second WAV file from the repository root in another terminal:

```bash
curl http://localhost:8091/health

curl -X POST http://localhost:8091/v1/audio/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "A gentle piano melody with soft room ambience",
    "audio_length": 10.0,
    "num_inference_steps": 50,
    "guidance_scale": 7.0,
    "negative_prompt": "Low quality, distorted, noisy",
    "seed": 42,
    "response_format": "wav"
  }' \
  --output examples/online_serving/stable_audio/piano_10s.wav
```

##### Verification

Check that:

- The offline command writes a valid WAV file.
- The server responds on `http://localhost:8091/health`.
- The online request writes a valid WAV file.
- The generated audio sample rate is 44.1 kHz.
- The generated duration is approximately 10 seconds.
- Peak sampled GPU memory is within the 24GB RTX 4090 budget. In the validated
  run, offline and online generation each peaked at about 12.6 GiB.

Validate the offline outputs:

```bash
ls -lh examples/offline_inference/text_to_audio/stable_audio_10s.wav

python - <<'PY'
import soundfile as sf

path = "examples/offline_inference/text_to_audio/stable_audio_10s.wav"
audio, sample_rate = sf.read(path)
print("sample_rate:", sample_rate)
print("shape:", audio.shape)
print("duration:", len(audio) / sample_rate)
PY
```

Validate the online output:

```bash
ls -lh examples/online_serving/stable_audio/piano_10s.wav

python - <<'PY'
import soundfile as sf

path = "examples/online_serving/stable_audio/piano_10s.wav"
audio, sample_rate = sf.read(path)
print("sample_rate:", sample_rate)
print("shape:", audio.shape)
print("duration:", len(audio) / sample_rate)
PY
```

##### Notes

- `stable-audio-open-1.0` can generate up to about 47 seconds of 44.1 kHz
  stereo audio. This recipe validates 10-second WAV outputs.
- `--cache-backend tea_cache` is supported and was used for the 10-second
  offline validation command.
- The model is gated on Hugging Face and requires license acceptance before
  download.
- If online serving fails while importing `torchaudio`, make sure the
  `torchaudio` wheel matches the installed PyTorch and CUDA build. The
  validated environment used `torch==2.11.0+cu130` and
  `torchaudio==2.11.0+cu130`.
- The `NIXL is not available`, `GLOO_SOCKET_IFNAME`, and `torchsde` boundary
  warnings observed during validation did not prevent successful generation.
- This recipe was validated on one RTX 4090 24GB GPU only. Other GPU counts,
  ROCm, XPU, and NPU setups are not covered here.
- Long generations, higher inference-step counts, and non-WAV response formats
  were not benchmarked in this recipe.
