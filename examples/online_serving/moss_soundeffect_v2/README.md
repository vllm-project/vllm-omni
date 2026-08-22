# MOSS-SoundEffect-v2.0 Online Serving

Serve `OpenMOSS-Team/MOSS-SoundEffect-v2.0` through the OpenAI-compatible
`/v1/audio/generate` endpoint. The model generates 48 kHz mono audio and
supports clips ending at or before 30 seconds.

This initial integration supports basic single-request inference. Parallelism
and cache-backend acceleration are not yet supported.

## Start The Server

From the repository root:

```bash
bash examples/online_serving/moss_soundeffect_v2/run_server.sh
```

The script uses port `8091` by default. Override its settings with environment
variables when needed:

```bash
MODEL=/path/to/MOSS-SoundEffect-v2.0 HOST=127.0.0.1 PORT=8091 \
  bash examples/online_serving/moss_soundeffect_v2/run_server.sh
```

## Generate Audio

### curl

```bash
curl -X POST http://localhost:8091/v1/audio/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The sound of a dog barking in a quiet park",
    "audio_length": 10.0,
    "guidance_scale": 5.0,
    "num_inference_steps": 10,
    "seed": 42,
    "response_format": "wav",
    "extra_params": {
      "sigma_shift": 5.0
    }
  }' \
  --output dog.wav
```

### Python Client

```bash
python examples/online_serving/moss_soundeffect_v2/openai_client.py \
  --text "Thunder and rain outside a wooden cabin" \
  --audio_length 10.0 \
  --guidance_scale 5.0 \
  --num_inference_steps 10 \
  --sigma_shift 5.0 \
  --seed 42 \
  --output thunder.wav
```

The Python client sends the same JSON request as the curl example.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | required | Text description of the sound to generate. |
| `audio_length` | float | 10.0 | Duration of the returned clip in seconds. |
| `audio_start` | float | 0.0 | Start offset of the returned clip. `audio_start + audio_length` must not exceed 30 seconds. |
| `negative_prompt` | string | null | Optional classifier-free guidance negative prompt. |
| `guidance_scale` | float | 5.0 | Classifier-free guidance scale. |
| `num_inference_steps` | integer | 10 | Number of denoising steps. |
| `seed` | integer | random | Seed used to initialize diffusion noise. |
| `response_format` | string | `wav` | Output format: `wav`, `pcm`, `flac`, `mp3`, `aac`, or `opus`. |
| `extra_params.sigma_shift` | float | 5.0 | Flow-matching scheduler shift. |

`audio_start` selects a segment from the generated timeline. For example,
`audio_start=2` and `audio_length=10` returns the interval `[2, 12)`.

## Verify The Output

```bash
python - <<'PY'
import soundfile as sf

audio, sample_rate = sf.read("dog.wav")
print("sample_rate:", sample_rate)
print("shape:", audio.shape)
print("duration:", len(audio) / sample_rate)
PY
```

The expected sample rate is 48000 Hz and the expected duration is approximately
the requested `audio_length`.

## Troubleshooting

- Check server health with `curl http://localhost:8091/health`.
- Reduce `audio_length` or `num_inference_steps` if a request times out.
- Ensure `audio_start + audio_length <= 30`.
- Use `extra_params` for model-specific values such as `sigma_shift`; unknown
  top-level request fields are rejected by the API schema.

## See Also

- [Offline text-to-audio example](../../offline_inference/text_to_audio/README.md)
- [Audio generation API](../../../docs/serving/audio_generate_api.md)
- [Model card](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
