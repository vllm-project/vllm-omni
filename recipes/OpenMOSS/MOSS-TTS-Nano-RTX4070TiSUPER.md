# MOSS-TTS-Nano on RTX 4070 Ti SUPER 16GB

> Single-stage 0.1B AR TTS with a mandatory reference clip, measured on one
> 16 GB consumer GPU.

## Summary

- Vendor: OpenMOSS
- Model: `OpenMOSS-Team/MOSS-TTS-Nano` (`MossTTSNanoForCausalLM`)
- Task: Multilingual text-to-speech, voice cloned from a reference clip
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API
- Hardware: 1x NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB
- Maintainer: Community

## When to use this recipe

Use this when you want measured numbers for MOSS-TTS-Nano on a single 16 GB
consumer NVIDIA card. The model is also documented for Ascend NPU in
[`MOSS-TTS-Nano-NPU.md`](./MOSS-TTS-Nano-NPU.md), and general usage — launch,
requests, streaming PCM and the Gradio demo — is covered in the shared
[text-to-speech online serving guide](../../examples/online_serving/text_to_speech/README.md#moss-tts-nano).
This recipe adds only the hardware qualification for this card.

## Supported model contract

| Property | Value |
| --- | --- |
| Checkpoint | `OpenMOSS-Team/MOSS-TTS-Nano` (0.1B AR LM + MOSS-Audio-Tokenizer-Nano codec) |
| Input | Text plus a mandatory `ref_audio` clip; there are no built-in speaker presets |
| Output | 48 kHz mono WAV, or PCM when streaming |
| Qualified profile | Single GPU, one request at a time, as set by the bundled deploy config |

The OpenAI-schema `voice` and `ref_text` fields are accepted but ignored; see the
shared guide linked above for why.

## References

- Shared example and request shapes:
  [`examples/online_serving/text_to_speech/README.md`](../../examples/online_serving/text_to_speech/README.md#moss-tts-nano)
- Deploy config: [`vllm_omni/deploy/moss_tts_nano.yaml`](../../vllm_omni/deploy/moss_tts_nano.yaml)
- Ascend NPU profile: [`MOSS-TTS-Nano-NPU.md`](./MOSS-TTS-Nano-NPU.md)
- Supported models table: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)

## Hardware

- Accelerator: 1x NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB
- Number of devices: 1
- Interconnect: not applicable to a single-device profile
- Host memory: 32 GB, of which the WSL2 VM was allocated 15 GiB
- Qualification scope: runtime-qualified for non-streaming `/v1/audio/speech`

## Software environment

- OS: Ubuntu 24.04.4 LTS on WSL2, kernel 6.6.114.1-microsoft-standard-WSL2
- Python: 3.12.3
- PyTorch: 2.13.0+cu132
- Driver / runtime: NVIDIA 610.60 / CUDA 13.2
- vLLM version: 0.29.0
- vLLM-Omni version or commit: `85da94339130337b49e610d08276aedcf84f6b96`

## Command

The deploy config at
[`vllm_omni/deploy/moss_tts_nano.yaml`](../../vllm_omni/deploy/moss_tts_nano.yaml)
auto-loads from the model registry; no `--deploy-config`, `--trust-remote-code`
or `--enforce-eager` flag is needed, and nothing in it was overridden for this
qualification.

```bash
vllm serve OpenMOSS-Team/MOSS-TTS-Nano --omni --port 8091
```

## Verification

Every request must carry a reference clip. This run used `zh_1.wav` from the
upstream repository. The Base64 clip is about 419,000 characters, which exceeds
the per-argument limit of `execve` (`MAX_ARG_STRLEN`, 128 KiB), so the request
body is written to a file and posted with `--data-binary` instead of being
inlined in `-d`.

```bash
REF_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/moss-tts-nano"
mkdir -p "$REF_DIR"
REF_WAV="$REF_DIR/zh_1.wav"
[ -s "$REF_WAV" ] || curl -L -o "$REF_WAV" \
    https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS-Nano/main/assets/audio/zh_1.wav

python3 - "$REF_WAV" > request.json <<'PY'
import base64, json, sys
clip = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
print(json.dumps({
    "input": "Hello, this is MOSS-TTS-Nano running on a sixteen gigabyte consumer card.",
    "ref_audio": "data:audio/wav;base64," + clip,
    "response_format": "wav",
}, ensure_ascii=False))
PY

curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    --data-binary @request.json --output output.wav
```

Three inputs of different lengths were sent against one server instance. All
returned HTTP 200 with decodable 48 kHz mono WAV:

| Input | Wall clock | Audio duration | Size |
| --- | ---: | ---: | ---: |
| `你好，这是在 RTX 4070 Ti SUPER 上运行的语音合成测试。` (first request) | 8.29 s | 5.92 s | 568,364 B |
| `Hello, this is MOSS-TTS-Nano running on a sixteen gigabyte consumer card.` | 3.91 s | 5.04 s | 483,884 B |
| `短句测试。` | 1.02 s | 1.12 s | 107,564 B |

These are single observations, not a benchmark: the first request also absorbs
warmup and the three inputs differ in length. Before startup, a 40-second
desktop baseline averaged 13% GPU utilisation and 24 W power, so the card was
not otherwise idle.

## Notes

- Memory usage: `nvidia-smi` was sampled continuously across startup, model
  load, idle and the three requests (281 samples). The highest whole-device
  sample was **2211 MiB / 16376 MiB**. Desktop baselines taken immediately
  before startup and after shutdown averaged 1351 MiB and 1375 MiB, so the
  server's own contribution is estimated at about **848 MiB**, leaving roughly
  13.8 GiB of the card free. The sampler polls and then sleeps 0.2 s, so a
  shorter transient peak between samples would not appear in these figures.
- The load phase peaked at 2198 MiB and the request phase at 2211 MiB, so on
  this run the two were close; the engine reported 0.31 GiB for model loading,
  which took 5.26 s.
- Startup to `Application startup complete` took 42 s.
- Output determinism: the deploy config sets `seed: 42`. Two requests with
  identical input earlier in this qualification produced byte-identical WAV
  files, and the three different inputs above produced three different outputs.
- Audio content was checked by listening. The English and short Chinese samples
  were judged acceptable. The longer Chinese sample had an unclear opening
  greeting and unreliable pronunciation of "4070 Ti SUPER"; Whisper
  transcription of the same clip returned the greeting but also garbled the
  model name. Voice timbre was judged acceptable, and no obvious pops,
  distortion or repetition were reported. Some PCM samples reached full scale:
  84 points (0.030%) in the longest clip with a longest run of 10 samples
  (0.21 ms), 47 points in the second and none in the short clip. The cause of
  the pronunciation issues was not investigated.
- `max_num_seqs: 1` in the deploy config is a correctness constraint, not a
  memory one: the config notes that the remote model and audio tokenizer hold
  process-wide RNG and streaming decode state, so generation must stay serial.
  The spare memory on a 16 GB card does not lift that limit.

## Supported features

| Feature | Status for this profile | Guide |
| --- | --- | --- |
| Non-streaming `/v1/audio/speech` | Measured, three requests | [Speech API](../../docs/serving/speech_api.md) |
| Streaming PCM | Implemented upstream, not exercised here | [shared TTS guide](../../examples/online_serving/text_to_speech/README.md#moss-tts-nano) |
| `ref_audio` voice cloning | Required on every request; no speaker presets | [shared TTS guide](../../examples/online_serving/text_to_speech/README.md#moss-tts-nano) |
| Concurrent generation | Capped at one sequence by the deploy config, for correctness rather than memory | [`moss_tts_nano.yaml`](../../vllm_omni/deploy/moss_tts_nano.yaml) |
| CUDA graph capture | Disabled by the deploy config (`enforce_eager: true`) | [`moss_tts_nano.yaml`](../../vllm_omni/deploy/moss_tts_nano.yaml) |
