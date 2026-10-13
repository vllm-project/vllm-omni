# Chatterbox Turbo and Original — RTX A5000

## Summary

- Vendor: Resemble AI
- Model: `ResembleAI/chatterbox-turbo` (Turbo) and `ResembleAI/chatterbox` (English Original)
- Task: English text-to-speech with the checkpoint's built-in voice or a voice cloned from one recording
- Mode: offline `Omni.generate`; online `/v1/audio/speech` (complete or streamed audio) and the sentence-segmented WebSocket frontend
- Hardware: 1x NVIDIA RTX A5000 24 GB
- Maintainer: @will-rice

## When to use this recipe

One 24 GB card serving a handful of concurrent streams with the default
profiles. The reference encoders run on the CPU of the API process, so the
GPU holds only the two model stages.

## Supported model contract

| Task | Entrypoint |
| --- | --- |
| Built-in voice | `voice: "default"` on `/v1/audio/speech`; `VoiceConditioning.from_builtin` offline |
| Voice cloning | one `ref_audio` recording (URL or data URL), no transcript; `VoiceConditioner.prepare` offline |
| Long text | the shared WebSocket frontend with `split_granularity: "sentence"` ([speech API](../../docs/serving/speech_api.md#streaming-text-input-websocket)) |

| Input | Limit |
| --- | --- |
| Text | English. Turbo: the 2048-token stage 0 context (prompt plus output); Original: at most 2048 text tokens |
| Reference recording | mono; Turbo: longer than 5 s; Original: the shared 1 s minimum. T3 conditions on the first 15 s (Turbo) or 6 s (Original), S3Gen on the first 10 s |
| Original controls | `extra_params.exaggeration` and `extra_params.cfg_weight`, finite and nonnegative, default 0.5; Turbo rejects them |
| Speech tokens | `max_new_tokens` default 1000 (40 s at 25 tokens/s), API ceiling 4096, bounded by the stage 0 context |

| Output | Value |
| --- | --- |
| Audio | mono, 24 kHz, float32 offline; WAV, PCM, MP3 or FLAC online; streamed PCM or WAV as delta chunks |
| Streaming | first chunk 15 tokens plus reference alignment, then 30 and 60; a fixed 50-token left context and position-aligned flow noise keep chunks consistent |

| Profile on this hardware | Status |
| --- | --- |
| `chatterbox_turbo.yaml` (default): BF16 T3 with CUDA graphs, FP32 eager S3Gen, 8 AR slots, Model Runner V2 | runtime-qualified, both runners and both handoff modes |
| `chatterbox.yaml` (default Original): the same with the Llama T3 and paired classifier-free guidance (two AR slots per guided request) | runtime-qualified offline and at 64 WebSocket sessions with a 128-token cap |
| `*_high_concurrency.yaml` and `*_high_concurrency_bf16_mps.yaml` (64 requests, GPU conditioning, BF16 flow) | configuration-only here; qualified in the [H100 recipe](Chatterbox-H100.md) |

## References

- [Upstream implementation](https://github.com/resemble-ai/chatterbox) (`chatterbox-tts` 0.1.7)
- [Offline examples](../../examples/offline_inference/text_to_speech/README.md#chatterbox-turbo-and-original)
- [Online examples](../../examples/online_serving/text_to_speech/README.md#chatterbox-turbo-and-original)
- [Speech API contract](../../docs/serving/speech_api.md#chatterbox-turbo-and-original)
- [Supported models](../../docs/models/supported_models.md)
- [Async chunk streaming](../../docs/design/feature/async_chunk.md)

## Hardware

- Accelerator model and per-device memory: NVIDIA RTX A5000, 24 GB
- Number of devices: 1 (`devices: "0"` for both stages)
- Device interconnect: none used; PCIe host attachment
- Host memory: the reference encoders (voice encoder, S3 tokenizer, CAMPPlus) and the built-in voice live in the API process on the CPU; peak host memory was not recorded
- Qualification scope: the default profiles, offline and online, on CUDA only

## Software environment

| Field | Value |
| --- | --- |
| OS | Ubuntu 24.04 |
| Python | 3.12 |
| Driver / runtime | 580.159.04, CUDA 13.0; PyTorch 2.13.0+cu130, torchaudio 2.11.0+cu130 |
| vLLM | 0.31.0 |
| vLLM-Omni | the head of [#8700](https://github.com/vllm-project/vllm-omni/pull/8700) |
| Checkpoints | Turbo `749d1c1a46eb10492095d68fbcf55691ccf137cd`; Original `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` |

The 64-session WebSocket run below predates this environment and its
software versions were not retained.

## Command

Install the reference-conditioning extra, then serve:

```bash
uv pip install -e '.[chatterbox]'
vllm serve ResembleAI/chatterbox-turbo --omni --host 127.0.0.1 --port 8091
# Original: vllm serve ResembleAI/chatterbox --omni --host 127.0.0.1 --port 8091
```

Built-in voice:

```bash
curl --fail http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ResembleAI/chatterbox-turbo",
    "input": "Hello, welcome to the speech synthesis demonstration.",
    "voice": "default",
    "response_format": "wav",
    "seed": 42
  }' --output chatterbox.wav
```

Cloned voice, and the Original's controls:

```json
{
  "model": "ResembleAI/chatterbox",
  "input": "Please read this new sentence in the reference voice.",
  "ref_audio": "data:audio/wav;base64,<base64-encoded-reference>",
  "extra_params": {"exaggeration": 0.7, "cfg_weight": 0.5},
  "response_format": "wav"
}
```

For streamed audio add `"stream": true`, `"stream_format": "audio"` and
`"response_format": "pcm"` (signed 16-bit little-endian, 24 kHz) or `"wav"`.
Pass `--no-async-chunk` to decode each finished utterance once instead.

Offline, with the same tokenizer, prompt and conditioning code serving uses:

```bash
uv run python - <<'PYTHON'
import soundfile as sf
import torch
from vllm_omni import Omni
from vllm_omni.model_executor.models.chatterbox.conditioning import (
    VoiceConditioning, build_prompt, load_text_tokenizer, tokenize_text,
)
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

model, variant = "ResembleAI/chatterbox-turbo", "turbo"
# For Original: model, variant = "ResembleAI/chatterbox", "original"
config = ChatterboxConfig(variant)
tokenizer = load_text_tokenizer(model, variant)
voice = VoiceConditioning.from_builtin(model)
ids = tokenize_text("Hello, this is Chatterbox speaking.", tokenizer, variant)
prompt = build_prompt(ids, voice, config)
engine = Omni(model=model)
try:
    output = list(engine.generate([prompt]))[0].outputs[0].multimodal_output
    audio = output["audio"]
    if isinstance(audio, list):
        audio = torch.cat([chunk.flatten() for chunk in audio])
    sf.write("output.wav", audio.flatten().numpy(force=True), config.sample_rate)
finally:
    engine.close()
PYTHON
```

To clone offline, read a recording with `soundfile.read` and replace `voice`
with `VoiceConditioner(model, config, torch.device("cpu")).prepare(wav, sample_rate)`.

## Verification

```bash
pytest tests/e2e/offline_inference/test_chatterbox.py -m advanced_model --run-level advanced_model
```

Expected: every parametrized case (both checkpoints, both runners, chunked
and whole-utterance) passes, each returning finite, nonempty mono 24 kHz
audio of at least one second for two distinct prompts. The curl command
above writes a playable WAV; `soundfile.info("chatterbox.wav")` reports
`samplerate=24000, channels=1`.

## Notes

- Memory usage: stage 0 reserves 35 % and stage 1 15 % of the card for 8
  slots; the encoders add their weights to host memory, not the GPU.
- Key flags: `model_runner: v1` in a copy of the deploy file selects the
  default runner (both are supported; V2 is the default on CUDA). Stage 1
  accepts `hf_overrides: {flow_dtype: bfloat16, flow_attention_backend: cudnn}`
  for BF16 flow with native cuDNN attention; HiFT and the output stay FP32,
  and the audio changes numerically, so validate quality before enabling it.
- Deploy relations: keep stage 1's `max_model_len` at least stage 0's and its
  `max_num_batched_tokens` at least `max_num_seqs * max_model_len`; the
  stages refuse to start otherwise. Prefix caching must stay off: prompts are
  placeholder ids whose embeddings differ per request.
- Original: each guided request holds two AR slots; `cfg_weight: 0` drops the
  companion. Prefix caching, chunked prefill and asynchronous scheduling stay
  disabled for its atomic scheduler.
- Known limitations: English only (the multilingual checkpoint is not
  served); no VoiceDesign or free-form instructions; no prefix caching. Text
  that needs more than 1000 speech tokens is cut at the cap, as upstream's
  `generate` is; both Omni and upstream produce repetition or unrelated speech
  on unsplit narration of several hundred words, so send long text through
  the sentence frontend. A WebSocket connection may hold 8 MiB or 4096
  queued messages; exceeding either aborts the session with close code 1008.

## Supported features

| Feature | Status |
| --- | --- |
| Built-in voice, voice cloning | both variants; [request contract](../../docs/serving/speech_api.md#chatterbox-turbo-and-original) |
| Async chunk streaming | delta audio on both runners; [feature guide](../../docs/design/feature/async_chunk.md) |
| Model Runner V2, native data plane | default on CUDA; V1 supported |
| AR CUDA graphs | on by default; S3Gen stays eager |
| Original CFG and exaggeration | supported; Turbo rejects the fields |
| BF16 flow, cuDNN attention, CUDA MPS | opt-in overlays, qualified only on H100 |
| Multilingual, VoiceDesign, instructions, prefix caching | unsupported |
| NPU, XPU, ROCm, MUSA | configuration-only (`model_runner: v1` kept there); not qualified |

## Qualification evidence

Turbo, default profile, streaming, short English reference clip, three
sentences scored with Whisper (`small`), medians of three runs over two
engine starts:

| Runner | First audio, 1 stream | RTF, 1 stream | First audio, 8 streams | RTF, 8 streams |
| --- | --- | --- | --- | --- |
| V1 | 0.31–0.38 s | 0.17–0.20 | 1.03–1.04 s | 0.44 |
| V2 | 0.32–0.40 s | 0.17–0.21 | 1.08–1.09 s | 0.44–0.45 |

Transcripts over eight runs per runner: mean WER 0.089 (V1, 72 utterances)
and 0.087 (V2, 48 utterances); 7 of 72 and 4 of 48 utterances exceeded 0.17,
nearly all on product-name words. Streaming also completed at 12 concurrent
streams with no error line. This is a bounded qualification run, not a
benchmark; per-utterance WER varies with the talker's sampling.

Original, an earlier revision (`223f3c5f6`) with a 128-token cap: 576
WebSocket sessions and 1152 sentence segments completed at 64 clients across
three waves each of cold distinct references, a repeated reference and the
built-in voice, with no failure; p95 session completion 111.6, 106.4 and
66.3 s. Shared-memory entries returned to zero afterwards. Not every wave
reached 128 simultaneous CFG branches, and the run does not establish the
default 1000-token capacity.
