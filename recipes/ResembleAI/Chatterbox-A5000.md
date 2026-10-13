# Chatterbox Turbo and Original — RTX A5000

The model integration includes its required adapter extension hooks.
[PR #8842](https://github.com/vllm-project/vllm-omni/pull/8842) contains independent
shared runtime fixes. Historical measurements below used the combined implementation
and do not qualify the standalone branch.

[Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) generates
mono 24 kHz English speech. The integration uses vLLM's GPT-2 backbone for
speech-token generation and S3Gen meanflow with HiFT for waveform decoding.
It supports the checkpoint's built-in voice and voice cloning from one recording.

## Summary

| Field | Value |
| --- | --- |
| Vendor / checkpoints | ResembleAI: `chatterbox-turbo`, `chatterbox` |
| Runtime / mode | vLLM-Omni, two-stage offline and streaming speech serving |
| Hardware | One NVIDIA RTX A5000, 24 GB, PCIe; no inter-GPU communication |
| Maintainer | PR author; see [#8700](https://github.com/vllm-project/vllm-omni/pull/8700) |

## Offline inference

The production tokenizer and conditioning helpers also prepare offline requests.
Run from the repository root after installing the Chatterbox extra:

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

For cloning, load a recording with `soundfile.read` and replace `voice` with
`VoiceConditioner(model, config, torch.device("cpu")).prepare(wav, sample_rate)`.
The recording must be mono and meet the variant's reference-duration minimum.
The interactive streaming player lives at `apps/chatterbox/gradio_demo.py`.

## Supported model contract

| Surface | Contract |
| --- | --- |
| Input | English text; optional single reference recording, no transcript |
| Reference limits | Turbo: more than 5 seconds; Original: shared API minimum 1 second |
| Output | Mono 24 kHz; non-streaming WAV/PCM/MP3/FLAC; streamed PCM/WAV |
| Default profiles | `chatterbox_turbo.yaml`, `chatterbox.yaml`: BF16 AR, FP32 decoder, 8 AR slots; Original CFG uses two slots per request |
| High-concurrency profiles | 64 requests; A5000 qualification below uses a reduced 128-token cap, not the default full-context allowance |
| Unsupported | Multilingual checkpoint, VoiceDesign, free-form instructions, prefix caching |

## Hardware and software qualification

The A5000 qualification below used one 24 GB device and CPU reference staging.
Host-memory peak, OS, Python, driver and framework versions were not retained
with that historical run; do not infer them from the separate H100 environment.
Revision `223f3c5f6` identifies that run, which predates the full-batch decoder.
The current head has unit coverage; fresh cold-conditioning and WebSocket load
qualification of the full-batch decoder is still outstanding.

## References

- [Upstream implementation](https://github.com/resemble-ai/chatterbox)
- [Offline examples](../../examples/offline_inference/text_to_speech/README.md#chatterbox-turbo-and-original)
- [Online examples](../../examples/online_serving/text_to_speech/README.md#chatterbox-turbo-and-original)
- [Speech API](../../docs/serving/speech_api.md)
- [Supported models](../../docs/models/supported_models.md)

## Installation and serving

Follow the source installation guide, then install the reference-conditioning extra:

```bash
uv pip install -e '.[chatterbox]'
vllm-omni serve ResembleAI/chatterbox-turbo --omni --host 127.0.0.1 --port 8091
```

The default deployment uses one CUDA GPU with 24 GB, Model Runner V2,
BF16 autoregressive generation with CUDA graphs, and eager FP32 waveform decoding.
Both stages also support Model Runner V1. Set `model_runner: v1` in a copy of
`vllm_omni/deploy/chatterbox_turbo.yaml` and pass it with `--deploy-config`.
Only the CUDA deployment has been validated.

Stage 1 can opt into BF16 flow autocast with
`hf_overrides: {flow_dtype: bfloat16}` while retaining FP32 HiFT and FP32
output audio. The default remains `float32`. This is a numerical tradeoff:
validate speech and speaker quality for the intended workload before enabling
it. Decoder-only measurements are not end-to-end performance guarantees.
On CUDA, add `flow_attention_backend: cudnn` to the same `hf_overrides` to
select native cuDNN SDPA with the existing flow masks. This requires BF16 flow;
unsupported devices or attention shapes fail. The default backend is `auto`.

For the measured H100 throughput settings, use the opt-in
`vllm_omni/deploy/chatterbox_turbo_high_concurrency_cuda.yaml` profile. It inherits
the 64-request profile and enables BF16 flow, native cuDNN attention, and CUDA
MPS across the API conditioner and model stages. It requires
`nvidia-cuda-mps-control` on PATH. HiFT remains FP32. The ordinary profiles retain
FP32 flow. See the [profiling report](Chatterbox-H100.md) for batch-size
measurements, warmup requirements, numerical checks, and latency limitations.

```bash
curl --fail http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ResembleAI/chatterbox-turbo",
    "input": "Hello, welcome to the speech synthesis demonstration.",
    "voice": "default",
    "response_format": "wav",
    "max_new_tokens": 250,
    "seed": 42
  }' --output chatterbox.wav
```

## Reference voices

Supply `ref_audio` as an audio URL or base64 data URL, either a string or a
one-item list. The shared speech API's audio-access restrictions apply.
A transcript is unnecessary. Reference recordings must contain more than five
seconds of audio. Conditioning uses up to fifteen seconds for T3 and ten seconds
for S3Gen, with loudness normalization to -27 LUFS.

```json
{
  "model": "ResembleAI/chatterbox-turbo",
  "input": "Please read this new sentence in the reference voice.",
  "ref_audio": "data:audio/wav;base64,<base64-encoded-reference>",
  "response_format": "wav"
}
```

Each request carries its own voice tensors. Requests without `ref_audio` use
`conds.pt` from the selected model directory or Hub snapshot. Reference encoding
runs outside the API event loop; the encoder is initialized on its first
reference request. A bounded queue admits up to 64 conditioning requests, and
one encoder owner processes batches of up to eight. Identical cache misses share
one encoding operation, and a 128-entry cache stores completed conditioning.
The initialization lock does not serialize request generation.

## Chatterbox Original

Serve `ResembleAI/chatterbox` with the same command to select the Original
deployment (`vllm_omni/deploy/chatterbox.yaml`). It uses a Llama backbone,
classifier-free guidance, and ten-step S3Gen instead of Turbo's GPT-2 and
meanflow decoder. Both deployments expose the same speech and streaming API.
Original uses six seconds of reference audio for T3, ten seconds for S3Gen,
and does not apply Turbo's loudness normalization or five-second minimum.
Its opt-in acceleration profile is
`vllm_omni/deploy/chatterbox_high_concurrency_cuda.yaml`. It enables the same
BF16 flow, cuDNN attention and MPS settings as Turbo. Admitting 64 requests
does not guarantee real-time Original audio; consult the profiling report.
This overlay starts with 30-token chunks to reduce early flow work; the ordinary
profiles keep the 15-token initial setting.

Original accepts `extra_params` with `exaggeration` and `cfg_weight`, both
defaulting to 0.5. Values must be finite and nonnegative. Setting `cfg_weight`
to zero disables the unconditional companion. Otherwise each request occupies
two autoregressive sequence slots; the atomic scheduler keeps both branches
together, and only the conditional branch produces audio. Keep prefix caching,
chunked prefill, and asynchronous scheduling disabled for this deployment.

```json
{
  "model": "ResembleAI/chatterbox",
  "input": "Hello, welcome to the speech synthesis demonstration.",
  "voice": "default",
  "extra_params": {"exaggeration": 0.7, "cfg_weight": 0.5},
  "response_format": "wav"
}
```

Turbo supports English and checkpoint vocal-event tags such as `[laugh]`.
It does not support VoiceDesign, free-form `instructions`, speaker embeddings
from other models, or the Original model's exaggeration and CFG controls.

## Streaming and limits

For raw streamed audio, request `"stream": true`, `"stream_format": "audio"`,
and `"response_format": "pcm"` or `"wav"`. PCM is signed 16-bit little-endian
audio at 24 kHz. Complete responses consolidate the same internal chunks.
Set `--no-async-chunk` to decode each finished utterance once instead.

The deployment starts with 15-token chunks, ramps to 30 and 60, and waits for
the flow decoder's three-token lookahead. Reference-length alignment can
increase the first chunk. The decoder keeps a fixed 50-token left context
and stable noise positions for each request.

`max_new_tokens` caps speech tokens at 25 tokens per second, including EOS.
The public API accepts at most 4096 new speech tokens for either variant.
Original enforces the same 4096-token ceiling for offline sampling, including
unguided and stage-0-only requests; an explicit unlimited output budget is unsupported.
Original also accepts at most 2048 text tokens because its text positions are learned.
The default is 1000, subject to the remaining 2048-token stage-0 context.
The prompt occupies one speaker slot, up to 375 reference tokens, the tokenized
text, and one start token. Prefix caching must remain disabled because these
positions use placeholder IDs with request-specific embeddings.

When tuning the deployment, keep stage 1's `max_model_len` at least as large
as stage 0's and `max_num_batched_tokens` at least
`max_num_seqs * max_model_len`. These bounds cover the full utterance even
when a request raises the default output cap. Stage 1 requires FP32 for the
vocoder. Memory and latency depend on concurrency, text length, and reference
length; tune on the intended hardware.

## Conditioning load validation at 64 clients

Use the public streaming API harness for both deployments on the same hardware.
Unlike a configuration limit, this exercises reference loading, reference conditioning,
autoregressive generation, and waveform decoding together. The opt-in
`chatterbox_turbo_high_concurrency.yaml` and `chatterbox_high_concurrency.yaml`
profiles admit 64 requests (128 Original CFG branches), batch conditioning in
groups of eight and decode the scheduled waveform batch together. They explicitly
set stage 0 `hf_overrides.conditioning_device: cuda:0`: the API process and both
stages must share visible GPU zero on one host. The standard profiles retain CPU
conditioning. Include encoder weights and activations in the GPU memory budget.

Conditioning has one encoder owner, a 64-request admission bound, batches of up
to eight references, and a 128-entry content-keyed cache. Concurrent identical
references share encoding work; each caller receives separate conditioning
tensors. Cancellation retains admission until its encoder work finishes.

On an RTX A5000 (24 GB), revision `223f3c5f6` completed 576 Original WebSocket
sessions and 1152 sentence segments at 64 clients, with a 128-token cap, across
three waves each of cold distinct-reference, warm repeated-reference, and
built-in-voice workloads. All streams passed; speech shared-memory entries
returned to zero while the server remained running. Cold/warm/built-in session
p95 completion times were 111.64/106.40/66.27 seconds. These short runs do not
establish the default 1000-token capacity or latency parity with Qwen3-TTS.
The strict per-wave scheduler-capacity check was false: not every sampled wave
reached 128 simultaneous Original CFG branches.

H100 results, including historical microbatch measurements, are recorded in the
[H100 recipe](Chatterbox-H100.md). They do not qualify this A5000 profile.

These sessions were driven by a Chatterbox-specific load harness that has
since been removed in favour of the shared `benchmarks/tts/bench_tts.py`
driver (see the [TTS benchmark README](../../benchmarks/tts/README.md)).
Reproduce the HTTP workload from a fresh server, so the first wave includes
conditioner initialization:

```bash
python benchmarks/tts/bench_tts.py \
  --model ResembleAI/chatterbox-turbo \
  --task voice_clone --locale en \
  --dataset-path benchmarks/build_dataset/seed_tts_smoke \
  --host 127.0.0.1 --port 8091 \
  --concurrency 64 --num-prompts 192 --num-warmups 64 \
  --output-dir results/turbo-c64
```

Repeat with `ResembleAI/chatterbox` after restarting the server for Original.
Keep hardware, reference recordings, concurrency and caps identical when
comparing models. The driver reports first-audio latency, real-time factor and
streaming continuity (`audio_underrun`, `audio_continuity_ok`); it does not
sample scheduler occupancy, so client completion alone does not establish
active capacity. The sentence-segmented WebSocket frontend has no shared load
driver.

### Sentence-segmented WebSocket sessions

The WebSocket receiver continues draining input during generation so protocol
keepalive replies remain readable. Each connection permits up to 8 MiB of pending
messages and 4096 queued messages, in addition to the existing per-message limits.
Exceeding either backlog bound aborts active generation and closes with code 1008.

## Supported features and verification

| Feature | Status |
| --- | --- |
| Built-in voice / reference voice | Both variants; [request contract](../../docs/serving/speech_api.md#chatterbox-turbo-and-original) |
| Async chunk streaming | Delta audio; [shared streaming guide](../../docs/design/feature/async_chunk.md) |
| AR CUDA graphs | vLLM runner capture; decoder remains eager |
| Original CFG / exaggeration | Supported; Turbo rejects these controls |
| Other accelerator platforms | Configuration-only; no qualification claimed |

For offline verification, use the [Chatterbox example](../../examples/offline_inference/text_to_speech/README.md#chatterbox-turbo-and-original).
Expect a finite, nonempty mono waveform at 24 kHz. For browser playback use the
[streaming demo](../../examples/online_serving/text_to_speech/README.md#chatterbox-turbo-and-original).
MP3/FLAC and eager-versus-graph checks passed on the fresh H100 integration
run; browser playback is qualified separately in the
[H100 recipe](Chatterbox-H100.md). These are not new A5000 measurements.
