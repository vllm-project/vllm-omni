# Breeze-TTS-2

[Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) generates 24 kHz
English and Chinese speech. The integration supports Voice Design from text
instructions, Voice Clone from a reference recording and transcript, and
Voice Direction combining reference conditioning with instructions.

## Installation and serving

Follow the project's source installation guide for vLLM and vLLM-Omni.
The integration loads the released checkpoint directly. The original Breeze
repository and the `qwen-tts` package are not required. The common requirements
include `soxr` to reproduce reference-audio resampling.

```bash
vllm-omni serve BreezeBlue/Breeze-TTS-2 --omni --host 127.0.0.1 --port 8091
```

The default configuration uses one CUDA GPU, BF16 generation and FP32 reference
encoding and waveform decoding. It targets GPUs with at least 16 GiB available
for the deployment; other workloads consume additional memory.

On WSL2, set vLLM's supported pinned-memory option before starting the server:

```bash
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
```

Native Linux enables pinned memory by default. Startup warmup exercises plain
generation, CFG, reference conditioning and two-request batches. New text
lengths or batch sizes can require additional graph
capture or compilation; warm the workload before measuring steady performance.

## Voice Design

```bash
curl --fail http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BreezeBlue/Breeze-TTS-2",
    "input": "你好，欢迎使用语音合成系统。",
    "instructions": "年轻女性，声音温暖自然，使用标准普通话。",
    "extra_params": {"guidance_scale": 4.0},
    "response_format": "wav",
    "seed": 42,
    "max_new_tokens": 250
  }' --output breeze.wav
```

Omitting `instructions` uses `Speak clearly and naturally.`. An explicitly
empty instruction remains empty. Vocal-event tokens, including `(laughs)`
and `[笑]`, are passed to the checkpoint unchanged.

## Voice Clone and Voice Direction

Add `ref_audio` and its exact `ref_text` transcript. Reference audio accepts the
shared speech API's audio URLs and base64 data URLs. The shared resolver's
format, duration and local-file access restrictions apply.

```json
{
  "model": "BreezeBlue/Breeze-TTS-2",
  "input": "Please read this new sentence clearly and naturally.",
  "ref_audio": "data:audio/wav;base64,<base64-encoded-reference>",
  "ref_text": "The words actually spoken in the reference recording.",
  "instructions": "Keep the reference voice. Speak gently and calmly.",
  "extra_params": {"guidance_scale": 4.0},
  "response_format": "wav",
  "seed": 42
}
```

Use `guidance_scale: 1.0` with the default instruction for ordinary cloning.
Reference audio plus a delivery instruction and CFG provides Voice Direction.
When explicitly specified, `task_type` is `Base` for reference conditioning
and `VoiceDesign` without reference audio.

Stereo references are averaged to mono and resampled to 24 kHz with soxr HQ.
Reference text is encoded independently from the target text. The reference
codec preserves full causal attention during offline encoding, as used by the
released Breeze runtime; newer Transformers sliding-window defaults would
change long references. Reference convolution uses full FP32 precision.

## Sampling and output

Supported `extra_params`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `guidance_scale` | 1.0 | Any finite positive value; values other than 1 use two CFG branches |
| `temperature` | 0.9 | Zero selects greedy generation |
| `top_k` | 50 | Zero or -1 disables top-k filtering |
| `top_p` | 1.0 | Nucleus sampling probability in (0, 1] |
| `repetition_penalty` | 1.1 | Positive penalty applied to the first codebook's generated history |

CFG combines conditional and unconditional logits in the Qwen3 output head
and all 15 remaining depth codebooks. Both branches retain reference audio
and reference text; the unconditional branch removes only target instructions.

`max_new_tokens` limits generated audio frames, including EOS. Each frame is
80 ms; the default is 1500 tokens, subject to the remaining 2048-token context.
Both CFG branches stop at the same generation limit despite unequal prompts.
Each request owns its seed, generator, audio-code history and codec state.
Different GPU kernels and batch shapes can still change sampled outputs.

For streamed PCM, use `"stream": true`, `"stream_format": "audio"`, and
`"response_format": "pcm"`. Output is mono signed 16-bit little-endian PCM at
24 kHz. Complete WAV responses accumulate the same internal streaming output.
The first chunks contain 1, 2, 4 and 5 codec frames, then remain at 5 frames.

## Pipeline and optimization

Stage 0 batches T5Gemma2 text encoding, optional reference-audio encoding,
vLLM's native Qwen3 backbone with paged KV caching, and Breeze's Llama depth
decoder. Text segments up to 128 tokens use padded CUDA Graphs with separate
attention masks. Longer segments use dynamically compiled encoder layers
without retaining a separate graph workspace for each input length.
The depth decoder uses fused QKV/MLP projections, compiled layer kernels,
fixed frame-local KV buffers and a CUDA Graph covering all 15 depth steps.
Temperature, top-k and top-p are mutable GPU inputs to the captured sampler.
Changing those settings, or changing a positive CFG scale within the paired
path, reuses the graph. Each request owns its random-number generator.

Reference encoding uses dynamic-shape compilation with CUDA Graphs disabled
for that path, so large reference-convolution workspaces remain reusable by
the allocator. Every convolution retains each recording's actual boundary;
the final downsampler repeats its last valid state for partial frames, matching
independent encoding. Static convolution metadata removes GPU-to-CPU padding
calculations. Startup compiles reference batches of one and two.
Waveform storage is aligned to 1920-sample frames, with that frame width
marked static for compilation. Actual sample lengths still govern every
convolution boundary. This avoids nested dynamic ceil/mod expressions that
otherwise make compiler tiling spend minutes simplifying reference shapes.

Stage 1 reuses Qwen3-TTS's stateful codec and its CUDA Graph decoder with the
checkpoint's `audio_tokenizer/` weights. Shared-memory chunks connect stages.
Only the generated-code decode path is captured: reference conditioning was
already consumed in stage 0, so the Qwen ICL reference-prefix graphs and their
large convolution workspaces are unnecessary for Breeze.

The Qwen3 backbone uses compiled piecewise prefill and full decode graphs.
Stage 0 uses synchronous scheduling to keep CFG branches at the same
generation step; stage 1 uses asynchronous scheduling. CFG admission reserves
enough KV pages for both branches' complete generation and preserves their
actual position IDs. Cancellation finishes both branches.

The default admission policy lets active requests produce eight audio frames
before adding another prefill. This gives playback a buffer before text and
reference encoding share the device with an existing stream. It increases a
queued request's first-audio latency; benchmark it separately from an idle server.

The default `max_num_seqs: 2` counts physical branches: it permits two plain
requests or one CFG request at a time. Additional CFG requests queue.
Prefix caching and chunked prefill are disabled because prompt token IDs
reserve positions for externally encoded embeddings. The deployment requires
`async_chunk: true`, including for complete HTTP responses.

The supported execution scope is one CUDA GPU, TP=1, PP=1 and unquantized
released weights. Named voices, speaker embeddings, dual CFG and multiple
reference recordings are not supported. The original public fast runtime
also does not expose dual CFG.

## Offline inference and validation

`vllm_omni.model_executor.models.breeze_tts.prompt.build_breeze_prompt`
accepts a checkpoint tokenizer, text, instructions, sampling parameters,
and optional `ref_audio=(waveform, sample_rate)` plus `ref_text`. Pass its
result to `Omni.generate()` using the default Breeze deployment. Waveform
arrays use soundfile's `(samples, channels)` convention.

CPU tests compare cached depth attention and CFG against Hugging Face Llama
and cover request-owned sampling, payload transport, chunk delivery and CFG
admission. GPU tests compare padded batched text graphs with independent
eager encoders and exercise depth graphs across CFG scales and batch shapes.

```bash
pytest -v tests/model_executor/models/test_breeze_tts.py \
  tests/core/sched/test_omni_cfg_ar_scheduler.py -m 'core_model and cpu'
pytest -v tests/model_executor/models/test_breeze_tts_graphs.py -m 'core_model and cuda'
pytest -v tests/e2e/online_serving/test_breeze_tts_2.py \
  -m 'core_model and tts' --run-level=core_model
pytest -v tests/e2e/online_serving/test_breeze_tts_2.py \
  -m 'advanced_model and tts' --run-level=advanced_model
```

Serving tests use the shared client for Chinese CFG design, English PCM,
concurrent CFG requests, reference cloning and direction. Core tests use dummy
weights and eight-frame requests; advanced tests use real weights and the
shared content checks. These checks do not establish perceptual quality parity.
The Chinese content check uses Whisper large-v3 as its primary transcriber:
small misrecognizes homophones on the fixed test clip. The expected text and
shared similarity threshold are unchanged; there is no failed-check retry.
Ensure enough host memory and disk space for the large-v3 test model when
Whisper runs on CPU alongside the serving process.

The English PCM test uses a 0 dB harmonic-to-noise floor because the original
eager runtime scores 0.56 dB on that same prompt and seed, below the shared
1 dB floor. Raw PCM is not transcribed by the shared helper. Separate matched
evaluations must retain audio and compare intelligibility, speaker similarity,
instruction effects, first audio, RTF and playback underruns.

The shared serving benchmark supports default speech, Voice Design and
Voice Clone. For Voice Design:

```bash
python benchmarks/tts/bench_tts.py \
  --model BreezeBlue/Breeze-TTS-2 --task voice_design --locale en \
  --host 127.0.0.1 --port 8091 --concurrency 1 2 \
  --num-prompts 4 --num-warmups 1 --request-seed 42 --output-len 250 \
  --output-dir results/breeze
```

For a local model path, add `--served-model-name /path/to/Breeze-TTS-2`.
For cloning, select `--task voice_clone --dataset-path /path/to/seed-tts-eval`
with the desired locale. Reference conditioning with instructions and other
CFG scales also require identical requests and sampling settings in the
reference runtime and this deployment.

The reference source and released weights have separate licenses. Consult
the [reference repository](https://github.com/breezeblue-ai/breeze-tts) and
checkpoint model card.
