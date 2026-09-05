# Breeze-TTS-2

[Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) generates 24 kHz
English and Chinese speech. This integration supports reference-free voice
design with optional natural-language instructions and CFG fixed to 1.

The first stage runs the T5Gemma2 text encoder, vLLM's Qwen3 backbone with
paged KV caching, and Breeze's per-frame depth decoder. The second stage
reuses the Qwen3 codec bundled in the checkpoint's `audio_tokenizer/` directory.
Each request owns its sampling state and generated-code history. The codec
retains its streaming state per request.

## Installation and serving

Install vLLM and vLLM-Omni following the project's source installation guide.
The integration uses existing dependencies; the Breeze reference repository
and `qwen-tts` Python package are not required.

```bash
vllm-omni serve BreezeBlue/Breeze-TTS-2 --omni --host 127.0.0.1 --port 8091
```

The default deployment uses one GPU, two active requests, BF16 generation,
FP32 codec decoding, and eager execution. Prefix caching and chunked prefill
are disabled because scheduled prompt tokens are placeholders for the
T5Gemma2 embeddings. The provided configuration targets GPUs with at least
16 GiB of memory; other GPU workloads reduce the available capacity.

```bash
curl --fail http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BreezeBlue/Breeze-TTS-2",
    "input": "你好，欢迎使用语音合成系统。",
    "instructions": "年轻女性，声音温暖自然，使用标准普通话。",
    "response_format": "wav",
    "seed": 42,
    "max_new_tokens": 250
  }' --output breeze.wav
```

For streamed PCM, add `"stream": true`, `"stream_format": "audio"`, and
`"response_format": "pcm"`. The returned PCM is mono, signed 16-bit little
endian at 24 kHz. Complete HTTP responses use the same chunked internal
pipeline and accumulate audio before returning it.

Generation temperature, top-k, top-p, and repetition penalty come from stage
0's `default_sampling_params` in the deployment YAML. A temperature of zero
selects greedy decoding for both the backbone head and the depth decoder.
The maximum token count is the maximum number of generated audio frames;
each frame corresponds to 80 ms of audio, and EOS also consumes one token.

## Offline inference

The production helper
`vllm_omni.model_executor.models.breeze_tts.prompt.build_breeze_prompt`
accepts the checkpoint tokenizer, input text, instructions, and sampling
parameters. Pass its result to `Omni.generate()` with the default Breeze
deployment. Use the same seed in the stage-0 `SamplingParams` to reproduce
sampling independently of other requests. Numerical differences between GPU
types or batch shapes can still change autoregressive outputs.

## Supported scope

- English and Chinese voice design, with or without instructions.
- Streaming and complete `/v1/audio/speech` responses.
- Single-GPU eager execution and concurrent requests.
- The released checkpoint layout, without checkpoint conversion.

Reference-audio cloning/direction, CFG other than 1, tensor/pipeline
parallelism, quantization, and CUDA Graph capture are not implemented in this
initial integration. The speech adapter rejects reference-audio and named-voice
requests. The deployment requires `async_chunk: true` even when the HTTP
request asks for a complete response.

## Validation

CPU checks compare the depth transformer's cached attention with Hugging Face
Llama's full-sequence attention, including a two-row batch, and check that
interleaved frame generation preserves request-local sampling state.

```bash
pytest -v tests/model_executor/models/test_breeze_tts.py -m 'core_model and cpu'
```

The serving tests require a CUDA GPU and the checkpoint available through the
Hugging Face cache or the test harness's `MODEL_PREFIX` environment variable.

```bash
pytest -v tests/e2e/online_serving/test_breeze_tts_2.py \
  -m 'core_model and tts' --run-level=core_model

pytest -v tests/e2e/online_serving/test_breeze_tts_2.py \
  -m 'advanced_model and tts' --run-level=advanced_model
```

The core-model run uses the test harness's dummy weights and eight-frame
requests. The advanced-model run loads real weights and also checks audio
content through the shared test client.

The English PCM test uses a 0 dB harmonic-to-noise floor: the original eager
Breeze runtime scores 0.56 dB on the same text, instructions, seed (42), and
CFG (1), below the shared helper's default 1 dB threshold. This is a
sample-specific catastrophic-distortion check. The shared helper checks
transcripts for the Chinese and concurrent English WAV cases; it does not
transcribe raw PCM. These checks do not establish perceptual quality parity.

The reference inference source and the released weights have separate
licenses. See the [reference repository](https://github.com/breezeblue-ai/breeze-tts)
and the checkpoint's model card for their terms.

## Serving benchmark

Use the shared TTS benchmark against the running server. For a local model
directory, add `--served-model-name /path/to/Breeze-TTS-2`.

```bash
python benchmarks/tts/bench_tts.py \
  --model BreezeBlue/Breeze-TTS-2 --task voice_design --locale en \
  --host 127.0.0.1 --port 8091 --concurrency 1 2 \
  --num-prompts 4 --num-warmups 1 --request-seed 42 --output-len 250 \
  --output-dir results/breeze
```

### Initial eager measurements

Measured on one RTX 5070 Ti (16 GiB), Ubuntu 24.04 under WSL2, with vLLM
0.28.0, PyTorch 2.13.0+cu130, Transformers 5.14.1, and the default deployment
above. Each concurrency setting used one warmup and four measured English
voice-design prompts, repeated in two rounds. All 16 measured requests
succeeded. Ranges below are the two rounds' means.

| Concurrent requests | RTF | First audio (ms) | End-to-end latency (s) | Audio seconds per wall second |
| --- | --- | --- | --- | --- |
| 1 | 1.80–1.93 | 774–809 | 6.81–7.28 | 0.520–0.555 |
| 2 | 3.36–3.42 | 1471–1627 | 13.90–15.29 | 0.534–0.538 |

RTF above 1 means synthesis is slower than realtime. The depth decoder
currently processes each request separately within a backbone step, so this
initial implementation makes no throughput or speedup claim. Sampling state
is independent, but floating-point differences between batch shapes can
change generated frame counts.

A 50-second device-memory sample during the second round peaked at 10.79 GiB.
This includes Windows desktop applications, which used 3.55 GiB after the
server stopped; it is not an isolated process-memory measurement. Initial
requests can also incur kernel compilation that is absent from warmed runs.
