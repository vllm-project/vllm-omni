# Chatterbox Turbo

[Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) generates
mono 24 kHz English speech. The integration uses vLLM's GPT-2 backbone for
speech-token generation and S3Gen meanflow with HiFT for waveform decoding.
It supports the checkpoint's built-in voice and voice cloning from one recording.

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
A transcript is unnecessary. Reference recordings must contain at least five
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
`conds.pt` from the selected model directory or Hub snapshot. CPU reference
encoding runs outside the API event loop; the encoder is initialized on its
first reference request. Four preprocessing workers can encode references
concurrently; the initialization lock is released before encoding begins.

## Chatterbox Original

Serve `ResembleAI/chatterbox` with the same command to select the Original
deployment (`vllm_omni/deploy/chatterbox.yaml`). It uses a Llama backbone,
classifier-free guidance, and ten-step S3Gen instead of Turbo's GPT-2 and
meanflow decoder. Both deployments expose the same speech and streaming API.
Original uses six seconds of reference audio for T3, ten seconds for S3Gen,
and does not apply Turbo's loudness normalization or five-second minimum.

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
