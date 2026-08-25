# MiniCPM-o 4.5 Code2Wav early setup

Status: implemented and statically validated; A3 overlap, score, and complete
quality evidence are pending.

## Hypothesis

Seed-TTS uses a different reference audio per request. Reusing a cross-request
initial-state template therefore has poor expected hit rate, but prompt feature
extraction and the initial CFM state can run after the first codec token while
Talker is still accumulating the 25-code audio window. Moving that required
work earlier should reduce first-audio latency without changing the amount or
math of Code2Wav work.

## Protocol

On the first non-duplex codec delta, Stage 1 sends one internal setup packet
when fewer than 25 codec frames are ready:

- "meta.init_only=true", "code_flat_numel=0", and a one-token transport
  placeholder make the existing async generation request schedulable;
- "chunk_seq" remains zero and the producer's "codec_end" remains zero;
- the first request reference audio and sample rate move on this packet;
- pending codec tokens remain in Stage 1 and the first real audio packet still
  uses "chunk_seq=0".

Stage 2 prepares prompt features and the request-owned Token2Wav initial state,
stores it with an internal sequence position of -1, emits no PCM, and accepts
the first real codec packet at sequence zero. Duplicate or malformed setup
packets fail explicitly. Abort and normal request cleanup use the existing
request-state and runtime-prompt release paths.

The output processor recognizes the internal marker, drains the empty audio
delta, keeps the Stage 2 streaming request active, and publishes no client
request output. Therefore the packet cannot become an empty audio SSE or alter
the official TTFP endpoint.

## Activation and rollback

The path is MiniCPM-o async-chunk source behavior and defaults on because the
official evaluator supplies its own deploy YAML. Precedence is:

1. connector extra "code2wav_early_setup" when explicitly present;
2. "VLLM_OMNI_MINICPMO45_EARLY_CODE2WAV_SETUP";
3. default true.

Set the environment variable to 0 and cleanly restart for rollback. Native
full-duplex turns and non-async full-payload execution retain the parent path.

## Promotion gate

- Parent/candidate codec sequence, audio, state shape/dtype, and final output
  match on A3.
- Timeline proves setup moves before the first 25-code window and does not emit
  PCM or client output.
- RTF remains non-inferior within 0.5% and TTFP improves at least 5%, or mean
  RTF improves at least 2% with TTFP/TTFT no more than 1% worse.
- Peak HBM growth is at most 5%; request, stream, and decodable-audio success
  remain 100%; all four complete quality gates pass.
