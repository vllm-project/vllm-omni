# Lychee-FD

Lychee-FD is a speech-to-speech model with separate semantic, acoustic and
interaction-control branches. This integration runs its autoregressive model
through Model Runner V2 (MRV2), its sessions through Unified Duplex, and its
streaming Token2Wav decoder as a native second Omni stage.

The public model is [HIT-TMG/Lychee-FD](https://huggingface.co/HIT-TMG/Lychee-FD).
The checkpoint keeps the `step_audio_2_full_duplex` model type; architecture
aliases `LycheeFullDuplex`, `StepAudio2FullDuplex` and
`LycheeFullDuplexForConditionalGeneration` select the same native implementation.
The separate vocoder is the `token2wav/` bundle from
[Step-Audio-2-mini](https://huggingface.co/stepfun-ai/Step-Audio-2-mini/tree/main/token2wav).

## Deployment

See the [single-A100 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/HIT-TMG/Lychee-FD-A100.md)
for checkpoint downloads, required environment variables and the standard
`vllm-omni serve --omni` command. The recipe uses user-selected model directories;
it does not require the released demo server or an HTTP Token2Wav sidecar.

Use `vllm_omni/deploy/lychee_fd_single_gpu.yaml` for one session on one GPU.
The explicit `lychee_fd_c2_single_gpu.yaml` and `lychee_fd_c4_single_gpu.yaml`
profiles admit two and four sessions on the same replica. Both stages must have
matching capacity. Their explicit 4 GiB AR KV budgets differ from the C1 default
memory-utilization budget; select a profile that fits the host.

All three profiles use eager execution, TP=PP=1 and synchronous scheduling.
Prefix caching and native preemption are disabled. CUDA Graphs, multi-GPU
execution and AR/Token2Wav overlap are outside the qualified deployment scope.
The C2/C4 profiles establish functional support, not a throughput guarantee.

## Session and audio contract

Connect to `/v1/realtime?duplex=1` using the
[Realtime Duplex API](../serving/realtime_duplex_api.md). Input is mono 16 kHz
PCM16, processed in 400 ms model windows. Output is mono 24 kHz PCM16 with
response ownership, ordered chunk sequences and a final event. The production
vocoder uses a 10-codec hop and three-codec lookahead. Keep microphone audio
flowing while the model speaks so the model can make its native listen/speak
and backchannel decisions.

The plugin supports input append, model-native turn control, client commit,
explicit barge-in, response cancellation, playback ACK, engine-managed admission
and reconnect/resume while the serving process remains alive. Playback ACK
reports progress; it does not rewind the model. Resume does not restore a
session after a process restart.

Image input, text-only turns, chat completions, audio truncation, checkpoint
rollback and concurrent turns within one session are disabled. A duplex route
being mounted does not make every OpenAI endpoint supported by this model.
Inspect the capabilities returned in `session.created` before using optional
operations.

## Validation scope

Migration evidence was collected on one NVIDIA A100 80 GiB with a local
`dzall-10004-release` checkpoint, using vLLM 0.31.0 and the tested Torch
2.13.0+cu132 runtime. It covers native runner selection, checkpoint mapping,
327 committed windows, ordinary C2/C4 admission and mixed batches,
cancel/barge-in/close/reopen, and WebSocket resume/replay/ACK. Fault-injection
and late-output tests are separate correctness diagnostics.

Small public metadata checks found the same 982 weight-key names, six shards,
28/4/4/4 decoder layout and encoder configuration as the validated local
checkpoint. Architecture names and dtype serialization differ; the public
configuration also omits top-level EOS/PAD IDs present in the local checkpoint.
These checks do not establish equal numerical weights or qualify execution of
the publicly downloaded checkpoint. Keep public-checkpoint validation separate
from the local migration results.

The standalone Token2Wav steady run passed 1803 seconds with 64 measured cycles
and 192 natural final responses, bounded request retirement and a memory
plateau. It reused one recorded 187-codec fixture and does not qualify the full
service or general answer quality.

Complete-answer quality, the standard server smoke, full-service steady-state
progress and uncontaminated realtime performance remain explicit release checks.
Shared-GPU correctness measurements are not realtime benchmarks. No speedup,
realtime SLA, broader hardware support or released-demo quality equivalence is
claimed here.
