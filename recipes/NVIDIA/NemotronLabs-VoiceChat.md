# NemotronLabs VoiceChat 11B

> Speech-to-speech on a 3-stage vLLM-Omni pipeline: offline single-turn
> inference and experimental model-native, frame-locked Realtime serving.

## Summary

- Vendor: NVIDIA
- Model: [`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
- Task: Speech-to-speech voice chat. The model runs a frame-locked 12.5 Hz
  timeline: a Conformer + NemotronH hybrid-Mamba thinker emits one text token
  per acoustic frame, a Gemma3-1B EAR-TTS talker turns the text timeline into
  31-quantizer RVQ code stacks, and an RVQ-VAE codec decodes them to audio.
- Mode: Offline single-turn inference (batch=1), plus experimental native
  duplex Realtime serving at one 80 ms microphone frame per scheduler wake.
- Maintainer: [`@yuekaizhang`](https://github.com/yuekaizhang)

## When to use this recipe

Use this recipe to run a single-turn voice-chat exchange with
`NVIDIA-NemotronLabs-VoiceChat-11B` on one GPU: you provide a user utterance as
a WAV file (any sample rate; it is resampled to 16 kHz mono) plus an optional
spoken-style system prompt, and get back the agent's reply as text and a
22.05 kHz WAV. The integration is NeMo-free at runtime — the perception
Conformer, EAR-TTS talker, and RVQ-VAE codec are vendored, dependency-stripped
NeMo modules (`nemo_vendored/`), so no `nemo_toolkit` install is needed.

## References

- Offline example:
  [`examples/offline_inference/nemotron_voicechat/end2end.py`](../../examples/offline_inference/nemotron_voicechat/end2end.py)
- Model modules (thinker / talker / code2wav / vendored NeMo):
  [`vllm_omni/model_executor/models/nemotron_voicechat/`](../../vllm_omni/model_executor/models/nemotron_voicechat/)
- Staged pipeline config (NeMo bit-parity path):
  [`vllm_omni/deploy/nemotron_labs_voicechat.yaml`](../../vllm_omni/deploy/nemotron_labs_voicechat.yaml)
- Native duplex config and E2E probe:
  [`vllm_omni/deploy/nemotron_labs_voicechat_duplex.yaml`](../../vllm_omni/deploy/nemotron_labs_voicechat_duplex.yaml),
  [`tests/e2e/online_serving/nemotron_voicechat_realtime_duplex.py`](../../tests/e2e/online_serving/nemotron_voicechat_realtime_duplex.py)
- Nightly model-level gate:
  [`tests/e2e/online_serving/test_nemotron_voicechat_duplex.py`](../../tests/e2e/online_serving/test_nemotron_voicechat_duplex.py)
- Upstream model card:
  [`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
- Reference implementation: NVIDIA-NeMo/Speech, branch `nemotron-labs-voicechat`

## Pipeline

| stage | arch | dtype | role |
|---|---|---|---|
| 0 thinker | `NemotronVoiceChatThinkerForConditionalGeneration` (LLM_AR) | fp32 | WAV + system prompt -> frame-locked text-token timeline (+ function channel) |
| 1 talker | `NemotronVoiceChatTalker` (LLM_AR) | fp32 | text timeline -> 31-quantizer RVQ code stacks (one per 80 ms frame) |
| 2 code2wav | `NemotronVoiceChatCode2Wav` (LLM_GENERATION) | fp32 | RVQ-VAE decode -> 22.05 kHz PCM |

The deploy yamls default to a fast execution profile; every fast setting
carries a "PARITY:" comment whose edits restore the eager fp32 reference
execution that matches the NeMo implementation token for token (see
Performance below).

## Performance

Each deploy yaml defaults to a fast execution profile, mirroring how NVIDIA's
own serving stack runs this model (bf16 LLM + CUDA-graph decode, EAR-TTS on a
vLLM engine, incremental codec cache):

- `nemotron_labs_voicechat.yaml` (offline sync): bf16 + CUDA-graph thinker and
  the talker's Gemma3 backbone as a native vLLM model
  (`hf_overrides.use_native_talker`) — PagedAttention KV, vLLM decode CUDA
  graphs, and the MoG sampling step captured into a single graph. RTF ~0.23 on
  the 196-frame acceptance fixture (1x H100, warm).
- `nemotron_labs_voicechat_streaming.yaml` (offline streaming): the vendored
  per-frame EAR-TTS step captured into CUDA graphs
  (`hf_overrides.use_talker_cuda_graphs`), 5-frame codec chunks decoded
  incrementally with a per-request causal-conv cache
  (`use_incremental_codec_cache`, O(T) instead of O(T^2) codec work). First
  audio chunk ~1.5 s after submission on the same fixture.
- `nemotron_labs_voicechat_duplex.yaml` (realtime): native talker + bf16
  thinker; audio is delivered as one 80 ms packet per frame at the frame clock
  (median inter-chunk gap ~81 ms on the bundled fixture) and the session
  length is unbounded (paged KV, no StaticCache bucket cap).

Notes on the fast settings:

- Thinker: `dtype: bfloat16` + `enforce_eager: false`. The 9B NemotronH decode
  is memory- and launch-bound, so both are enabled together; weights fit in
  ~21 GB, so the pipeline also runs on sub-80 GB parts. Text stays coherent
  but greedy decoding can legitimately diverge from the fp32 reference
  (verified turn-for-turn coherent and ASR-clean on the bundled fixtures).
- Talker (captured-step, streaming yaml): the whole EAR-TTS frame step
  (Gemma3 backbone via HF StaticCache + CFG batch doubling + MoG iterative
  unmasking) replays as one CUDA graph per cache-size bucket. MoG noise draws
  from the CUDA-graph Philox stream: audio is equivalent but not
  bit-identical to the eager path. Capture failures and oversize sessions
  fall back to the eager step automatically.
- Talker (native, sync/duplex yamls): engine positions map 1:1 onto timeline
  steps; fused input embeddings are numerically verified against the
  reference forward, and the reference's EOS-silence code feedback
  (`inference_force_speech_silence_on_eos`) is applied. Supports
  multi-session batching (`max_num_seqs > 1`).

Bit parity with the NeMo reference is preserved behind the `PARITY:` comments
in each yaml: applying those edits runs the eager fp32 reference execution
(md5-anchored WAV on the acceptance fixture; the drain-all talker scheduling
added for performance was verified WAV-byte identical against it).

## Hardware Support

## GPU

### 1x H100 80GB

#### Environment

- OS: Linux
- Python: 3.12
- vLLM version: 0.27.0
- vLLM-Omni version or commit: this PR / current `main`

#### Command

```bash
# Tokenizer: the checkpoint ships no HF tokenizer; it resolves from the
# nvidia/NVIDIA-Nemotron-Nano-9B-v2 HF id automatically. For air-gapped runs,
# point NEMOTRON_VOICECHAT_LLM_PATH at a local snapshot of that repo instead.
python examples/offline_inference/nemotron_voicechat/end2end.py \
    --checkpoint /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
    --wav /path/to/user_question.wav \
    --output-dir results/nemotron_voicechat
```

#### Verification

```bash
ls results/nemotron_voicechat
# <stem>_output.txt          the agent reply as text
# <stem>_output.wav          the agent reply as 22.05 kHz audio
# <stem>_text_tokens.json    the frame-locked text-token timeline
```

The reply text should read as a coherent spoken-style answer to the question in
the input WAV, and the WAV should transcribe to (approximately) the same text
with any ASR model.

#### Native duplex serving

The checkpoint does not contain the underlying Nemotron text tokenizer. Set
`NEMOTRON_VOICECHAT_LLM_PATH` to a local snapshot of
`nvidia/NVIDIA-Nemotron-Nano-9B-v2` before starting an offline deployment.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NEMOTRON_VOICECHAT_LLM_PATH=/path/to/NVIDIA-Nemotron-Nano-9B-v2

vllm-omni serve /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
  --omni \
  --served-model-name nemotron-voicechat \
  --deploy-config vllm_omni/deploy/nemotron_labs_voicechat_duplex.yaml
```

The single-GPU profile runs all three engine processes on one device.

In another shell, stream only the user channel of one of NVIDIA's bundled
stereo conversation fixtures:

```bash
python tests/e2e/online_serving/nemotron_voicechat_realtime_duplex.py \
  --model nemotron-voicechat \
  --input-wav /path/to/NVIDIA-NemotronLabs-VoiceChat-11B/turn_taking.wav \
  --input-channel 0 \
  --output-dir results/nemotron_voicechat_duplex \
  --timeout-s 600
```

The probe requires a completed response, non-silent 22.05 kHz output, and the
advertised native 80 ms append capabilities. Add `--expect-function-call` and
use `tool_call.wav` to validate the function-call channel. To validate the
Realtime tool round trip, also provide the expected arguments and return a
tool result:

```bash
python tests/e2e/online_serving/nemotron_voicechat_realtime_duplex.py \
  --model nemotron-voicechat \
  --input-wav /path/to/NVIDIA-NemotronLabs-VoiceChat-11B/tool_call.wav \
  --output-dir results/nemotron_voicechat_tool_call \
  --expect-function-call \
  --expected-function-name generate_random_number \
  --expected-function-arguments '{"min":1,"max":50}' \
  --function-output 20 \
  --expected-post-tool-text "random number"
```

Probe caveats observed on the bundled `turn_taking.wav`: stream the FULL
fixture (a `--max-frames` truncation cuts the conversation mid-turn, the model
then never closes its last response, and the probe times out waiting for a
`response.done`), and note that the pass/fail condition "a response completes
after the final commit" is timing-sensitive — a faster pipeline can legitimately
complete every turn before the commit lands (use
`--allow-incomplete-response` when measuring latency rather than protocol
completion).

#### Duplex performance profile

`nemotron_labs_voicechat_duplex.yaml` defaults to the fast profile: the
talker's Gemma3 backbone on PagedAttention + vLLM CUDA graphs (MoG sampling
in a captured step graph, unbounded session length) and a bf16 + CUDA-graph
thinker inherited from the streaming base. On the bundled `turn_taking.wav`
fixture (1x H100) the client receives one 80 ms packet per frame at the frame
clock (median inter-chunk gap ~81 ms / p95 ~110 ms) with sub-second first-packet
latency per turn; playback clients should still keep a small (~200-300 ms)
prebuffer to absorb scheduling jitter.

The bf16 thinker's greedy text stream can legitimately diverge from fp32
mid-conversation. On the bundled fixture it was verified turn-for-turn:
coherent transcripts, every turn reaching EOS/`response.done`, ASR-clean
output. Apply the PARITY edits in the yaml to bit-match the fp32 reference
thinker instead.

Classifier-free guidance on the native talker
(`hf_overrides.native_talker_guidance: true` + `gpu_memory_utilization: 0.18`
on stage 1): the conditional stream stays on vLLM paged KV while the
unconditional stream mirrors it on the vendored HF backbone (a captured
StaticCache step), and the pair is blended with NeMo's `generate_step` math
(hidden-space blends in `lm_head` and inside `MoGHead.infer`). The per-frame
cost stays within the 80 ms duplex budget, and a paired 4-fixture ASR study
matched the guided reference. Off by default since guidance at this
checkpoint's scale (0.2) is ASR-indistinguishable; enable it for strict parity
with NVIDIA's guided deployment or for listening tests. Under CFG the
unconditional stream's StaticCache is bucketed (largest 4096 positions ≈ 5.4
minutes); one session at a time uses the captured step, extra concurrent
sessions fall back to eager stepping.

#### Notes

- Memory usage: the shipped yamls run all three stages on one GPU
  (`gpu_memory_utilization` 0.62 / 0.12 / 0.06). The default bf16 thinker fits
  in ~21 GB of weights; the fp32 PARITY thinker needs roughly 43 GB of weights
  alone (9B backbone + 587M `embed_tokens` + 587M `function_head` + 0.6B
  Conformer), so the PARITY execution requires an 80 GB part.
- Input sizing: the timeline is frame-locked, so the reply budget IS the input
  duration. The acoustic channel trails the text channel; if the WAV does not
  carry enough trailing silence for the reply to finish, the spoken answer is
  truncated silently. Leave generous trailing silence (a question ending at
  ~4.5 s truncated in an 8 s WAV but completed cleanly in 16 s); the offline
  example warns when the text channel is still speaking near the last frame.
- Key flags: sampling is greedy end to end. The thinker is frame-locked —
  `max_tokens` equals the acoustic frame count with `ignore_eos=True`. Do NOT
  set `min_tokens` on the thinker: the tokenizer's EOS token is also the
  frame-locked PAD/silence token, so masking it forces the model to babble
  instead of pausing.
- The talker's `max_tokens` is 16383 (its stage prompt is one placeholder
  token, and the stage context is 16384).
- The 80 ms frame period is a model/protocol contract, not a throughput
  guarantee. The default duplex profile keeps up with the frame clock on 1x
  H100; the eager fp32 PARITY execution is functionally streaming but not
  wall-clock realtime (the serial 8-step EAR-TTS unmasking dominates, and
  placing the three stages on separate GPUs does not remove it).
- Known limitations: the offline pipeline is single-request per call
  (the native talker supports engine-level multi-session batching via
  max_num_seqs > 1). The native duplex deployment allows one
  active session and does not support barge-in. Tool execution remains
  client-owned: the server emits function-call events, accepts a validated
  `function_call_output`, and resumes the live model with the returned result;
  it does not execute arbitrary tools itself.
