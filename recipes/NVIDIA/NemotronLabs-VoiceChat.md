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
- Staged pipeline config:
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

Stages 0/1 default to fp32 for exact parity with the NeMo reference
implementation (greedy decoding matches it token for token on the acceptance
fixture). The deploy yaml documents a ~2x-faster bf16 thinker option whose
output stayed within one word of the reference in testing.

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

The single-GPU profile runs three eager fp32 engine processes on one device.

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

#### Notes

- Memory usage: the shipped yaml runs all three stages on one GPU
  (`gpu_memory_utilization` 0.62 / 0.12 / 0.06); peak usage is dominated by the
  fp32 thinker. The fp32 default has a hard floor of roughly 43 GB of thinker
  weights alone (9B backbone + 587M `embed_tokens` + 587M `function_head` +
  0.6B Conformer), so 48 GB cards cannot run it — use the bf16 thinker option
  documented in the deploy yaml on anything smaller than an 80 GB part.
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
  guarantee. In an eager fp32 H200 profile, Stage 2 emitted 80 ms packets at
  roughly 0.87--0.88 s intervals (RTF about 10.8--10.9); placing the three
  stages on separate GPUs did not remove the serial 8-step EAR-TTS bottleneck.
  The current native duplex path is functionally streaming but not wall-clock
  realtime.
- Known limitations: batch=1 only. The native duplex deployment allows one
  active session and does not support barge-in. Tool execution remains
  client-owned: the server emits function-call events, accepts a validated
  `function_call_output`, and resumes the live model with the returned result;
  it does not execute arbitrary tools itself.
