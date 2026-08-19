# NemotronLabs VoiceChat 11B

> Offline speech-to-speech: a 16 kHz user utterance in, the agent's spoken reply
> (text + 22.05 kHz WAV) out, on a 3-stage vLLM-Omni pipeline.

## Summary

- Vendor: NVIDIA
- Model: [`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
- Task: Speech-to-speech voice chat. The model runs a frame-locked 12.5 Hz
  timeline: a Conformer + NemotronH hybrid-Mamba thinker emits one text token
  per acoustic frame, a Gemma3-1B EAR-TTS talker turns the text timeline into
  31-quantizer RVQ code stacks, and an RVQ-VAE codec decodes them to audio.
- Mode: Offline single-turn inference (batch=1). Online/duplex serving,
  function calling, and batch>1 are documented follow-ups.
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
- vLLM version: 0.26.0
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
- Known limitations: batch=1 only; offline single turn (no online/duplex
  serving yet); the function-call channel is decoded but not yet acted on.
