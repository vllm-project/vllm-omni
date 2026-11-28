# Step-Audio-2-mini-Base

`stepfun-ai/Step-Audio-2-mini-Base` shares the `StepAudio2` thinker, Token2Wav,
and `step_audio_2` pipeline with `Step-Audio-2-mini`. Only the prompt format
differs: Base is a pretrained checkpoint without `<|BOT|>`/`<|EOT|>` turn
markers and is the upstream checkpoint used for verbatim TTS (`tts_test`) and
T2ST (`t2st_test`) alongside ASR, S2TT, and S2ST.

Existing `Step-Audio-2-mini` launch commands, deploy configs, templates, and
serving behavior remain unchanged.

## Launch the Server

```bash
vllm serve stepfun-ai/Step-Audio-2-mini-Base --omni --port 8092 \
    --deploy-config vllm_omni/deploy/step_audio_2_base.yaml \
    --chat-template vllm_omni/transformers_utils/chat_templates/step_audio2_base.jinja \
    --trust-remote-code
```

`--chat-template` is required: the checkpoint's `tokenizer_config.json` bundles
a generic Qwen `<|im_start|>` chat template rather than Step-Audio2's format.

## Variant Configuration

`vllm_omni/deploy/step_audio_2_base.yaml` inherits all runtime settings from
`step_audio_2.yaml` via `base_config` and adds a single stage-0 override:

```yaml
base_config: step_audio_2.yaml

stages:
  - stage_id: 0
    hf_overrides:
      step_audio2_variant: base
```

- `/v1/audio/speech` reads `hf_config.step_audio2_variant` to choose the speech
  prompt builder. When unset, it defaults to `mini`. Unsupported variant values
  raise a `ValueError`.
- `/v1/chat/completions` formatting is controlled solely by `--chat-template`.

## TTS via `/v1/audio/speech` (Recommended)

`/v1/audio/speech` follows the
[upstream Base TTS example](https://github.com/stepfun-ai/Step-Audio2/blob/main/examples-base.py):
`StepAudio2Adapter` tokenizes `instructions + "\n"` and `input + "<tts_start>"`
separately (`add_special_tokens=False`) and concatenates the two ID lists into
`prompt_token_ids`. This preserves upstream tokenization boundaries even when
`input` starts with whitespace or a newline.

```bash
curl -X POST http://localhost:8092/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "stepfun-ai/Step-Audio-2-mini-Base",
      "input": "你好呀，我是你的AI助手，很高兴认识你！",
      "instructions": "以自然的语速读出下面的文字。",
      "voice": "default"
    }' --output output_tts.wav
```

When `instructions` is omitted, the adapter defaults to
`"以自然的语速读出下面的文字。"`.

## Multimodal Tasks via `/v1/chat/completions`

`vllm_omni/transformers_utils/chat_templates/step_audio2_base.jinja`
concatenates message contents in order without roles or `<|BOT|>`/`<|EOT|>`
markers:

- Each input audio part emits one `<audio_patch>`, which
  `StepAudio2MultiModalProcessor` expands into 25-second
  `<audio_start>...<audio_end>` chunks.
- Callers provide any required spaces or newlines between text segments.
- For speech output (e.g., S2ST), place `<tts_start>` in the final `assistant`
  message and pass `continue_final_message=True` and
  `add_generation_prompt=False` in `extra_body`.
- Generation stops at the checkpoint's native EOS token `<|endoftext|>`
  (`151643`). Do not add Mini's `<|EOT|>` stop token (`151665`).

See [upstream `StepAudio2Base`](https://github.com/stepfun-ai/Step-Audio2/blob/main/stepaudio2.py)
for the reference prompt layout.

### Tokenization Boundary Notes

| Scenario | Parity with upstream `StepAudio2Base` |
| --- | --- |
| `/v1/audio/speech` (TTS) | Exact token-for-token match (two-segment encoding) |
| `/v1/chat/completions` with audio (ASR, S2TT, Audio Caption, S2ST) | Exact token-for-token match (`<audio_patch>` and `<tts_start>` are special tokens and prevent cross-segment BPE merges) |
| `/v1/chat/completions` with adjacent plain-text messages | Rendered to a single string before tokenization; may differ if the boundary can trigger a Qwen2 BPE merge (for example `"。\n"` + `"\n你好"` or `"Translate: "` + `"hello"`) |

For strict token-level parity on TTS, use `/v1/audio/speech`. To replay raw
mixed text/audio token histories (upstream `{"type": "token", ...}` entries),
pass `prompt_token_ids` and `multi_modal_data` to offline `Omni.generate`.
