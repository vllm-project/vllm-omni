# Step-Audio-2-mini-Think

`stepfun-ai/Step-Audio-2-mini-Think` shares the `StepAudio2` thinker,
Token2Wav, `step_audio_2` pipeline, and native `<|BOT|>`/`<|EOT|>` chat
template (`step_audio2.jinja`) with `Step-Audio-2-mini`. It adds deep audio
reasoning before generating the spoken or text response.

Existing `Step-Audio-2-mini` and `Step-Audio-2-mini-Base` deployments and
behavior remain unchanged.

## Launch the Server

```bash
vllm serve stepfun-ai/Step-Audio-2-mini-Think --omni --port 8092 \
    --deploy-config vllm_omni/deploy/step_audio_2_think.yaml \
    --chat-template vllm_omni/transformers_utils/chat_templates/step_audio2.jinja \
    --trust-remote-code
```

`--chat-template` is required: the checkpoint's `tokenizer_config.json` bundles
a generic Qwen `<|im_start|>` template rather than Step-Audio2's `<|BOT|>` /
`<|EOT|>` format.

## Variant and Stop-Token Configuration

`vllm_omni/deploy/step_audio_2_think.yaml` inherits all settings from
`step_audio_2.yaml` via `base_config` and overlays stage 0:

```yaml
base_config: step_audio_2.yaml

stages:
  - stage_id: 0
    hf_overrides:
      step_audio2_variant: think
    default_sampling_params:
      max_tokens: 2048
      stop_token_ids: [151665]  # <|EOT|>
```

- **Why `stop_token_ids: [151665]` is required**: The Think checkpoint's
  `tokenizer_config.json` sets `eos_token` to `<|endoftext|>` (`151643`), while
  upstream `StepAudio2.__init__` overrides EOS to `<|EOT|>` (`151665`). Setting
  `stop_token_ids: [151665]` in stage 0's `default_sampling_params` restores
  `<|EOT|>` termination for both `/v1/chat/completions` and offline
  `Omni.generate`.
- **`max_tokens: 2048`**: Matches
  [upstream `examples-think.py`](https://github.com/stepfun-ai/Step-Audio2/blob/main/examples-think.py)
  for reasoning and audio token output.
- **`/v1/audio/speech`**: Requests against a `think` deployment raise a
  `ValueError` directing callers to `/v1/chat/completions` (or
  `Step-Audio-2-mini-Base` for verbatim TTS).

## Two-Pass Reasoning via `/v1/chat/completions`

Following upstream `think_test_vllm` in `examples-think.py`, the two-pass
reasoning + speech loop is orchestrated on the client side over standard
`/v1/chat/completions`:

1. **Pass 1 (reasoning, text-only)**: Append `{"role": "assistant", "content": "<think>"}`
   and call `/v1/chat/completions` with `modalities=["text"]`,
   `stop=["</think>"]`, `continue_final_message=True`, and
   `add_generation_prompt=False`.
   - `<think>` and `</think>` are regular text tokens (not special tokens) in
     the Think vocabulary, so string `stop=["</think>"]` stops right before
     `</think>`.
   - Passing `modalities=["text"]` avoids running Stage 1 Token2Wav during the
     reasoning pass.
2. **Pass 2 (speech response)**: Append `think_text + "</think>\n\n<tts_start>"`
   to the final `assistant` message and call `/v1/chat/completions` again with
   `continue_final_message=True` and `add_generation_prompt=False`. Generation
   stops automatically at `<|EOT|>` (`151665`).
3. **Multi-turn history**: Strip the `<think>...</think>` reasoning block from
   the assistant turn before starting the next round, keeping only the final
   response.

```python
SYSTEM = (
    "你的名字叫小跃，你是由阶跃星辰(StepFun)公司训练出来的语音大模型，"
    "你能听见用户的声音特征并在思维过程中描述出来，"
    "请激活深度思考模式，通过逐步分析、逻辑推理来解决用户的问题。"
)
history = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": audio_data_uri}}]},
    {"role": "assistant", "content": "<think>"},
]

# Pass 1: reasoning only
r1 = client.chat.completions.create(
    model="stepfun-ai/Step-Audio-2-mini-Think",
    messages=history,
    modalities=["text"],
    stop=["</think>"],
    temperature=0.7,
    max_tokens=2048,
    extra_body={"continue_final_message": True, "add_generation_prompt": False},
)
think_text = r1.choices[0].message.content

# Pass 2: spoken response (uses stage-0 default repetition_penalty=1.05 and stops at <|EOT|>)
history[-1]["content"] += think_text + "</think>\n\n<tts_start>"
r2 = client.chat.completions.create(
    model="stepfun-ai/Step-Audio-2-mini-Think",
    messages=history,
    temperature=0.7,
    max_tokens=2048,
    extra_body={"continue_final_message": True, "add_generation_prompt": False},
)
```
