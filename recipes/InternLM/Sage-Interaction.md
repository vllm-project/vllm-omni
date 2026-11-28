# Sage-Interaction

> Real-time full-duplex omni assistant with native `<|listen|>` / `<|speak|>` / `<|delegate|>` control actions and asynchronous tool-result injection over `/v1/realtime?duplex=1`

## Summary

- Vendor: InternLM (Shanghai AI Laboratory)
- Model: [`internlm/Sage-Interaction`](https://huggingface.co/internlm/Sage-Interaction) (also mirrored at `Shanghai_AI_Laboratory/Sage-Interaction` on ModelScope)
- Task: Full-duplex streaming audio-video dialogue with 24 kHz mono speech output and native asynchronous task delegation (`<|delegate|>`)
- Mode: Online full-duplex serving over `/v1/realtime?duplex=1` (alias `/v1/duplex`) on the three-stage MRv2 pipeline (Thinker -> Talker -> Code2Wav)
- Maintainer: Community

## When to use this recipe

Use this recipe to serve `internlm/Sage-Interaction` on vLLM-Omni. Sage-Interaction builds on the MiniCPM-o 4.5 three-stage architecture (Thinker + MiniCPMTTS Talker + Token2Wav Code2Wav) and adds:

1. Audio-unit boundary markers (`<|audio_start|>` / `<|audio_end|>`) inside every `<unit>` chunk;
2. A 3-way per-unit control vocabulary (`<|listen|>`, `<|speak|>`, and `<|delegate|>` at token ID `151748`);
3. Asynchronous tool delegation (`tool_mode: "delegate_only"`): `<|delegate|>` units short-circuit Stage 1 Talker and Stage 2 Code2Wav and publish standard OpenAI Realtime function-call events (`response.function_call_arguments.done`);
4. Structured `<|im_start|>tool\n{"type":"delegate_result",...}<|im_end|>` injection before the next `<unit>` when the client submits `conversation.item.create` (`function_call_output`).

## Hardware & Environment

- Validated GPU: 1x NVIDIA RTX 6000D (85 GiB VRAM), CUDA 13.0, Model Runner V2 across all 3 stages
- Disk: ~20 GB for the BF16 checkpoint (`internlm/Sage-Interaction`)
- Python packages: `vllm==0.34.0`, `vllm-omni`, `step-audio2==1.0.0`, `onnxruntime`, `hyperpyyaml`, `librosa`

> [!NOTE]
> The Hugging Face repository `internlm/Sage-Interaction` bundles `assets/token2wav/*` but does **not** ship a default speaker prompt WAV (`assets/HT_ref_audio.wav` or `system_ref_audio.wav`). Pass a per-session `ref_audio` URI (such as `data:audio/wav;base64,...`) in `session.update`, or place a `system_ref_audio.wav` file in the checkpoint root directory.

## Launch

Auto-detection routes `internlm/Sage-Interaction` (`model_type="minicpmo"`, `version="4.5"`, `vocab_size=151749`) directly to the `sage_interaction` pipeline, or pass `--deploy-config` explicitly:

```bash
vllm serve internlm/Sage-Interaction --omni \
  --deploy-config vllm_omni/deploy/sage_interaction.yaml \
  --port 8091
```

For ordinary non-duplex turn serving:

```bash
vllm serve internlm/Sage-Interaction --omni \
  --deploy-config vllm_omni/deploy/sage_interaction_turn.yaml \
  --port 8091
```

## Realtime Duplex & Delegation Protocol

Connect to `ws://127.0.0.1:8091/v1/realtime?duplex=1&model=internlm/Sage-Interaction&autostart=0` and open the session with `session.update`:

```json
{
  "type": "session.update",
  "session": {
    "model": "internlm/Sage-Interaction",
    "modalities": ["audio", "text"],
    "input_audio_format": "pcm_f32le",
    "output_audio_format": "pcm16",
    "sample_rate_hz": 16000,
    "ref_audio": "data:audio/wav;base64,<WAV_BASE64>",
    "extra_body": {
      "auto_response": true,
      "realtime_tools": [{"type": "function", "name": "delegate"}]
    }
  }
}
```

Stream 1-second 16 kHz `pcm_f32le` audio chunks via `input_audio_buffer.append`.

- **Direct spoken reply (`<|speak|>`)**: Streams `response.output_audio.delta` (24 kHz PCM16) and `response.output_audio_transcript.delta`.
- **Delegation (`<|delegate|>`)**: Emits `response.output_item.added` (`item.name == "delegate"`) and `response.function_call_arguments.done` (`call_id`, `arguments='{"query": "..."}'`).
- **Returning a tool result**: Send `conversation.item.create` with `type: "function_call_output"`, matching `call_id`, and `output` (plain text or JSON `delegate_result` envelope). The plugin formats `<|im_start|>tool\n{"type":"delegate_result",...}<|im_end|>` and prepends it ahead of the next audio `<unit>` once the foreground turn is idle; the model then speaks the result automatically.

## Validation Summary

Verified on 1x NVIDIA RTX 6000D (CUDA, MRv2 3-stage):

- `/health` HTTP 200 and `/v1/realtime?duplex=1` handshake reporting `tool_mode="delegate_only"`, `max_tools=1`, `function_call_output_continuation="automatic"`, `supports_core_resumable_request=true`, `supports_multi_session=true`.
- Voiced speech turn (`hello_16k.wav`): 3.88 s voiced 24 kHz PCM (`abs(pcm).max() = 0.7164`) with coherent transcript.
- Live delegation round-trip (`delegate_weather_16k.wav` -> `response.function_call_arguments.done` -> `function_call_output` -> spoken Tokyo weather reply): 6.04 s voiced 24 kHz PCM (`abs(pcm).max() = 0.7049`) announcing the injected result.
- Malformed input resilience (`unsupported_tools` rejected cleanly; malformed JSON / invalid base64 do not poison active sessions) and concurrent voiced session isolation.

## References

- Deploy configs: [`vllm_omni/deploy/sage_interaction.yaml`](../../vllm_omni/deploy/sage_interaction.yaml), [`vllm_omni/deploy/sage_interaction_turn.yaml`](../../vllm_omni/deploy/sage_interaction_turn.yaml)
- Pipeline & duplex plugin: [`vllm_omni/model_executor/models/sage_interaction/`](../../vllm_omni/model_executor/models/sage_interaction/)
- Upstream repository: [`InternLM/Sage-Interaction`](https://github.com/InternLM/Sage-Interaction)
- Hugging Face checkpoint: [`internlm/Sage-Interaction`](https://huggingface.co/internlm/Sage-Interaction)
