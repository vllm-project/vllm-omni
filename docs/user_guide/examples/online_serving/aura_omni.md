# AURA Omni Native Pipeline

`aura_omni` wires ASR, AURA, and Qwen3-TTS into one vLLM-Omni pipeline:

```text
ASR -> AURA -> Qwen3-TTS Talker -> Code2Wav
```

Qwen3-TTS remains two engine stages so the pipeline reuses the existing native
Talker and Code2Wav implementation.

```bash
vllm serve aurateam/AURA \
  --omni \
  --deploy-config vllm_omni/deploy/aura_omni.yaml \
  --served-model-name aura_omni \
  --trust-remote-code
```

Configure local checkpoints by editing per-stage `model` values in
`vllm_omni/deploy/aura_omni.yaml`. The deploy file sets
`pipeline: aura_omni`, so the four-stage topology is used even if the
command-line `--model` points at one of the component checkpoints.

Send requests with `"model": "aura_omni"`. The ASR, AURA, and Qwen3-TTS
checkpoint paths are internal stage models from the deploy YAML, not the
OpenAI-facing served model name.

The AURA stage can emit `<|silent|>`. Silent outputs are treated as a gate:
they produce no Qwen3-TTS Talker input, so no audio is synthesized for that
turn.

## TTS Modes

`aura_omni` can pass AURA text to Qwen3-TTS in two task modes:

- `Base`: voice clone from `tts_ref_audio` with ICL enabled in the AURA
  pipeline. Provide both `tts_ref_audio` and `tts_ref_text`. Set
  `tts_x_vector_only_mode=true` to disable ICL and use speaker embedding only.
- `CustomVoice`: predefined speaker mode. Use a Qwen3-TTS CustomVoice
  checkpoint for stages 2 and 3 in `aura_omni.yaml`, then pass
  `tts_task_type=CustomVoice` and `tts_speaker`.

Experimental token passthrough is available with `tts_use_aura_token_ids=true`.
It trims AURA response boundary tokens and passes the generated token ids into
Qwen3-TTS directly instead of re-tokenizing the response text.
