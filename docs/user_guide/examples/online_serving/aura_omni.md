# AURA Omni: Online serving

`aura_omni` wires ASR, AURA, and Qwen3-TTS into one vLLM-Omni pipeline:

```text
ASR -> AURA -> Qwen3-TTS Talker -> Code2Wav
```

**Primary online path:** Realtime duplex at `/v1/realtime?duplex=1`.

The default deploy profile (`vllm_omni/deploy/aura_omni.yaml`) sets
`session_mode: duplex`. The `aura_omni` pipeline declares `duplex_plugin`, and
`DuplexOmniEngine` requires that mode (same pattern as MiniCPM-o duplex
deploys). Do not treat turn-based `chat/completions` as the primary AURA
online serve path with this profile.

```bash
vllm serve aurateam/AURA \
  --omni \
  --deploy-config vllm_omni/deploy/aura_omni.yaml \
  --served-model-name aurateam/AURA \
  --trust-remote-code
```

Configure local checkpoints by editing per-stage `model` values in the deploy
YAML. The file sets `pipeline: aura_omni`, so the four-stage topology is used
even if the command-line `--model` points at one component checkpoint.

For a local-weight smoke serve + WS client:

```bash
bash examples/online_serving/aura_omni/run_duplex_smoke_serve.sh
python examples/online_serving/aura_omni/smoke_duplex_realtime_client.py
```

Silent Stage1 outputs (`<|silent|>` / id `151669`) skip TTS for that turn.

## GPU Utilization Recommendation

`gpu_memory_utilization` in `vllm_omni/deploy/aura_omni.yaml` controls how much
VRAM each stage can reserve. Start with this split for a single GPU:

- Stage 0 (ASR): `0.10`
- Stage 1 (AURA): `0.40`
- Stage 2 (Qwen3-TTS Talker): `0.20`
- Stage 3 (Qwen3-TTS Code2Wav): `0.20`

## TTS modes (stage extras)

When the duplex session supplies TTS extras, AURA text can feed Qwen3-TTS as:

- `Base`: voice clone from `tts_ref_audio` (optional x-vector-only mode)
- `CustomVoice`: predefined speaker (`tts_speaker`) with a CustomVoice checkpoint
  on stages 2 and 3

Optional `tts_pass_token_ids` passes AURA assistant token ids into Talker
instead of detokenized text.

## Turn-based examples (not primary)

`examples/online_serving/aura_omni/` still contains chat-completions, curl, and
Gradio helpers from the older turn-based online path. They are **not** the
supported primary path for the duplex deploy profile; use the Realtime duplex
smoke client for online verification.
