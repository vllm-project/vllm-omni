# AURA Omni: Online serving

`aura_omni` wires ASR, AURA, and Qwen3-TTS into one vLLM-Omni pipeline:

```text
ASR -> AURA -> Qwen3-TTS Talker -> Code2Wav
```

**Supported online path:** Realtime duplex at `/v1/realtime?duplex=1`.

The default deploy profile (`vllm_omni/deploy/aura_omni.yaml`) sets
`session_mode: duplex`. The `aura_omni` pipeline declares `duplex_plugin`, and
`DuplexOmniEngine` requires that mode (same pattern as MiniCPM-o duplex
deploys). Turn-based `chat/completions` is not a supported AURA online mode
against this profile.

`async_chunk` defaults to `false` in that file (safer if a turn-based
`OmniOrchestrator` ever loads it). Duplex Realtime Stage2→3 still needs
async chunks — use the smoke deploy below, or set `async_chunk: true`
yourself.

```bash
bash examples/online_serving/aura_omni/run_duplex_smoke_serve.sh
python examples/online_serving/aura_omni/smoke_duplex_realtime_client.py
```

Configure local checkpoints by editing per-stage `model` values in the deploy
or smoke YAML. The file sets `pipeline: aura_omni`, so the four-stage topology
is used even if the command-line `--model` points at one component checkpoint.

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

## Old chat / Gradio / curl scripts (unsupported)

`examples/online_serving/aura_omni/` still contains chat-completions, curl, and
Gradio helpers from an older turn-based path. They do **not** work against the
shipped duplex profile. That path was an incomplete stand-in before streaming
I/O and session history were available; it is not a supported AURA mode. Use
the Realtime duplex smoke client for online verification.
