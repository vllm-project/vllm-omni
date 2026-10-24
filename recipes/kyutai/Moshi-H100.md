# Moshi — H100

> kyutai Moshi full-duplex speech dialogue on one H100, through the unified
> duplex stack and the PersonaPlex pipeline.

## Summary

- Vendor: kyutai
- Model: [`kyutai/moshiko-pytorch-bf16`](https://huggingface.co/kyutai/moshiko-pytorch-bf16)
  (ungated, CC-BY 4.0)
- Task: Full-duplex speech-to-speech. 24 kHz audio in, Moshi's speech out, one
  80 ms frame per step.
- Mode: Live duplex WebSocket serving (`/v1/realtime?duplex=1`, alias `/v1/duplex`)
- Hardware: 1x NVIDIA H100 80GB
- Maintainer: [`@ischencheng`](https://github.com/ischencheng)

## When to use this recipe

Use it to run kyutai's Moshi as a live voice agent that serves several sessions
on one GPU. Moshi shares PersonaPlex's architecture (Helium temporal
transformer, 8-step depformer, Mimi codec), so it runs on the PersonaPlex
pipeline and deploy. A checkpoint with an 8-step depformer is served the way
kyutai's `moshi.server` runs it: a session starts from Moshi's initial-token
frame with no voice prompt or persona.

## Supported model contract

| Item | Value |
| --- | --- |
| Entrypoint | `WS /v1/realtime?duplex=1` (`/v1/duplex`) |
| Input | `pcm_f32le`, 24 kHz mono, framed into 1920-sample (80 ms) frames |
| Output | `pcm16`, 24 kHz mono |
| Session fields | Leave `voice` and `instructions` unset; a session that sets them is refused (`voice_unsupported`, `instructions_unsupported`) |
| Decoding | Greedy (text and depformer) |
| Context | 3000 frames (4 min) of KV per session |
| Deploy profile | `vllm_omni/deploy/personaplex.yaml` (2 sessions); raise `duplex_session.max_sessions` and both stages' `max_num_seqs` for more |

## References

- Moshi paragraph in the Stage 0 runtime section of
  [`docs/design/fullduplex-personaplex.md`](../../docs/design/fullduplex-personaplex.md)
- Online serving example:
  [`examples/online_serving/personaplex/`](../../examples/online_serving/personaplex/)
- Upstream model and reference server: [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)
- Related recipe: [`NVIDIA/PersonaPlex.md`](../NVIDIA/PersonaPlex.md)

## Hardware

- Accelerator model and per-device memory: NVIDIA H100 80GB HBM3
- Number of devices: 1 (both stages on GPU 0)
- Device interconnect: not used
- Qualification scope: the commands below, in a container

## Software environment

- Python: 3.12
- Driver / runtime: NVIDIA 610.57.04, CUDA 13.0
- vLLM version: 0.32.0 (`vllm/vllm-openai:v0.32.0`)
- vLLM-Omni version or commit: `df1c1b059` + the Moshi support PR

## Command

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve kyutai/moshiko-pytorch-bf16 --omni \
  --deploy-config vllm_omni/deploy/personaplex.yaml
```

More sessions: an overlay such as

```yaml
base_config: personaplex.yaml
duplex_session:
  max_sessions: 16
stages:
  - stage_id: 0
    max_num_seqs: 16
  - stage_id: 1
    max_num_seqs: 18
```

## Verification

```bash
python tests/e2e/online_serving/personaplex_realtime_duplex.py \
  --model kyutai/moshiko-pytorch-bf16 \
  --input-wav tests/assets/minicpmo_4_5/response_required_16k.wav \
  --voice "" --persona "" --secondary-persona "" --replacement-persona "" \
  --url "ws://127.0.0.1:8000/v1/realtime?duplex=1"
```

It opens two paced sessions, checks overflow admission and slot reuse, and
requires audible whole-frame output. `--sessions N` switches it to the paced
multi-session load mode.

## Notes

- Memory usage: with the overlay above, `nvidia-smi` shows 35,461 MiB from 1 to
  16 sessions. Stage 0 logs `GPU KV cache size: 33,238 tokens` (one token per
  frame), about 11 sessions at the full 3000-frame window; raise its
  `gpu_memory_utilization` for more long sessions.
- Load: the driver's `--sessions 1` and `--sessions 4` pass all its checks,
  including audio continuity. With 8 or 16 sessions in one driver process, the
  driver reports `client_pacing_warning` (its own sends run 330-430 ms late),
  which by itself fails the continuity check; that run does not tell whether
  the server would keep up with a client that paces cleanly.
- Key flags: `max_num_seqs` of Stage 0 should be a CUDA graph capture size
  (1, 2, 4, then multiples of 8).
- Known limitations: greedy decoding only (kyutai's reference samples by
  default); the first 80 ms of each session is not read, as in `moshi.server`.

## Supported features

| Feature | Status |
| --- | --- |
| Multi-session batching | Yes (see Notes) |
| Depformer / Mimi CUDA graphs | Yes (deploy default) |
| Model Runner V2 (Stage 1) | Runs, `personaplex_duplex_mrv2.yaml` (one session tested) |
| `kyutai/moshika-pytorch-bf16` | Runs with the same commands (one session tested) |
| Voice prompt, persona | No (Moshi has none) |
| Sampling | No |
| TP / quantization | No |
