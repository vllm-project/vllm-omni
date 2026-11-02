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
- vLLM-Omni version or commit: `dc68e90b9` + the Moshi support PR (tested at `8bf25e889`)

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

- Memory usage: with the overlay above, `nvidia-smi` shows at most 35,725 MiB
  from 1 to 16 sessions. Stage 0 logs `GPU KV cache size: 33,238 tokens` (one token per
  frame), about 11 sessions at the full 3000-frame window; raise its
  `gpu_memory_utilization` for more long sessions.
- Load: the driver's `--sessions 1`, `4` and `8` pass all its checks,
  including audio continuity. With 16 sessions in one driver process, the
  driver reports `client_pacing_warning` (its own sends run up to about 350 ms
  late), which by itself fails the continuity check; that run does not tell
  whether the server would keep up with a client that paces cleanly.
- Key flags: `max_num_seqs` of Stage 0 should be a CUDA graph capture size
  (1, 2, 4, then multiples of 8).
- Known limitations: greedy decoding only (kyutai's reference samples by
  default); the first 80 ms of each session is not read, as in `moshi.server`.
- Repo id with `--download-dir`: the duplex plugin reads the checkpoint from
  the default Hub cache, else from the Hub, and remembers it once read. Until
  then, a session that opens while the Hub is unreachable is refused
  (`prefill_unavailable`), and the next open checks again. To avoid that, set
  `HF_HUB_CACHE` to that directory in place of `--download-dir`, or serve the
  local snapshot path.
- Not bit-matched to `moshi.server`: greedy decoding follows small numeric
  differences here. On the tested input the first reply lands on the same
  frames as kyutai's bf16 reference, which starts its second reply at step 134;
  this deploy starts it at step 192 and says less. kyutai's reference run in
  fp32 does the same as this deploy (step 192, 99.7% text-token agreement).

## Supported features

| Feature | Status |
| --- | --- |
| Multi-session batching | Yes (see Notes) |
| Depformer / Mimi CUDA graphs | Yes (deploy default) |
| Model Runner V2 (Stage 1) | Yes, `personaplex_duplex_mrv2.yaml`: same Stage 0 tokens and output audio as the default deploy (one session tested) |
| `kyutai/moshika-pytorch-bf16` | Runs with the same commands (one session tested) |
| Hub repo id or local path | Both (tested: a local snapshot path, and the repo id with the Hub online, with and without `--download-dir`) |
| Voice prompt, persona | No (Moshi has none) |
| Sampling | No |
| TP / quantization | No |
