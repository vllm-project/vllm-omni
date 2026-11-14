# PersonaPlex for live full-duplex speech-to-speech — RTX 5080 Laptop 16 GB

## Summary

- Vendor: NVIDIA
- Model: [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)
  (gated; a Moshi finetune)
- Task: Full-duplex speech-to-speech. 24 kHz mic audio in, agent speech plus
  inner-monologue text out, in 80 ms lockstep, with per-session voice and
  persona control.
- Mode: Live duplex WebSocket serving through the unified duplex stack
  (`/v1/realtime?duplex=1`, alias `/v1/duplex`)
- Hardware: 1x NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB (16303 MiB visible),
  Ubuntu 24.04 on WSL2
- Maintainer: [`@ziyuanLi-alex`](https://github.com/ziyuanLi-alex)

## When to use this recipe

In BF16, startup fails because the weights and KV cache couldn't fit in a single 16 GB GPU.
This profile loads eligible Stage 0 layers in online FP8 (cutting Stage 0 load
to 8.74 GiB) and sets the two stage budgets to 0.80 and 0.15, leaving enough
Stage 0 KV cache while keeping Stage 1 resident on the same card for up to
two concurrent sessions.

## Supported model contract

| Task | Input | Output | Profile |
| --- | --- | --- | --- |
| Live full-duplex speech-to-speech | 24 kHz mono PCM stream (80 ms frames) + per-session `voice` and `instructions` | 24 kHz mono PCM stream + inner-monologue text delta, 80 ms lockstep | `personaplex_rtx5080_16gb.yaml` |

Up to two sessions run concurrently; a third is rejected with
`resource_exhausted` until a slot frees. With the inherited
`max_model_len: 3000` and 3000-frame attention window, each session can run for
at most 3000 frames (~240 s) before ending with `context_length_exceeded` (see
[Notes](#notes)). Decoding is greedy only (`temperature` 0, `top_k` 1); Stage 0
loads eligible layers in FP8 while Stage 1 (Mimi codec) stays unquantized. See
[`PersonaPlex.md`](./PersonaPlex.md) and
[`docs/serving/realtime_duplex_api.md`](../../docs/serving/realtime_duplex_api.md)
for the wire protocol.

## References

- Model recipe and architecture notes: [`PersonaPlex.md`](./PersonaPlex.md)
- Original deployment:
  [`vllm_omni/deploy/personaplex.yaml`](../../vllm_omni/deploy/personaplex.yaml)
- This profile:
  [`vllm_omni/deploy/personaplex_rtx5080_16gb.yaml`](../../vllm_omni/deploy/personaplex_rtx5080_16gb.yaml)
- Realtime driver used for verification:
  [`tests/e2e/online_serving/personaplex_realtime_duplex.py`](../../tests/e2e/online_serving/personaplex_realtime_duplex.py)
- Design notes:
  [`docs/design/fullduplex-personaplex.md`](../../docs/design/fullduplex-personaplex.md)
- Upstream model card:
  [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1)

## Checkpoint and setup

`nvidia/personaplex-7b-v1` is gated: accept the license on the model page and
export `HF_TOKEN` with access to the repo. The measurements below were taken
on snapshot revision `fdaf4090a61cb315c138a1faee287ffd6c716309`:

```bash
export HF_TOKEN=...
hf download nvidia/personaplex-7b-v1 \
  --revision fdaf4090a61cb315c138a1faee287ffd6c716309
```

The first run also fetches `tokenizer_spm_32k_3.model`, `voices.tgz` and the
Mimi reference checkpoint.

## Hardware

- Accelerator model and per-device memory: NVIDIA GeForce RTX 5080 Laptop GPU,
  16303 MiB visible to the process
- Number of devices: 1
- Device interconnect: none (single device)

## Software environment

- OS: Ubuntu 24.04.5 LTS on WSL2
- Python: 3.12.3
- Driver / runtime: NVIDIA 610.53, CUDA UMD 13.3
- vLLM: 0.33.0
- vLLM-Omni: `0.33.0rc2.dev97+gf6a7f54a5` (commit
  `f6a7f54a545839b860c09e20eb09482d83672aab`)
- Model revision: `fdaf4090a61cb315c138a1faee287ffd6c716309`
- Support packages: torch 2.14.0+cu130, transformers 5.19.0,
  flashinfer-python 0.7.2.post1

## Command

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/personaplex-7b-v1 \
  --omni \
  --deploy-config vllm_omni/deploy/personaplex_rtx5080_16gb.yaml \
  --port 8000
```

The server exposes `WS /v1/realtime?duplex=1`. Voice and persona are selected
per session in `session.update`; audio is paced at 80 ms per input frame by
the client.

## Verification

```bash
# Ready
curl -fsS http://127.0.0.1:8000/health

# GPU-free contract tests for the duplex plugin and Stage 0 runtime
pytest tests/model_executor/models/personaplex/duplex/ \
  tests/engine/duplex/test_session_runner_personaplex.py -q

# GPU end to end: paced 24 kHz PCM, two concurrent sessions, overflow
# admission, non-silent output
python tests/e2e/online_serving/personaplex_realtime_duplex.py \
  --model /path/to/personaplex-7b-v1 \
  --input-wav /path/to/24khz-monophonic-speech.wav \
  --output-dir /tmp/personaplex-rtx5080
```

Expected on this profile: `/health` returns 200 after both stages initialize,
the lifecycle driver reports `ok=true` with two accepted sessions and one
rejected third session, and output WAVs contain non-silent agent audio.

## Notes

- `engine_args.quantization: fp8` applies only to Stage 0 and only to layers
  the upstream vLLM FP8 path supports; the custom depformer and Mimi modules
  are not quantized by it.
- Session length: Stage 0 runs on vLLM paged attention
  instead of the wrapping ring KV cache from NVIDIA's original implementation.
  To run longer sessions, raise Stage 0 `max_model_len` and
  `hf_overrides.temporal_config.max_position_embeddings`. On this profile the
  Stage 0 KV pool holds ~7666 tokens, so a longer limit needs fewer sessions
  or more memory.
