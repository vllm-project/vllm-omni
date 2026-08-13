# MiniMax-H3 on a single Ascend 950PR

This recipe runs MiniMax-H3 with online INT8 quantization on one Ascend 950PR
NPU (128 GB HiBL 1.0 HBM). It covers the single-card T2VA configuration at
1024x576. For the eight-card Atlas 800I A3 BF16 route at 768P, see
[MiniMax-H3-NPU.md](MiniMax-H3-NPU.md).

## Capacity requirements

| Resource | Requirement |
| --- | ---: |
| NPU | 1x Ascend 950PR |
| NPU HBM | 128 GiB (131,072 MiB reported by `npu-smi`) |
| Checkpoint storage | 135 GiB per partition |
| Container shared memory | 8 GiB minimum |

The container's `/dev/shm` is load-bearing: the decoded video is handed back to
the API server through POSIX shared memory, and a 5 s clip at this shape
produces an 837 MiB payload. Docker's 64 MB default cannot back it, and the
failure is expensive — every denoise step completes first, then the worker dies
with `Bus error (core dumped)` during the handoff, with no Python traceback.
Start the container with `--shm-size=8g` or `--ipc=host`.

## Environment

- Host architecture: x86_64
- Ascend driver: 25.7.rc1.6 (ascendhal 7.35.23)
- Ascend firmware: 9.0.0.105.229
- CANN toolkit: 9.1.0 (`/usr/local/Ascend/cann-9.1.0`)
- npu-smi: 25.7.rc1.6
- Python: 3.12.13
- PyTorch: 2.10.0+cpu
- torch_npu: 2.10.0.post2
- vLLM: 0.26.0
- vLLM-Omni: 0.26.1.dev103+g584d78c67.npu (commit `584d78c6`)

Install vLLM-Omni from a checkout with MiniMax-H3 support:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Start a server

```bash
export MODEL=/path/to/MiniMax-H3/FL2VA
export PORT=8000
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 1 \
  --tensor-parallel-size 1 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 1 \
  --vae-patch-parallel-size 1 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --init-timeout 14400 \
  --stage-init-timeout 14400 \
  --quantization int8
```

Do not pass `--diffusion-attention-backend CUDNN_ATTN`. cuDNN attention has no
NPU implementation and fails at the first denoise step. Leave the backend
unset: it resolves to `TORCH_SDPA`, or to `FLASH_ATTN` when MindIE-SD is
installed (see [MiniMax-H3-NPU.md § Environment](MiniMax-H3-NPU.md#environment)).

H3 is CFG-distilled, so `--cfg-parallel-size` must remain 1.

## Validated evidence

Measured on one Ascend 950PR with the configuration above, generating a 5 s
1024x576 T2VA clip at 60 requested steps (59 denoise updates). The request
returned `200 OK` with a playable MP4.

| Measurement | Result |
| --- | ---: |
| End-to-end request | 472.47 s |
| Denoise (59 updates) | 435 s at 7.39 s/update |
| Server-reported `denoise_step_latency_ms` | 7,874.4 ms |
| MP4 muxing | 0.59 s |
| Generated audio | 5.175 s at 32 kHz (165,600 samples) |
| Peak HBM | 81,059 MiB of 131,072 MiB |

The server-reported per-step latency divides the whole stage time by the 60
requested steps, so it reads higher than the 7.39 s measured per actual denoise
update. The roughly 37 s outside the denoise loop covers text encoding, VAE
decode, the 837 MiB worker-to-scheduler transfer, and MP4 muxing.

This run was taken in a container whose `/dev/shm` was the 64 MB default and
could not be remounted, so the oversized transfer used a file-backed fallback
rather than shared memory; about 28 s of the end-to-end time went to writing
and reading back that payload. A container started with `--shm-size=8g` keeps
the transfer in shared memory and avoids that cost.

Peak HBM is sampled externally with `npu-smi`. Re-measure for longer outputs,
a different output shape, or concurrency greater than one.

## T2VA request example

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"

curl -sS --max-time 1800 -X POST "${API_URL}" \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F 'width=1024' \
  -F 'height=576' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=60' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

## Known limitations

- INT8 online quantization is validated for T2VA on this card. Use BF16 for
  FL2VA and Ref2VA, or re-measure before relying on it.
- Single-card serving loads one task partition at a time. For Ref2VA, stop the
  server and restart it against the `Ref2VA` directory.
- Video outputs at this shape exceed the default container shared-memory mount.
  See [§ Capacity requirements](#capacity-requirements).
- The image ships a `triton` package whose Ascend backend is not built
  (`No module named 'triton._C.libtriton.ascend'`). vLLM logs this as an error
  at startup and disables Triton; the diffusion path does not need it and the
  run completes normally.

## Additional resources

- [MiniMax-H3.md](MiniMax-H3.md) — full GPU guide
- [MiniMax-H3-NPU.md](MiniMax-H3-NPU.md) — eight-card Atlas 800I A3 BF16 guide
- [Int8 quantization](../../docs/user_guide/quantization/int8.md)
- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
