# MiniMax-H3 on RTX 4090

This recipe uses BF16 weights, tiled VAE decode, tensor parallelism across two
GPUs, and distributed layerwise offload (DLO). It is the 24 GiB sibling of
[MiniMax-H3-5090.md](MiniMax-H3-5090.md): the topology is identical, but the
resident DiT-block count is lower so the per-rank peak fits in 24 GiB. Lower
resident counts reduce HBM use and increase CPU-to-GPU transfer time.

## Capacity requirements

| Resource | Two RTX 4090s |
| --- | ---: |
| GPU HBM | 24 GiB per GPU |
| Checkpoint storage | 135 GiB per partition |
| Available system RAM | 200 GiB minimum |
| Recommended system RAM | 384 GiB |

`FL2VA` and `Ref2VA` are separate 135 GiB checkpoint partitions. Start one
server at a time. DLO keeps rank-local weights in pinned host memory; increasing
`--dlo-resident-layers` improves latency but does **not** reduce host RAM in the
current implementation because resident layers retain pinned CPU master copies.

A single RTX 4090 is not covered here. The one-GPU DLO profile in the RTX 5090
recipe peaked at 26.50 GiB with 12 resident layers, which exceeds 24 GiB. On one
4090, use the model-level CPU offload command in
[MiniMax-H3.md](MiniMax-H3.md#single-gpu-accuracy-and-memory-first) instead, or
lower the resident count and re-measure before trusting it.

> **Modular H3:** after #5720 lands, preserve this recipe's one-partition
> behavior with `--task-type fl2va` or `--task-type ref2va`.

## Two RTX 4090s: 1024x576, 5 seconds

Use TP2 and 12 resident DiT layers. This is the `rtx4090` profile from the main
recipe. On two RTX 4090s it peaked at 15,620 MiB per rank for T2VA and
14,918 MiB per rank for Ref2VA, leaving roughly 9 GiB of headroom per card. See
[Target-hardware validation](#target-hardware-validation) for the full numbers.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400

CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
  --usp 1 --ring 1 --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 12 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

For Ref2VA, stop the FL2VA server and restart the same command with
`/path/to/MiniMax-H3/Ref2VA`. Ref2VA reference image count and prompt length can
increase activation memory; begin with one request at a time.

`VLLM_OMNI_VIDEO_SYNC_TIMEOUT` matters more here than on larger cards. DLO
streams non-resident DiT blocks over PCIe on every denoising step, so a 60-step
request takes several minutes and would otherwise hit the 600-second default.

## Target-hardware validation

Measured on 2 x RTX 4090 (24,564 MiB each, driver 580.126.09) with the serve
command above, at vLLM-Omni `0.26.1.dev55+g81b48e83e`, vLLM `0.26.0`, and
PyTorch `2.11.0+cu130`. Both GPUs were dedicated to this server. `Client E2E`
and `Peak per GPU` come from the `/v1/videos/sync` response headers
`x-inference-time-s` and `x-peak-memory-mb`; the latter is the rank-0 CUDA
reserved high-water mark, so it is a per-GPU figure.

| Task | Shape | Frames | Steps | Client E2E | Peak per GPU | Output validation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| T2VA (`FL2VA`) | 1024x576 | 124 at 24 FPS | 60 | 7 min 9 s | 15.3 GiB | H.264 video + 32 kHz stereo AAC; full `ffmpeg` decode passed |
| Ref2VA | 1024x576 | 124 at 24 FPS | 60 | 14 min 52 s | 14.6 GiB | H.264 video + 32 kHz stereo AAC; full `ffmpeg` decode passed |

Both rows use `seed=1101`, `flow_shift=12`, and `audio_flow_shift=3.0`, and each
was run twice. T2VA measured 434.9 s and 429.3 s; Ref2VA measured 891.9 s and
892.1 s. Peak memory was identical across repeats for both tasks, and the two
outputs of each pair agreed to within 115 bytes. These are single-request
validation runs, not concurrent throughput benchmarks.

## Request examples

`t2va` requires an explicit `aspect_ratio`. Without it the request fails with
`t2va requires an explicit aspect_ratio`, even when `width` and `height` are set.

Text to video and audio, against the `FL2VA` server:

```bash
curl -sS -D headers_t2va.txt -o out_t2va.mp4 --max-time 14400 \
  -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=傍晚的小厨房，夕阳余晖从窗边洒进来，旧木桌上放着洗到一半的马克杯和起雾的玻璃瓶，悬挂的抹布轻轻晃动。画面带有手持拍摄的轻微晃动和逆光曝光波动，生活感十足。环境音是安静的厨房底噪。' \
  -F 'aspect_ratio=16:9' \
  -F 'width=1024' -F 'height=576' -F 'fps=24' \
  -F 'num_inference_steps=60' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5,"audio_flow_shift":3.0}'
```

Reference to video and audio, against the `Ref2VA` server:

```bash
curl -sS -D headers_ref2va.txt -o out_ref2va.mp4 --max-time 14400 \
  -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=傍晚的小厨房，夕阳余晖从窗边洒进来，旧木桌上放着洗到一半的马克杯和起雾的玻璃瓶，悬挂的抹布轻轻晃动。画面带有手持拍摄的轻微晃动和逆光曝光波动，生活感十足。环境音是安静的厨房底噪。' \
  -F 'input_reference=@/path/to/reference.jpg;type=image/jpeg' \
  -F 'audio_reference=</path/to/audio_reference.json' \
  -F 'width=1024' -F 'height=576' -F 'fps=24' \
  -F 'num_inference_steps=60' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"ref2va","duration":5,"audio_flow_shift":3.0}'
```

`audio_reference` is a JSON object, which is why the example loads it from a file
with curl's `<` syntax. Its `audio_url` must be an `http(s)` URL or a data URL;
a bare filesystem path is rejected with `Invalid audio_reference.audio_url`.

```json
{"audio_url": "https://example.com/reference.wav"}
```

Reference images are validated before inference: the short edge must be at least
256 pixels, the long edge at most 5760, and the aspect ratio must fall between
0.4 and 2.5.

The response headers carry the measurements used in the table above:

```bash
cat headers_t2va.txt
ffmpeg -v error -i out_t2va.mp4 -f null - && echo DECODE_OK
```
