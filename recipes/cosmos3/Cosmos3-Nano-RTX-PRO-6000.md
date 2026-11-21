# Cosmos3-Nano text-to-video: RTX PRO 6000 Blackwell Server Edition

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Cosmos3-Nano` (BF16 `main` branch and the ModelOpt FP8 `fp8` branch)
- Task: Text-to-video (T2V) at 1280x720, 189 frames
- Mode: Online serving (`POST /v1/videos/sync`)
- Hardware: 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), PCIe, no NVLink
- Recommended deployment: one server per GPU for node throughput; one server on all eight GPUs for per-clip latency
- Maintainer: Community

## When to use this recipe

Use this recipe to serve Cosmos3-Nano video generation on an 8-GPU RTX PRO 6000 Blackwell Server Edition node and choose between node throughput and per-clip latency. The node has no NVLink, so the throughput profile runs one independent server per GPU and keeps GPU-to-GPU traffic off PCIe. All other Cosmos3-Nano modes (T2I, I2V, V2V, transfer, sound, action) are documented in [`Cosmos3-Nano.md`](./Cosmos3-Nano.md) and were not measured on this hardware.

## Supported model contract

| Item | Value |
| --- | --- |
| Measured task | T2V, `POST /v1/videos/sync` |
| Measured output | 1280x720, 189 frames at 24 fps, 35 steps, guidance 6.0, `flow_shift=10.0`, `max_sequence_length` 4096 (the Cosmos3 default) |
| Measured checkpoints | `main` at `e59a53c2` (BF16); `fp8` at `1af44289` (ModelOpt FP8) |
| Measured prompt | One prompt: the model repository's `assets/example_t2v_prompt.json` with `assets/negative_prompt.json`, sent as compact JSON strings |
| Other sizes and modes | Per [`Cosmos3-Nano.md`](./Cosmos3-Nano.md); not measured here |
| Guardrails | Off in every measurement below (`--no-guardrails`); see the licensing note in [`Cosmos3-Nano.md`](./Cosmos3-Nano.md) |

Deployment profiles on this node, each from one run of 24 timed clips with one request in flight per server (seeds 17, 23, 41):

| Profile | Servers x GPUs each | Median s/clip | Clips per node-hour | GPU memory in use, GiB/GPU | LPIPS vs BF16 reference (mean, range) | Distinct outputs (of 24 clips) |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Throughput, BF16 | 8 x 1 | 371.93 | 77.0 | 48.3 | 0.176 (0.112-0.242), BF16 servers that differ from it | 9 |
| Throughput, BF16 + SeaCache | 8 x 1 | 203.77 | 140.6 | 49.7 | 0.180 (0.140-0.215) | 3 |
| Throughput, FP8 checkpoint | 8 x 1 | 333.46 | 85.9 | 35.6 | 0.328 (0.236-0.413) | 9 |
| Balanced, CFG-parallel 2 | 4 x 2 | 197.77 | 72.6 | 48.6 | 0.165 (0.114-0.209) | 12 |
| Latency, CFG-2 x Ulysses-4 x HSDP-8 | 1 x 8 | 74.41 | 48.4 | 25.6 | 0.197 (0.144-0.228) | 3 |

- **Median s/clip** is client wall time per request.
- **Clips per node-hour** is the number of valid clips (ffprobe: 1280x720, 24 fps, 189 frames) divided by the timed window, from the first timed request's start to the last one's finish.
- **GPU memory in use** is the highest `nvidia-smi` `memory.used` sample on any GPU (sampled about every 10 s). It was the same at every sample from the start of the timed window, so it is the memory each server reserves, including CUDA context and allocator cache.
- **LPIPS** is AlexNet LPIPS per frame, averaged over the 189 frames, against the reference: the most common BF16 output at the same seed. The mean and range (minimum to maximum) cover every clip that is not byte-identical to the reference, pooled over the three seeds: 6 clips for BF16, 18 for CFG-parallel 2, all 24 for the other rows. LPIPS measures distance from the BF16 output, not quality.
- **Distinct outputs** counts different videos among the 24 clips; SeaCache and the 8-GPU layout produced one video per seed.

## References

- Full Cosmos3-Nano recipe: [`Cosmos3-Nano.md`](./Cosmos3-Nano.md)
- Model card: [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano)
- Cosmos3 SeaCache: [#6922](https://github.com/vllm-project/vllm-omni/pull/6922); FP8 mixed-precision schedule: [#6560](https://github.com/vllm-project/vllm-omni/pull/6560) and the [ModelOpt guide](../../docs/user_guide/quantization/modelopt.md#cosmos3-mixed-precision-schedule)
- Fix that lets `--revision fp8` load the FP8 transformer weights: [#9436](https://github.com/vllm-project/vllm-omni/pull/9436) (open)
- Earlier-build records cited in Notes (tensor parallel, Cache-DiT, guardrails, 480p): [nebius/nebius-physical-ai#969](https://github.com/nebius/nebius-physical-ai/pull/969)
- Community recipes tracker: [#2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware

- Accelerator: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, compute capability 12.0, 188 SMs
- Number of devices: 8
- Interconnect: PCIe, no NVLink. Two PCIe switches of four GPUs (0-3 and 4-7), one per CPU socket. NCCL all-reduce (BF16, 1 GiB messages, earlier image with the same NCCL 2.29.7) measured about 41.6 GB/s bus bandwidth inside a switch and 27.8 GB/s across all eight.
- Host: 2-socket Intel Xeon 6776P, 192 vCPU
- Qualification scope: the T2V shape above, guardrails off, one node, one run of 24 timed clips per profile; each command block below was also run as written through one Verification clip

## Software environment

- OS: Ubuntu 24.04 (container base image)
- Python: 3.12
- Driver / runtime: NVIDIA driver 580.173.02, CUDA 13.0, torch 2.13.0+cu130, cuDNN 9.20, NCCL 2.29.7
- vLLM version: 0.32.0
- vLLM-Omni version or commit: 0.32.0rc2.dev145+g637695435, container `vllm/vllm-omni:nightly-637695435c98bc11a76930cbe81b2671acd51bf2` (`sha256:c1fc4d40b7aada840e7602b4487f4e8c190ca18549c69aa422295222962bf220`), for every profile in the table

## Command

Guardrails are off in every command below. Cosmos3-Nano is released under the NVIDIA Open Model License, and with guardrails off you are responsible for compliance; see [`Cosmos3-Nano.md`](./Cosmos3-Nano.md).

Start the container with all GPUs and host networking, then run one profile's block inside it:

```bash
docker run --rm -it --gpus all --network host --ipc=host --shm-size 32g \
  --ulimit nofile=1048576:1048576 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-omni:nightly-637695435c98bc11a76930cbe81b2671acd51bf2 bash
```

Every block sets `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=5400` (default 600 s; the slowest request in the table took 375 s) and pins `--diffusion-attention-backend CUDNN_ATTN`, the platform default on this GPU. This image logs that `--diffusion-attention-backend` is deprecated in favor of `diffusion_attention_config`. Each multi-server block also gives every server its own `MASTER_PORT`: the measured runs did not, and one 8-server launch failed when two servers picked the same random rendezvous port (`EADDRINUSE`). Put a load balancer or a client-side round robin in front of ports 8100-8107.

Throughput, BF16: one server per GPU.

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$((29500 + 100 * i)) VLLM_OMNI_VIDEO_SYNC_TIMEOUT=5400 \
  vllm serve nvidia/Cosmos3-Nano --omni \
    --host 0.0.0.0 --port $((8100 + i)) --init-timeout 1800 \
    --deploy-config /app/vllm-omni/vllm_omni/deploy/cosmos3_omni.yaml --no-guardrails \
    --diffusion-attention-backend CUDNN_ATTN &
done
wait
```

Throughput, BF16 + SeaCache: the same loop with SeaCache. The `--cache-config` values are the SeaCache defaults at this commit, passed explicitly in the measurement.

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$((29500 + 100 * i)) VLLM_OMNI_VIDEO_SYNC_TIMEOUT=5400 \
  vllm serve nvidia/Cosmos3-Nano --omni \
    --host 0.0.0.0 --port $((8100 + i)) --init-timeout 1800 \
    --deploy-config /app/vllm-omni/vllm_omni/deploy/cosmos3_omni.yaml --no-guardrails \
    --diffusion-attention-backend CUDNN_ATTN \
    --cache-backend sea_cache \
    --cache-config '{"sea_threshold":0.25,"sea_residual_order":1,"sea_max_consecutive_cached":2,"sea_power_exp":3.0}' &
done
wait
```

Throughput, FP8 checkpoint: the `fp8` branch is a ModelOpt FP8 checkpoint whose transformer config enables the [Cosmos3 mixed-precision schedule](../../docs/user_guide/quantization/modelopt.md#cosmos3-mixed-precision-schedule): W8A16 for the first 3 and last 3 denoising steps, W8A8 for the middle steps, A16 for the reasoner. In this image, `vllm serve nvidia/Cosmos3-Nano --revision fp8` loads the `main` BF16 transformer weights and stops at startup (fixed by [#9436](https://github.com/vllm-project/vllm-omni/pull/9436), not in this image), so download the branch and serve the snapshot directory under the Hub name. With `--quiet`, `hf download` prints only the snapshot path.

```bash
FP8=$(hf download nvidia/Cosmos3-Nano --revision 1af44289002ad8afe2567dd85e34fe5dea21e523 --quiet)
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$((29500 + 100 * i)) VLLM_OMNI_VIDEO_SYNC_TIMEOUT=5400 \
  vllm serve "$FP8" --served-model-name nvidia/Cosmos3-Nano --omni \
    --host 0.0.0.0 --port $((8100 + i)) --init-timeout 1800 \
    --deploy-config /app/vllm-omni/vllm_omni/deploy/cosmos3_omni.yaml --no-guardrails \
    --diffusion-attention-backend CUDNN_ATTN &
done
wait
```

Balanced, CFG-parallel 2: four servers, each on two GPUs inside one PCIe switch (0,1 / 2,3 / 4,5 / 6,7).

```bash
for p in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((2 * p)),$((2 * p + 1)) MASTER_PORT=$((29500 + 100 * p)) VLLM_OMNI_VIDEO_SYNC_TIMEOUT=5400 \
  vllm serve nvidia/Cosmos3-Nano --omni \
    --host 0.0.0.0 --port $((8100 + p)) --init-timeout 1800 \
    --deploy-config /app/vllm-omni/vllm_omni/deploy/cosmos3_omni.yaml --no-guardrails \
    --diffusion-attention-backend CUDNN_ATTN \
    --cfg-parallel-size 2 &
done
wait
```

Latency: one server on all eight GPUs.

```bash
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=5400 \
vllm serve nvidia/Cosmos3-Nano --omni \
  --host 0.0.0.0 --port 8100 --init-timeout 1800 \
  --deploy-config /app/vllm-omni/vllm_omni/deploy/cosmos3_omni.yaml --no-guardrails \
  --diffusion-attention-backend CUDNN_ATTN \
  --cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp --hsdp-shard-size 8
```

## Verification

Run from a client with `curl` and `ffprobe`:

```bash
curl -sS --fail-with-body -X POST http://localhost:8100/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Nano" \
  -F "prompt=A robot arm is cleaning a plate in the kitchen" \
  -F "negative_prompt=blurry, distorted, low quality, jittery, deformed" \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" \
  -F "num_inference_steps=35" -F "guidance_scale=6.0" \
  -F "flow_shift=10.0" -F "seed=17" \
  -o cosmos3_t2v.mp4
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames -of csv=p=0 cosmos3_t2v.mp4
# expected: 1280,720,24/1,189
```

## Notes

- **Choosing a profile.** One server per GPU gives the most clips per node-hour and moves no data between GPUs. Against uncached BF16 one-per-GPU (77.0 clips per node-hour), SeaCache raises throughput 1.83x, and the FP8 checkpoint raises it 1.12x and frees about 13 GiB per GPU. CFG-parallel 2 on GPU pairs inside one switch cuts per-clip time 1.88x for 5.7% less node throughput. One server on all eight GPUs is 5.0x faster per clip than a single GPU in the one-per-GPU layout, at 63% of that layout's throughput (34% of SeaCache's). For FP8, the [ModelOpt guide](../../docs/user_guide/quantization/modelopt.md#cosmos3-mixed-precision-schedule) describes the schedule's dequantization as a correctness baseline with no speedup implied; the 1.12x here was measured on this hardware with this image only.
- **Server-to-server output variation.** Without caching, BF16 servers started the same way did not all produce the same video for a seed. Each server repeated its own output exactly (warmup and timed clips match), but at each seed six of the eight servers agreed and two produced different videos, which differ from the majority by LPIPS 0.15-0.21 (seed 17), 0.11-0.14 (seed 23) and 0.20-0.24 (seed 41). In an earlier launch on the earlier build (see below), a different set of four servers differed. The two minority videos were not compared with each other. Compare a cache, layout or quantization against this spread rather than against one uncached server.
- **Quality.** At each seed, SeaCache, CFG-parallel 2 and the 8-GPU layout are within 0.05 of the BF16 servers' range (above it by up to 0.047, SeaCache at seed 23: 0.184 against 0.112-0.137) or below it. The FP8 checkpoint is above that range at every seed, by 0.03-0.28 (seed 23: 0.40-0.41). That BF16 range comes from two servers in one launch with one prompt; in an earlier launch on the same node and an earlier build, the BF16 variants at seed 23 reached 0.179, so treat it as approximate. In the one pair reviewed side by side (seed 23, BF16 against FP8, the largest LPIPS), FP8 shifted the composition and details without visible quality loss.
- **How these were measured.** Each server ran in its own container (`nerdctl`, host networking, `--ipc=host`, `--shm-size 32g`, `--ulimit nofile=1048576:1048576`), with the GPU assigned by the container runtime, `--host 127.0.0.1`, and no `MASTER_PORT`. One-per-GPU servers also passed `--tensor-parallel-size 1` (the default). BF16 servers passed `--revision e59a53c25979a090fa8706c9acc0c254a6e89b92`, the current `main`; in this image that pins only the config reads (see #9436). FP8 servers served the same `fp8` snapshot from the Hugging Face cache. A logging-only `sitecustomize` was on `PYTHONPATH` so worker logs print (see [#9261](https://github.com/vllm-project/vllm-omni/pull/9261)). A client sent the prompt above with `width`/`height` fields. Requests went out in rounds: every server received one request per round, seeds cycled 17, 23, 41, after one warmup round at seed 17. The timed part was 3 rounds for 8 servers, 6 for 4 servers and 24 for the 8-GPU server.
- **Earlier build.** Earlier measurements on the same node used `vllm/vllm-omni:cosmos3` (`sha256:86fb713b…`, vLLM 0.31.0, vLLM-Omni 0.32.dev2+g4f62d6809). For BF16, SeaCache, FP8 and CFG-parallel 2 they were within 1% of the table on median s/clip, clips per node-hour and memory (LPIPS differed: SeaCache 0.206, CFG-parallel 2 0.154). The 8-GPU layout measured 71.25 s per clip there with a different launch: no logging shim, backend pin, `--ipc=host` or `--init-timeout`, and guardrails off through the deploy config only. The following items come from that build.
    - Tensor parallelism returned less than the layouts above: TP-2 (4 x 2) 243.32 s at 59.0 clips/node-hour, TP-4 (2 x 4) 171.02 s at 42.1.
    - `--cache-backend cache_dit` with its defaults measured 201.01 s per clip (1.85x the uncached throughput) with LPIPS 0.275 (0.241-0.301), and logged the `torch._dynamo` recompile-limit warning during server startup on every server.
    - With guardrails on, one server per GPU measured 408.29 s per clip (70.3 clips/node-hour, 54.4 GiB per GPU, slowest request 413 s), against 370.35 s (77.2) with guardrails off in the same run. Turning them on took a copy of the deploy config with `guardrails: true` and a Hugging Face token with access to the gated guardrail model; dropping `--no-guardrails` alone is not enough because the stock deploy config sets `guardrails: false`.
    - At 832x480, one server per GPU measured 108.65 s per clip.
- Not measured on this hardware: I2V, V2V, T2I, transfer, sound, action, more than one request in flight per server, online FP8 (`--quantization fp8`), more than one prompt, repeated runs.

## Supported features

| Feature | Status on this hardware | Guide |
| --- | --- | --- |
| CFG parallel | Measured, 4 x 2 and within the 8-GPU layout | [CFG parallel](../../docs/user_guide/diffusion/parallelism/cfg_parallel.md) |
| Ulysses sequence parallel + HSDP | Measured, 8-GPU layout | [Sequence parallel](../../docs/user_guide/diffusion/parallelism/sequence_parallel.md), [HSDP](../../docs/user_guide/diffusion/parallelism/hsdp.md) |
| Tensor parallel | Serves; lower throughput than the layouts above (earlier build) | [Tensor parallel](../../docs/user_guide/diffusion/parallelism/tensor_parallel.md) |
| SeaCache | Measured | [`Cosmos3-Nano.md`](./Cosmos3-Nano.md), [#6922](https://github.com/vllm-project/vllm-omni/pull/6922) |
| Cache-DiT | Serves; recompile-limit warning, higher LPIPS (earlier build) | [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) |
| FP8 ModelOpt checkpoint + mixed-precision schedule | Measured from the `fp8` snapshot; output differs from BF16 (see Notes) | [ModelOpt](../../docs/user_guide/quantization/modelopt.md#cosmos3-mixed-precision-schedule) |
| Pipeline parallel, CPU offload | Not measured | |
