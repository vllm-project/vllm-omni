# MiniMax-H3 on NVIDIA H20-3e GPUs

This recipe runs MiniMax-H3 on 141 GiB H20-3e GPUs. It contains four validated
BF16 configurations — TP2 x Ulysses1 on two GPUs, TP2 x Ulysses2 on four GPUs,
and both TP2 x Ulysses4 and TP1 x Ulysses8 on eight GPUs — plus an optional
online FP8 route on eight GPUs. No configuration requires layerwise offload:
the 141 GiB frame buffer keeps every partition resident.

H20-3e pairs modest per-card compute with a full NVLink mesh. The eight-GPU
routes reach 96% scaling efficiency and land within 2% of the per-step latency
of a considerably faster PCIe-only card, so eight GPUs is the intended
operating point for this platform.

## Capacity requirements

| Resource | Requirement |
| --- | ---: |
| GPUs | 2, 4, or 8 x NVIDIA H20-3e |
| GPU HBM | 140.4 GiB per GPU (143,771 MiB) |
| Checkpoint storage | 135 GiB per partition |
| Container `/dev/shm` | 8 GiB minimum, 32 GiB recommended |
| Recommended system RAM | 384 GiB |

`FL2VA` and `Ref2VA` are separate checkpoint partitions. Start one server at a
time.

## NVLink topology

Confirm the mesh before starting the server:

```bash
nvidia-smi topo -m
```

On the validated host every GPU pair reports `NV18`, an eighteen-link bonded
NVLink connection, with no PCIe fallback anywhere in the matrix. GPUs 0-3 are
attached to NUMA node 0 and GPUs 4-7 to NUMA node 1, but GPU-to-GPU collectives
never traverse the host interconnect, so no `CUDA_VISIBLE_DEVICES` reordering
is needed. Unlike PCIe-only platforms, device order does not affect throughput
here.

## Shared memory

vLLM-Omni transfers the rendered result from the diffusion worker to the API
server through shared memory. A 1344x768 payload is roughly 1.4 GiB, far above
the 64 MiB Docker default. With an undersized `/dev/shm` the worker either
spills to disk and exceeds the 30 second result-handoff timeout, or aborts with
`Bus error (core dumped)` and takes the server down mid-request.

Start the container with at least:

```bash
docker run --gpus all --ipc=host --shm-size=32g ...
```

## Software versions

| Component | Version |
| --- | --- |
| Container image | `vllm/vllm-omni:minimax-h3` |
| vLLM-Omni | `0.1.dev2381+g310b4b477` |
| vLLM | `0.26.0` |
| NVIDIA driver | `580.105.08` |

The BF16 configurations run unmodified from the published image. The optional
FP8 route is not available in that image and requires a source build; see
[Optional: online FP8](#optional-online-fp8).

The image emits a `vLLM and vLLM-Omni appear to have mismatched major/minor
versions` warning at startup. It is benign on this build.

## Recommended serving configurations

All configurations share the same model path, VAE tiling settings, and
attention backend. Selecting `FLASH_ATTN` explicitly keeps the recipe
independent of platform-default backend changes.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091
```

### Two GPUs

```bash
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

The measurement below was taken without `--text-encoder-tp-size 2`. GPU0 then
holds the entire 47 GiB text encoder and peaks about 48 GiB above GPU1, which
is what produced the 100.3 GiB figure in the results table. The flag is
included above because it balances the two ranks; the balanced variant was not
re-measured.

### Four GPUs

```bash
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --tensor-parallel-size 2 \
  --usp 2 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

### Eight GPUs, TP2 x Ulysses4

The recommended default for long-lived services. It is the most memory
efficient BF16 route and leaves the largest headroom for longer outputs.

```bash
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 8 \
  --tensor-parallel-size 2 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

### Eight GPUs, TP1 x Ulysses8

Steady-state throughput is equivalent to TP2 x Ulysses4 within 0.7%, and peak
memory is 23.8 GiB higher, so this route is not a throughput upgrade. Its
advantage is start-up cost: removing tensor parallelism eliminates two
all-reduce collectives from each of the 50 DiT blocks, which shrinks the
regional-compile graph by roughly an order of magnitude. The first request
after start-up costs 28.9 s of extra denoise time instead of 239.1 s.

Prefer this route for elastic, preemptible, or frequently restarted
deployments, and TP2 x Ulysses4 for steady long-running services.

```bash
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 8 \
  --tensor-parallel-size 1 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

Do not add `--enforce-eager` to any of these routes. Warm the server with at
least one request before measuring so regional compilation falls outside the
measured request.

For Ref2VA, stop the FL2VA server and restart the same command with
`MODEL="${MODEL_ROOT}/Ref2VA"`.

## Optional: online FP8

Online FP8 quantizes the DiT at load time and leaves the text encoder and both
VAEs in BF16. On eight GPUs it is the fastest and the most memory efficient
configuration in this recipe, beating every BF16 route on both axes at once.

This route is **not available in `vllm/vllm-omni:minimax-h3`**. The published
image at `0.1.dev2381+g310b4b477` contains no FP8 code path for MiniMax-H3:
`Fp8PerTensorOnlineLinearMethod` is absent from the package and the H3 model
directory has no FP8 branch. Passing `--quantization fp8` to that image does
not enable quantization. A source build is required:

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
git checkout bbe6ccc51
python3 -m pip install -e . --no-build-isolation --no-deps
```

`--no-deps` is required. Without it pip upgrades vLLM away from the `0.26.0`
that the image ships, which breaks the runtime. Confirm the result before
serving:

```bash
python3 -c "import vllm_omni, vllm; print(vllm_omni.__version__, vllm.__version__)"
```

Then add two flags to the eight-GPU TP1 x Ulysses8 command:

```bash
  --quantization fp8 \
  --init-timeout 3600
```

`--init-timeout 3600` is not optional. Weight quantization runs during model
load and is slow; the default timeout aborts start-up partway through.

Two log lines confirm that quantization actually engaged:

```
Building quantization config: fp8
Selected CutlassFP8ScaledMMLinearKernel for Fp8PerTensorOnlineLinearMethod
```

Online FP8 is incompatible with H3 layerwise offload, whose weight stride is
rejected by the Cutlass FP8 kernel. That restriction is irrelevant on this
platform, where no route needs offload.

## Target-hardware validation

All configurations were exercised on a single host with eight H20-3e GPUs, NVLink
NV18 full mesh, driver 580.105.08, 1344x768 output, 24 fps, 124 frames, seed
1101, `flow_shift=12`, and `duration=5.0`. Each run performed two warmups
followed by three measured requests; steady-state figures are the mean of the
last four requests.

MiniMax-H3 requests 50 denoise steps and executes 49 denoise updates, so
per-step latency is `denoise / 49`.

| GPUs | Parallelism | Precision | E2E (s) | Denoise (s) | VAE decode (s) | Per step (ms) | Peak memory (GiB) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | TP2 x Ulysses1 | BF16 | 637.85 | 625.35 | 10.60 | 12,762 | 100.3 |
| 4 | TP2 x Ulysses2 | BF16 | 327.94 | 320.81 | 5.10 | 6,547 | 68.8 |
| 8 | TP2 x Ulysses4 | BF16 | 168.09 | 163.26 | 2.975 | 3,332 | 70.6 |
| 8 | TP1 x Ulysses8 | BF16 | 166.97 | 162.40 | 2.973 | 3,314 | 94.4 |
| 8 | TP1 x Ulysses8 | FP8 | **135.01** | **130.39** | 2.973 | **2,661** | **63.5** |

E2E is the engine-reported request time. MP4 muxing adds a further 1.5-2.7 s
before the HTTP response completes. Peak memory is the maximum per-GPU value
sampled externally with `nvidia-smi` at 2 s intervals; device capacity is
140.4 GiB, so the tightest route still leaves 40.1 GiB free.

Text encode is 0.04-0.07 s in the steady state on all eight-GPU routes. The
first request additionally pays 3.9-5.1 s of text-encoder warm-up.

Run-to-run variation is very low. The four steady-state requests of the
eight-GPU BF16 TP1 x Ulysses8 route spanned 100 ms end to end, and the FP8
route spanned 40 ms.

The FP8 row was collected on the source build described above; all BF16 rows
were collected on the published image. A same-version BF16 control run
established that the version difference accounts for 0.84% of denoise time, so
the FP8 comparison is not materially affected. See
[Version sensitivity](#version-sensitivity).

## Scaling behaviour

| Denoise scaling | Speed-up | Efficiency |
| --- | ---: | ---: |
| 2 -> 4 GPUs | 1.95x | 97.5% |
| 4 -> 8 GPUs | 1.96x | 98.2% |
| 2 -> 8 GPUs | 3.83x | 95.8% |

The full NVLink mesh is what makes this possible; measure `nvidia-smi topo -m`
before assuming the same behaviour on a partially connected host.

No thermal throttling was observed. Maximum GPU temperature was 66 C and
average board power 322-342 W across every configuration, with no drift across
the five-request sequence.

## Memory behaviour

Quantization saves considerably more memory than adding tensor parallelism on
this model:

| Action | Memory saved per GPU |
| --- | ---: |
| TP1 -> TP2 (eight GPUs) | 18.1 GiB |
| BF16 -> FP8 (eight GPUs, TP1) | 30.3 GiB |

Online FP8 halves 30,978 MiB of DiT weights, which implies a DiT of about
60.5 GiB and matches the 62 GiB component figure documented for the model. The
tensor-parallel split, however, only moves 37.1 GiB. The remaining ~23 GiB of
the DiT is replicated on every rank regardless of TP size, which is consistent
with the AdaLN projections and the condition projection being outside the
tensor-parallel partition.

Plan capacity from quantization first and parallelism second.

## FP8 output equivalence

FP8 and BF16 produce visually equivalent output of comparable quality, but
they do **not** produce the same sample from the same seed.

At identical seed, version, and request parameters, FP8 and BF16 rendered
entirely different — though equally coherent and equally well composed —
scenes. Pixel metrics between them are correspondingly low, at 21.55 dB PSNR
and 0.798 SSIM. This is trajectory divergence, not degradation: FP8 perturbs
each of the 50 sampling steps enough that the trajectory settles into a
different mode of the distribution.

PSNR and SSIM are therefore not meaningful accuracy metrics for this
quantization. Evaluate FP8 on perceptual quality and prompt adherence, not on
pixel fidelity, and do not use FP8 where frame-exact reproducibility against a
BF16 reference is required.

## Version sensitivity

A BF16 control run on the same source build used for the FP8 route isolates
what the version change contributes:

| Metric | Image `dev2381` | Source `bbe6ccc51` | Delta |
| --- | ---: | ---: | ---: |
| Denoise (s) | 162.40 | 161.04 | -0.84% |
| VAE decode (s) | 2.973 | 2.975 | 0.0% |
| Post-load memory (MiB) | 81,971 | 81,793 | -178 |
| Peak memory (MiB) | 96,673 | 96,523 | -150 |
| MP4 muxing (s) | 2.727 | 1.511 | -44.6% |

Compute and memory are effectively unchanged. The only material difference is
MP4 muxing, which is CPU-side post-processing that quantization cannot affect;
the FP8 route reports 1.537 s for the same stage, confirming the improvement
belongs to the source build rather than to FP8.

The two BF16 builds render the same scene from the same seed with only local
differences, at 32.03 dB PSNR and 0.934 SSIM. A version change perturbs the
sample locally; FP8 relocates it entirely.

## T2VA request example

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"

curl -sS --max-time 1800 -X POST "${API_URL}" \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

Verify the output with `ffprobe` rather than relying on the server metrics.
The `image_pixels` field in `StageRequestStats` reports 499,968 for this
request, which is the latent token count multiplied by four
(`84 x 48 x 31 x 4`), not the 1,032,192 output pixels of a 1344x768 frame.

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,avg_frame_rate \
  -of default=nw=1 t2va.mp4
```
