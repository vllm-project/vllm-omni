# Pika H3 Flex

Serve the official FL2VA or Ref2VA adapter with the matching original native
MiniMax-H3 partition. Pika uses six transformer forwards with fresh noise between
steps, rather than H3's ordinary Euler update. Video and stereo audio are sampled
together. The integration uses H3's existing LoRA manager and serving APIs.

For official numerical comparison, start with the [reference configuration](#compare-with-official-numerical-results) and [complex people / 2K examples](#complex-people-examples-and-numerical-evidence).

## Download (optional)

Serving `Pika-Labs/Pika-H3-Flex` resolves the matching adapter and original
native H3 base automatically, using the revisions below. It downloads only
the selected task partition. To download ahead of time:

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include 'Ref2VA/*' --local-dir /models/MiniMax-H3
hf download Pika-Labs/Pika-H3-Flex \
  --revision 3ff166447d5edb1406107ab8370c42000ce173e4 \
  --include 'ref2va/*' LICENSE NOTICE --local-dir /models/Pika-H3-Flex
```

For T2VA or first/last-frame conditioning, download `FL2VA/*` and `fl2va/*`
instead. Keep both `manifest.json` and `adapter.safetensors` in the selected
adapter directory. The loader verifies the manifest, checksum, all 624 tensors,
rank/alpha 128, and complete binding. Do not substitute a fused FastH3 checkpoint
for the original base. The Pika repository itself does not contain a native base.

## Serve

Use a vLLM/PyTorch environment matching this vLLM-Omni checkout. This example
uses four CUDA GPUs, dense attention and BF16 LoRA weights:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_WORKER_MULTIPROC_METHOD=spawn \
OMP_NUM_THREADS=8 VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
VLLM_OMNI_VAE_LEGACY_TEMPORAL=1 \
vllm serve Pika-Labs/Pika-H3-Flex --omni --trust-remote-code \
  --task-type ref2va --num-gpus 4 --tensor-parallel-size 4 \
  --text-encoder-tp-size 4 --usp 1 --ring 1 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --enforce-eager --diffusion-attention-backend CUDNN_ATTN \
  --diffusion-quantization-config '{"transformer":null,"text_encoder":null,"video_vae":null,"audio_vae":null}' \
  --additional-config '{"minimax_h3_vae_audio_overlap":true}' \
  --host 127.0.0.1 --port 8000
```

Use `--task-type fl2va` for T2VA/FL2VA; `auto` defaults to FL2VA for the
repository root, or follows the manifest when serving a local adapter directory.
A local Pika release root or its `fl2va`/`ref2va` directory can replace the model ID.
To reuse an existing native base, add
`"pika_base_model_path":"/models/MiniMax-H3"` to `--additional-config`.
This local override is a trusted checkpoint and must match the released base.
Each
server uses one partition: a Ref2VA adapter cannot bind the second transformer
of a combined server. GPU count here describes a validation configuration,
not a universal minimum-memory requirement. The legacy temporal flag selects
the original checkpoint decoder and avoids the old uint8 patch's callback ABI.

## Generate

Direct Pika deployment pins its bundled adapter at scale 1. Requests default
to six transformer forwards and do not need a LoRA field:

```bash
curl --fail-with-body --max-time 1800 http://127.0.0.1:8000/v1/videos/sync \
  --form-string 'prompt=The person in Image 1 smiles and waves. Natural motion and soft room ambience.' \
  -F 'input_reference=@reference.png' \
  -F seconds=5 -F size=832x480 -F fps=24 -F seed=42 \
  -F 'extra_params={"task":"ref2va","flow_shift":12,"audio_flow_shift":3}' \
  -o pika.mp4
```

Against an FL2VA server, use `task=t2va` without a reference and include
`"aspect_ratio":"16:9"` in `extra_params`, or use `task=fl2va` with H3's
existing keyframe inputs. See the
[H3 conditioning API](MiniMax-H3.md) for first/last frames and
multimodal references. A five-second request produces 124 frames at 24 fps,
with 32 kHz stereo audio.

The published native output range is 480p/768p, for integer durations of 5–15
seconds. The six-step contract fixes video/audio shifts to 12/3 and LoRA scale
to 1. Direct Pika deployment rejects startup and request LoRA overrides. Initial
noise, reference augmentation and intermediate noise use independent,
request-owned device generators and the official external latent layouts.
Both ordinary and step execution retain that state, including packed batching.
Offline step-execution callers must explicitly set `num_inference_steps=6`.

### Compare with official numerical results

The four-GPU configuration above uses optimized BF16 arithmetic and cuDNN
attention. For comparison with the official T2VA pipeline, use the following
single-GPU configuration with the original unquantized FL2VA base:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=8 \
VLLM_OMNI_VAE_LEGACY_TEMPORAL=1 VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve /models/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --task-type fl2va --num-gpus 1 --tensor-parallel-size 1 \
  --text-encoder-tp-size 1 --usp 1 --ring 1 \
  --vae-patch-parallel-size 1 --vae-parallel-mode tile --vae-use-tiling \
  --enforce-eager --diffusion-attention-backend TORCH_SDPA \
  --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}' \
  --lora-path /models/Pika-H3-Flex/fl2va --lora-scale 1 \
  --additional-config '{"minimax_h3_reference_numerics":true}' \
  --host 127.0.0.1 --port 8000
```

Select the adapter in each request as described below. This mode preserves
the official BF16 rounding in text normalization, DiT normalization,
modulation, SwiGLU and LoRA accumulation. It selects PyTorch Flash SDPA and
keeps the original video decoder in FP32. It requires eager execution,
unquantized TP=SP=1 and no approximate cache backend. Text-encoder CPU
offload reduces residency; it still needs a large GPU and sufficient host RAM.
It is a comparison configuration, with different performance from the
four-GPU optimized configuration.

### Request-switchable adapters on an ordinary H3 server

To retain the original H3 model and switch adapters per request, replace the
served model with `/models/MiniMax-H3/Ref2VA` and add
`--lora-path /models/Pika-H3-Flex/ref2va --lora-scale 1` to the server command.
Then add these fields to each Pika request:

```bash
-F num_inference_steps=6 \
-F 'lora={"name":"pika","path":"/models/Pika-H3-Flex/ref2va","scale":1}'
```

On this ordinary H3 endpoint, scale 0 disables Pika and restores the base H3
sampling contract. Select the matching FL2VA paths for T2VA/FL2VA.

## Optional 2K output

Pika's public weights do not include its hosted 2K generator. For local 2K
output, use the existing H3 latent upscaler after Pika's six steps and before
video decoding. This is an additional community model, not native 2K Pika or
the hosted Draft-to-Video service.

```bash
hf download LBH-123-AI/Minimax_h3_latent_Upscaler \
  --revision 3f941d5d182014dd5c0a5e16330420ee2d4aa0c6 \
  --include 'minimax_h3_latent_upscaler_3d_conv_v1/*bf16.safetensors' \
  --local-dir /models/h3-upscaler
```

Replace the server's `--additional-config` value with:

```json
{
  "minimax_h3_vae_audio_overlap": true,
  "latent_upscaler_path": "/models/h3-upscaler/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
}
```

Keep the request's generation canvas at 832x480 or a published 768p bucket and
add the output target to `extra_params`:

```json
{
  "task": "ref2va",
  "flow_shift": 12,
  "audio_flow_shift": 3,
  "latent_upscale": {"width": 2560, "height": 1440}
}
```

Audio and frame count stay unchanged. Upscale-only runs no extra denoising
steps. `latent_refine` is rejected for an active Pika adapter: H3's truncated
Euler refinement cannot replace Pika's released six-step sampler. A separate
[Open-H3-Regenerate-2K server](../Efficient-Large-Model/Open-H3-Regenerate-2K-H200.md)
is another existing route for re-rendering drafts, with its own weights and
sampling contract; its combination with Pika is not qualified here.

## Reused optimizations and boundaries

| Optimization | Behavior |
| --- | --- |
| Exact AdaLN projection reuse | Already in vLLM-Omni; reused with request/adapter invalidation. On validated SM120 deployments, mainline can also offload these weights between cache misses. |
| VAE spatial tiling and temporal chunks | Reuse the existing H3 implementation for larger output canvases. |
| Audio/video decode overlap | Optional `minimax_h3_vae_audio_overlap=true` runs audio on a side stream and holds both components resident until it drains, including failure exits. Default is false; CUDA only. It trades some memory for concurrency. |
| Tensor/sequence parallelism | Reuse H3's existing execution and LoRA sharding. Noise remains request-local and identical across ranks. |
| MP4 output streaming | The existing `preencode_mp4` route has its own staged output handling; the new decode-overlap option applies to ordinary `decode()`. |

MXFP8, sparse attention, approximate caches and fixed-shape VAE experiments
are not enabled by this recipe. Their numerical or hardware assumptions need
separate qualification; the historical FastH3 benchmark numbers do not apply
to this Pika configuration. Continuing clips, latent editing, ControlNet,
timeline guides and locked driving audio are outside the released Pika contract.

## Validation

Validated against upstream `7afd00e982dfa93c365eee5475a5707904b181e6`
with vLLM 0.32.0, PyTorch 2.13.0+cu132 and Transformers 5.14.1. Optimized functional
serving used four SM120 CUDA GPUs, TP4, dense BF16 and the pinned adapters
above. The independent official comparison below used one SM120 GPU with
`minimax_h3_reference_numerics=true`, unquantized TP=SP=1 and eager execution. The FL2VA base was converted from the original Diffusers tensors to
the native H3 layout using the existing weight-mapping helpers; Ref2VA used
the original native partition.

- H3, LoRA and SDPA CPU regression after the numerical fixes: 1322 passed,
  2 skipped, 59 deselected. CUDA SDPA/compile checks: 101 passed.
- Pika and decode-overlap coverage: 33 passed, including the CUDA stream test.
  These tests overlap the CPU total; the counts are not additive.
- The published reference sampler and this implementation produced identical
  video/audio rows on CPU and CUDA with fixed synthetic velocity inputs.
  This validates sampler math and RNG layout, not full-model pixel parity.
- Real-weight HTTP generation covered T2VA at 480p and 768p, first-frame,
  last-frame and first/last-frame FL2VA, and image-conditioned Ref2VA.
  FL2VA and Ref2VA both produced
  2560x1440 output with the community upscaler. Five-second samples contained
  124 frames at 24 fps and 32 kHz stereo audio; all streams decoded fully.
- For each adapter, decoding the audio from the matching 480p and 2K outputs
  produced identical PCM hashes. Invalid step counts and generic refinement
  returned HTTP 400.
- Direct model-ID deployment also generated Ref2VA from a combined image,
  video-with-audio and audio reference at 832x480 and 2560x1440. These runs used
  the pinned adapter cache and an explicit local original-H3 base override;
  cold network downloading of the base was not exercised. The two outputs
  again had identical decoded audio. Request adapter overrides were rejected.
- Independent official and native HTTP runs now cover three complex adult
  scenes with the FL2VA adapter: a watchmaker's tool handoff, three chefs
  working through occlusion and steam, and three full-body dancers crossing
  paths. Each uses seed 42, six steps, 832x480 and 124 frames. Text embeddings,
  all 18 video/audio denoising outputs, final latents and pre-AAC waveforms
  match exactly in reference mode. This comparison uses the native text
  encoder, sampler, DiT and VAE; it does not inject official intermediate
  tensors into the service.

### Complex people examples and numerical evidence

The comparison videos place the official output on the left and native
vLLM-Omni on the right. Exact prompts, request payloads, source revisions,
per-step metrics and reproduction scripts are included in the
[validation bundle](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/h3-people-validation.tar.gz).

| Scene | Official / native comparison | Native 2K output |
| --- | --- | --- |
| Watch repair: three adults, overlapping hands and tool handoff | [Video](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/pika-people-workshop-comparison.mp4) | [2560x1440 video](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/pika-people-workshop-2k.mp4) |
| Kitchen: three chefs, bowl handoff, steam and camera movement | [Video](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/pika-people-kitchen-comparison.mp4) | — |
| Dance: three adults, full bodies, turns and crossing paths | [Video](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/pika-people-dance-comparison.mp4) | — |

Across the eight native-resolution Pika/DMAD comparisons, pre-encoding RGB
values differ by at most 1/255, with minimum PSNR 99.18 dB.
Audio waveforms match exactly. RGB comparison is before H.264 compression;
the two MP4 encoders need not produce identical bytes. The 2K case preserves
the corresponding native-resolution latent and waveform, then applies the
community upscaler; there is no official local Pika 2K target for pixel parity.
See the [metrics](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/h3-people-comparison.json) and
[reproduction guide](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/h3-people-README.txt).

To reproduce the watch-repair prompt against the reference-mode FL2VA
server above, download the retained prompts and submit the same parameters
(requires `curl` and `jq`):

```bash
curl --fail -L -o h3-people-prompts.json \
  https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/h3-people-prompts.json
curl --fail http://127.0.0.1:8000/v1/videos/sync \
  --form-string "prompt=$(jq -r .workshop h3-people-prompts.json)" \
  -F 'seconds=5' -F 'size=832x480' -F 'fps=24' -F 'seed=42' \
  -F 'num_inference_steps=6' \
  -F 'lora={"name":"pika","path":"/models/Pika-H3-Flex/fl2va","scale":1}' \
  -F 'extra_params={"task":"t2va","aspect_ratio":"16:9","flow_shift":12,"audio_flow_shift":3}' \
  --output pika-people-workshop.mp4
```

Select `.kitchen` or `.dance` for the other prompts. For 2K, configure the
server's community upscaler and add `latent_upscale` as described above.

The earlier image-conditioned examples are also available:
[FL2VA 2K](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/pika-fl2va-2k.mp4),
[Ref2VA image 2K](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/pika-ref2va-image-2k.mp4), and
[Ref2VA mixed-reference 2K](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/pika-ref2va-mixed-2k.mp4).
Their full original request payloads were not retained. These historical
functional samples do not establish Ref2VA numerical parity.

Reference-mode results do not qualify the optimized cuDNN/fused-operator
configuration for bitwise equivalence or establish a perceptual-quality
score. Other hardware, Ref2VA, quantization and distributed execution need
separate numerical qualification. Tracing overhead makes these accuracy
runs unsuitable as latency benchmarks.

Run the small contract tests from the repository root with the checkout's
normal development dependencies installed. The CPU cases need no model
weights; the CUDA decode test needs one GPU:

```bash
pytest -q tests/diffusion/models/minimax_h3/test_pika_h3.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_reference_numerics.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_decode_overlap.py \
  -m 'core_model and cpu' --run-level core_model
pytest -q tests/diffusion/models/minimax_h3/test_minimax_h3_decode_overlap.py \
  -m 'core_model and cuda' --run-level core_model
```

The existing CPU and diffusion-model GPU sweeps in `test-ready.yml` collect
these tests. Real-weight runs above are local smoke tests, not a benchmark or
a complete hardware, duration, resolution and reference-modality matrix.
GPU sequence parallelism and real-weight continuous batching need separate
Pika qualification; request-isolation and step-sampler coverage here use CPU
stand-in transformers. No new performance or perceptual-quality claim is made.

## References

- [Pika model card](https://huggingface.co/Pika-Labs/Pika-H3-Flex)
- [Pika sampler reference](https://github.com/Pika-Labs/Pika-H3-Flex/tree/06bd434ce0b6e287e6f6c4058420460f47ba031b)
- [Community latent upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
