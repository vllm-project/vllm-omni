# Device-side video postprocessing

vLLM-Omni can convert decoded WAN2.2 video from normalized floating-point
`[B, C, T, H, W]` tensors to contiguous `uint8` `[B, T, H, W, C]` frames before
the worker-to-engine device-to-host transfer. This reduces the transported video
payload by four times when the original IPC path widens bfloat16 to float32.

The optimization is disabled by default. Enable it with a structured diffusion
configuration:

```python
from vllm_omni import Omni

engine = Omni(
    model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    video_output_transport={"enable_device_postprocess": True},
)
```

For the server CLI, pass the same object as JSON:

```bash
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
  --video-output-transport '{"enable_device_postprocess": true}'
```

For a deploy configuration, place it on the diffusion stage:

```yaml
stages:
  - stage_id: 0
    video_output_transport:
      enable_device_postprocess: true
```

## Runtime contract

WAN2.2 emits decoded video through the typed `DiffusionOutput.media` field. The
model runner validates and splits the batch into request-local tensors before
preparing each video for transport. The worker then performs the D2H copy and
shared-memory IPC using the prepared representation. The engine uses the generic
media finalizer instead of a model-specific postprocessor.

The runtime leaves the video in normalized floating-point form when:

- `enable_device_postprocess` is false;
- the requested output type is not `np`; or
- frame interpolation still requires the floating-point tensor.

These are policy fallbacks, not malformed-contract errors. If the device cannot
allocate the temporary conversion buffer, the runner logs a warning and prepares
the request-local float representation instead. Invalid tensor layout, encoding,
value range, lifecycle state, and non-memory runtime failures still fail before
IPC.

## Precision

Device preparation converts to float32 before denormalization and quantization.
For bfloat16 WAN output this is more precise than the historical host path,
which denormalizes in bfloat16. Relative to that path, values may differ by at
most one level out of 255. Float32 inputs remain byte-identical after
quantization.

## Scope

WAN2.2 is the reference migration for the first contract version. Other video
pipelines continue using their existing output paths until they can declare the
same typed media contract without bypassing model-specific float consumers such
as safety checks or audio/video packaging.

See [RFC #6541](https://github.com/vllm-project/vllm-omni/issues/6541) for the
contract, lifecycle, batching rules, and migration plan.

## Experimental registered-SHM video transport

`video_output_transport.enable_registered_shm` is a separate, default-off option
for CUDA request-mode workers. It copies eligible video tensors directly into
CUDA-registered shared memory and returns borrowed CPU views, removing the
intermediate pinned-to-SHM and SHM-to-consumer host copies. The D2H transfer still
occurs.

`video_output_transport.enable_borrowed_frames` independently enables borrowed
RGB AVFrames in the Videos API MP4 response encoder, removing the third host
copy into PyAV frame buffers. It also defaults to false. Eligible uint8 RGB
frames are wrapped with `VideoFrame.from_numpy_buffer`; other dtypes/layouts use
the existing encoder path. Color conversion, compression and muxing still occur.

Enable it through the same CLI, Python or deploy-stage configuration:

```bash
--video-output-transport '{"enable_registered_shm": true, "enable_borrowed_frames": true}'
```

Registered SHM applies only to prepared typed video media (currently WAN2.2) and
explicitly marked legacy video outputs (currently MiniMax-H3). H3 marks the video
entry of its `(video, audio)` output; audio, trajectory data, images and unmarked
legacy outputs keep their existing transport. Device-side postprocessing is
independent and does not need to be enabled for registered SHM.
Pre-encoded MP4 outputs bypass raw-frame transport and API-side encoding.

The existing SHM routing threshold is unchanged: the tensor or its backing
storage must exceed 1,000,000 bytes. This is not a measured performance crossover.
SHM is allocated and registered on every transfer, without pooling; small outputs
can regress and registration latency can vary. Benchmark representative output
sizes before opting in. CPU outputs, non-CUDA backends, and synchronous/step-mode
transfers keep their existing path.

Borrowed views keep the unlinked mapping alive until their last user releases
it. CUDA registration/copy failures fail the request; if CUDA cleanup cannot be
confirmed, the worker retains the mapping for safety and must be restarted before
further registered-SHM transfers.
