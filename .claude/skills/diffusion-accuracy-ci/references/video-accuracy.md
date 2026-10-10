# Video accuracy cases

Use this reference when adding or extending a real-model video accuracy pytest.
Apply [the reference contract](reference-contract.md) for provenance, shared
inputs, repeated-run calibration, and thresholds. Choose the comparison boundary
before implementing the runner: raw decoded outputs and encoded files differ.

## Choose a reference pattern

| Pattern | Repository example | Comparison boundary |
| --- | --- | --- |
| Runtime Diffusers reference | [Wan I2V](../../../../tests/e2e/accuracy/wan22_i2v/test_wan22_i2v_video_similarity.py) | All decoded MP4 frames with FFmpeg SSIM/PSNR |
| Pinned official implementation | [LTX official accuracy](../../../../tests/e2e/accuracy/ltx/test_ltx_official_similarity.py) and [runner](../../../../tests/e2e/accuracy/ltx/run_ltx_reference.py) | Raw video arrays and audio arrays before output encoding |
| Frozen reference artifacts | [SANA golden accuracy](../../../../tests/e2e/accuracy/sana_video/test_sana_video_golden.py) | Manifest-verified transformer outputs and encoded pipeline video |

Reuse their mechanics, not their sampling values or thresholds. Select the actual
reference pipeline and supported task; a runner filename does not prove its
parallelism, reference independence, or version pinning.

## Own generation, completion, and cleanup

- Create both outputs in one self-contained test or a shared fixture with explicit
  dependencies and per-run paths. Retain the request and result metadata together.
  Test order or old files must not determine whether a comparison can run.
- Use [OmniServer and server fixtures](../../../../tests/helpers/runtime.py).
  For the asynchronous video API, reuse
  [`send_video_request_with_timeout`](../../../../tests/e2e/accuracy/helpers.py):
  it submits `/v1/videos`, checks the creation response, waits for completion,
  then downloads the completed video's content. A returned job ID is not output.
- Choose initialization, reference-subprocess, and job-completion timeouts from
  the case's resource needs. Failures and timeouts must preserve logs and run
  teardown; do not leave a server or reference process holding the comparison GPU.
- Close the server before starting a same-device reference when both need the
  device. For direct offline runners, release the engine in `finally`, as the
  LTX runner does with `omni.shutdown()`.

## Align geometry and compare the complete clip

Align prompt, input image bytes and preprocessing, effective width and height,
frame count, FPS, seed/generator device, scheduler, steps, and task-specific
guidance. Check temporal/VAE shape constraints instead of rounding one side
silently. For I2V, preserve the same conditioning frame and any codec preprocessing.

For encoded output, reuse `probe_video` and `assert_video_metadata` before
`assert_video_similarity_metrics` from [helpers](../../../../tests/e2e/accuracy/helpers.py).
Assert both clips' requested dimensions, FPS, and frame count as well as equality
between them. These helpers use FFprobe and FFmpeg; missing binaries cause skips,
which must not be reported as accuracy passes.

The FFmpeg gate compares video streams across the complete clip. Preserve frame
order and timing; do not trim, resize, resample, or compare only a few preview
frames to make mismatched clips pass. Aggregate SSIM/PSNR may hide a short local
failure; add per-frame diagnostics or gates when the chosen contract requires it.

An MP4 gate includes codec, pixel format, chroma subsampling, and serialization
effects. Align and record export settings, or compare raw decoded model output
when the intent is model numerical fidelity. Preserve raw outputs where available
to distinguish denoising/VAE drift from encoding drift; previews are inspection
artifacts and do not replace an all-frame gate.

## Wan runtime reference pitfalls

The current Wan file splits offline generation, online generation, and similarity
into three tests. Similarity reads their persisted paths and skips if artifacts
are absent; it does not establish that existing files came from this run.
Use an explicit current-run fixture or one test when creating a new case.

The [Wan reference runner](../../../../tests/e2e/accuracy/wan22_i2v/run_wan22_i2v_diffusers_cp.py)
uses one GPU despite its `_cp` name. It converts I2V input to RGB, resizes with
LANCZOS, and configures UniPC `flow_shift` plus the model's `boundary_ratio` and
dual guidance. Align these model-specific choices with the native path.
It currently calls `from_pretrained` without an immutable revision; the parent
test also prepends a sibling `diffusers/src` to `PYTHONPATH`. Resolve and record
the actual imported reference version, and pin the reference/model for new cases.

## LTX official raw video and audio

Follow the LTX case definitions to pin official source, model, checkpoint,
upsampler, and LoRA revisions separately. The official runner executes in a
subprocess through `uv run --no-project --with` a pinned OpenImageIO version;
keep reference-only dependency handling separate from Omni's runtime dependencies.
Verify the actual source revision and use the shared request JSON for both runners.

The runner saves `video.npy`, `audio.npy`, and metadata. Canonical video is
frame-major RGB float data in `[0, 1]`; assert identical shapes and compare every
frame. `_video_metrics` records mean/min SSIM, mean PSNR, and absolute error.
First/middle/last PNGs are previews only. This gate measures pre-output-encoding
arrays. LTX's I2V reference also controls conditioning-image encoding with `crf=0`.

When audio is part of the model contract, assert sample-rate and canonical
waveform-shape equality before relative L2/cosine or other calibrated gates.
Do not trim, shift, or resample to force alignment. The FFmpeg video helper does
not validate audio; explicitly state whether the case covers audio.

## SANA frozen artifacts and stage diagnosis

Follow [the SANA golden contract](../../../../tests/e2e/accuracy/sana_video/README.md)
and its generator for immutable revisions, normalized I2V input, metadata, and
manifest SHA256/size checks. Its artifacts include pre-MP4 uint8 frames and final
latents as well as MP4 output. Missing `SANA_VIDEO_GOLDEN_BASE_URL` causes a skip.
Validate downloaded provenance before inference; native weights must match it too.

Use [pipeline alignment](../../../../tests/e2e/accuracy/sana_video/test_sana_video_pipeline_alignment.py)
or [I2V alignment](../../../../tests/e2e/accuracy/sana_video/test_sana_video_i2v_alignment.py)
for model-specific stage diagnosis: initial latents, conditioning, timesteps,
transformer/CFG, and scheduler steps. Retain both implementation outputs and
stage metrics alongside the final clip. These checks complement the final gate.

Golden generation and publication are separate actions. Generate reviewable local
artifacts when authorized; publish only within the user's requested scope. Use a
new version prefix for a changed reference, and never overwrite a frozen golden
automatically. Calibrate each new case via [the reference contract](reference-contract.md).
