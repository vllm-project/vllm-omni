# JoyAI-Image-Edit E2E Smoke Guide

This guide is for running the first cloud-side e2e smoke tests for
`JoyImageEditPipeline` after the JoyAI-Image-Edit vLLM-Omni integration.

The target model for vLLM-Omni is the Diffusers-format checkpoint:

- Hugging Face Diffusers checkpoint:
  <https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers>
- Original Hugging Face checkpoint:
  <https://huggingface.co/jdopensource/JoyAI-Image-Edit>
- Pipeline class expected by `model_index.json`: `JoyImageEditPipeline`
- vLLM-Omni model class: `JoyImageEditPipeline`

The Hugging Face model card currently lists the Diffusers checkpoint as an
Apache-2.0 licensed image-to-image Diffusers model, and shows
`JoyImageEditPipeline` usage with `guidance_scale=4.0`. In vLLM-Omni, use
`true_cfg_scale` / `--cfg-scale` as the canonical classifier-free guidance
field. `guidance_scale` is accepted only as a Diffusers compatibility alias
when `true_cfg_scale` is absent; `guidance_scale=1.0` is treated as the shared
image-edit CLI's disabled/default guidance value.

## Scope

This guide verifies that a single-GPU cloud environment can load the model and
complete a minimal image-editing request. It is not a quality benchmark.

Validated behavior for this smoke:

- Single prompt.
- Exactly one input image.
- `height=512`, `width=512`.
- `num_inference_steps=2` for first smoke.
- Output image exists and can be opened.
- Multi-image input fails clearly.
- Conflicting CFG aliases fail clearly.

Out of scope for this smoke:

- Full 40-50 step visual quality validation.
- Diffusers pixel or latent parity.
- Tensor parallel, USP/SP, HSDP.
- Cache-DiT, TeaCache, LoRA.
- Multi-image editing.

## Recommended Machine

Start with one NVIDIA A100 40GB or larger.

The first run should use CPU offload and VAE tiling to reduce risk while
validating the request path. After that passes, run the same 2-step smoke
without offload.

## Preflight

Run from the repository root:

```bash
set -o pipefail
pwd
git status --short
nvidia-smi
```

Confirm dependencies:

```bash
python - <<'PY'
import importlib.util

for name in ["torch", "diffusers", "transformers", "huggingface_hub", "vllm"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {spec.origin if spec else 'missing'}")
PY
```

Required:

- `vllm` imports successfully.
- `diffusers` imports successfully.
- `transformers` supports `Qwen3VLProcessor`.
- The machine can access Hugging Face, or the model is already present in the
  Hugging Face cache.

Optional cache setup:

```bash
export HF_HOME=${HF_HOME:-/mnt/hf_cache}
export HF_HUB_ENABLE_HF_TRANSFER=1
```

If the environment requires a token, set:

```bash
export HF_TOKEN=<token>
```

## Unit Gate

Run the Joy unit tests before model e2e:

```bash
pytest -q tests/diffusion/models/joy_image/test_joy_image.py
```

Expected:

```text
31 passed
```

Also run the focused static checks:

```bash
uvx ruff check \
  vllm_omni/diffusion/models/joy_image \
  vllm_omni/diffusion/models/qwen3_vl \
  vllm_omni/diffusion/models/internvla_a1/adapter_qwen3_vl.py \
  tests/diffusion/models/joy_image/test_joy_image.py \
  vllm_omni/diffusion/registry.py \
  vllm_omni/diffusion/model_metadata.py \
  examples/offline_inference/image_to_image/image_edit.py

git diff --check
```

Expected:

```text
All checks passed!
```

`git diff --check` should produce no output.

## Offline Smoke 1: Conservative Load And Generate

Use a small image and two denoise steps. The checked-in `docs/assets/WeChat.jpg`
is sufficient for shape/path validation.

```bash
mkdir -p /tmp/joyai_e2e

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
PYTHONPATH=/root/vllm-omni \
python examples/offline_inference/image_to_image/image_edit.py \
  --model jdopensource/JoyAI-Image-Edit-Diffusers \
  --image docs/assets/WeChat.jpg \
  --prompt "Change the background to a clean studio while preserving the subject." \
  --height 512 \
  --width 512 \
  --num-inference-steps 2 \
  --cfg-scale 4.0 \
  --output /tmp/joyai_e2e/joyai_edit_smoke_offload.png \
  --enable-cpu-offload \
  --disable-pin-cpu-memory \
  --vae-use-tiling \
  2>&1 | tee /tmp/joyai_e2e/offline_smoke_offload.log
```

On this 32 GiB RTX 4080 SUPER environment, keep
`--disable-pin-cpu-memory` enabled for model-level offload. The default pinned
CPU memory path made the dummy warmup spend many minutes in CPU-side module
movement after denoising had completed, with GPU memory already released.

Validate the output:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image

path = Path("/tmp/joyai_e2e/joyai_edit_smoke_offload.png")
assert path.exists(), f"missing output: {path}"
img = Image.open(path)
print(img.mode, img.size)
assert img.size == (512, 512), img.size
PY
```

Pass criteria:

- Process exits with code 0.
- Output file exists.
- PIL opens the image.
- Output size is `(512, 512)`.

## Offline Smoke 2: No Offload

After the conservative smoke passes, run the same case without CPU offload and
without VAE tiling. This checks whether A100 40GB can carry the v1 single-card
path directly.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
PYTHONPATH=/root/vllm-omni \
python examples/offline_inference/image_to_image/image_edit.py \
  --model jdopensource/JoyAI-Image-Edit-Diffusers \
  --image docs/assets/WeChat.jpg \
  --prompt "Change the background to a clean studio while preserving the subject." \
  --height 512 \
  --width 512 \
  --num-inference-steps 2 \
  --cfg-scale 4.0 \
  --output /tmp/joyai_e2e/joyai_edit_smoke_no_offload.png \
  2>&1 | tee /tmp/joyai_e2e/offline_smoke_no_offload.log
```

Validate:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image

path = Path("/tmp/joyai_e2e/joyai_edit_smoke_no_offload.png")
assert path.exists(), f"missing output: {path}"
img = Image.open(path)
print(img.mode, img.size)
assert img.size == (512, 512), img.size
PY
```

If this run OOMs but the offload run passed, record the OOM details and keep
the offload run as the first v1 functional smoke result.

## Offline Smoke 3: Ten-Step Sanity

Run this only after a 2-step smoke passes.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
PYTHONPATH=/root/vllm-omni \
python examples/offline_inference/image_to_image/image_edit.py \
  --model jdopensource/JoyAI-Image-Edit-Diffusers \
  --image docs/assets/WeChat.jpg \
  --prompt "Change the background to a clean studio while preserving the subject." \
  --height 512 \
  --width 512 \
  --num-inference-steps 10 \
  --cfg-scale 4.0 \
  --seed 42 \
  --output /tmp/joyai_e2e/joyai_edit_10step.png \
  2>&1 | tee /tmp/joyai_e2e/offline_10step.log
```

This is still a smoke test. Do not use it as a quality benchmark.

## Negative Smoke 1: Multi-Image Rejection

Joy v1 supports exactly one input image. This command should fail before GPU
generation proceeds.

```bash
set +e
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
PYTHONPATH=/root/vllm-omni \
python examples/offline_inference/image_to_image/image_edit.py \
  --model jdopensource/JoyAI-Image-Edit-Diffusers \
  --image docs/assets/WeChat.jpg docs/assets/WeChat.jpg \
  --prompt "Combine these images." \
  --height 512 \
  --width 512 \
  --num-inference-steps 2 \
  --cfg-scale 4.0 \
  --output /tmp/joyai_e2e/joyai_edit_should_fail_multi.png \
  2>&1 | tee /tmp/joyai_e2e/negative_multi_image.log
status=$?
set -e
test "$status" -ne 0
grep -Ei "exactly one image|multiple input images|one input image" \
  /tmp/joyai_e2e/negative_multi_image.log
```

Pass criteria:

- Command exits non-zero.
- Log clearly states that Joy v1 supports one image.

## Negative Smoke 2: Conflicting Guidance Fields

In vLLM-Omni, `--cfg-scale` maps to `true_cfg_scale`. Joy accepts
`guidance_scale` as a Diffusers compatibility alias only when `true_cfg_scale`
is absent. A non-default conflicting value should fail.

```bash
set +e
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
PYTHONPATH=/root/vllm-omni \
python examples/offline_inference/image_to_image/image_edit.py \
  --model jdopensource/JoyAI-Image-Edit-Diffusers \
  --image docs/assets/WeChat.jpg \
  --prompt "Change the background to a clean studio while preserving the subject." \
  --height 512 \
  --width 512 \
  --num-inference-steps 2 \
  --cfg-scale 4.0 \
  --guidance-scale 3.0 \
  --output /tmp/joyai_e2e/joyai_edit_should_fail_guidance.png \
  2>&1 | tee /tmp/joyai_e2e/negative_guidance_conflict.log
status=$?
set -e
test "$status" -ne 0
grep -Ei "compatibility alias|true_cfg_scale|guidance_scale" \
  /tmp/joyai_e2e/negative_guidance_conflict.log
```

Pass criteria:

- Command exits non-zero.
- Log clearly reports ambiguous/conflicting guidance semantics.

## Negative Smoke 3: Token Budget Rejection

Joy v1 caps image tokens at 4096. For the current VAE and patch configuration,
that corresponds to `(height / 16) * (width / 16) <= 4096`. An explicit
`2048x2048` request should fail during preprocessing.

```bash
set +e
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
PYTHONPATH=/root/vllm-omni \
python examples/offline_inference/image_to_image/image_edit.py \
  --model jdopensource/JoyAI-Image-Edit-Diffusers \
  --image docs/assets/WeChat.jpg \
  --prompt "Change the background to a clean studio while preserving the subject." \
  --height 2048 \
  --width 2048 \
  --num-inference-steps 2 \
  --cfg-scale 4.0 \
  --output /tmp/joyai_e2e/joyai_edit_should_fail_size.png \
  2>&1 | tee /tmp/joyai_e2e/negative_token_budget.log
status=$?
set -e
test "$status" -ne 0
grep -Ei "token budget|4096|exceed" /tmp/joyai_e2e/negative_token_budget.log
```

Pass criteria:

- Command exits non-zero.
- Log mentions the token budget or max image sequence length.

## Optional Online Smoke

Start a server:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
vllm serve jdopensource/JoyAI-Image-Edit-Diffusers \
  --omni \
  --port 8092 \
  --enable-cpu-offload \
  --vae-use-tiling \
  2>&1 | tee /tmp/joyai_e2e/online_server.log
```

In another shell:

```bash
IMG_B64=$(base64 -w0 docs/assets/WeChat.jpg)

cat > /tmp/joyai_e2e/request_chat.json <<EOF
{
  "model": "jdopensource/JoyAI-Image-Edit-Diffusers",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Change the background to a clean studio while preserving the subject."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$IMG_B64"}}
    ]
  }],
  "extra_body": {
    "height": 512,
    "width": 512,
    "num_inference_steps": 2,
    "true_cfg_scale": 4.0,
    "seed": 42
  }
}
EOF

curl -sS http://localhost:8092/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/joyai_e2e/request_chat.json \
  | tee /tmp/joyai_e2e/online_chat_response.json
```

Extract and validate the image:

```bash
python - <<'PY'
import base64
import json
from pathlib import Path

from PIL import Image

response = json.loads(Path("/tmp/joyai_e2e/online_chat_response.json").read_text())
content = response["choices"][0]["message"]["content"]
url = content[0]["image_url"]["url"]
payload = url.split(",", 1)[1]
out = Path("/tmp/joyai_e2e/online_chat_output.png")
out.write_bytes(base64.b64decode(payload))
img = Image.open(out)
print(img.mode, img.size)
assert img.size == (512, 512), img.size
PY
```

Pass criteria:

- HTTP response has one output image.
- Output image can be decoded and opened.
- Output image size is `(512, 512)`.

## Logs And Artifacts To Keep

Archive these files after the run:

```text
/tmp/joyai_e2e/offline_smoke_offload.log
/tmp/joyai_e2e/offline_smoke_no_offload.log
/tmp/joyai_e2e/offline_10step.log
/tmp/joyai_e2e/negative_multi_image.log
/tmp/joyai_e2e/negative_guidance_conflict.log
/tmp/joyai_e2e/negative_token_budget.log
/tmp/joyai_e2e/joyai_edit_smoke_offload.png
/tmp/joyai_e2e/joyai_edit_smoke_no_offload.png
/tmp/joyai_e2e/joyai_edit_10step.png
```

For failures, also collect:

```bash
nvidia-smi > /tmp/joyai_e2e/nvidia_smi_after_failure.txt
python - <<'PY' > /tmp/joyai_e2e/dependency_versions.txt
import importlib.metadata as md

for pkg in ["torch", "vllm", "diffusers", "transformers", "huggingface_hub"]:
    try:
        print(pkg, md.version(pkg))
    except md.PackageNotFoundError:
        print(pkg, "missing")
PY
```

## Final Pass Criteria

The first cloud-side e2e smoke is acceptable when all of the following are
true:

- Joy unit tests pass in the server environment.
- Focused ruff check passes.
- `git diff --check` passes.
- Offline 2-step offload smoke passes.
- Offline 2-step no-offload smoke either passes, or fails only with a recorded
  memory limitation.
- Multi-image negative smoke fails with a clear one-image error.
- Conflicting guidance negative smoke fails with a clear guidance error.
- Oversized explicit resolution negative smoke fails with a clear token budget
  error.

## Run Notes: 2026-06-05

Environment setup for the current e2e attempt:

- Repository: `/root/vllm-omni`, commit `87e84afa feat: joyai image`.
- Runtime environment: `/root/autodl-tmp/vllm-omni-venv`.
- Hugging Face cache: `/root/autodl-tmp/huggingface`.
- Model path used for runtime smoke:
  `/autodl-fs/data/joyai_models/JoyAI-Image-Edit-Diffusers`.
- e2e artifact target: `/root/autodl-tmp/joyai_e2e`.
- GPU: NVIDIA GeForce RTX 4080 SUPER, 32760 MiB.
- `huggingface.co` was not reachable from this host during the run; use
  `HF_ENDPOINT=https://hf-mirror.com` for model download and e2e commands in
  this environment.

Completed gates:

```text
pytest -q tests/diffusion/models/joy_image/test_joy_image.py
31 passed

pytest -q \
  tests/diffusion/models/glm_image/test_glm_image_sp.py \
  tests/diffusion/models/joy_image/test_joy_image.py \
  tests/diffusion/offloader/test_sequential_backend.py
38 passed, 1 skipped

ruff check vllm_omni/diffusion/models/joy_image \
  vllm_omni/diffusion/models/qwen3_vl \
  vllm_omni/diffusion/models/internvla_a1/adapter_qwen3_vl.py \
  vllm_omni/diffusion/models/glm_image/pipeline_glm_image.py \
  vllm_omni/diffusion/offloader/sequential_backend.py \
  tests/diffusion/models/joy_image/test_joy_image.py \
  tests/diffusion/models/glm_image/test_glm_image_sp.py \
  tests/diffusion/offloader/test_sequential_backend.py \
  vllm_omni/diffusion/registry.py \
  vllm_omni/diffusion/model_metadata.py \
  examples/offline_inference/image_to_image/image_edit.py
All checks passed!

git diff --check
<no output>
```

Implementation changes made during e2e debugging:

```text
- examples/offline_inference/image_to_image/image_edit.py:
  added --height/--width forwarding for explicit JoyAI output size.
- examples/offline_inference/image_to_image/image_edit.py:
  added --init-timeout and --stage-init-timeout forwarding for large model
  startup.
- examples/offline_inference/image_to_image/image_edit.py:
  added --disable-pin-cpu-memory forwarding to set pin_cpu_memory=False for
  offload runs on this 32 GiB card.
- vllm_omni/diffusion/models/joy_image/pipeline_joy_image_edit.py:
  deferred text encoder and VAE device placement when CPU/layerwise offload or
  HSDP is active.
- vllm_omni/diffusion/models/joy_image/pipeline_joy_image_edit.py:
  treated Joy VAE as a model-level offload peer of the text encoder instead of
  a permanently resident VAE.
- vllm_omni/diffusion/models/joy_image/pipeline_joy_image_edit.py:
  forwarded mm_token_type_ids into Qwen3-VL and cast processor pixel tensors to
  the encoder dtype.
- vllm_omni/diffusion/models/joy_image/pipeline_joy_image_edit.py:
  cast VAE image latents to the requested diffusion dtype and moved the VAE to
  the active device before decode when offload placement is deferred.
- vllm_omni/diffusion/models/joy_image/pipeline_joy_image_edit.py:
  skipped VAE decode for the internal dummy warmup request by returning latent
  output directly; this keeps startup warmup from spending memory and time on a
  throwaway image decode.
- vllm_omni/diffusion/models/glm_image/pipeline_glm_image.py and
  tests/diffusion/models/glm_image/test_glm_image_sp.py:
  made the existing GLM dummy-warmup request check tolerate lightweight
  request objects in the same way as Joy's dummy request helper, and removed
  the GLM test file's dependency on the external `pytest-mock` fixture.
- vllm_omni/diffusion/models/joy_image/pipeline_joy_image_edit.py:
  made the internal dummy warmup return latents after denoising instead of
  decoding an image, so JoyAI initialization does not get stuck in VAE
  decode/offload movement before serving the first real request.
- vllm_omni/diffusion/models/qwen3_vl/adapter_qwen3_vl.py and
  vllm_omni/diffusion/models/internvla_a1/adapter_qwen3_vl.py:
  adjusted Qwen3-VL mask handling needed by JoyAI text encoding.
- joyai_image_edit_e2e_guide.md:
  documented the explicit `PYTHONPATH=/root/vllm-omni` requirement for direct
  example-script execution in this environment, because invoking
  `examples/offline_inference/image_to_image/image_edit.py` directly otherwise
  puts only the example directory on `sys.path`.
- vllm_omni/diffusion/offloader/sequential_backend.py:
  made model-level CPU offload use blocking device-to-CPU copies when
  `pin_cpu_memory=False`, because non-blocking copies are only useful with
  pinned CPU memory and can leave large swaps outstanding behind Python control
  flow.
- tests/diffusion/offloader/test_sequential_backend.py:
  added coverage for non-pinned CPU offload using blocking copies.
- vllm_omni/diffusion/models/joy_image/pipeline_joy_image_edit.py:
  after the internal dummy warmup returns latent output, explicitly offloaded
  the transformer back to CPU when deferred/offload placement is active.
  Without this, skipping VAE decode left the DiT resident on GPU and forced the
  first real image-edit request to start by moving the full transformer back to
  CPU before text encoding.
- tests/diffusion/models/joy_image/test_joy_image.py:
  extended the dummy-warmup regression test to assert both that VAE decode is
  skipped and that the deferred/offload path moves the transformer to CPU
  before returning latent warmup output.
```

Latest e2e smoke results from this environment:

```text
/root/autodl-tmp/joyai_e2e/offline_smoke_offload_final.log
- Command used `/root/autodl-tmp/vllm-omni-venv/bin/python`,
  `PYTHONPATH=/root/vllm-omni`, local model path
  `/autodl-fs/data/joyai_models/JoyAI-Image-Edit-Diffusers`,
  `--enable-cpu-offload`, `--vae-use-tiling`, `--disable-pin-cpu-memory`,
  `--enforce-eager`, `--init-timeout 1200`, and `--stage-init-timeout 900`.
- Model weights loaded successfully.
- Model-level offloading initialized successfully:
  `transformer <-> text_encoder, vae`.
- Worker reached `ready to receive requests via shared memory`.
- Dummy warmup denoising reached 100% for 1/1 step.
- No output image was produced; the log stopped after the dummy warmup progress
  line and the process exited non-zero without an additional Python traceback.

/root/autodl-tmp/joyai_e2e/offline_smoke_offload_after_dummy_skip.log
- After adding the dummy-warmup decode skip, the offload run progressed past
  warmup and printed `Pipeline loaded`.
- The real 2-step request reached `Processed prompts: 0%`.
- No output image was produced at
  `/root/autodl-tmp/joyai_e2e/joyai_edit_smoke_offload_after_dummy_skip.png`.
- The log ended with Python `resource_tracker` warnings about leaked semaphore
  and shared-memory objects, without a Python traceback or generation timing.
- Because the smoke commands pipe through `tee`, use `set -o pipefail` or
  inspect `${PIPESTATUS[0]}` when collecting exit status; otherwise `tee` can
  mask the Python process failure.

/root/autodl-tmp/joyai_e2e/offline_smoke_no_offload_final.log
- No-offload run failed during model construction with CUDA OOM on the
  32 GiB RTX 4080 SUPER:
  `Tried to allocate 128.00 MiB`, with only `30.62 MiB` free and about
  `31.44 GiB` already in use by the process.

/root/autodl-tmp/joyai_e2e/offline_smoke_offload_after_dummy_offload.log
- Command used model-level CPU offload, VAE tiling, disabled pinned CPU memory,
  eager execution, the local model path, and the new dummy latent-return
  transformer offload fix.
- Dummy warmup completed and the worker initialized successfully.
- The real request still did not produce an output image. The log stopped at
  `Processed prompts: 0%`, then the process exited without a Python traceback
  and without creating
  `/root/autodl-tmp/joyai_e2e/joyai_edit_smoke_offload_after_dummy_offload.png`.
- Runtime observation while it was stuck: the diffusion subprocess used about
  99% CPU, GPU memory dropped to about 354 MiB, and `/proc/<pid>/io` showed
  tens of GiB of shared-memory/page reads. This indicates the remaining
  failure is still in CPU/offload/shared-memory movement before real denoising,
  not in VAE decode.

/root/autodl-tmp/joyai_e2e/offline_smoke_layerwise.log
- Command used `--enable-layerwise-offload`, VAE tiling, eager execution, and
  the local model path.
- The process loaded the initial 750 weights, but did not reach the layerwise
  backend initialization log or produce an output image.
- After about 11 minutes, the log had not advanced beyond the initial
  `Loading weights: 100%` line; the subprocess was in `D` state with no image
  output. The run was terminated and
  `/root/autodl-tmp/joyai_e2e/joyai_edit_smoke_layerwise.png` was not created.
```

Current e2e status:

```text
The 32 GiB server environment is not yet an acceptable first JoyAI e2e pass.
No-offload is memory-limited on this GPU. Model-level CPU offload reaches model
load, offload setup, worker readiness, dummy denoising, orchestrator readiness,
and the real request submission path, but the real 2-step edit exits before
producing an image. Layerwise offload also did not complete initialization on
this host. No valid JoyAI output image has been produced in this environment.
```

## Run Notes: 2026-06-06 H800 No-Offload

Environment setup for the H800 e2e attempt:

- Repository: `/root/vllm-omni`, commit `87e84afa feat: joyai image`.
- Runtime environment: `/root/autodl-tmp/vllm-omni-venv`.
- Hugging Face cache: `/root/autodl-tmp/huggingface`.
- Model path used for runtime smoke:
  `/autodl-fs/data/joyai_models/JoyAI-Image-Edit-Diffusers`.
- e2e artifact target: `/root/autodl-tmp/joyai_e2e_h800`.
- GPU: NVIDIA H800 PCIe, `81559 MiB`, driver `580.82.07`, CUDA `13.0`.
- System memory at preflight: `MemTotal: 1056444608 kB`,
  `MemAvailable: 979008556 kB`.
- No GPU processes were running before the H800 smoke, and `nvidia-smi` showed
  no GPU processes after the runs.
- The tested model I/O is image editing: one input image plus one text edit
  instruction, producing one edited image. The smoke input was
  `docs/assets/WeChat.jpg`; the prompt was
  `Change the background to a clean studio while preserving the subject.`;
  the validated output size was `(512, 512)`.
- CPU offload and layerwise offload were intentionally not used on H800. The
  previous 48 GiB/32 GiB attempts showed that CPU offload can move tens of GiB
  of model state through CPU RAM and pin/copy paths. For the H800 run, the
  model was kept resident on GPU to avoid that CPU memory pressure.

H800 no-offload resource observations:

```text
/root/autodl-tmp/joyai_e2e_h800/offline_smoke_no_offload.log
- Model loading took 46.8600 GiB and 168.755791 seconds on the first H800 run.
- Process-scoped GPU memory after model loading: 47.40 GiB.
- Dummy warmup completed successfully.
- The real 2-step request completed successfully.
- Total generation time: 1.1050 seconds (1105.03 ms).
- Output image:
  /root/autodl-tmp/joyai_e2e_h800/joyai_edit_smoke_no_offload.png.
- PIL validation: RGB (512, 512).
```

H800 ten-step sanity:

```text
/root/autodl-tmp/joyai_e2e_h800/offline_10step.log
- Command used no CPU offload, no layerwise offload, local model path,
  `--height 512`, `--width 512`, `--num-inference-steps 10`,
  `--cfg-scale 4.0`, `--seed 42`, `--enforce-eager`,
  `--init-timeout 1200`, and `--stage-init-timeout 900`.
- Model loading took 46.8600 GiB and 14.824385 seconds with warm cache.
- Process-scoped GPU memory after model loading: 47.40 GiB.
- Dummy warmup completed successfully.
- The real 10-step request completed successfully.
- Total generation time: 4.3149 seconds (4314.92 ms).
- Output image:
  /root/autodl-tmp/joyai_e2e_h800/joyai_edit_10step.png.
- PIL validation: RGB (512, 512).
```

H800 negative smoke results:

```text
/root/autodl-tmp/joyai_e2e_h800/negative_multi_image.log
- Exit status recorded in
  /root/autodl-tmp/joyai_e2e_h800/negative_multi_image.status:
  PYTHON_STATUS=1.
- The request failed clearly with:
  `Received multiple input images. JoyAI-Image-Edit v1 supports exactly one image.`

/root/autodl-tmp/joyai_e2e_h800/negative_guidance_conflict.log
- Exit status recorded in
  /root/autodl-tmp/joyai_e2e_h800/negative_guidance_conflict.status:
  PYTHON_STATUS=1.
- The request failed clearly with:
  `JoyAI-Image-Edit treats guidance_scale as a Diffusers compatibility alias
  for true_cfg_scale. Provide only one value, or provide matching values.`

/root/autodl-tmp/joyai_e2e_h800/negative_token_budget.log
- Exit status recorded in
  /root/autodl-tmp/joyai_e2e_h800/negative_token_budget.status:
  PYTHON_STATUS=1.
- The request failed clearly with:
  `height and width exceed JoyAI-Image-Edit token budget:
  (2048 / 16) * (2048 / 16) = 16384, max 4096.`
```

Current H800 e2e status:

```text
The H800 no-offload path is an acceptable functional JoyAI e2e smoke for this
environment: the 2-step image-edit request produced a valid 512x512 PNG, the
10-step sanity request produced a valid 512x512 PNG, and all three negative
smokes failed with clear errors. CPU-offload validation remains intentionally
out of scope for this H800 rerun because the operator constraint was to avoid
loading the model into CPU memory.
```
