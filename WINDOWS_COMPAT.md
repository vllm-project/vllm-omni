# Windows Compatibility Work

This fork tracks native Windows fixes needed by OmniChat's embedded vLLM-Omni experiments.

## Current Branch

Branch: `windows-compat`

## Changes

- Replace POSIX-only top-level `fcntl` imports with platform-gated locking.
- Move shared-memory lock files from hardcoded `/dev/shm` to `%TEMP%\vllm_omni_shm` on Windows.
- Move device initialization lock files from hardcoded `/tmp` to `%TEMP%\vllm_omni_locks` on Windows.
- Avoid registering `multiprocessing.Process.sentinel` handles in `zmq.Poller` on Windows.
- Set `asyncio.WindowsSelectorEventLoopPolicy()` before creating the vLLM-Omni orchestrator loop on Windows, because pyzmq async sockets require `add_reader`.
- Apply the same sentinel polling fix to diffusion stage handshake code.
- Preserve non-final `OmniRequestOutput` chunks so async audio/text streams surface before terminal completion.
- Keep abort-after-shutdown idempotent for `janus` queues used by `AsyncOmni`.
- Support Windows shared-memory chunk transfer through a file-backed payload fallback.
- Add Windows-compatible Qwen3-Omni single-GPU smoke deploy:
  `vllm_omni/deploy/qwen3_omni_moe_windows_single_gpu.yaml`.
- Add a Windows Qwen3-Omni audio-in/audio-out streaming probe:
  `examples/offline_inference/qwen3_omni/windows_audio_stream_probe.py`.

## Start From Scratch

These notes assume PowerShell on Windows, Visual Studio 2022 Build Tools, CUDA
13, and an NVIDIA GPU with enough VRAM for the selected model. The full
Qwen3-Omni audio-in/audio-out smoke below was validated on an RTX PRO 6000
Blackwell-class GPU.

1. Create the shared venv:

```powershell
py -3.12 -m venv C:\tmp\vllmvenv
C:\tmp\vllmvenv\Scripts\python.exe -m pip install -U pip setuptools wheel ninja cmake
```

2. Clone and build the Windows vLLM fork first:

```powershell
cd C:\Users\ericl\Documents\ai-agents\Claude
git clone -b windows-compat https://github.com/ericleigh007/vllm-windows.git
cd vllm-windows

$env:CUDA_HOME = "C:\tmp\cuda13"
$env:CUDA_PATH = "C:\tmp\cuda13"
$env:CUDACXX = "C:\tmp\cuda13\bin\nvcc.exe"
$env:VLLM_TARGET_DEVICE = "cuda"
$env:MAX_JOBS = "4"
$env:NVCC_THREADS = "1"
$env:FETCHCONTENT_BASE_DIR = "C:\tmp\vllm_deps"
$env:CMAKE_ARGS = "-DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler"

C:\tmp\vllmvenv\Scripts\python.exe setup.py build_ext --inplace
C:\tmp\vllmvenv\Scripts\python.exe -m pip install -e .
```

Use the CUDA install path appropriate for your machine. On systems where CUDA is
installed under `C:\Program Files\...`, create a junction such as
`C:\tmp\cuda13` to avoid spaces in toolchain paths.

3. Clone and install the Windows vLLM-Omni fork:

```powershell
cd C:\Users\ericl\Documents\ai-agents\Claude
git clone -b windows-compat https://github.com/ericleigh007/vllm-omni-windows.git
cd vllm-omni-windows
C:\tmp\vllmvenv\Scripts\python.exe -m pip install -e .
```

4. Use these runtime variables for local source runs:

```powershell
$env:PATH = "C:\tmp\cuda13\bin;C:\tmp\vllmvenv\Lib\site-packages\torch\lib;" + $env:PATH
$env:CUDA_HOME = "C:\tmp\cuda13"
$env:CUDA_PATH = "C:\tmp\cuda13"
$env:PYTHONPATH = "C:\Users\ericl\Documents\ai-agents\Claude\vllm-omni-windows"
$env:HF_HUB_DISABLE_SYMLINKS = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

The Hugging Face symlink variables avoid `WinError 1314` on Windows machines
without Developer Mode or Administrator symlink privileges.

## Smoke Tests

Run focused unit coverage first:

```powershell
C:\tmp\vllmvenv\Scripts\python.exe -m pytest `
  tests/distributed/omni_connectors/test_chunk_transfer_adapter.py `
  tests/engine/test_async_omni_engine_outputs.py
```

Then run the full Qwen3-Omni streaming smoke. It exercises audio input,
thinker-to-talker async chunk transfer, code2wav output, and stops after the
first streamed audio tensor without writing a WAV:

```powershell
C:\tmp\vllmvenv\Scripts\python.exe `
  examples/offline_inference/qwen3_omni/windows_audio_stream_probe.py `
  --stop-after-audio-chunks 1 `
  --out windows_qwen3_omni_audio_stream_probe.json
```

On the validated workstation this produced `ok: true`, recognized the built-in
`mary_had_lamb` audio input, and emitted one stage-2 audio tensor with shape
`[1, 47445]`.

For Blackwell Windows systems, keep
`mm_encoder_attn_backend: TORCH_SDPA` in the Windows Qwen3-Omni deploy config.
FlashAttention for the multimodal encoder hit `cudaErrorUnsupportedPtxVersion`
in local testing.

## Validation

- Focused unit suite passes under `C:\tmp\vllmvenv`.
- `python -m py_compile` passes for modified vLLM-Omni files.
- Qwen3-TTS CustomVoice streams non-final audio chunks under Windows.
- Qwen3-Omni thinker mode streams image-to-text and audio-to-text under Windows.
- Qwen3-Omni full 3-stage audio-in/audio-out emits streamed stage-2 audio on
  Windows with the single-GPU deploy config above.

## Runtime Evidence

These changes now go beyond the earlier runtime monkeypatches: native Windows
vLLM-Omni can initialize Qwen3-TTS and Qwen3-Omni from source, preserve
non-final async outputs, and stream audio tensors directly from `AsyncOmni`
without writing an output WAV.
