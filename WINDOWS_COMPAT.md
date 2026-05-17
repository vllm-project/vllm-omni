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

## Validation

- `python -m py_compile` passes for all modified vLLM-Omni files.
- Direct source import against the local installed vLLM wheel is currently blocked by a vLLM/vLLM-Omni main-branch API mismatch (`split_routed_experts`), so this branch still needs a matched source build / wheel CI pass.

## Runtime Evidence

These changes mirror the monkeypatches that allowed native Windows vLLM-Omni to initialize `Qwen/Qwen2.5-Omni-3B` Stage 0, Stage 1, and Stage 2 and complete embedded text plus text-to-audio generation without an HTTP server.
