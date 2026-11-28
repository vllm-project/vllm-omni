# SAIL PPU Ring attention

vLLM-Omni can use the SAIL SDK's FlashAttention 3 (FA3) kernel for diffusion
Ring attention. Ring combines each K/V block's attention output and softmax
log-sum-exp across ranks.

## Requirements

- A [SAIL environment](https://github.com/vllm-project/vllm-sail/blob/main/docs/getting_started/installation.md)
  with the vLLM SAIL plugin active and the matching PPU FA3 package installed.
- A diffusion model configured for [Ring sequence parallelism](sequence_parallel.md).
  Set `ring_degree > 1` and select the `FLASH_ATTN` backend.

## Backend selection

On SAIL PPU, Ring prefers the SDK's FA3 kernel when its forward interface is
available. For SDK builds that expose `window_size`, vLLM-Omni passes a tuple
of the left and right window bounds. Builds with separate window arguments
use `window_size_left` and `window_size_right`.

## Validation

Run the SAIL routing and interface tests in a vLLM-Omni development
environment. These tests use mocks and run on CPU.

```bash
python -m pytest -q tests/platforms/sail/test_ring_fa3.py -m "core_model and cpu"
```
