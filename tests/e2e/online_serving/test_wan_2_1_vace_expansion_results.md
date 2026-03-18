## Purpose

Add L4 e2e online serving tests for Wan2.1-VACE, covering all diffusion acceleration features
required by the [Model×Feature RFC (#1832)](https://github.com/vllm-project/vllm-omni/issues/1832).

Uses the 1.3B variant (`Wan-AI/Wan2.1-VACE-1.3B-diffusers`) for faster CI testing.

## Test Plan

| Test ID | GPUs | Features |
|---------|------|----------|
| `single_card_001` | 1 | Cache-DiT + layerwise CPU offload |
| `parallel_001` | 2 | Cache-DiT + Ulysses-SP = 2 |
| `parallel_002` | 2 | Cache-DiT + Ring = 2 |
| `parallel_003` | 2 | Cache-DiT + CFG-Parallel = 2 |
| `parallel_004` | 2 | Cache-DiT + TP = 2 + VAE-Patch-Parallel = 2 |
| `parallel_005` | 2 | Cache-DiT + HSDP shard = 2 + VAE-Patch-Parallel = 2 |

> **Note:** TeaCache is not supported for Wan2.1-VACE; Cache-DiT is used in its place.

## Test Result

### Collection check

```
$ pytest tests/e2e/online_serving/test_wan_2_1_vace_expansion.py --collect-only -m "diffusion and advanced_model"

collected 6 items

  <Function test_wan_2_1_vace[single_card_001]>
  <Function test_wan_2_1_vace[parallel_001]>
  <Function test_wan_2_1_vace[parallel_002]>
  <Function test_wan_2_1_vace[parallel_003]>
  <Function test_wan_2_1_vace[parallel_004]>
  <Function test_wan_2_1_vace[parallel_005]>

========================== 6 tests collected in 0.01s ==========================
```

### Full L4 test run (2× H200)

```
$ CUDA_VISIBLE_DEVICES=0,1 pytest tests/e2e/online_serving/test_wan_2_1_vace_expansion.py -v

tests/e2e/online_serving/test_wan_2_1_vace_expansion.py::test_wan_2_1_vace[single_card_001] PASSED [ 16%]
tests/e2e/online_serving/test_wan_2_1_vace_expansion.py::test_wan_2_1_vace[parallel_001] PASSED [ 33%]
tests/e2e/online_serving/test_wan_2_1_vace_expansion.py::test_wan_2_1_vace[parallel_002] PASSED [ 50%]
tests/e2e/online_serving/test_wan_2_1_vace_expansion.py::test_wan_2_1_vace[parallel_003] PASSED [ 66%]
tests/e2e/online_serving/test_wan_2_1_vace_expansion.py::test_wan_2_1_vace[parallel_004] PASSED [ 83%]
tests/e2e/online_serving/test_wan_2_1_vace_expansion.py::test_wan_2_1_vace[parallel_005] PASSED [100%]

================== 6 passed, 17 warnings in 313.23s (0:05:13) ==================
```
