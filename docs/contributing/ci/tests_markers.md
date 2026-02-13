# Markers for Tests

By adding markers before test functions, tests can later be executed uniformly by simply declaring the corresponding marker type.

## Current Markers
Defined in `pyproject.toml`:

| Marker             | Description                                               |
| ------------------ | --------------------------------------------------------- |
| `core_model`       | L1&L2 tests (run in each PR)                              |
| `advanced_model`   | L3&L4 level tests (run in each merge or nightly)          |
| `diffusion`        | Diffusion model tests                                     |
| `omni`             | Omni model tests                                          |
| `cache`            | Cache backend tests                                       |
| `parallel`         | Parallelism/distributed tests                             |
| `cpu`              | Tests that run on CPU                                     |
| `gpu`              | Tests that run on GPU *                                   |
| `cuda`             | Tests that run on CUDA *                                  |
| `rocm`             | Tests that run on AMD/ROCm *                              |
| `npu`              | Tests that run on NPU/Ascend *                            |
| `H100`             | Tests that require H100 GPU  *                            |
| `L4`               | Tests that require L4 GPU *                               |
| `MI325`            | Tests that require MI325 GPU (AMD/ROCm) *                 |
| `A2`               | Tests that require A2 NPU *                               |
| `A3`               | Tests that require A3 NPU *                               |
| `distributed_cuda` | Tests that require multi cards on CUDA platform *         |
| `distributed_rocm` | Tests that require multi cards on ROCm platform  *        |
| `distributed_npu`  | Tests that require multi cards on NPU platform  *         |
| `skipif_cuda`      | Skip if the num of CUDA cards is less than the required * |
| `skipif_rocm`      | Skip if the num of ROCm cards is less than the required * |
| `skipif_npu`       | Skip if the num of NPU cards is less than the required *  |
| `slow`             | Slow tests (may skip in quick CI)                         |
| `benchmark`        | Benchmark tests                                           |

\* Means those markers are auto-added, and they will be added by the `@hardware_test` decorator.

### Example usage for markers

```python
from tests.utils import hardware_test

@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(
   res={"cuda": "L4", "rocm": "MI325", "npu": "A2"},
   num_cards=2,
)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_audio()
    ...
```
### Decorator: `@hardware_test`

This decorator is intended to make hardware-aware, cross-platform test authoring easier and more robust for CI/CD environments. The `hardware_test` decorator in `vllm-omni/tests/utils.py` performs the following actions:

1. **Applies platform and resource markers**  
   Adds the appropriate pytest markers for each specified hardware platform (e.g., `cuda`, `rocm`, `npu`) and resource type (e.g., `L4`, `H100`, `MI325`, `A2`, `A3`).
   ```
   @pytest.mark.cuda
   @pytest.mark.L4
   ```
2. **Handles multi-card (distributed) scenarios**  
   For tests requiring multiple cards, it automatically adds distributed markers such as `distributed_cuda`, `distributed_rocm`, or `distributed_npu`.
   ```
   @pytest.mark.distributed_cuda(num_cards=num_cards)
   ```
3. **Supports flexible card requirements**  
   Accepts `num_cards` as either a single integer for all platforms or as a dictionary with per-platform values. If not specified, defaults to 1 card per platform.

4. **Integrates resource validation**  
   On CUDA, adds a skip marker (`skipif_cuda`) if the system does not have the required number of devices.
   Support for `skipif_rocm` and `skipif_npu` will be implemented later.


5. **Works with pytest filtering**  
   Allows tests to be filtered and selected at runtime using standard pytest marker expressions (e.g., `-m "distributed_cuda and L4"`).

#### Example usage for decorator
- Single call for multiple platforms:
    ```python
    @hardware_test(
        res={"cuda": "L4", "rocm": "MI325", "npu": "A2"},
        num_cards={"cuda": 2, "rocm": 2, "npu": 2},
    )
    ```
    or
    ```python
    @hardware_test(
        res={"cuda": "L4", "rocm": "MI325", "npu": "A2"},
        num_cards=2,
    )
    ```
- `res` must be a dict; supported resources: CUDA (L4/H100), ROCm (MI325), NPU (A2/A3)
- `num_cards` can be int (all platforms) or dict (per platform); defaults to 1 when missing
- Distributed markers (`distributed_cuda`, `distributed_rocm`, `distributed_npu`) are auto-added for multi-card cases
- Filtering examples:
    - CUDA only: `pytest -m "distributed_cuda and L4"`
    - ROCm only: `pytest -m "distributed_rocm and MI325"`
    - NPU only: `pytest -m "distributed_npu"`

## Add Support for a New Platform

If you want to add support for a new platform (e.g., "tpu" for a new accelerator), follow these steps:

1. **Extend the marker list in your pytest config** so that platform/resource markers are defined:
   ```toml
   # In pyproject.toml or pytest.ini
   [tool.pytest.ini_options]
   markers = [
       # ... existing markers ...
       "tpu: Tests that require TPU device",
       "TPU_V3: Tests that require TPU v3 hardware",
       "distributed_tpu: Tests that require multiple TPU devices",
   ]
   ```
2. **Implement a marker construction function for your platform** in `vllm-omni/tests/utils.py`:
   ```python
   # In vllm-omni/tests/utils.py

   def tpu_marks(*, res: str, num_cards: int):
       test_platform = pytest.mark.tpu
       if res == "TPU_V3":
           test_resource = pytest.mark.TPU_V3
       else:
           raise ValueError(
               f"Invalid TPU resource type: {res}. Supported: TPU_V3")

       if num_cards == 1:
           return [test_platform, test_resource]
       else:
           test_distributed = pytest.mark.distributed_tpu(num_cards=num_cards)
           # Optionally: add skipif_tpu when implemented
           return [test_platform, test_resource, test_distributed]
   ```
3. **Update `hardware_test` to recognize your new platform**:
    In the relevant place (see the `hardware_test` implementation), add:
    ```python
    if platform == "tpu":
        marks = tpu_marks(res=resource, num_cards=cards)
    ```
4. **(Recommended) Add a test using your new markers**:
   ```python
   @hardware_test(
       res={"tpu": "TPU_V3"},
       num_cards=2,
   )
   def test_my_tpu_feature():
       ...
   ```

**Summary**:  
- Add pytest markers for your new platform/resources  
- Implement a marker function (`xxx_marks`)  
- Plug into `hardware_test`  
- You're done: tests decorated with `@hardware_test` using your platform now automatically get the correct markers, distribution, and isolation!

See code in `vllm-omni/tests/utils.py` for existing examples (`cuda_marks`, `rocm_marks`, `npu_marks`).
