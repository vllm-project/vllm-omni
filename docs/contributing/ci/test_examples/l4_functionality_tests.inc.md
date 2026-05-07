**Scope**

For diffusion models, the L4 functionality test covers all or common *diffusion features* that are supported by this model, including several [parallelism acceleration methods](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/parallelism_acceleration/), [CPU offloading](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cpu_offload_diffusion/), [TeaCache](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/teacache/) and [Cache-DiT](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_dit_acceleration/) cache backends, [quantization methods](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/quantization/overview/).

**Test Case Design**

For a *high priority* model (currently listed in [issue #1832](https://github.com/vllm-project/vllm-omni/issues/1832)), we use several test cases, each with multiple features turned on, so that each supported feature is tested in at least one test case. This is to relieve the GPU workload on the CI machine. The suggested test case combination is as follows:

- If the model can fit into 4 L4 GPU (with quantization and tensor parallel always on) (20GB RAM each)
    - (1 GPU) TeaCache + Layerwise CPU offloading + GGUF
    - (4 GPUs) CacheDiT + Ulysses=2 + TP=2 + VAE=2 + FP8
    - (4 GPUs) CacheDiT + Ring=2 + HSDP=2 + VAE=2 + GGUF
    - (4 GPUs) TeaCache + CFG=2 + TP=2 + VAE=2 + FP8
- Otherwise, consider 2 H100 GPU environment (80GB RAM each) with the following tests
    - (1 GPU) TeaCache + Layerwise CPU offloading + GGUF
    - (2 GPUs) CacheDiT + Ulysses=2 + FP8
    - (2 GPUs) CacheDiT + Ring=2 + GGUF
    - (2 GPUs) TeaCache + CFG=2 + FP8
    - (2 GPUs) CacheDiT + TP=2 + VAE=2 + FP8
    - (2 GPUs) CacheDiT + HSDP=2 + VAE=2 + GGUF
- If 2 H100 GPU cannot handle the model either (e.g., HunyuanImage 3.0)
    - Still design tests and feature combinations that can best fit real-world scenario.
    - Do not include it in CI (or exclude it from the CI's filtering criteria). Instead, relevant PR authors are suggested to run these tests locally.
- Fallback plan
    - If the model does not support layerwise CPU offloading, replace the corresponding test case with module-wise offloading
    - If the model only supports specific or no caching feature, use this option or remove this option in all test cases.
    - If the model only supports specific or no quantization feature, use this option or remove this option in all test cases.
    - If the model does not support certain other features, remove this option from that test case. If, consequently, the coverage of this modified test case completely overlaps with others, remove this test case.

For a *normal priority* model, further reduce the number of test cases.

- Only write one or two test cases for the most common feature combinations.
- The author can explore themselves to see which feature combination balances output quality and performance. Alternatively, the author can refer to any example code in the PR that adds the model, or the example code in the PR that adds a feature (if the code involves this model of interest).

Currently all the features are available in online serving mode. Hence, only need to add `tests/e2e/online_serving/test_{model}_expansion.py`.

**Code Style**

- Validation: test that the multimodal output files of your model have the correct shapes. `OpenAIClientHandler.send_diffusion_request` should have taken care of this.
- Test marks: always add `full_model` and `diffusion` for L4 nightly `test_*_expansion.py` cases. Add GPU-related marks if needed. Ref: [Markers for Tests](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/ci/tests_markers/).
- To maximize code reuse, you may refer to
    - `tests/conftest.py` for `omni_server` (running server in subprocess) and `openai_client` fixtures (sending requests and validating output), `generate_synthetic_image` and `assert_XXX_valid` helper.
    - `tests/helpers/mark.py` for `@hardware_test(...)` and `hardware_marks`.
    - [Parametrizing tests (pytest doc)](https://docs.pytest.org/en/stable/example/parametrize.html) to reuse test function implementation for different cases.
- Doc: add a concise docstring for each test function.
- Reference L4 test implementation: [tests/e2e/online_serving/test_qwen_image_edit_expansion.py](https://github.com/vllm-project/vllm-omni/blob/main/tests/e2e/online_serving/test_qwen_image_edit_expansion.py).
