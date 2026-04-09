**Preferred Test Strategy**

Use one of the following patterns depending on page type:

- **Dynamic code-block extraction (preferred for offline docs)**
    - Extract Python/Bash code blocks from markdown AST analyzer, then execute them directly in tests.
    - Benefit: test logic stays automatically aligned with docs.
    - Basic idea: Use `ReadmeSnippet.extract_readme_snippets` to extract a list of code blocks as a global variable in file,
    use this list as `pytest.mark.parametrize` parameters, and pass each snippet item to `example_runner.run` inside the parametrized test.
    Additionally pass an `output_subfolder` argument for the 2nd-level output folder explained in **Output Directory Structure** below.
    If any extra environment variable is need for a test (e.g., the example script reads it), `example_runner.run` also accepts a 3rd `env` parameter.
    - See [tests/examples/offline_inference/test_text_to_image.py](https://github.com/vllm-project/vllm-omni/blob/main/tests/examples/offline_inference/test_text_to_image.py) for reference implementation.

- **Explicit copied scripts (used by online docs for now until further update)**
    - For online serving pages, it is acceptable to copy code from docs into dedicated test functions, because only client-side, request-sending scripts are tested.
    - Benefit: dynamic extraction is overly complex: need to tell server-launch and client-request scripts.
    - Requirement: copied test code must be kept in sync with doc updates.

**Test Case Naming Convention**

- Dynamic code extraction (auto-generated internally):
    - `test_{single_function_name_matching_file_name}[h2_heading_00X]`
    - Example: `test_text_to_image[basic_usage_001]`
- Explicit copied scripts:
    - `test_{h2_heading_00X}[{dummy_param_id_for_omni_server}]`
    - Example: `test_api_calls_001[omni_server0]`

**Runtime Configuration**

In the example code tests, do **not** reduce `num_inference_steps` just to speed up the tests  unless there is a strong CI reliability reason to do otherwise.

**Skipping Rules**

You may skip examples falling in the following categories using `pytest.mark.skip` or `pytest.skip`:

- Gradio UI scripts
- Scenarios that significantly overlap with existing tests and add little new coverage.

**Output Directory Structure**

Use a three-layer output structure to store output artifacts:

1. Root output directory
    - Auto-detected from `OUTPUT_DIR` env var or auto-generated under `/tmp`.
2. Doc-page directory
    - Define and use a clear page-level folder name in each `test_*.py` yourself (abbreviations are acceptable, e.g., `example_offline_t2i`).
3. Test-case directory
    - Must match the case identifier (e.g., `basic_usage_001`).
    - Auto-generated for dynamic extracted tests.
