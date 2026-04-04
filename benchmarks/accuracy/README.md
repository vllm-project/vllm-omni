# Accuracy Benchmarks

This directory hosts accuracy benchmark integrations that run entirely through a
local `vllm-omni serve` deployment.

Current integrations:

- `text_to_image/`: GEBench generation + local judge scoring flow.
- `image_to_image/`: GEdit-Bench generation + local VIEScore-style scoring flow.

Design notes:

- Generation is executed through the OpenAI-compatible endpoints exposed by
  `vllm-omni serve`.
- Evaluation is also executed through a local OpenAI-compatible judge model
  served by `vllm-omni`.
- Both generation and judge requests accept either `http://host:port` or
  `http://host:port/v1`.
- Output directory layout intentionally stays close to the upstream repos.

Test guidance:

- Local static/self-checks live in `tests/benchmarks/test_accuracy_bench_utils.py`.
- End-to-end generation/evaluation should be validated in a remote GPU
  environment. In the current repo marker system there is `L4` but no `L5`
  marker, so benchmark smoke tests should be wired as `advanced_model +
  benchmark + L4` when GPU capacity is available.
