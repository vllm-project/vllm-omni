# LingBot-VLA 2.0 Benchmarks

This manual benchmark compares direct-policy latency of the LeRobot LingBot-VLA
2.0 implementation with vLLM-Omni on the same device and runtime. Run it from
the repository root. It does not run in CI.

| Benchmark | Script | Purpose |
| --- | --- | --- |
| Direct-policy latency | `bench_policy.py` | Check parity, then time model-only and processor-plus-model inference for both implementations |

## Requirements

- One CUDA GPU with room for two FP32 copies of the policy (about 50 GB).
- The LeRobot LingBot implementation installed in the same environment, pinned
  to `huggingface/lerobot@f58f7fb9dbb241304df96e967aea0f5b42ff9a4d`. Its Qwen3-VL
  code calls vision methods that Transformers removed in 5.17, so install
  `transformers==5.16.1`.
- A local `lerobot/lingbot_vla_v2_robotwin` snapshot at revision
  `54f79a4b6e9166bda91b5010088097ff3191bca2`, and a local directory holding the
  `Qwen/Qwen3-VL-4B-Instruct` `config.json`.

## Run

```bash
python benchmarks/lingbot_vla_v2/bench_policy.py \
  --model ./lingbot-robotwin --backbone-config ./qwen3-vl-config \
  --output lingbot-policy-benchmark.json
```

The script first checks that both implementations produce identical model
inputs and matching robot actions (`atol=1e-4`, `rtol=1e-5`) on a fixed
three-camera observation. It then times each scope with fixed noise, FP32,
batch 1, ten flow steps and 50-action chunks: three warmups per path, followed
by ten interleaved ABBA blocks. Matmul TF32 is off and cuDNN TF32 is on.
Loading, networking and simulation are excluded.

## Output

The JSON file records, for `model_only` and `processor_and_model`:

- `samples_ms` and `median_ms` for each implementation;
- `paired_ratio_median` and `paired_ratio_range`, the per-block LeRobot to
  vLLM-Omni time ratio.

It also records the parity error, the device name, the model path, the
vLLM-Omni, PyTorch and Transformers versions and the precision settings.
