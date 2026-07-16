When you want to add L4-level ***performance test*** cases, add entries to JSON files under `tests/dfx/perf/tests/` and run them via `tests/dfx/perf/scripts/run_benchmark.py` (omni / TTS) or `run_diffusion_benchmark.py` (diffusion).

**Config file layout (in-tree examples)**

| Model type | Runner | Example JSON files |
| ---------- | ------ | ------------------ |
| Omni | `run_benchmark.py` | `test_qwen3_omni_no_async_chunk.json`, `test_qwen3_omni_async_chunk.json`, `test_qwen3_omni_vllm_text.json`, `test_qwen3_omni_multi_replicas.json` |
| TTS | `run_benchmark.py` | `test_tts.json`, `test_voxcpm2.json`, `test_higgs_audio_v3.json` |
| Diffusion | `run_diffusion_benchmark.py` | `test_qwen_image_vllm_omni.json`, `test_bagel_vllm_omni.json`, `test_wan22_i2v_vllm_omni.json`, `test_cosmos3_vllm_omni.json`, … |

**How runners pick cases**

Without **`--test-config-file`**, each runner scans all `*.json` under `tests/dfx/perf/tests/` but only keeps its own model type:

- **`run_benchmark.py`**: omni and TTS cases only (skips diffusion JSON).
- **`run_diffusion_benchmark.py`**: diffusion cases only (skips omni / TTS JSON).

Diffusion cases are detected when the JSON has `server_type` (typically `"vllm-omni"`) or `"diffusion"` in the `mark` array. Omni / TTS JSON has neither.

**Running perf cases**

Pass **`--test-config-file`** to run one JSON file as-is, or omit it for the bulk scan above and filter with pytest **`-m`** on each case's JSON `mark` (for example `pytest … run_benchmark.py -m "full_model and tts and H100"`).

```JSON
{
    "test_name": "test_qwen3_omni_async_chunk",
    "mark": [
        {"hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 2}},
        "full_model",
        "omni"
    ],
    "server_params": {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "stage_config_name": "qwen3_omni.yaml"
    },
    "benchmark_params": [
        {
            "name": "random_10p",
            "dataset_name": "random",
            "num_prompts": [10, 20],
            "max_concurrency": [1, 4],
            "random_input_len": 2500,
            "random_output_len": 900,
            "ignore_eos": true,
            "percentile-metrics": "ttft,tpot,itl,e2el,audio_rtf,audio_ttfp,audio_duration",
            "baseline": {
                "mean_ttft_ms": [500, 800],
                "mean_audio_ttfp_ms": [2000, 3500],
                "mean_audio_rtf": [0.25, 0.35]
            }
        }
    ]
}
```

**Parameter Explanation**

*Overview*

| Field            | Required | Description                                                     |
| ---------------- | -------- | --------------------------------------------------------------- |
| test_name        | Yes      | Unique identifier for the test case                             |
| mark             | No       | Pytest marks for this case (see **`mark` field** below). Omit only for configs not meant to be filtered by `-m`. |
| server_params    | Yes      | Server-side configuration parameters                            |
| benchmark_params | Yes      | Benchmark running parameters (supports multiple configurations) |
| server_type      | Diffusion only | When set (e.g. `"vllm-omni"`), routes the case to `run_diffusion_benchmark.py`. |
| benchmark_endpoint | Diffusion only | API path for the benchmark client (e.g. `/v1/videos`, `/v1/images/generations`). |

**`mark` field**

Optional top-level field on each perf JSON **case object** (one per `test_name`). `run_benchmark.py` and `run_diffusion_benchmark.py` read it via `tests.dfx.conftest.resolve_pytest_marks` and attach the marks to the corresponding `pytest.param`, so you can filter locally with `-m`.

When `mark` is present, it must be an **array** with exactly one ``hardware_marks`` object (same shape as `@hardware_test` / `hardware_marks()` in `tests/helpers/mark.py`), followed by registered pytest marker name strings such as `full_model`, `omni`, `tts`, `diffusion`, or `local_model`.

Supported form:

| Form | Example | Effect |
| ---- | ------- | ------ |
| Array (single platform) | `"mark": [{"hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 2}}, "full_model", "diffusion"]` | Hardware marks from `hardware_marks(...)`; `num_cards` > 1 adds `distributed_*` (+ CUDA `skipif_cuda` when applicable). String entries become extra pytest marks. |
| Array (multi-platform) | `"mark": [{"hardware_marks": {"res": {"cuda": "H100", "rocm": "MI325", "npu": "A2"}, "num_cards": {"cuda": 2, "rocm": 2, "npu": 1}}}, "full_model", "omni"]` | Same as above, but declares **multiple platforms** in one case. Filter examples: `-m "full_model and H100 and cuda"`, `-m "full_model and MI325 and rocm"`. |

Recommended for L4 perf cases:

```JSON
{
    "test_name": "test_bagel_single_device_single_stage_t2i",
    "mark": [
        {"hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 1}},
        "full_model",
        "diffusion"
    ],
    "server_type": "vllm-omni",
    "server_params": { "...": "..." },
    "benchmark_params": [ { "name": "1024x1024_steps20", "...": "..." } ]
}
```

Multi-GPU diffusion (example: Cosmos3 with `cfg-parallel-size=2`):

```JSON
{
    "test_name": "test_cosmos3_t2i_official_demo_2gpu",
    "mark": [
        {"hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 2}},
        "full_model",
        "diffusion"
    ],
    "server_type": "vllm-omni",
    "benchmark_endpoint": "/v1/images/generations",
    "server_params": { "...": "..." },
    "benchmark_params": [ { "name": "1024x1024_steps4", "...": "..." } ]
}
```

- Omni perf: include `"omni"` in the `mark` array
- TTS perf: include `"tts"` in the `mark` array
- Diffusion perf: include `"diffusion"` in the `mark` array
- HunyuanImage local-weight cases: add `"local_model"` to the `mark` array

**Parametrization IDs**

Each `(server, benchmark index)` pair becomes one `pytest.param` with id `{test_name}-{suffix}`. The suffix comes from `benchmark_params[].name` when set (for example `test_tts-p0`, `test_omni-p1`); otherwise it is derived from fields like `task` / `eval_phase`.

**Benchmark result filenames**

Result files use the **runtime** hardware label from `get_runtime_resource_label()` (detected GPU/NPU name on the machine that ran the job), **not** `mark.hardware_marks.res`. On the default H100 CI pool, `H100` is **omitted** from filenames (`resource_label_for_filename`).

Examples:

- Omni/TTS: `result_{test_name}_{optional_hw}_{dataset}_....json` under `BENCHMARK_DIR`
- Diffusion: one aggregate `diffusion_result_{config_stem}_{optional_hw}_{timestamp}.json` per source JSON file (array of all runs from that file)

**Local commands**

```bash
# Bulk load + filter by JSON mark
pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py -m "full_model and H100 and diffusion"
pytest -s -v tests/dfx/perf/scripts/run_benchmark.py -m "full_model and omni and H100"

# Single file (same as nightly CI Perf steps)
pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_bagel_vllm_omni.json
pytest -s -v tests/dfx/perf/scripts/run_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_qwen3_omni_async_chunk.json

# Optional baseline assertion (default off)
pytest -sv tests/dfx/perf/scripts/run_diffusion_benchmark.py --assert-baseline \
  --test-config-file tests/dfx/perf/tests/test_qwen_image_vllm_omni.json
```

See also [Markers for Tests](./tests_markers.md) for registered hardware markers (`H100`, `L4`, `cuda`, `distributed_cuda`, …).

**`server_params` Configuration**

*Basic Parameters*

| Parameter         | Required | Example                            | Description                   |
| ----------------- | -------- | ---------------------------------- | ----------------------------- |
| model             | Yes      | "Qwen/Qwen3-Omni-30B-A3B-Instruct" | Model name or path            |
| stage_config_name | Yes      | "qwen3_omni.yaml"                  | Stage configuration file name |

*Dynamic Configuration (update/delete)*

Supports incremental modifications based on the basic configuration:

| Operation | Description                          |
| --------- | ------------------------------------ |
| update    | Update or add configuration items    |
| delete    | Delete specified configuration items |

**Example**:

```
"update": {
    "async_chunk": true,  // Enable asynchronous chunk processing
    "stage_args": {
        "0": {
            "engine_args.custom_process_next_stage_input_func": "vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_async_chunk"
        }
    }
},
"delete": {
    "stage_args": {
        "2": ["custom_process_input_func"]  // Delete this configuration for stage 2
    }
}
```

**`benchmark_params` Configuration**

You can add any benchmark running parameters you need here. For all optional parameters, refer to the [benchmark documentation](https://github.com/vllm-project/vllm-omni/blob/main/docs/cli/bench/serve.md). General modifications are as follows:

1.  Change the --xxx-xx-xx running parameters to xxx_xx_xx format and fill them as keys in the JSON file.
2.  For boolean variables in the running parameters, modify them to forms such as ignore_eos: true/false and fill them into the JSON file.
3.  Optionally add a `baseline` object (see **Baseline thresholds** below). If you omit `baseline` or leave it empty, the performance test still runs but does not assert metric thresholds from this field.
4.  Set `"name"` on each `benchmark_params` entry for stable pytest ids and readable result keys.
5.  The qps and concurrency modes are recommended to be mutually exclusive. For detailed explanations, see the table below:

| Parameter       | Type        | Required | Example/Values  | Description                                                                                                                                                                                                                                                          |
| --------------- | ----------- | -------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| name            | string      | No       | `"1024x1024_steps4"` | Stable suffix for pytest param id and logs. |
| num_prompts     | int / array | Yes      | 10,[10, 20, 30] | Number of requests. Supports single values or arrays. If a single value is used, it will be automatically expanded to match the number of qps or max_concurrency, e.g., [10,10,10]. If an array is used, its length must match the number of qps or max_concurrency. |
| request_rate    | float / array | No  | 0.5, [0.5, 1, inf] | Queries per second. Supports single values or arrays. If a single value is used, it will be automatically expanded to match the number of num_prompts, e.g., [1,1,1]. If an array is used, its length must match the number of num_prompts.                          |
| max_concurrency | int / array | No       | 1, [1, 2, 3]    | Maximum concurrent in-flight requests. Same array / expansion rules as `request_rate` (mutually exclusive with QPS mode).                                                                                                                                                                                                             |
| baseline        | object      | No       | see above       | Optional per-metric thresholds; keys must match benchmark output fields. Scalar, list (per sweep step), or object (keyed by concurrency or QPS string).  
