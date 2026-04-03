When you want to add L4-level ***performance test*** cases, you can refer to the following format for case addition in tests/dfx/perf/tests/test.json:

```JSON
{
    "test_name": "test_qwen3_omni",
    "server_params": {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "stage_config_name": "qwen3_omni.yaml"
    },
    "benchmark_params": [
        {
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
| server_params    | Yes      | Server-side configuration parameters                            |
| benchmark_params | Yes      | Benchmark running parameters (supports multiple configurations) |

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
4.  The qps and concurrency modes are recommended to be mutually exclusive. For detailed explanations, see the table below:

| Parameter       | Type        | Required | Example/Values  | Description                                                                                                                                                                                                                                                          |
| --------------- | ----------- | -------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| num_prompts     | int / array | Yes      | 10,[10, 20, 30] | Number of requests. Supports single values or arrays. If a single value is used, it will be automatically expanded to match the number of qps or max_concurrency, e.g., [10,10,10]. If an array is used, its length must match the number of qps or max_concurrency. |
| request_rate    | float / array | No  | 0.5, [0.5, 1, inf] | Queries per second. Supports single values or arrays. If a single value is used, it will be automatically expanded to match the number of num_prompts, e.g., [1,1,1]. If an array is used, its length must match the number of num_prompts.                          |
| max_concurrency | int / array | No       | 1, [1, 2, 3]    | Maximum concurrent in-flight requests. Same array / expansion rules as `request_rate` (mutually exclusive with QPS mode).                                                                                                                                                                                                             |
| baseline        | object      | No       | see above       | Optional per-metric thresholds; keys must match benchmark output fields. Scalar, list (per sweep step), or object (keyed by concurrency or QPS string).  
