# Prefill-Decode (PD) Disaggregation

PD disaggregation splits the Qwen3-Omni thinker into separate prefill and decode
stages so prompt processing and token generation can run on different workers.

This is documented as a stage-config recipe instead of a bundled YAML because the
deployment-specific values usually change per environment:

- GPU placement
- `tensor_parallel_size`
- connector backend and connector ports
- connector IPs or bootstrap addresses

Start from the [default Qwen3-Omni stage config](gh-file:vllm_omni/model_executor/stage_configs/qwen3_omni_moe.yaml)
and copy it to your own file, for example `qwen3_omni_pd.yaml`. Then apply the
changes below.

## Requirements

- 3+ GPUs for a basic layout: prefill, decode, and talker+code2wav
- A KV connector supported by vLLM, such as `MooncakeConnector`
- Matching `tensor_parallel_size` on the prefill and decode thinker stages

## 1. Split the thinker into prefill and decode stages

Replace the original thinker stage with two stages:

```yaml
stage_args:
  - stage_id: 0
    stage_type: llm
    is_prefill_only: true
    runtime:
      devices: "0"
    engine_args:
      max_num_seqs: 16
      model_stage: thinker
      model_arch: Qwen3OmniMoeForConditionalGeneration
      worker_type: ar
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      gpu_memory_utilization: 0.9
      enforce_eager: true
      trust_remote_code: true
      engine_output_type: latent
      distributed_executor_backend: "mp"
      enable_prefix_caching: false
      max_num_batched_tokens: 32768
      hf_config_name: thinker_config
      tensor_parallel_size: 1
      kv_transfer_config:
        kv_connector: "MooncakeConnector"
        kv_role: "kv_producer"
        kv_rank: 0
        kv_parallel_size: 2
        kv_connector_extra_config:
          mooncake_bootstrap_port: 25201
    final_output: false
    is_comprehension: true
    default_sampling_params:
      temperature: 0.4
      top_p: 0.9
      top_k: 1
      max_tokens: 2048
      seed: 42
      detokenize: True
      repetition_penalty: 1.05

  - stage_id: 1
    stage_type: llm
    is_decode_only: true
    runtime:
      devices: "1"
    engine_args:
      max_num_seqs: 64
      model_stage: thinker
      model_arch: Qwen3OmniMoeForConditionalGeneration
      worker_type: ar
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      gpu_memory_utilization: 0.9
      enforce_eager: true
      trust_remote_code: true
      engine_output_type: latent
      distributed_executor_backend: "mp"
      enable_prefix_caching: false
      max_num_batched_tokens: 32768
      hf_config_name: thinker_config
      tensor_parallel_size: 1
      kv_transfer_config:
        kv_connector: "MooncakeConnector"
        kv_role: "kv_consumer"
        kv_rank: 1
        kv_parallel_size: 2
        kv_connector_extra_config:
          mooncake_bootstrap_port: 25202
    engine_input_source: [0]
    final_output: true
    final_output_type: text
    is_comprehension: true
    default_sampling_params:
      temperature: 0.4
      top_p: 0.9
      top_k: 1
      max_tokens: 2048
      seed: 42
      detokenize: True
      repetition_penalty: 1.05
```

Notes:

- `is_prefill_only: true` marks the thinker stage that only saves KV.
- `is_decode_only: true` marks the thinker stage that resumes from remote KV.
- `kv_transfer_config` is required on both stages.
- The orchestrator forces the prefill stage to run with `max_tokens=1`, so the
  prefill side only processes the prompt and exports KV.

## 2. Shift the downstream stages by one index

After inserting the extra thinker stage, renumber the remaining stages:

```yaml
  - stage_id: 2
    runtime:
      devices: "2"
    engine_input_source: [1]
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker

  - stage_id: 3
    runtime:
      devices: "2"
    engine_args:
      max_num_seqs: 1
    engine_input_source: [2]
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav
```

Compared with the default Qwen3-Omni config:

- the talker becomes stage `2` instead of stage `1`
- the code2wav stage becomes stage `3` instead of stage `2`
- the talker now reads from decode stage `1`

## 3. Add runtime edges for the four-stage pipeline

```yaml
runtime:
  enabled: true
  edges:
    - from: 0
      to: 1
    - from: 1
      to: 2
    - from: 2
      to: 3
```

## 4. Launch with your custom config

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --stage-configs-path /path/to/qwen3_omni_pd.yaml
```

## Operational Notes

- `MooncakeConnector` does not support heterogeneous TP sizes across the PD
  pair. Keep prefill and decode at the same `tensor_parallel_size`.
- If the thinker requires TP=2, both thinker stages must use TP=2 and be given
  separate GPU sets, for example `"0,1"` for prefill and `"2,3"` for decode.
- Choose connector ports and addresses that match your deployment. The values
  shown above are examples only.
