# Prefill-Decode (PD) Disaggregation

PD disaggregation splits the Qwen3-Omni thinker into separate prefill and decode
stages so prompt processing and token generation can run on different workers.

This is documented as a stage-config recipe instead of a bundled YAML because the
deployment-specific values usually change per environment:

- GPU placement
- `tensor_parallel_size`
- connector backend and connector ports
- connector IPs or bootstrap addresses

Start from the [default Qwen3-Omni stage config](gh-file:vllm_omni/deploy/qwen3_omni_moe.yaml)
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
    sync_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_token_only

  - stage_id: 3
    runtime:
      devices: "2"
    engine_args:
      max_num_seqs: 1
    engine_input_source: [2]
    sync_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_token_only
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

## 5. Scale beyond 1P1D — M-P-1-D / 1P-N-D / M-P-N-D

`vllm-omni` 的 PD 路由层支持任意 K 个 prefill stage 与 N 个 decode stage 共存
的拓扑（K ≥ 1, N ≥ 1）：

- **M-P-1-D**：K 个 prefill stage 共享 1 个 decode stage（吞吐瓶颈在 prefill 时
  最常见）。参考 [vllm_omni/deploy/qwen3_tts_pd_mp1d.yaml](gh-file:vllm_omni/deploy/qwen3_tts_pd_mp1d.yaml)。
- **1P-N-D**：1 个 prefill stage 喂多个 decode stage（瓶颈在 decode 时使用）。
- **M-P-N-D**：上面两种的组合，K > 1 且 N > 1。

### 5.1 拓扑识别

`PDDisaggregationMixin._init_pd_state` 会自动从 stage_configs 中识别 K / N 与
prefill→decode 的绑定关系，写入下列字段：

- `_pd_prefill_ids`：所有 prefill stage 索引（按 stage_configs 顺序）。
- `_pd_decode_ids`：所有 decode stage 索引。
- `_pd_decode_to_prefill`：每个 decode 接受哪些 prefill 作为
  `engine_input_source`（用于 N > 1 的过滤）。
- `_pd_separation_pair`：仅当 K=1 ∧ N=1 时返回 `(prefill_idx, decode_idx)`，
  其它拓扑下为 `None`（向后兼容 1P1D 调用方）。

### 5.2 调度策略

K > 1 时编排器需要在 K 个 prefill 中为每个请求选一个；N > 1 时同理在 N 个
decode 中选一个。可通过环境变量或部署 YAML 顶层字段控制：

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `pd_prefill_pick_strategy` (YAML) / `VLLM_OMNI_PD_PREFILL_PICK_STRATEGY` (env) | `round_robin` (默认) / `least_inflight` | K > 1 时挑选 prefill |
| `pd_decode_pick_strategy` (YAML) / `VLLM_OMNI_PD_DECODE_PICK_STRATEGY` (env)   | `round_robin` (默认) / `least_inflight` | N > 1 时挑选 decode  |

- **优先级**：环境变量 > YAML 顶层字段 > `round_robin` 默认。
- **`least_inflight`**：调度器维护一份 `_pd_prefill_inflight` /
  `_pd_decode_inflight` 计数器，每个请求绑定到当前 inflight 最少的 stage，
  请求结束后递减。适合负载不均衡的场景。
- **`round_robin`**：用单调递增的内部 counter 取模选择，适合负载均衡的
  匀速负载。

### 5.3 YAML 顶层字段示例

```yaml
# 顶层（与 stage_args 同级）
pd_prefill_pick_strategy: least_inflight   # 可选
pd_decode_pick_strategy:  round_robin      # 可选

stage_args:
  - stage_id: 0
    is_prefill_only: true
    # ...
  - stage_id: 1
    is_prefill_only: true
    # ...
  - stage_id: 2
    is_decode_only: true
    engine_input_source: [0, 1]   # 同时接两个 prefill
    # ...
```

> 互链：参考 [`examples/online_serving/qwen3_tts_pd/README.md`](gh-file:examples/online_serving/qwen3_tts_pd/README.md)
> 中的 1P1D 探针脚本 `probe_1p1d.py`，并对照 [`vllm_omni/deploy/qwen3_tts_pd_mp1d.yaml`](gh-file:vllm_omni/deploy/qwen3_tts_pd_mp1d.yaml)
> 的 M-P-1-D 部署 YAML 与外部 `qwen3_tts_pd_ar_only_3p1d.yaml`（M-P-1-D, 3P+1D）/
> `qwen3_tts_pd_ar_only_1p3d.yaml`（1P-N-D, 1P+3D）做对比。

### 5.4 注意事项

- 每个 prefill stage 必须使用**唯一**的 `mooncake_bootstrap_port`
  （`runtime.env.VLLM_MOONCAKE_BOOTSTRAP_PORT` 或
  `engine_args.kv_transfer_config.kv_connector_extra_config.mooncake_bootstrap_port`）。
  端口冲突会让 `MooncakeBootstrapServer.start` 报错并 fail-fast。
- decode stage 不会启动 bootstrap server，但其 env 端口仍应填写并保持唯一，
  以便诊断时区分进程。
- M-P-N-D 拓扑下 `kv_parallel_size` 应等于 K + N。
- 当 K > 1 时，caller (`Omni`/`AsyncOmni`) 通过 `_pick_prefill_stage` 选完
  prefill 后会把 `bound_prefill_stage_id` 顺路传给 orchestrator，避免编排器
  自己再走一次 round_robin 导致与 caller 不一致。这是 M-P-N-D 主线的核心
  路由不变量，禁止在 caller 不传 `bound_prefill_stage_id` 的同时
  让 orchestrator 自己 pick。
