# Prefill-Decode (PD) Disaggregation

Splits the thinker into separate prefill and decode stages for higher throughput:

```
Stage 0: Prefill (KV producer) --[KV via MooncakeConnector]--> Stage 1: Decode (KV consumer) --> Stage 2: Talker --> Stage 3: Code2Wav
```

## Requirements

- 3+ GPUs (1 prefill, 1 decode, 1 talker+code2wav)
- `mooncake-transfer-engine` installed
- Same `tensor_parallel_size` on both prefill and decode stages

## Usage

```python
from vllm_omni import OmniLLM

omni = OmniLLM(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    stage_configs_path="vllm_omni/model_executor/stage_configs/qwen3_omni_moe_pd_multiconnector.yaml",
)
```

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_omni_moe_pd_multiconnector.yaml
```

## Key YAML Fields

```yaml
stage_args:
  - stage_id: 0
    is_prefill_only: true        # prompt processing only, saves KV cache
    engine_args:
      kv_transfer_config:
        kv_connector: "MooncakeConnector"
        kv_role: "kv_producer"
        kv_rank: 0
        kv_parallel_size: 2
        kv_connector_extra_config:
          mooncake_bootstrap_port: 25201

  - stage_id: 1
    is_decode_only: true         # loads KV cache, generates tokens
    engine_args:
      kv_transfer_config:
        kv_connector: "MooncakeConnector"
        kv_role: "kv_consumer"
        kv_rank: 1
        kv_parallel_size: 2
        kv_connector_extra_config:
          mooncake_bootstrap_port: 25202
```

For TP>1, assign multiple GPUs per stage (e.g. `devices: "0,1"`) and set matching `tensor_parallel_size` on both.
