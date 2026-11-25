# MiniCPM-o 4.5 on two Ascend cards

## Features

- The Thinker runs on the second visible NPU with the model's 40,960-token context.
- Talker and Code2Wav have one replica per card. Talker uses K16 codec feedback.
  Code2Wav uses five CFM steps and micro-batches of eight and four.
- The profile uses MRv2 and serves ordinary chat and native duplex sessions.
  The public checkpoint uses Ngram 7 by default.

## Start the service

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 vllm serve openbmb/MiniCPM-o-4_5 \
  --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/minicpmo_4_5_2gpu.yaml
```

Use a local model directory in place of the model ID for offline serving.
Wait for `/health` before sending requests. The deployment qualifies prefill
graphs during startup.

## Optional Thinker DSpark

Set the following on stage 0 under `platforms.npu.stages` in the deployment
YAML. An upstream model ID uses the normal Hub cache and download path.
You can also download the draft with `hf download` and set `model` to the
local directory.

```yaml
async_scheduling: false
speculative_config:
  method: dspark
  model: Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm
  num_speculative_tokens: 7
  draft_tensor_parallel_size: 1
```

DSpark can accelerate ordinary chat generation. Native duplex requests use
their session sampler and skip drafts.
