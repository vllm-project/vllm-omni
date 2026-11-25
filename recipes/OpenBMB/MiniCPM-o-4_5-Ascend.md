# MiniCPM-o 4.5 on two Ascend cards

This recipe runs one 40,960-token Thinker, two 4,096-token Talker replicas
with K16, and two Code2Wav replicas with five-step FP32 CFM. All stages use
MRv2 and the stock vLLM-Ascend allocator.

- [Two-card deployment template](../../vllm_omni/deploy/minicpmo_4_5_2gpu.yaml)
- [Launcher](serve_minicpmo_ascend.sh)

## Dependencies

Use two Ascend 910B3 cards (64 GiB each), preferably on the same NUMA node.
Start with the `quay.nju.edu.cn/atlas-ci/vllm-ascend:v0.34.0` image, the matching
Ascend driver, and `numactl`. The measured image digest is
`sha256:4ec880dc132cc9c3993d914039b2879db5db4f432ef36211417fcdcb9249d590`.
It contains vLLM `62d4bc3c7`, vLLM-Ascend `19c063b54`, and CANN 9.3.
Mount the checkout and your Hugging Face cache or local models into the
container. Keep the image's CANN environment and Python paths.

## Launch with the original weights

From the repository root, run:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export NUMA_NODE=0
export PORT=8000
export PROFILING_MODE=dynamic TASK_QUEUE_ENABLE=1
bash recipes/OpenBMB/serve_minicpmo_ascend.sh
```

The default target is the public [OpenBMB MiniCPM-o 4.5 checkpoint][base].
The service downloads the model on first use if needed. To use an existing
local snapshot, set `MODEL_ROOT=/models/MiniCPM-o-4_5` before starting the
launcher. The default Thinker uses Ngram speculation with the target
weights and committed token history.

## Optional Thinker DSpark

The launcher can switch Thinker speculation to the separate public
[DSpark draft][draft] and propose up to seven tokens per draft. Native duplex
WebSocket decoding still skips drafts. The draft can be used by ordinary chat
requests on the same service.

For automatic download, pass the Hugging Face repository ID:

```bash
DRAFT_MODEL=Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm \
  bash recipes/OpenBMB/serve_minicpmo_ascend.sh
```

The launcher pins the revision below in stage 0's `speculative_config`.
vLLM downloads it into the Hugging Face cache as needed. For a manual
download or offline serving, use a local directory:

```bash
hf download Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm \
  --revision c4c310aca2c768343165261cb22c21b90c1f79e0 \
  --local-dir /models/minicpmo-thinker-dspark
DRAFT_MODEL=/models/minicpmo-thinker-dspark \
  bash recipes/OpenBMB/serve_minicpmo_ascend.sh
```

For a custom deploy YAML, set `speculative_config` on the entry with
`stage_id: 0` under `platforms.npu.stages`. Set `async_scheduling: false`
on the same entry. The model can be the pinned Hub repo or a local directory.
Leave out `revision` for a local path.

```yaml
speculative_config:
  method: dspark
  model: Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm
  revision: c4c310aca2c768343165261cb22c21b90c1f79e0
  num_speculative_tokens: 7
  draft_tensor_parallel_size: 1
```

In a historical Ascend C1 comparison on 32 streaming Seed-TTS requests with
the same W8A8 Thinker, 16K context, K16 Talker, and five CFM steps, enabling
DSpark 7 reduced mean RTF from 0.1794 to 0.1580 (11.9%) and mean time to first
audio from 443 to 347 ms (21.8%). The control used no Thinker speculation.
The gain over default Ngram or on the original BF16 weights has yet to be
measured in a matching run.

## Optional INT8 W8A8 Thinker

The five-tier performance report used INT8 Thinker weights with dynamic
per-token INT8 activation quantization. To export from the pinned public
checkpoint, run the following on a CPU host with PyTorch and safetensors.
The conversion processes one shard at a time. Allow space for both
checkpoints.

```bash
hf download openbmb/MiniCPM-o-4_5 \
  --revision a0b28789837801b80ca674a178cc5cb27843fba8 \
  --local-dir /models/minicpmo-original
OMP_NUM_THREADS=4 python3 tools/quantize_minicpmo_ascend.py \
  /models/minicpmo-original /models/minicpmo-original/thinker-w8a8
MODEL_ROOT=/models/minicpmo-original THINKER_W8A8=1 \
  DRAFT_MODEL=Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm \
  bash recipes/OpenBMB/serve_minicpmo_ascend.sh
```

The exporter quantizes the Thinker attention and MLP projections with
symmetric per-output-channel scales. Re-exporting the four weight shards
from the pinned revision reproduced their SHA256 digests from the benchmark.
The exporter copies the model code and tokenizer from the public snapshot.
The historical deployment used different model Python assets. See the PR's
test results for the measured configuration.

[base]: https://huggingface.co/openbmb/MiniCPM-o-4_5
[draft]: https://huggingface.co/Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm/tree/c4c310aca2c768343165261cb22c21b90c1f79e0

## Operation

Set `RUNTIME_DIR` to a unique path to keep the rendered deployment YAML and
per-service mode files.

Thinker and Talker allow 16 sequences per worker, and Code2Wav allows 32. The
Talker splits new requests evenly. Code2Wav routes by weighted active and
reserved work, with CFM micro-batches of eight and four. Each request keeps
its replica for subsequent chunks. To keep a single Code2Wav replica on the
first visible card, override that stage's placement at launch.

```bash
bash recipes/OpenBMB/serve_minicpmo_ascend.sh \
  --stage-overrides '{"2":{"devices":"0","num_replicas":1}}'
```

Startup graph capture can take several minutes. Wait for `/health` before
sending requests. The launcher qualifies common short prefill shapes ahead
of readiness. Other eligible graphs qualify against reference results before
replay. A shape or numerical mismatch uses the reference path. The Talker
uses short-context graphs when eligible and retains its full context limit.
