# BAGEL-7B-MoT on Ascend NPU

> Single-stage BAGEL text-to-image serving on Ascend 910C

## Summary

- Vendor: ByteDance Seed
- Model: [`ByteDance-Seed/BAGEL-7B-MoT`](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
- Task: Text-to-image
- Mode: OpenAI-compatible online serving
- Hardware: 1x Ascend 910C, 64 GB
- Maintainer: Community

This recipe covers the qualified single-stage online text-to-image path. Other
BAGEL tasks and deployment topologies are not covered by this NPU qualification.
For CUDA deployment examples, see [BAGEL-7B-MoT.md](BAGEL-7B-MoT.md).

## References

- Upstream model: [`ByteDance-Seed/BAGEL-7B-MoT`](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT)
- Single-stage deploy config: [`bagel_single_stage.yaml`](../../vllm_omni/deploy/bagel_single_stage.yaml)
- NPU installation guide: [Installation on NPU](../../docs/getting_started/installation/npu.md)

## Hardware

- Accelerator: Ascend 910C, 64 GB HBM
- Number of devices: 1 visible NPU
- Qualification scope: online text-to-image with the single-stage deploy config

## Software environment

- OS: Linux
- Python: 3.12
- Driver / runtime: CANN 9.1.0 with ATB
- vLLM: 0.30.0
- vLLM-Omni: checkout containing the BAGEL NPU RMSNorm implementation

Activate both CANN and ATB before launching vLLM-Omni:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export ASCEND_RT_VISIBLE_DEVICES=0
```

Install the current checkout in the NPU environment if needed:

```bash
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

## Command

Set `MODEL` to the local BAGEL checkpoint directory. The verified host used
`/workspace/bagel`:

```bash
export MODEL=/path/to/BAGEL-7B-MoT

vllm serve "${MODEL}" \
  --omni \
  --deploy-config vllm_omni/deploy/bagel_single_stage.yaml \
  --trust-remote-code \
  --port 8091
```

## Verification

A 512x512 text-to-image request with two inference steps returned HTTP 200 and
a generated image on the single-stage server.
