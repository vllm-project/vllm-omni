# OpenVLA-7B

> OpenVLA-7B served as an autoregressive robot policy over the OpenPI WebSocket protocol

## Summary

- Vendor: OpenVLA (Stanford / Google DeepMind / Toyota Research Institute)
- Model: `openvla/openvla-7b`
- Task: Vision-Language-Action (VLA) inference for robot manipulation
- Mode: Online serving via the OpenPI WebSocket endpoint
- Hardware used for the numbers below: 1x NVIDIA A800 80GB

## When to use this recipe

Use this when you want OpenVLA to serve more than one robot, or when the
reference implementation's control rate is not enough. OpenVLA is the first
robot policy in this repository whose actions come out of an autoregressive
decoder rather than a denoiser: the seven action dimensions are seven
discretised tokens from the tail of the Llama-2 vocabulary. That makes it an
ordinary `LLM_AR` stage, so it gets paged KV, continuous batching and CUDA
graphs — none of which the reference implementation has. The reference
implementation cannot batch at all: `modeling_prismatic.py` raises
`ValueError("Generation with batch size > 1 is not currently supported!")`.

## References

- Upstream model: <https://huggingface.co/openvla/openvla-7b>
- Upstream codebase: <https://github.com/openvla/openvla>
- OpenPI client library: <https://github.com/Physical-Intelligence/openpi>
- Model class: upstream vLLM's `vllm.model_executor.models.openvla.OpenVLAForActionPrediction`
- Pipeline: [`vllm_omni/model_executor/models/openvla/pipeline.py`](../../vllm_omni/model_executor/models/openvla/pipeline.py)
- Action decode: [`vllm_omni/model_executor/models/openvla/action_decode.py`](../../vllm_omni/model_executor/models/openvla/action_decode.py)
- Deploy config: [`vllm_omni/deploy/openvla.yaml`](../../vllm_omni/deploy/openvla.yaml)

## Environment

- OS: Linux
- Python: 3.11+
- Driver / runtime: NVIDIA CUDA
- Hardware: 1 NVIDIA GPU. bf16 weights are 14.05 GiB; measured on an A800 80GB.
- `timm` is required — the two vision towers are timm ViTs
  (`vit_large_patch14_reg4_dinov2.lvd142m` and `vit_so400m_patch14_siglip_224`).

## Start server

```bash
vllm serve openvla/openvla-7b \
  --omni \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name openvla
```

The pipeline declares `openvla.yaml` as its default deploy config, so no
`--deploy-config` flag is needed. The WebSocket endpoint is
`ws://127.0.0.1:8000/v1/realtime/robot/openpi`.

Notes:

- **The handshake is derived from the checkpoint, not configured.** An
  `LLM_AR` stage may not carry the `model_config:` block the diffusion robot
  deploys use, so `policy_server_config` is built from the checkpoint's own
  `norm_stats`: `action_dim`, `action_horizon: 1`, the selected `unnorm_key`,
  and the full list of supported embodiments. By construction a fine-tuned
  OpenVLA with a different embodiment set should advertise itself with no
  config change, but only `openvla/openvla-7b` has been run — that is a design
  property, not a measurement.
- **Pick an embodiment.** `openvla-7b` ships action statistics for 25
  datasets, so one has to be chosen. The deploy config sets
  `hf_overrides: {unnorm_key: bridge_orig}`; a client may override it per
  observation by putting `unnorm_key` in the observation dict.
- The observation needs an RGB image (looked up under `image`,
  `observation/image`, `observation.image`,
  `observation/exterior_image_1_left`, `base_image`, `primary_image`) and a
  language instruction (`prompt`, `instruction` or `task`). The response is a
  `(1, 7)` float32 array — action horizon 1, seven dimensions.

## Verification

Numerical agreement with the reference implementation, on real
`bridge_orig` episodes from `IPEC-COMMUNITY/bridge_orig_lerobot`:

- 18 WebSocket round trips across three episodes: **17 of 18 actions
  bit-identical** to `OpenVLAForActionPrediction.predict_action` under
  transformers 4.40.1 on the same frames; the remaining one differs by 0.0134,
  a single bin.
- Offline, over 10 held-out observations (5 photographs, 5 synthetic):
  **8/10 free-running token sequences identical**, and **69/70 teacher-forced
  next-token argmax positions identical**. Both disagreements are on
  synthetic-noise images at positions where the reference's own top-1/top-2
  logprob gap is 0.0625–0.125, i.e. bf16 rounding resolution.

CPU-only unit tests:

```bash
pytest tests/model_executor/models/test_openvla_action_decode.py \
       tests/model_executor/models/test_openvla_registration.py \
       tests/entrypoints/openai_api/test_openpi_ar_action_policy.py
```

## Performance

Measured on 1x A800 80GB, bf16, TP=1, greedy, seven action tokens per step.

**Measure with a distinct image per request.** vLLM caches by image hash, so a
benchmark that reuses one frame skips the vision encoder and reports a latency
no robot ever sees — 89 ms instead of 145 ms here, a 39% error. (The towers
themselves are only ~24 ms of the step, so encoder-output reuse saves more than
the towers alone; the rest is unexplained and I did not chase it.)

Single-robot latency:

| | reference (transformers) | offline engine | over the OpenPI endpoint |
|---|---|---|---|
| one robot | 206.6 ms (4.84 Hz) | 145.4 ms (6.88 Hz) | 183.4 ms (5.45 Hz) |

The endpoint is what a robot actually gets; 19.1 ms of the 38 ms it adds is
transport + msgpack + dispatch.

Concurrency — N observations submitted together on one served instance. Batching
trades per-robot control rate for fleet throughput; the reference has no batch
axis at all, so N robots need N model copies:

| N | aggregate | per robot |
|---|---|---|
| 1 | 6.9 actions/s | 6.88 Hz |
| 2 | 11.5 | 5.77 Hz |
| 4 | 17.8 | 4.44 Hz |
| 8 | 26.0 | 3.25 Hz |
| 16 | 32.9 | 2.06 Hz |

Where a single control step goes (batch 1, distinct images):

| part | time |
|---|---|
| vision towers (DINOv2 + SigLIP, 256 tokens each) | 14–24 ms |
| rest of prefill | 31–38 ms |
| 6 decode steps | 68–76 ms |

## Notes

- **Batch size is a real knob here.** The other robot deploys in this repo
  (`Gr00tN1d7.yaml`, `dreamzero.yaml`, `pi0.yaml`) pin `max_num_seqs: 1`.
  OpenVLA is a plain AR
  stage, so `max_num_seqs` behaves normally and concurrent robots share the GPU.
- The 256 image tokens are inserted immediately after BOS, which is where the
  reference splices them. That means the part of the prompt that changes every
  control step is at the *front*, so prefix caching cannot reuse anything; the
  deploy config turns it off rather than paying for the bookkeeping.
- The checkpoint expects a trailing empty token (id `29871`) after the prompt —
  the reference re-inserts it in `predict_action` because the training-time
  template ended with a space that `get_prompt()` strips. The serving path
  builds prompt token ids directly for this reason; no chat template produces it.
