# MiniCPM-o 4.5 Code2Wav NPUGraph

Status: implemented and statically validated; A3 parity, memory, and score
evidence are pending.

## Boundary

Only the deterministic tensor body of the CFM DiT estimator is captured for an
exact input signature. Timestep embedding, flow encoding, HiFT, request parsing,
state ownership, and output publication remain eager. Cached and uncached CFM
calls use separate graph keys, and request-owned outputs are cloned before a
later replay can overwrite graph-persistent buffers.

The implementation is a manual competition-baseline port of upstream vLLM-Omni
PR #5604. Stage 2 remains outer-eager; Stage 0/1 PIECEWISE graph behavior is not
changed.

## Activation and fallback

The NPU platform patch defaults this optimization on for MiniCPM-o 4.5 because
official evaluation supplies the baseline deploy YAML and ignores a candidate
YAML as an activation mechanism. Precedence is:

1. Stage `additional_config.code2wav_enable_npu_graph` when explicitly present;
2. `VLLM_OMNI_MINICPMO45_CODE2WAV_NPU_GRAPH`;
3. NPU default `true`.

The graph cache defaults to 32 signatures and can be changed with explicit
stage config or `VLLM_OMNI_MINICPMO45_CODE2WAV_MAX_NPU_GRAPHS`. Setting the
enable environment variable to `0` is the score-safe rollback.

Missing NPUGraph APIs, launch-blocking mode, incompatible runtime configuration,
a non-NPU backend, or a training-mode flow fall back to eager before capture.
Once capture has started, a capture failure is process-fatal because torch-npu
may leave allocator/capture state invalid; restart with graph disabled.

## Promotion gate

- A3 eager/graph cached and uncached outputs: `rtol=1e-3`, `atol=1e-3`, finite.
- Graph cache HBM increase <= 5%; unseen signatures fall back after the cap.
- Official Chinese c=1 mean RTF improves by at least 2%, with TTFP/TTFT no more
  than 1% worse, before integration.
- All four accuracy gates and 100% request/audio success remain mandatory.
