# MiniCPM-o 4.5 numerical lab

Status: implemented; static gates pass and focused CPU tests await a local
PyTorch runtime. This branch is always Draft. A3 performance, waveform parity,
full Seed-TTS quality, Daily-Omni, Video-MME, Demo, and stability evidence are
pending.

## Default contract

With no explicit configuration or environment variables, Code2Wav retains the
competition baseline:

- CFM Euler steps: 10;
- Flow/CFM precision: FP32;
- HiFT, CFG/Euler integration, prompt features, and final waveform: FP32;
- public API, chunking, noise, CFG rate, sampling, and weights: unchanged.

This PR does not edit the official deploy YAML. Experimental settings are
resolved once during Stage 2 construction with this priority:

1. explicit connector `extra.token2wav_n_timesteps` /
   `extra.token2wav_float16`;
2. numerical-lab environment variable;
3. 10-step/FP32 default.

Malformed or unsupported values fail during model construction. CFM steps are
restricted to the finite quality grid 10, 8, or 6.

## Experiment switches

```bash
# Baseline: leave both unset.
unset VLLM_OMNI_MINICPMO45_CFM_STEPS
unset VLLM_OMNI_MINICPMO45_FLOW_FP16

# Example isolated step candidate.
export VLLM_OMNI_MINICPMO45_CFM_STEPS=8
unset VLLM_OMNI_MINICPMO45_FLOW_FP16

# Example isolated NPU Flow precision candidate.
export VLLM_OMNI_MINICPMO45_CFM_STEPS=10
export VLLM_OMNI_MINICPMO45_FLOW_FP16=1
```

Every setting change requires a clean service restart. Do not compare arms by
hot-switching one process: Graph captures, allocator state, and request caches
would contaminate the result.

## NPU Flow precision boundary

On Ascend, enabling Flow FP16 keeps model weights available in FP32 and applies
NPU autocast only to the Flow encoder and CFM DiT estimator. The following stay
FP32:

- speaker projection and prompt conditioning;
- random-noise state, cosine timeline, Euler updates, and CFG combination;
- HiFT and emitted waveform conversion.

The exact-shape NPUGraph estimator body enters the same autocast context during
capture, so eager and Graph paths test the same precision policy. Graph cache
keys include tensor dtype, shape, and effective Flow precision. If autocast
later falls back, its FP32 epoch cannot replay an earlier FP16 capture.

If the initial NPU autocast capability probe cannot enter its context, the
backend falls back once to FP32 and records Host-only telemetry:

```text
requested_dtype, effective_dtype,
fallback_count, fallback_reason, fallback_error_type
```

Flow operator errors after context entry are not swallowed. A formal FP16 arm
is invalid unless `effective_dtype=float16` and `fallback_count=0` throughout
the run. If an autocast entry fails after FP16 already executed, Stage 2 fails
and must restart rather than reusing mixed-precision request caches. Opt-in
timeline CFM/HiFT events include step count and precision telemetry without
device synchronization.

## A3 experiment ladder

Run one variable at a time before combinations:

| Arm | Steps | NPU Flow | Purpose |
| --- | ---: | --- | --- |
| B10 | 10 | FP32 | frozen baseline |
| S8 | 8 | FP32 | first step-reduction gate |
| S6 | 6 | FP32 | only after S8 quality passes |
| H10 | 10 | FP16 | isolated precision gate |
| H8 | 8 | FP16 | only after S8 and H10 pass independently |
| H6 | 6 | FP16 | last/highest-risk candidate |

For each arm:

1. compare eager and NPUGraph fixed inputs (`rtol=1e-3`, `atol=1e-3`) and
   reject NaN/Inf;
2. run Chinese Seed-TTS c=1 with two warmups and `B-C-C-B-B-C`, three formal
   repetitions per arm;
3. repeat English c=1/32 prompts for compatibility;
4. run c=4/8 success, continuity, HBM, and tail-latency guardrails;
5. run full Daily-Omni, Video-MME, Seed-TTS ASV/WER, Demo, and stability gates
   on the exact candidate commit/settings.

The numerical branch cannot enter `vllm-omni-v0-ultra` unless all conservative
quality gates pass: Daily-Omni >= 78%, Video-MME >= 68%, ASV >= 0.689, and WER
<= 1.56%, with 100% request success, stream continuity, and decodable audio.

## Promotion and rollback

RTF is the primary score. Prefer a candidate with at least 2% RTF improvement
and no more than 1% TTFP/TTFT regression. A step/precision arm that misses any
quality, continuity, fallback, NaN/Inf, HBM, or stability gate remains Draft
and is not stacked into the integration branch.

Rollback by unsetting both environment variables (or restoring explicit
10/false configuration) and restarting. No persistent numerical cache survives
the process restart.
