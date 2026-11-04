# Debugging Multi-Stage Omni Pipelines

## Contents
- First steps
- Symptom table
- Error contracts
- Startup knobs
- Audio-output pitfalls

## First steps

1. Identify the failing stage: the logs tag stage ids; `EngineDeadError` logs carry
   `error_stage_id`.
2. Re-run a single request with `--log-stats` to see per-stage wall time and
   connector transfer timings.
3. Reduce to the smallest reproduction: one request, text-only output
   (`"modalities": ["text"]`) to isolate the thinker, then add audio.
4. Toggle one suspect at a time: `--enforce-eager` (CUDA graphs), stage overrides,
   connector settings, `async_chunk`.

Environment variable contract: `docs/configuration/environment_variables.md` and
`vllm_omni/config/environment_variable_inventory.py`.

## Symptom table

| Symptom | Likely cause | Where to check |
|---------|--------------|----------------|
| Server never becomes ready | Stage registration or init timeout (slow checkpoint load, sequential GPU init) | "Timed out waiting for registration" in logs; raise `--stage-init-timeout` (default 300 s) / `--init-timeout` (default 600 s); `vllm_omni/engine/stage_engine_startup.py`, `omni_engine_base.py` |
| CUDA/device errors when a stage spawns | Wrong `devices` in deploy YAML or bad replica split | Deploy YAML `devices`; `split_devices_for_replicas` / `setup_stage_devices` in `vllm_omni/engine/stage_init_utils.py` |
| OOM at load or first request | Co-located stages exceed memory; CUDA graph capture memory | Per-stage `gpu_memory_utilization` (fraction of total GPU memory) must sum to ≤ 1.0 on a shared GPU, see `docs/configuration/gpu_memory_utilization.md`; try `enforce_eager` on the suspect stage |
| Client gets 500, engine dead | A stage process crashed | `vllm_omni/entrypoints/serve/utils/errors.py` log lines; first traceback in the stage process log |
| Client gets 400 | Request-scoped `OmniClientError` / guardrail | `vllm_omni/errors.py`; validate the payload |
| Request stalls mid-pipeline | Downstream stage waiting for upstream payload; connector or NIXL timeout | `VLLM_OMNI_INPUT_WAIT_TIMEOUT_S` (default 600), `VLLM_OMNI_CONNECTOR_RECV_POLL_MS`, `VLLM_OMNI_NIXL_*`; `--log-stats` transfer timings |
| `NativeKVHandoffError` | A request lost its producer binding or transfer metadata | `vllm_omni/engine/errors.py`; check replica routing and aborts |
| High TTFT while GPUs are idle | Orchestrator loop saturation or queue backlog | `--enable-orch-monitor`; `VLLM_OMNI_EVENT_DRIVEN_ORCH`; `docs/contributing/profiling.md` |
| High AUDIO_TTFP | Talker/code2wav slow to first chunk; first-chunk size too large | `--print-stage`; connector `initial_codec_chunk_frames`; `VLLM_OMNI_TALKER_FIRST_AUDIO` |
| AUDIO_RTF ≥ 1 under load | Talker or code2wav saturated; batch limits too low | Per-stage `max_num_seqs`; stage replicas; profile the slow stage |
| Good RTF but audible gaps | Bursty chunk delivery | Bench `audio_continuity_ok_rate`; codec chunk/context frames in the connector config |
| Truncated or repeated streaming audio | Delta vs cumulative audio confusion in output handling | `vllm_omni/outputs/output_processor.py`; model output path |
| Voices or content mixed across concurrent requests | Per-request state not keyed by request id | Model-side caches in talker/code2wav; reproduce with concurrency ≥ 2 |
| Results differ with CUDA graphs on vs off | Graph capture/replay bug (stale buffers, autocast mismatch) | Compare with `--enforce-eager`; model-specific graph code (see the model docs in `models/`) |
| Accuracy collapse on Daily-Omni | Thinking mode consuming the answer budget; wrong pack mode | Disable thinking via `--extra-body`; `--daily-omni-pack-mode` |
| Bench cannot load local media | Server missing `--allowed-local-media-path` | Serve command |

## Error contracts

- Request-scoped failures (`OmniClientError`, `GuardrailViolationError`) → 4xx.
- Engine death / generate failures → 500 without hanging other requests.
- OOM and signal behavior expectations: `docs/user_guide/fault_injection_reliability_matrix.md`.

A fix that converts a hang into a clear error is progress; a fix that swallows an
error to keep the server up usually is not.

## Startup knobs

| Knob | Default |
|------|---------|
| `--stage-init-timeout` | 300 s per stage |
| `--init-timeout` | 600 s overall |
| `--shm-threshold-bytes` | 65536 |
| Worker start method | Forced to `spawn` |

## Audio-output pitfalls

Debug bad audio in this order: RoPE → normalization → hop length → codec token
mapping → sampling params → tensor layout → dtype. Compare against the
HuggingFace reference on the same input before touching runtime code. Avoid
`.item()` / `.cpu()` in hot loops; host syncs show up as TTFP regressions.
