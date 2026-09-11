# PersonaPlex ModelScope acquisition and H200 checks (2026-09-07)

## Result and boundary

The cancelled-close/four-frame-tail status below is historical and is
superseded by the [generated-audio drain follow-up](personaplex_stream_drain_20260907.md).

The missing-weight blocker is resolved. PersonaPlex now loads entirely from
local assets in the existing H200 container and vLLM 0.28.0 environment.
Both the instrumented and unchanged-client dual-session smoke pass after
fixing an implicit MiniCPM silence-continuation policy in the shared entrypoint.

This is **not** strict lossless-close acceptance. The existing driver permits
four missing tail frames, and all three sessions actually finish with
`cancelled/session_close`, not `completed/stream_drained`. No throughput,
answer-quality, H20/H100/NPU, or fleet-wide sign-off is claimed.

## Model provenance

- Public mirror: [nv-community/personaplex-7b-v1](https://modelscope.cn/models/nv-community/personaplex-7b-v1).
  The ModelScope `nvidia/personaplex-7b-v1` ID does not exist.
- Official comparison: [NVIDIA model manifest](https://huggingface.co/api/models/nvidia/personaplex-7b-v1/revision/fdaf4090a61cb315c138a1faee287ffd6c716309?blobs=true).
- All 16 mirror files downloaded using the existing ModelScope 1.39.1 SDK.
  No credential, environment, or container was replaced.
- Main weights, Mimi checkpoint, SentencePiece tokenizer, and `voices.tgz`
  match the official sizes and SHA-256 digests. `config.json` matches its
  official Git blob. The voice archive contains 18 presets, including NATF2.
- Use remains subject to the NVIDIA Open Model License stated by the model
  card. Downloading this mirror did not repair the existing HF credentials.

Container model directory:

```text
/root/.cache/huggingface/modelscope/models/nv-community--personaplex-7b-v1/snapshots/master
```

Host equivalent is under `/var/lib/docker/huggingface/modelscope/models/`.
The verifier and full manifests are saved under the workspace artifact
directory `duplex_official_alignment_20260907/`.

## Shared-entrypoint fix

The first real-model run exposed an invalid 16 kHz synthetic append after
24 kHz PersonaPlex input. Shared native-duplex handling had implicitly applied
MiniCPM's silent decision-unit policy to every model-turn adapter.

Synthetic silence now requires the serving adapter to explicitly declare
`supports_silence_continuation = True`. MiniCPM opts in and keeps its previous
behavior. PersonaPlex does not opt in. The existing continuous-stream guard
also remains in force for Nemotron. KV retention alone never authorizes
generating additional model input.

Two regression variants, before and after a visible response exists, fail
against the pre-fix runtime and pass against the fixed runtime. Existing
MiniCPM silence-continuation tests remain passing.

## Evidence

- PersonaPlex L1 before the policy change: 50 passed, 1 deselected.
- Real-asset CPU checks: NATF2/persona prefill loads (77 slots); Mimi encodes
  one frame into `(1, 8)` codes and decodes finite `(1, 1920)` PCM.
- Shared handler and model-adapter regression after the fix: 310 passed,
  1 deselected. No new CI job is needed; the existing CPU sweep collects it.
- GPU input: NVIDIA's 40-second `assets/test/input_assistant.wav` at official
  commit `3428dfd95309a7f3c84fd93259ded0f810d1ff91`, plus the driver's 2-second
  silence tail. Git blob: `998d7db975afe7ce970990d8a0b2c3aec168ea60`.
- Instrumented run: two concurrent sessions; third session rejected with
  `resource_exhausted`; close primary, admit replacement, and keep the survivor
  on its original response and physical KV request.

| Session | Append receipts | Audio frames | Context tokens | Voiced frames |
| --- | ---: | ---: | --- | ---: |
| Primary | 525 | 521 | 78 to 602 | 269 |
| Survivor, including slot reuse | 1050 | 1046 | 78 to 1127 | 529 |
| Replacement | 525 | 521 | 78 to 602 | 233 |

Each session's receipt sequence is contiguous and its context monotonically
increases. Physical request IDs are distinct across sessions and stable within
each session. No successful-session protocol errors occurred. The deliberately
rejected overflow session is excluded from that statement.

The first, pre-fix run also timed out on secondary frame coverage. One later
passing run alone does not establish that every possible missing-frame race
has been eliminated. The original client was rerun without receipt logging
and also passed all existing gates, with the same output-frame counts and
audio RMS values. This checks for instrumentation-related timing effects; it
does not constitute a stress test.

Runtime Python fingerprint, excluding unrelated profiler work:

```text
ef14c8def60eabbf4e74aaf04ba405d9f5b6572038fcef5c3499afa60f5dd328
```

Instrumented launch record: `results/personaplex-silence-policy-fixed/launch.json`.
Uninstrumented launch record: `results/personaplex-policy-fixed-uninstrumented/launch.json`.
Both runs have matching before/after runtime fingerprints and empty
`remaining_owned` lists. Shutdown logs include a Python resource-tracker
warning about one semaphore; no task-owned process remains after cleanup.
Full close events are preserved in `c2/trace-*-events.jsonl`; the stock driver's
earlier audio snapshots alone do not prove graceful response completion.

## Reproduction

Prerequisites: matching Linux vLLM 0.28.0/Omni environment and pytest development
dependencies. L1 uses no weights or GPU. From the repository root:

```bash
# Focused local regression.
python -m pytest -q tests/entrypoints/openai/test_continuous_duplex.py

# CI-like shared-entrypoint and adapter regression.
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python -m pytest -q \
  tests/entrypoints/openai/test_continuous_duplex.py \
  tests/entrypoints/openai_api/test_duplex_handler.py \
  tests/model_executor/models/personaplex/duplex \
  tests/model_executor/models/nemotron_voicechat/duplex \
  -m 'core_model and cpu' --run-level=core_model
```

H200 uses the existing `vllm-minghui` container. The ownership-contained artifact
launcher, rather than global test-fixture cleanup, checks GPU occupancy and
cleans only PID/create-time-matched processes. To reproduce the traced smoke
inside that container:

```bash
/root/full-duplex-kvappend-fix-20260905/.venv/bin/python \
  /root/duplex-official-alignment.2uKOQy/scripts/run_e2e.py \
  --arm personaplex --label personaplex-recheck --gpu 0 \
  --source-dir /root/duplex-official-alignment.2uKOQy/source-personaplex \
  --input-wav /root/duplex-official-alignment.2uKOQy/personaplex_input_assistant.wav \
  --trace-personaplex --client-timeout 90
```

Use a fresh label; the launcher refuses to overwrite an existing result.
Omit `--trace-personaplex` to use the unchanged in-repo client. This command
requires a free H200 GPU and the verified local model directory above.

## Remaining strict-close work

PersonaPlex's shared OpenAI adapter still declares the default `model_turn`
lifecycle. Its delayed codebooks and chunked codec output need an explicit
stream-finalization/drain contract before moving it to `continuous_stream`.
Removing the invalid implicit silence append is not that migration. Do not
hide tail loss by relaxing frame gates, pad invented audio, or relabel a
cancelled response as completed. The independent browser/FrameStepper backend
was not validated by this shared-engine smoke.
