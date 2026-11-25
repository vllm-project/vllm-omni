# Duplex-MPE — Multi-Party Full-Duplex Interaction Benchmark

**Duplex-MPE** ([arXiv:2609.31948](https://arxiv.org/abs/2609.31948), [Project Page](https://step-out.github.io/Duplex-MPE-Page/), [GitHub](https://github.com/step-out/Duplex-MPE), [Hugging Face Dataset](https://huggingface.co/datasets/ChengqianMa/Duplex-MPE)) evaluates **selective participation** for spoken full-duplex models in multi-party rooms where human speakers talk both to **each other** and to an ambient voice assistant ("Aria").

Instead of single-speaker prompt-response testing, Duplex-MPE streams continuous room audio on a real-time wall clock (`mpeval.unified.v3`) over vLLM-Omni's [`WS /v1/realtime?duplex=1`](../../docs/serving/realtime_duplex_api.md) endpoint and evaluates four core behaviors across **2,000 paired scenarios** (`4,000` total across `explicit` and `implicit` addressing; `20` pairs / `40` scenarios in `--mini` mode):

| Metric | Symbol | Event Window | Definition |
| --- | --- | --- | --- |
| **Fresh-Onset Target Response** | `TR` | `T` | Assistant begins a fresh spoken response within `5.0 s` after the addressed user turn `T` finishes (`T_HIT_5S`, plus fresh-onset `T_OVERLONG_60S` where `speaking_at_t_end=false`). |
| **Answer Accuracy (Cond. / Uncond.)** | `AA` | `T` | Assistant's spoken answer matches the multi-turn room facts (`gold_answer`) on responded `T` turns (`Cond.`) and across all `T` scenarios (`Uncond.`). |
| **Silence Preservation** | `SP` | `N1`, `N2`, `N3` | Assistant stays silent or emits only brief phatic backchannels (`BACKCHANNEL`) during human-to-human exchanges rather than `SUBSTANTIVE_INTRUSION`. |
| **Answering-Window Yield** | `AY` | `N4_Q -> N4_R` | When an unaddressed human question (`N4_Q`) draws the assistant into speaking during the `3.0 s` pause and another human answers (`N4_R`), the assistant yields the floor within `2.0 s` of `N4_R` onset. |

Auxiliary diagnostics `RP` (response presence, including continuations already active at `T` end) and `WR` (`N4` window response rate) are reported alongside the four primary capabilities.

---

## Evaluation Modes on vLLM-Omni

`benchmarks/duplex_mpe` inspects `session.created.session.capabilities` on `/v1/realtime?duplex=1` (with automatic fallback to turn-based `/v1/realtime` when `session_mode: turn` is deployed):

1. **Scored Native Full-Duplex (`--protocol native` / `auto`)**:
   - Models advertising `implementation_level: "model_native_duplex"` (e.g., `openbmb/MiniCPM-o-4_5`, `nvidia/personaplex-7b-v1`, Nemotron VoiceChat, Haru).
   - Opens `/v1/realtime?duplex=1&autostart=0` with `turn_detection: null`, `extra_body.auto_response: true`, and `extra_body.silence_continuation: false` so the model hears only the benchmark's continuous audio stream.
   - Reports full `scored_full_duplex` metrics (`TR`, `AA`, `SP`, `AY`, `RP`, `WR`) under `evaluation_clock: "realtime_wall_clock"` (`1.0x` real-time streaming, where end-to-end serving latency and underruns count toward the `5.0 s` `T` onset and `2.0 s` `N4_R` yield windows, distinct from offline in-process `audio_samples_excluding_compute_time` adapters).
2. **Unscored Turn-Based Probe (`--protocol turn_based` or `openai_turn`)**:
   - Turn-committed deployments (`Qwen/Qwen3-Omni-30B-A3B-Instruct` on `qwen3_omni_duplex_mrv2.yaml` or `session_mode: turn` `/v1/realtime`).
   - Uses `turn_detection: {"type": "server_vad", "silence_duration_ms": 500}` and omits `TR`, `AY`, and `RP` (`evaluation_mode: "unscored_turn_based_probe"`), consistent with Duplex-MPE Section 3.4.

---

## Prerequisites

Install the lightweight audio and VAD packages if not already present in your environment:

```bash
pip install "silero-vad==6.2.1" soundfile scipy httpx websockets huggingface_hub
```

Pinning `silero-vad==6.2.1` (and optionally passing `--strict-reference-vad-version`) locks causal speech-activity segmentation to the reference Duplex-MPE paper configuration (`onset=0.5`, `offset=0.35`, `hangover=800 ms`, `t_endpoint_hangover=2000 ms`, `min_speech=120 ms`).

---

## Step 1: Prepare and Validate the Pinned Dataset

`prepare` downloads revision `6d5f59fab1237c14ee8229ea36f111b7423b0fe6` of [`ChengqianMa/Duplex-MPE`](https://huggingface.co/datasets/ChengqianMa/Duplex-MPE) and verifies SHA-256 digests on `dataset.json`, both manifests, `audio_inventory.json`, and `audio/aria_duty_preamble.wav`:

```bash
# Deterministic 20-pair (40-scenario) smoke subset:
python -m benchmarks.duplex_mpe prepare \
    --dataset-root benchmarks/cache/duplex_mpe_mini \
    --mini

# Full 2,000-pair (4,000-scenario) benchmark:
python -m benchmarks.duplex_mpe prepare \
    --dataset-root benchmarks/cache/duplex_mpe

# Optional full WAV SHA-256 + header verification:
python -m benchmarks.duplex_mpe validate \
    --dataset-root benchmarks/cache/duplex_mpe_mini \
    --mini \
    --verify-audio-checksums
```

---

## Step 2: Launch Services

### 1. Full-duplex model under test (example: MiniCPM-o 4.5)

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
    --trust-remote-code \
    --port 8091
```

### 2. Window ASR service (`Qwen/Qwen3-ASR-1.7B`)

Any OpenAI-compatible `/v1/audio/transcriptions` endpoint serving `Qwen/Qwen3-ASR-1.7B` can be used:

```bash
vllm serve Qwen/Qwen3-ASR-1.7B \
    --host 127.0.0.1 --port 8092
```

### 3. Semantic judge service

Copy [`configs/judge.example.json`](configs/judge.example.json) and point `base_url` / `model` (and optional `api_key_env`) at an OpenAI-compatible `/v1/chat/completions` endpoint (default reference judge: `Qwen/Qwen3.8-27B` with `enable_thinking: false`).

---

## Step 3: Run Evaluation

Run all phases end-to-end (`generate -> asr -> judge -> score`), or run each phase independently with checkpoint resume:

```bash
python -m benchmarks.duplex_mpe run \
    --dataset-root benchmarks/cache/duplex_mpe_mini \
    --output-dir benchmarks/results/duplex_mpe_minicpmo_mini \
    --mini \
    --phase all \
    --base-url http://127.0.0.1:8091 \
    --model openbmb/MiniCPM-o-4_5 \
    --max-concurrency 1 \
    --warmup 1 \
    --asr-base-url http://127.0.0.1:8092/v1 \
    --judge-config benchmarks/duplex_mpe/configs/judge.example.json \
    --strict-reference-vad-version
```

### Phase-by-Phase Execution & Resume

- `--phase generate`: Streams scenarios over `/v1/realtime?duplex=1`, saves retained assistant audio under `<output-dir>/audio/<scenario_id>.wav`, and checkpoints timing windows in `<output-dir>/report.json` (failed scenarios are isolated in `failures.json` for clean resume).
- `--phase asr`: Slices voiced `T` and `N1`/`N2`/`N3` windows from retained WAVs and checkpoints transcripts in `<output-dir>/transcribed.json`.
- `--phase judge`: Evaluates answer correctness and bystander intrusion with SHA-256 prompt locking into `<output-dir>/labels.json` (`judge_failures.json` isolates transient HTTP errors).
- `--phase score`: Validates timing invariants, audio digests, and VAD settings offline and writes `<output-dir>/RESULTS.md` and `<output-dir>/duplex_mpe_results.json` (exits `0` only when all selected scenarios and semantic tasks are complete).

---

## Output Artifacts

| File | Purpose |
| --- | --- |
| `run.json` | Frozen run manifest (protocol, sample rates, dataset/scenario/audio SHA-256 bindings, sanitized launch command). |
| `audio/<scenario_id>.wav` | Continuous sample-aligned assistant output audio retained for audit and ASR slicing. |
| `report.json` | Per-scenario `mpeval.unified.v3` turn windows, `T` timing outcome, and Silero VAD segments. |
| `tasks.json` / `transcribed.json` | Deterministic semantic evaluation tasks and window ASR transcripts. |
| `labels.json` | Prompt-locked semantic judge verdicts bound to `assistant_audio_sha256`. |
| `RESULTS.md` / `duplex_mpe_results.json` | Markdown summary table and machine-readable metrics + host/package provenance. |

---

## Citation

```bibtex
@article{ma2026duplexmpe,
  title   = {Duplex-MPE: Benchmarking Multi-Party Interaction in Full-Duplex Dialogue},
  author  = {Ma, Chengqian and Feng, Wenhao and Jin, Weixuan and Dai, Gaole and Xie, Tianyu and Ma, Yuexiao and Kang, Zhaolu and Zhao, Xiangyu and Zheng, Xiawu and Chao, Fei},
  journal = {arXiv preprint arXiv:2609.31948},
  year    = {2026}
}
```
