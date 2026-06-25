# Local test scenarios (matrix)

Canonical **Common stack** snippet for the **Test Result** section in the release report (`compose_full_report.py` → `### Common stack (all rows)`).

Per-GPU **H200 / H800 / A100** tables use the same grouped Summary layout as the **nightly** report when you pass `--log-dir-h200`, `--log-dir-h800`, `--log-dir-a100` (see [nightly-local-log-layout.md](nightly-local-log-layout.md)). **Issue tracking** in the release report is generated separately (GitHub Search: `label:ci-failure` and **local test** in the title).

## Common stack (all rows)

Torch **2.11.0**, Python **3.12**, Diffusers **0.37.1**, transformers **5.8.0**, huggingface_hub **1.13.0**.
