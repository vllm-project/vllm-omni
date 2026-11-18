# MiniCPM-o 4.5 on two Ascend cards

This profile retains a 40,960-token Thinker, a 4,096-token Talker, MRv2,
two Talker replicas, two Code2Wav replicas, and five-step FP32 CFM.
It is an explicit deployment profile; it does not change the default recipe.

- [Deployment template](../../vllm_omni/deploy/minicpmo_4_5_ascend.yaml)
- [Launcher](serve_minicpmo_ascend.sh)
- [Qualified checkpoint manifest](minicpmo_ascend_checkpoint_manifest.json)

## Dependencies

The qualified vLLM 0.33 reproduction snapshot is commit `676a1a42f` on
`repro/minicpmo-ascend-v033`. Check out that snapshot before following the
version-pinned commands below. Upstream main subsequently moved to vLLM 0.34
and removed MRv1; the rebased PR retains that upstream change. Functional
results from the 0.33 snapshot do not qualify the current main-based source
with the older dependency set. The new main-based integration needs a separate
compatible-environment functional run; no performance/quality rerun is implied.

```bash
git fetch origin repro/minicpmo-ascend-v033
git switch --detach 676a1a42f
```

The functional reproduction uses the following source revisions and packages.
Use the unmodified upstream repositories, not an experiment worktree or an
Ascend source overlay. Older versions without the standardized shared KV
allocator are not covered by this profile.

| Component | Qualified version |
| --- | --- |
| Hardware | Two Ascend 910B3 cards, 64 GiB each, same NUMA node |
| CANN | 9.2.1 |
| Python | 3.12.16 |
| vLLM | `4ba787ec9959e566d3b587a796dff07673324da5`, version `0.33.0+empty` |
| vLLM-Ascend | `221fb096cd1cdf9f0b273da9b9624fbf4bb7f200` |
| PyTorch | `2.10.0+cpu` |
| torch-npu | `2.10.0.post7` |
| Transformers | `5.19.0` |
| Triton / Triton-Ascend | `3.6.0` / `3.6.0.post2` |
| compressed-tensors | `0.17.0` |
| safetensors | `0.8.0` |

Start with a compatible Ascend/CANN build environment and install its build
requirements using the upstream NPU installation instructions. For existing
matching binary dependencies, select the source revisions explicitly:

```bash
git clone https://github.com/vllm-project/vllm.git /opt/vllm
git -C /opt/vllm checkout 4ba787ec9959e566d3b587a796dff07673324da5
git clone https://github.com/vllm-project/vllm-ascend.git /opt/vllm-ascend
git -C /opt/vllm-ascend checkout 221fb096cd1cdf9f0b273da9b9624fbf4bb7f200
VLLM_TARGET_DEVICE=empty python3 -m pip install --no-deps --no-build-isolation -e /opt/vllm
python3 -m pip install --no-deps --no-build-isolation -e /opt/vllm-ascend
python3 -m pip install --no-deps --no-build-isolation -e /path/to/this/vllm-omni
```

`--no-deps` assumes the table's dependencies are already installed. It does not
install CANN, drivers or a compatible torch-npu build. Source CANN's environment
script before invoking the launcher; preserve its Python and library paths.
Install `numactl` for host placement. No edit to vLLM or vLLM-Ascend is required.

## Checkpoint assets

The historical optimized runs used a prepared **W8A8 Thinker checkpoint**,
not the original BF16 checkpoint with a runtime flag. The qualified checkpoint
uses compressed-tensors, symmetric per-channel INT8 weights and dynamic
per-token INT8 activations; the vision/audio encoders, Talker and Code2Wav
remain outside that conversion. Model assets are external data dependencies
and are not included in the source PR. Using another quantization or draft
checkpoint does not reproduce the historical measurements.

Prepare this layout, with the original tokenizer/processor assets also present
at `MODEL_ROOT`:

```text
MODEL_ROOT/
  config.json, tokenizer.json, processing_minicpmo.py, ...
  original/          # original openbmb/MiniCPM-o-4_5 checkpoint
  thinker-w8a8/      # qualified compressed-tensors checkpoint
DRAFT_MODEL/
  config.json
  model.safetensors
```

The Thinker draft source is
`Qun1023/MiniCPM-o-4_5-thinker-DSpark-mm`, revision
`c4c310aca2c768343165261cb22c21b90c1f79e0`. Its qualified weights have SHA256
`6189fd34847451de1527a76d68ba5811bbd6228411ecd5c979b14155e8a409ac`.
The manifest records the exact checkpoint configurations, Python assets and
weight hashes used for reproduction. Supply the qualified W8A8 artifact;
the original HF model download alone is insufficient for this deployment.

For example, create a new model root from existing checkpoints:

```bash
export ORIGINAL_MODEL=/models/MiniCPM-o-4_5
export QUANTIZED_MODEL=/models/MiniCPM-o-4_5-w8a8
export MODEL_ROOT=/models/minicpmo-serving
export DRAFT_MODEL=/models/minicpmo-thinker-dspark
mkdir -p "$MODEL_ROOT"
ln -s "$ORIGINAL_MODEL" "$MODEL_ROOT/original"
ln -s "$QUANTIZED_MODEL" "$MODEL_ROOT/thinker-w8a8"
for asset in "$ORIGINAL_MODEL"/*; do
  ln -s "$asset" "$MODEL_ROOT/$(basename "$asset")"
done
```

Verify the model assets from the repository root. This reads checkpoint files
but performs no inference or benchmark:

```bash
python3 - "$MODEL_ROOT" "$DRAFT_MODEL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path("recipes/OpenBMB/minicpmo_ascend_checkpoint_manifest.json").read_text())
roots = {"original": Path(sys.argv[1]) / "original",
         "thinker-w8a8": Path(sys.argv[1]) / "thinker-w8a8",
         "draft-thinker": Path(sys.argv[2])}
for group, files in manifest["assets"].items():
    for name, expected in files.items():
        path = roots[group] / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == expected["sha256"], str(path)
print("Checkpoint assets match")
PY
```

## Launch

Run from a checkout containing this PR. `PYTHONPATH` must contain only the
intended dependency/CANN paths; remove experiment source overlays first.
The launcher prepends this checkout and prints the rendered deployment path.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export NUMA_NODE=0
export MAX_NUM_SEQS=16
export TALKER_ROUTE=quarter VOCODER_ROUTE=quarter
export PORT=8000
bash recipes/OpenBMB/serve_minicpmo_ascend.sh
```

The initial graph compilation can take several minutes. The service is ready
when `/health` succeeds. Fast graph variants qualify on their first eligible
request in each worker: prefill compares hidden state and KV; fused decoding
compares codec tokens, stop decisions and KV; compact decoding compares all
frames in the first K-step. A mismatch retains the reference path. Exact
lengths or shapes never encountered remain unqualified. No private benchmark
warmup script or pre-existing mode files are required.

The launcher creates mode files in `RUNTIME_DIR`, or a new temporary directory.
Their paths must be unique per service. The historical environment variable
names containing `C1`/`C8` are retained for compatibility; file names do not use
concurrency suffixes. Do not modify them while collecting comparable results.

| Profile | `MAX_NUM_SEQS` in every stage | Talker / Code2Wav route |
| --- | --- | --- |
| Historical five-tier campaign | `8` | C1: `primary` / `primary`; C8–C64: `quarter` / `quarter` |
| Historical high-concurrency campaign | `16` | `quarter` / `quarter` |

`quarter` places approximately one quarter of new requests on the secondary
replica, with sticky routing for the request's streaming state. It is not an
ingress concurrency limit. Both profiles keep 40K, two Talker/Code2Wav replicas,
and five-step CFM. The high-concurrency profile improved historical throughput
but had a worse C64 streaming tail; the PR retains that tradeoff explicitly.
The old 40-request ingress cap is not part of this recipe.

For the historical C1 profile, set `MAX_NUM_SEQS=8`, `TALKER_ROUTE=primary` and
`VOCODER_ROUTE=primary` before launching. For the other five-tier cases, keep
`MAX_NUM_SEQS=8` and use `quarter` for both routes. These commands reproduce the
configuration; they do not assert that the rebased source reproduces old timing
or quality numbers without a new measurement campaign.

## Optimization inventory

The adopted runtime source delta was checked against the retained experiment
inventory. The following components are in this repository:

| Adopted component | Implementation / activation |
| --- | --- |
| Thinker W8A8 | Native Ascend compressed-tensors loader; `thinker-w8a8` checkpoint asset |
| Thinker DSpark 7 and native gather | `minicpmo_4_5_omni.py`; stage-0 DSpark config and launcher flag |
| Thinker/Talker exact prefill | `platforms/npu/worker/prefill_graph.py`; active mode with per-length shadow qualification |
| Talker K16 and terminal-tail skip | `worker/talker_multiframe.py`, `npu_omni_v2_model_runner.py`; ngram-15 scheduling config |
| Talker native FIA/KV capacity buckets | `platforms/npu/attention/static_shape_backend.py`; static graph setup, ordinary scatter retained |
| Talker fused sampling, feedback and 8+8 exit | `talker_fused_decode.py`, `talker_fused_sampling*.py`, `talker_exact_rotary.py`; fused mode |
| Compact multi-request Talker | `talker_batched_decode.py`; `active_fixed`, guarded by full-frame shadow checks |
| Short/long Talker context | `talker_context.py`; 512-slot fast graphs within a 4096-token model, long requests fall back |
| Replica placement and sticky routing | `engine/stage_pool.py`; two replicas and route files |
| Reference cache, early setup and host spill | `batched_token2wav.py`, `minicpmo_4_5_code2wav.py`, orchestrator; cache and secondary-memory flags |
| CFM, HiFT and encoder graph coverage | `cuda_graph_wrapper.py`, `flow_encoder_graph.py`; deployment graph buckets and tail precapture |
| Codec chunking and bounded copies | Stage input processor and `batched_token2wav.py`; 32-frame chunks, secondary micro-batch and split copies |
| 40K shared KV layout | `platforms/npu/minicpmo_shared_kv.py`; MiniCPM-o MRv2 eligibility, stock Ascend allocator |
| Native duplex continuation/terminal fixes | Duplex data plane, stage-0/window state and Talker per-segment reset |
| NUMA placement | Launcher `numactl`; choose the node local to the selected card pair |

The 512-token **model limit**, reduced Thinker context, ingress cap, reduced CFM
steps, experimental fused RoPE and rejected ready-merge variants are not adopted.
Some CFM/HiFT batching mechanisms already exist in upstream main; their presence
is not evidence of an additional isolated PR speedup. Individual historical
ablations in the PR cannot be added together across different deployments.

## External-code audit and validation

The Ascend checkout's tracked Python files are unmodified. The original
container's vLLM checkout contained two KV debug-log-only edits; neither changes
its executable allocation decisions. Reproduction uses clean vLLM source from
the pinned commit, without those edits. The official Triton-Ascend wheel
overlays parts of the base Triton wheel; those files match Triton-Ascend's
package RECORD and are not private patches. PyTorch, torch-npu, Transformers,
safetensors and compressed-tensors Python files match their installed distribution metadata.
The old editable Omni install and all experiment source overlays are excluded.

This audit covers the relevant source repositories and listed Python packages;
it is not a rebuild or byte-level certification of drivers/CANN binaries.
The checkpoint manifest makes model assets explicit rather than treating them
as source patches. No performance or accuracy datasets were rerun for this audit.

Focused qualification checks, with the dependency table and CANN paths installed:

```bash
pytest -q tests/platforms/npu/test_minicpmo_graph_qualification.py
pytest -q tests/platforms/npu/test_minicpmo_graph_qualification.py tests/platforms/npu/test_minicpmo_shared_kv.py -m 'core_model and npu and A3 and omni' --run-level core_model
```
