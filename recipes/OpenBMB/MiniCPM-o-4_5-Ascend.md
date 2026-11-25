# MiniCPM-o 4.5 on two Ascend cards

This recipe runs one 40,960-token Thinker with DSpark 7, two 4,096-token
Talker replicas with K16, and two Code2Wav replicas with five-step FP32 CFM.
All stages use MRv2. The shared KV layout is scoped to MiniCPM-o on Ascend and
uses the stock vLLM-Ascend allocator.

- [Two-card deployment template](../../vllm_omni/deploy/minicpmo_4_5_npu_2card.yaml)
- [Launcher](serve_minicpmo_ascend.sh)
- [Checkpoint manifest](minicpmo_ascend_checkpoint_manifest.json)

## Dependencies

Use two Ascend 910B3 cards (64 GiB each), preferably on the same NUMA node.
Start with the `quay.nju.edu.cn/atlas-ci/vllm-ascend:v0.34.0` image, the matching
Ascend driver, and `numactl`. The measured image digest is
`sha256:4ec880dc132cc9c3993d914039b2879db5db4f432ef36211417fcdcb9249d590`.
It contains vLLM `62d4bc3c7`, vLLM-Ascend `19c063b54`, and CANN 9.3.
Mount the vLLM-Omni checkout and model assets into the container. Preserve the
image's CANN environment and Python paths. The launcher prepends this checkout
to `PYTHONPATH`.

## Checkpoint assets

This deployment uses a prepared **W8A8 Thinker checkpoint** in compressed-tensors
format with symmetric per-channel INT8 weights and dynamic per-token INT8
activations. The vision/audio encoders, Talker, and Code2Wav retain their
original weights. Use the checkpoint manifest to verify the model assets.

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

Run from the repository root after setting `MODEL_ROOT` and `DRAFT_MODEL`:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export NUMA_NODE=0
export PORT=8000
export PROFILING_MODE=dynamic TASK_QUEUE_ENABLE=1
bash recipes/OpenBMB/serve_minicpmo_ascend.sh
```

The launcher renders `minicpmo_4_5_npu_2card.yaml` with the prepared W8A8
Thinker and local DSpark checkpoint. Using the YAML directly with the original
checkpoint selects the shared Ngram settings.
It creates mode files in a per-service temporary directory. Set `RUNTIME_DIR`
to a unique path to keep the rendered deployment and mode files.

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
