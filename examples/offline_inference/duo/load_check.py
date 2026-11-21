# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load the Duo Realization stage from a released bundle into a vLLM-Omni engine.

This is the smallest end-to-end check that the ported Realization pipeline
actually accepts the shipped weights: it builds the engine, streams all
backbone shards through the pipeline's strict loader, and reports the resulting
stage metadata. It does not run a duplex session.

Two things the released causal-base bundle needs that its config.json does not
state, both done here instead of being hidden in the loader:
1. The backbone config is Diffusers-style and carries no routing keys. The
   native loader selects a pipeline from a single-element ``architectures``
   entry naming DuoPipeline. Do not add ``model_type``: it makes
   ``AutoConfig`` reject the directory before the diffusion loader reads
   ``architectures``.
2. The trained student LoRA is **not** in the backbone. ``realization/``
   declares ``storage.format: dcp`` and the adapter lives in
   ``realization/dcp/__0_0.distcp`` (2.78 GB, rank 128). It is exported to a
   safetensors adapter with upstream's ``apps/duo/duo_dcp.py`` and passed via
   ``adapter_path`` with ``adapter_format="torchtitan_split_qkv"``. Do **not**
   set ``backbone_includes_adapter``: that tells the pipeline the adapter is
   already folded in and would silently run an untrained projection path.

Duo stage options ride on the engine args, e.g. ``model_config={"duo": {...}}``.
A ``model_config:`` block in a deploy YAML is not a ``StageDeployConfig`` field,
so it would land in ``engine_extras`` rather than the stage model config.


Usage::

    python examples/offline_inference/duo/load_check.py \
        --model /path/to/Duo-1.0-causal-base \
        --deploy-config vllm_omni/deploy/duo_causal_4gpu.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKBONE_ARCH = "DuoPipeline"

DUO_OPTIONS = {
    "compact_padding": True,
    # causal-base is not the DMD release: fused denoise hard-requires exactly
    # four NFEs, so it cannot be combined with the 50-step causal grid.
    "fused_denoise": False,
    "reuse_static_text": False,
    "reuse_time_conditioning": True,
    "kv_capacity": 131072,
    "page_size": 32,
    "kv_token_budget": 65536,
    "max_graphs": 8,
    "max_sessions": 1,
    "sampling_stage": "causal",
    "flow_shift": 1.0,
    "fp8_gemm": False,
    "joint_execution": False,
    "prefix_capacity": 0,
    # The trained causal-base LoRA is exported from realization/dcp/ and applied
    # on top of the frozen backbone; the backbone itself is unadapted.
    "adapter_format": "torchtitan_split_qkv",
    "rank": 128,
    "alpha": 128,
    "role_embedding": False,
}


def ensure_backbone_config(backbone: Path) -> None:
    """Give the Diffusers-style backbone config the key the native loader routes on.

    ``resolve_model_class_name`` returns a single-element ``architectures``
    verbatim, which is how DuoPipeline is selected. Do **not** add a
    ``model_type``: ``minimax_h3_dit`` is not a Transformers architecture, and
    writing it makes ``AutoConfig`` reject the directory before the diffusion
    loader ever looks at ``architectures``.
    """
    path = backbone / "config.json"
    cfg = json.loads(path.read_text())
    if cfg.get("architectures") == [BACKBONE_ARCH] and "model_type" not in cfg:
        return
    backup = backbone / "config.json.orig"
    if not backup.exists():
        backup.write_text(json.dumps(cfg, indent=2))
    # A stale model_type from an earlier patch attempt must not survive.
    cfg.pop("model_type", None)
    cfg["architectures"] = [BACKBONE_ARCH]
    path.write_text(json.dumps(cfg, indent=2))
    print(f"patched {path} with architectures=[{BACKBONE_ARCH}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="bundle root (contains realization/backbone)")
    ap.add_argument("--deploy-config", default="vllm_omni/deploy/duo_causal_4gpu.yaml")
    ap.add_argument(
        "--adapter",
        required=True,
        help=(
            "safetensors adapter exported from realization/dcp/ (upstream "
            "apps/duo/duo_dcp.py, layout=diffusers_split_qkv, rank=alpha=128)"
        ),
    )
    ap.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Realization TP; pass explicitly, the deploy config value alone did not take effect",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=50,
        help="causal-base ships Euler/50 at flow shift 1.0 (see realization/complete.json)",
    )
    args = ap.parse_args()

    root = Path(args.model)
    backbone = root / "realization" / "backbone"
    ensure_backbone_config(backbone)
    adapter = Path(args.adapter)
    if not adapter.is_file():
        raise SystemExit(f"adapter not found: {adapter}")
    options = dict(DUO_OPTIONS)
    options["adapter_path"] = str(adapter)
    options["adapter_manifest_path"] = str(root / "realization" / "complete.json")

    from vllm_omni.entrypoints.omni import Omni

    print(f"deploy={args.deploy_config}")
    print(f"model={backbone}")
    print(f"adapter={adapter}")
    engine = Omni(
        model=str(backbone),
        deploy_config_path=args.deploy_config,
        model_class_name=BACKBONE_ARCH,
        # Pass TP explicitly. Measured on 4xA100 with the released causal-base
        # bundle: with only the deploy config's `tensor_parallel_size: 4` the
        # engine resolved one stage at TP=1 and put all 66 GB on device 0
        # (OOM on the first denoise step); with this argument the four workers
        # report 16.6 GiB each.
        tensor_parallel_size=args.tensor_parallel_size,
        step_execution=True,
        # Duo stage options, see the module docstring.
        model_config={"duo": options},
        disable_log_stats=True,
    )
    print("engine constructed")
    print("stage0 metadata:", engine.engine.get_stage_metadata(0))
    print("LOAD CHECK DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
