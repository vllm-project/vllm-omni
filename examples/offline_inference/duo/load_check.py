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
2. ``realization/complete.json`` records ``"mode": "lora"`` with no staged
   adapter file: the adapter is folded into the backbone, so the loader must be
   told ``backbone_includes_adapter``.

Duo stage options have to ride on the engine args, because a deploy YAML's
``model_config`` block is silently dropped (``StageDeployConfig`` has no such
field).

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
    "backbone_includes_adapter": True,
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
        "--steps",
        type=int,
        default=50,
        help="causal-base ships Euler/50 at flow shift 1.0 (see realization/complete.json)",
    )
    args = ap.parse_args()

    root = Path(args.model)
    backbone = root / "realization" / "backbone"
    ensure_backbone_config(backbone)

    from vllm_omni.entrypoints.omni import Omni

    print(f"deploy={args.deploy_config}")
    print(f"model={backbone}")
    engine = Omni(
        model=str(backbone),
        deploy_config_path=args.deploy_config,
        model_class_name=BACKBONE_ARCH,
        # An engine-args field for a single-stage diffusion deploy.
        step_execution=True,
        # Duo stage options, see the module docstring.
        model_config={"duo": DUO_OPTIONS},
        disable_log_stats=True,
    )
    print("engine constructed")
    print("stage0 metadata:", engine.engine.get_stage_metadata(0))
    print("LOAD CHECK DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
