"""Run the AR → DiT (diffusers) pipeline using YAML configuration defaults."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import yaml

from vllm_omni import (
    DiTConfig,
    OmniLLM,
    create_ar_stage_config,
    create_dit_stage_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a two-stage AR → DiT pipeline with config-derived defaults."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "ar_dit_local.yaml",
        help="Path to a YAML configuration describing the pipeline stages.",
    )
    parser.add_argument(
        "--prompt",
        default="A scenic watercolor painting of a lighthouse at sunset",
        help="Prompt passed to the AR stage.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt forwarded to the diffusion stage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic diffusion sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./omni_dit_output.png"),
        help="Destination path for the generated image.",
    )
    return parser.parse_args()


def _normalize_modalities(values: List[str] | None, default: List[str]) -> List[str]:
    if not values:
        return default
    return list(values)


def _load_stage_configs_from_yaml(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    stages = config.get("stages", [])
    if not stages:
        raise ValueError(f"No stages defined in config: {config_path}")

    stage_configs = []
    for stage in stages:
        stage_id = stage["stage_id"]
        engine_type = stage["engine_type"]
        default_stage_args = stage.get("default_stage_args")

        if engine_type == "AR":
            stage_configs.append(
                create_ar_stage_config(
                    stage_id=stage_id,
                    model_path=stage["model_path"],
                    input_modalities=_normalize_modalities(
                        stage.get("input_modalities"), ["text"]
                    ),
                    output_modalities=_normalize_modalities(
                        stage.get("output_modalities"), ["text"]
                    ),
                    default_stage_args=default_stage_args,
                )
            )
        elif engine_type == "DiT":
            dit_cfg_dict: Dict = dict(stage.get("dit_config", {}))
            if "num_inference_steps" not in dit_cfg_dict:
                raise ValueError(
                    "DiT stage requires 'num_inference_steps' in dit_config"
                )
            dit_cfg = DiTConfig(**dit_cfg_dict)

            stage_configs.append(
                create_dit_stage_config(
                    stage_id=stage_id,
                    model_path=stage["model_path"],
                    input_modalities=_normalize_modalities(
                        stage.get("input_modalities"), ["text"]
                    ),
                    output_modalities=_normalize_modalities(
                        stage.get("output_modalities"), ["image"]
                    ),
                    dit_config=dit_cfg,
                    default_stage_args=default_stage_args,
                )
            )
        else:
            raise ValueError(f"Unsupported engine_type '{engine_type}' in config")

    return stage_configs


def _apply_env_overrides(stage_configs):
    ar_override = os.environ.get("AR_MODEL")
    dit_override = os.environ.get("DIT_MODEL")

    for stage_config in stage_configs:
        if ar_override and stage_config.engine_type == "AR":
            stage_config.model_path = ar_override
        if dit_override and stage_config.engine_type == "DiT":
            stage_config.model_path = dit_override


def main():
    args = parse_args()

    stage_configs = _load_stage_configs_from_yaml(args.config)
    _apply_env_overrides(stage_configs)

    omni = OmniLLM(stage_configs)

    stage_overrides: Dict[int, Dict[str, object]] = {}
    if args.seed is not None or args.negative_prompt:
        for stage_config in stage_configs:
            if stage_config.engine_type != "DiT":
                continue
            override = stage_overrides.setdefault(stage_config.stage_id, {})
            if args.seed is not None:
                override["seed"] = args.seed
            if args.negative_prompt:
                override["negative_prompt"] = args.negative_prompt

    outputs = omni.generate(
        prompt=args.prompt,
        stage_overrides=stage_overrides if stage_overrides else None,
    )

    image = None
    for request_output in outputs:
        for completion in getattr(request_output, "outputs", []) or []:
            candidate = getattr(completion, "image", None)
            if candidate is not None:
                image = candidate
                break
        if image is not None:
            break

    if image is None:
        print("No image found in outputs.")
        return

    out_path = args.output.resolve()
    try:
        image.save(out_path)
        print(f"Saved image to {out_path}")
    except Exception:
        print("Generated output is not a PIL.Image; skipping save.")


if __name__ == "__main__":
    main()
