from __future__ import annotations

import json
import shutil
from pathlib import Path


def resolve_voxcpm_model_dir(model: str) -> Path:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return model_path

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model))


def prepare_voxcpm_hf_config_dir(model_dir: str | Path, hf_config_dir: str | Path) -> Path:
    model_dir = Path(model_dir).expanduser()
    hf_config_dir = Path(hf_config_dir).expanduser()
    hf_config_dir.mkdir(parents=True, exist_ok=True)

    source_config_path = model_dir / "config.json"
    if not source_config_path.exists():
        raise FileNotFoundError(f"VoxCPM config.json not found under {model_dir}")

    config_path = hf_config_dir / "config.json"
    shutil.copy2(source_config_path, config_path)

    source_generation_config_path = model_dir / "generation_config.json"
    if source_generation_config_path.exists():
        shutil.copy2(source_generation_config_path, hf_config_dir / "generation_config.json")

    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    config_dict["model_type"] = "voxcpm"
    config_dict.setdefault("architectures", ["VoxCPMForConditionalGeneration"])
    config_path.write_text(json.dumps(config_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    return hf_config_dir


__all__ = [
    "prepare_voxcpm_hf_config_dir",
    "resolve_voxcpm_model_dir",
]
