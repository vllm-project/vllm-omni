# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Normalize a Ming TTS deploy/talker checkpoint for the engine."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from vllm import envs


def prepare_talker_checkpoint(source: str, cache_dir: str) -> str:
    """Link deployment assets into an engine config view; leave sources intact."""
    deployment = Path(source).resolve()
    # The engine may call the resolver again on our already normalized view.
    if (deployment / "config.json").is_file():
        config = json.loads((deployment / "config.json").read_text())
        if config.get("model_type") == "ming_flash_omni_talker":
            return str(deployment)
    root = (deployment / "talker").resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"Expected deploy/talker/config.json at {deployment}")
    with (root / "config.json").open() as handle:
        config = json.load(handle)
    if "BailingTalker2" not in config.get("architectures", []):
        raise ValueError(f"Expected a converted BailingTalker2 checkpoint at {root}")
    for required in ("llm/config.json", "vae/config.json", "vae/model.safetensors", "campplus.onnx"):
        if not (root / required).is_file():
            raise FileNotFoundError(f"Incomplete talker checkpoint: {root / required}")
    if not list(root.glob("model*.safetensors")):
        raise FileNotFoundError(f"No talker safetensors in {root}")
    with (root / "llm/config.json").open() as handle:
        config["llm_config"] = json.load(handle)
    config.update(
        model_type="ming_flash_omni_talker",
        strict_weight_loading=True,
        max_decode_steps=1000,
        max_cache_len=2304,
        # Converted exports retain their text frontend; use cached VAE decoding by default.
        normalize_text=False,
        scrub_markup=True,
        duration_limit=False,
        stream_decode=True,
        # None follows native Ming: trim full output, retain streaming chunks.
        trim_trailing_silence=None,
    )
    config_json = json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    # Rebuild older views that did not link the checkpoint's TN resource directory.
    digest = hashlib.sha256(("model-tn-assets-v2\0" + str(deployment) + config_json).encode()).hexdigest()[:20]
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / digest
    if target.is_dir():
        if (target / "config.json").read_text() != config_json:
            raise RuntimeError(f"Conflicting talker config cache at {target}")
        return str(target)
    temporary = Path(tempfile.mkdtemp(prefix=f".{digest}-", dir=cache))
    try:
        for entry in root.iterdir():
            if entry.name not in {"config.json", "data"}:
                (temporary / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        (temporary / "config.json").write_text(config_json)
        try:
            temporary.rename(target)
        except OSError:
            if not target.is_dir() or (target / "config.json").read_text() != config_json:
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(target)


def resolve_ming_talker_checkpoint(model: str, revision: str | None, task_type: str | None) -> str:
    """Resolve a converted Ming3 export from a local directory or Hub snapshot."""
    del task_type
    if "://" in model:
        raise ValueError("Ming3 TTS expects a local directory or a Hugging Face Hub model ID")
    source = model
    if not Path(source).is_dir():
        if Path(source).is_absolute() or source.startswith(("./", "../", "~")):
            raise FileNotFoundError(f"Ming3 TTS checkpoint directory does not exist: {source}")
        from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

        source = download_weights_from_hf_specific(
            model_name_or_path=model,
            cache_dir=None,
            allow_patterns=["config.json", "talker/**"],
            revision=revision,
            require_all=True,
        )
    # Generated config views use the runtime cache, not the HF weight cache.
    return prepare_talker_checkpoint(source, cache_dir=str(Path(envs.VLLM_CACHE_ROOT) / "ming_talker"))
