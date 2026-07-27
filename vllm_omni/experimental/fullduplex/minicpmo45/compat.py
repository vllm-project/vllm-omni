from __future__ import annotations

from typing import Any


def patch_minicpmo_remote_config(config: Any) -> None:
    """Fill config fields expected by the vLLM MiniCPM implementation."""
    seen: set[int] = set()

    def patch_cfg(cfg: Any) -> None:
        if cfg is None or id(cfg) in seen:
            return
        seen.add(id(cfg))
        if getattr(cfg, "base_model_tp_plan", None) is None:
            setattr(cfg, "base_model_tp_plan", {})
        for value in getattr(cfg, "__dict__", {}).values():
            if value is cfg:
                continue
            if hasattr(value, "__dict__") and (
                value.__class__.__name__.endswith("Config")
                or hasattr(value, "model_type")
                or hasattr(value, "base_model_tp_plan")
            ):
                patch_cfg(value)

    patch_cfg(config)
    tts_config = getattr(config, "tts_config", None)
    if tts_config is None:
        return
    defaults = {
        "top_p": 0.8,
        "top_k": 100,
        "temperature": 0.8,
        "repetition_penalty": 1.05,
    }
    for name, value in defaults.items():
        if isinstance(tts_config, dict):
            tts_config.setdefault(name, value)
        elif not hasattr(tts_config, name):
            setattr(tts_config, name, value)


__all__ = ["patch_minicpmo_remote_config"]
