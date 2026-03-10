import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional in some minimal environments
    hf_hub_download = None


def _to_plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_dict(value: Any) -> dict:
    plain = _to_plain(value)
    if isinstance(plain, dict):
        return plain
    return {}


def _node_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    try:
        return node.get(key, default)
    except Exception:
        return getattr(node, key, default)


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper() == "NONE":
        return None
    return text


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        text = _normalize_optional_text(value)
        if text is not None:
            return text
    return None


def resolve_model_cfg_block(config: Any) -> Any:
    model_cfg = getattr(config, "model", None)
    model_cfg_dict = _as_dict(model_cfg)
    if "dynin_omni" in model_cfg_dict:
        return getattr(model_cfg, "dynin_omni", model_cfg_dict["dynin_omni"])
    if "mmada" in model_cfg_dict:
        return getattr(model_cfg, "mmada", model_cfg_dict["mmada"])
    raise ValueError("Config is missing model.dynin_omni/model.mmada block.")


def resolve_model_pretrained_source(config: Any, default: str | None = None) -> str:
    model_cfg = resolve_model_cfg_block(config)
    source = _first_non_empty(
        _node_get(model_cfg, "repo_id", None),
        _node_get(model_cfg, "pretrained_model_path", None),
        default,
    )
    if source is None:
        raise ValueError(
            "Model source is missing. Set model.dynin_omni.repo_id (or model.mmada.repo_id) "
            "or keep legacy pretrained_model_path."
        )
    return source


def resolve_tokenizer_source(config: Any, default: str | None = None) -> str:
    model_cfg = resolve_model_cfg_block(config)
    source = _first_non_empty(
        _node_get(model_cfg, "tokenizer_repo_id", None),
        _node_get(model_cfg, "tokenizer_path", None),
        _node_get(model_cfg, "repo_id", None),
        _node_get(model_cfg, "pretrained_model_path", None),
        default,
    )
    if source is None:
        raise ValueError("Tokenizer source is missing. Set model.dynin_omni.tokenizer_repo_id (or tokenizer_path).")
    return source


def resolve_model_local_files_only(config: Any, default: bool = False) -> bool:
    model_cfg = resolve_model_cfg_block(config)
    return bool(_node_get(model_cfg, "local_files_only", default))


def resolve_model_type_from_pretrained(
    pretrained_source: str,
    *,
    local_files_only: bool = False,
) -> str:
    """
    Resolve model_type from config_vllm.json without dynamic AutoConfig imports.

    This avoids following `auto_map` pointers to external repositories.
    """
    local_source = Path(str(pretrained_source)).expanduser()
    local_config_path = local_source / "config_vllm.json"
    if local_config_path.is_file():
        with local_config_path.open("r", encoding="utf-8") as handle:
            cfg_json = json.load(handle)
        return str(cfg_json.get("model_type", ""))

    if hf_hub_download is None:
        raise RuntimeError(f"huggingface_hub is required to read remote config_vllm.json for '{pretrained_source}'.")

    try:
        config_path = hf_hub_download(
            repo_id=str(pretrained_source),
            filename="config_vllm.json",
            repo_type="model",
            local_files_only=bool(local_files_only),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download config_vllm.json from '{pretrained_source}'. "
            "If this is a private repo, run `huggingface-cli login` with an authorized token."
        ) from exc

    with open(config_path, encoding="utf-8") as handle:
        cfg_json = json.load(handle)
    return str(cfg_json.get("model_type", ""))


def resolve_vq_cfg_block(config: Any, modality: str = "image") -> Any:
    model_cfg = getattr(config, "model", None)
    model_cfg_dict = _as_dict(model_cfg)
    if modality == "image":
        if "vq_model_image" in model_cfg_dict:
            return getattr(model_cfg, "vq_model_image", model_cfg_dict["vq_model_image"])
        if "vq_model" in model_cfg_dict:
            return getattr(model_cfg, "vq_model", model_cfg_dict["vq_model"])
        raise ValueError("Config is missing model.vq_model_image/model.vq_model block.")
    if modality == "audio":
        if "vq_model_audio" in model_cfg_dict:
            return getattr(model_cfg, "vq_model_audio", model_cfg_dict["vq_model_audio"])
        raise ValueError("Config is missing model.vq_model_audio block.")
    raise ValueError(f"Unsupported modality '{modality}'.")


def resolve_vq_repo_source(vq_cfg: Any, default: str | None = None) -> str:
    source = _first_non_empty(
        _node_get(vq_cfg, "repo_id", None),
        _node_get(vq_cfg, "vq_model_name", None),
        default,
    )
    if source is None:
        raise ValueError("VQ model source is missing. Set repo_id (or legacy vq_model_name).")
    return source


def resolve_hf_cache_root(config: Any, project_root: str | None = None) -> str:
    dataset_cfg = getattr(config, "dataset", None)
    dataset_params = _as_dict(_node_get(dataset_cfg, "params", {}))
    dataset_hf_cfg = _as_dict(_node_get(dataset_cfg, "hf", {}))
    configured = _first_non_empty(
        dataset_hf_cfg.get("cache_dir"),
        dataset_params.get("hf_cache_dir"),
        os.getenv("DYNIN_OMNI_HF_CACHE_DIR"),
    )
    if configured is not None:
        cache_root = Path(configured).expanduser()
    else:
        anchor = Path(project_root).expanduser() if project_root else Path.cwd()
        cache_root = anchor / "datasets" / "huggingface"

    if not cache_root.is_absolute():
        anchor = Path(project_root).expanduser() if project_root else Path.cwd()
        cache_root = anchor / cache_root
    return str(cache_root.resolve(strict=False))


def configure_hf_cache_env(config: Any, project_root: str | None = None) -> str:
    cache_root = resolve_hf_cache_root(config, project_root=project_root)
    cache_root_path = Path(cache_root)
    os.environ["DYNIN_OMNI_HF_CACHE_DIR"] = cache_root
    os.environ["HF_HOME"] = cache_root
    os.environ["HF_DATASETS_CACHE"] = cache_root
    os.environ["HF_HUB_CACHE"] = str(cache_root_path / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
    # TRANSFORMERS_CACHE is deprecated and often preconfigured globally to
    # unwritable paths in shared clusters (e.g., /models).
    os.environ.pop("TRANSFORMERS_CACHE", None)
    cache_root_path.mkdir(parents=True, exist_ok=True)
    (cache_root_path / "hub").mkdir(parents=True, exist_ok=True)
    return cache_root


def _set_node_value(node: Any, key: str, value: Any) -> None:
    if isinstance(node, dict):
        node[key] = value
        return
    try:
        node[key] = value
        return
    except Exception:
        pass
    setattr(node, key, value)


def _ensure_mapping_node(node: Any, key: str) -> Any:
    existing = _node_get(node, key, None)
    if existing is not None:
        return existing
    _set_node_value(node, key, {})
    return _node_get(node, key, {})


def _canonical_t2i_id(entry: dict) -> str | None:
    source_id = (
        str(entry.get("id") or entry.get("name") or entry.get("dataset_id") or entry.get("repo_id") or "")
        .strip()
        .lower()
    )
    if not source_id:
        return None
    if "text-to-image-2m" in source_id or "text2image2m" in source_id:
        return "text2image2m"
    if "pickapic" in source_id:
        return "pickapic"
    if "flux-reason" in source_id or "flux_reason" in source_id or "fluxreason" in source_id:
        return "flux_reason"
    if "hq-edit" in source_id or "hq_edit" in source_id or "hqedit" in source_id:
        return "hqedit"
    if "ultraedit" in source_id:
        return "ultraedit"
    if "journeydb" in source_id:
        return "journeydb"
    return source_id


def _canonical_video_id(entry: dict) -> str | None:
    source_id = (
        str(entry.get("id") or entry.get("name") or entry.get("dataset_id") or entry.get("repo_id") or "")
        .strip()
        .lower()
    )
    if not source_id:
        return None
    if "llava-video" in source_id or "llavavid" in source_id:
        return "llavavid"
    if "webvid" in source_id:
        return "webvid10m"
    if "openvid" in source_id:
        return "openvid1m"
    if "panda70m" in source_id:
        return "panda70m"
    return source_id


def apply_dataset_sources(config: Any) -> None:
    """
    Harmonize deployment-friendly `dataset.sources` into legacy `dataset.params` keys.

    This keeps existing train code paths intact while allowing configs to stay concise.
    """
    dataset_cfg = getattr(config, "dataset", None)
    if dataset_cfg is None:
        return
    sources = _as_dict(_node_get(dataset_cfg, "sources", {}))
    if not sources:
        return

    params = _ensure_mapping_node(dataset_cfg, "params")

    # Speech sources -> params.audio_data
    speech_sources = _to_plain(_node_get(sources, "speech", [])) or []
    if isinstance(speech_sources, list) and speech_sources:
        audio_data = []
        for raw in speech_sources:
            if not isinstance(raw, dict):
                continue
            dataset_id = _first_non_empty(
                raw.get("dataset_id"),
                raw.get("name"),
                raw.get("repo_id"),
                raw.get("id"),
            )
            if dataset_id is None:
                continue
            entry = {"dataset_id": dataset_id}
            subset = _first_non_empty(raw.get("config"), raw.get("subset"))
            if subset is not None:
                entry["config"] = subset
            split = _normalize_optional_text(raw.get("split"))
            if split is not None:
                entry["split"] = split
            local_only = raw.get("local_files_only")
            if local_only is not None:
                entry["local_files_only"] = bool(local_only)
            path = _normalize_optional_text(raw.get("path"))
            if path is not None:
                lowered = dataset_id.strip().lower()
                if "common_voice" in lowered or "commonvoice" in lowered:
                    entry["commonvoice_path"] = path
                elif lowered == "jsonl":
                    entry["jsonl_path"] = path
            audio_data.append(entry)
        if audio_data:
            _set_node_value(params, "audio_data", audio_data)

    # Video sources -> params.video_caption_dataset / params.video_speech_dataset
    video_sources = _to_plain(_node_get(sources, "video", [])) or []
    if isinstance(video_sources, list) and video_sources:
        video_entry = next((v for v in video_sources if isinstance(v, dict)), None)
        if video_entry is not None:
            video_id = _canonical_video_id(video_entry)
            video_path = _first_non_empty(
                video_entry.get("path"),
                video_entry.get("dataset_id"),
                video_entry.get("repo_id"),
            )
            sample_method = _normalize_optional_text(video_entry.get("sample_method"))
            num_frames = video_entry.get("num_frames")
            local_only = video_entry.get("local_files_only")

            if _node_get(params, "video_caption_dataset", None) is not None:
                video_caption_cfg = _ensure_mapping_node(params, "video_caption_dataset")
                if video_id is not None:
                    _set_node_value(video_caption_cfg, "dataset_name", video_id)
                if video_path is not None:
                    if video_id == "llavavid":
                        _set_node_value(video_caption_cfg, "llavavid_path", video_path)
                    elif video_id == "webvid10m":
                        _set_node_value(video_caption_cfg, "webvid10m_path", video_path)
                    elif video_id == "openvid1m":
                        _set_node_value(video_caption_cfg, "openvid1m_path", video_path)
                    elif video_id == "panda70m":
                        _set_node_value(video_caption_cfg, "panda70m_path", video_path)
                if sample_method is not None:
                    _set_node_value(video_caption_cfg, "sample_method", sample_method)
                if num_frames is not None:
                    _set_node_value(video_caption_cfg, "num_frames", int(num_frames))
                if local_only is not None:
                    _set_node_value(
                        video_caption_cfg,
                        "llavavid_local_files_only",
                        bool(local_only),
                    )

            if _node_get(params, "video_speech_dataset", None) is not None:
                video_speech_cfg = _ensure_mapping_node(params, "video_speech_dataset")
                if video_id == "llavavid":
                    _set_node_value(video_speech_cfg, "use_llavavid", True)
                    _set_node_value(video_speech_cfg, "llavavid_dataset_name", "llavavid")
                    if video_path is not None:
                        _set_node_value(video_speech_cfg, "llavavid_path", video_path)
                    if sample_method is not None:
                        _set_node_value(video_speech_cfg, "v2t_sample_method", sample_method)
                    if num_frames is not None:
                        _set_node_value(video_speech_cfg, "num_frames", int(num_frames))
                    if local_only is not None:
                        _set_node_value(
                            video_speech_cfg,
                            "llavavid_local_files_only",
                            bool(local_only),
                        )

    # T2I/I2I sources -> params.t2i_dataset and HF repo-id fields
    t2i_sources = _to_plain(_node_get(sources, "t2i", [])) or []
    if isinstance(t2i_sources, list) and t2i_sources:
        t2i_ids: list[str] = []
        for raw in t2i_sources:
            if not isinstance(raw, dict):
                continue
            canonical = _canonical_t2i_id(raw)
            if canonical is None:
                continue
            dataset_id = _first_non_empty(raw.get("dataset_id"), raw.get("repo_id"))
            if canonical not in t2i_ids:
                t2i_ids.append(canonical)
            if canonical == "text2image2m" and dataset_id is not None:
                _set_node_value(params, "t2i_dataset_name", dataset_id)
            elif canonical == "pickapic" and dataset_id is not None:
                _set_node_value(params, "pickapic_dataset_name", dataset_id)
            elif canonical == "flux_reason" and dataset_id is not None:
                _set_node_value(params, "flux_reason_dataset_name", dataset_id)
            elif canonical == "ultraedit" and dataset_id is not None:
                _set_node_value(params, "ultraedit_dataset_name", dataset_id)
            elif canonical == "journeydb":
                path = _normalize_optional_text(raw.get("path"))
                image_root = _normalize_optional_text(raw.get("image_root"))
                if path is not None:
                    _set_node_value(params, "journeydb_jsonl_path", path)
                if image_root is not None:
                    _set_node_value(params, "journeydb_image_root", image_root)
        if t2i_ids:
            _set_node_value(params, "t2i_dataset", "+".join(t2i_ids))

    # LM sources -> params.hf_instruction_lm.sources (optional, consumed by train stages 2/3)
    lm_sources = _to_plain(_node_get(sources, "lm", [])) or []
    if isinstance(lm_sources, list) and lm_sources:
        hf_instruction_lm = _ensure_mapping_node(params, "hf_instruction_lm")
        normalized_lm_sources = [entry for entry in lm_sources if isinstance(entry, dict)]
        if normalized_lm_sources:
            _set_node_value(hf_instruction_lm, "sources", normalized_lm_sources)
