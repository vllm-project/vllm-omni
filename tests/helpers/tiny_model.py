"""Reusable utility for building tiny diffusion models with random weights.

Mirrors the component loading loop in DiffusionPipeline.from_pretrained,
but monkeypatches from_pretrained on ModelMixin and PreTrainedModel to
initialize with random weights instead of loading checkpoint files.
Components with vendored configs use those; others load configs from
the upstream HF model, optionally passed through a per-component transform
function first.
"""

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import torch
from diffusers import ModelMixin
from diffusers.pipelines.pipeline_loading_utils import _get_pipeline_class, simple_get_class_obj
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import PretrainedConfig, PreTrainedModel

TINY_MODEL_DIR = os.path.join(tempfile.gettempdir(), "vllm-omni-tiny-models")


def _get_tiny_model_path(name: str) -> str:
    path = os.path.join(TINY_MODEL_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def build_tiny_from_configs(
    pipeline_name: str,
    model_id: str,
    configs_dir: str | Path | None = None,
    transform: dict[str, Callable[[dict], dict]] | None = None,
) -> str:
    """Build a tiny diffusion model with random weights.

    Args:
        pipeline_name: Name of the pipeline (used as output directory name).
        model_id: HuggingFace model ID for loading upstream components
            (tokenizer, scheduler, and any component without a vendored
            config or transform) that don't have vendored configs.
        configs_dir: Path to a directory containing vendored config files
            (model_index.json and per-component config.json).
        transform: Maps a component (subfolder) name to a function that
            takes that component's raw config dict (fetched from model_id)
            and returns a shrunk one, applied only when that component has
            no vendored config in configs_dir. Lets each pipeline encode
            exactly the shrink it needs (layer counts, and any component-
            internal width that isn't read by another component), including
            keeping sibling config fields in sync (e.g. layer_types matching
            num_hidden_layers).

    Returns:
        Path to the saved tiny model directory with safetensors weights.
    """
    model_dir = _get_tiny_model_path(pipeline_name)
    config_dir = Path(configs_dir) if configs_dir is not None else None
    transform = transform or {}

    if config_dir is None and not transform:
        raise ValueError(
            f"Neither configs_dir nor transform given for {pipeline_name!r}. Pass a "
            "configs_dir with vendored configs, and/or transform={{name: fn}} to shrink "
            f"specific components fetched from the upstream model {model_id!r}."
        )

    def _diffusers_from_config(cls, pretrained_model_name_or_path, **kwargs):
        """Replacement for ModelMixin.from_pretrained that initializes random weights."""
        subfolder = kwargs.get("subfolder")
        load_kwargs = {"subfolder": subfolder} if subfolder else {}
        config = cls.load_config(pretrained_model_name_or_path, **load_kwargs)
        if subfolder in transform:
            config = transform[subfolder](dict(config))
        return cls.from_config(config)

    def _transformers_from_config(cls, pretrained_model_name_or_path, **kwargs):
        """Replacement for PreTrainedModel.from_pretrained that initializes random weights.

        Uses the model's own config_class (every PreTrainedModel subclass
        declares one) so the raw dict can be transformed before constructing
        the config object.
        """
        subfolder = kwargs.get("subfolder")
        load_kwargs = {"subfolder": subfolder} if subfolder else {}
        config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **load_kwargs)
        if subfolder in transform:
            config_dict = transform[subfolder](config_dict)
        config = cls.config_class.from_dict(config_dict)
        return cls(config)

    # model_index.json comes from the vendored configs_dir when it's there, otherwise
    # from model_id. Independent of the per-component checks below, so a configs_dir
    # can vendor just one component without also needing its own model_index.json.
    vendored_index = config_dir is not None and (config_dir / "model_index.json").exists()
    config_dict = DiffusionPipeline.load_config(config_dir if vendored_index else model_id)
    pipeline_cls = _get_pipeline_class(DiffusionPipeline, config=config_dict)

    init_dict, _, _ = pipeline_cls.extract_init_dict(config_dict)

    # Pop non-component entries (optional pipeline kwargs like is_distilled),
    # same as DiffusionPipeline.from_pretrained lines 345-350
    _, optional_kwargs = DiffusionPipeline._get_signature_keys(pipeline_cls)
    init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in init_dict}

    with (
        patch.object(ModelMixin, "from_pretrained", classmethod(_diffusers_from_config)),
        patch.object(PreTrainedModel, "from_pretrained", classmethod(_transformers_from_config)),
    ):
        for name, (library_name, class_name) in init_dict.items():
            cls = simple_get_class_obj(library_name, class_name)

            # Use vendored config dir if available, otherwise upstream model
            if config_dir is not None and (config_dir / name).exists():
                init_kwargs[name] = cls.from_pretrained(config_dir / name)
            else:
                init_kwargs[name] = cls.from_pretrained(model_id, subfolder=name)

    pipe = pipeline_cls(**init_kwargs)
    pipe.to(torch.bfloat16).save_pretrained(model_dir)
    return model_dir
