# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from transformers import AutoProcessor, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import cached_file
from vllm.logger import init_logger

from vllm_omni.diffusion.models.gr00t.configs.embodiment_configs import ModalityConfig
from vllm_omni.diffusion.models.gr00t.configs.gr00t_n1d7 import Gr00tN1d7Config
from vllm_omni.diffusion.models.gr00t.dataio.embodiment_tags import EmbodimentTag
from vllm_omni.diffusion.models.gr00t.dataio.state_action.state_action_processor import StateActionProcessor
from vllm_omni.diffusion.models.gr00t.dataio.utils import parse_modality_configs, to_json_serializable


class LetterBoxTransform:
    """Pad image to square dimensions by adding black bars to the smaller side."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        import math

        *leading_dims, c, h, w = img.shape
        if h == w:
            return img
        max_dim = max(h, w)
        pad_h, pad_w = max_dim - h, max_dim - w
        pad_top, pad_left = pad_h // 2, pad_w // 2
        pad_bottom, pad_right = pad_h - pad_top, pad_w - pad_left
        if leading_dims:
            batch_size = math.prod(leading_dims)
            img_r = img.reshape(batch_size, c, h, w)
            padded = transforms.functional.pad(img_r, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)
            return padded.reshape(leading_dims + [c, max_dim, max_dim])
        return transforms.functional.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)


def _build_eval_image_transform(
    image_target_size: list[int],
    image_crop_size: list[int],
) -> transforms.Compose:
    """Deterministic eval/inference image transform (letterbox → resize → centercrop → resize)."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            LetterBoxTransform(),
            transforms.Resize(size=image_target_size),
            transforms.CenterCrop(size=image_crop_size),
            transforms.Resize(size=image_target_size),
        ]
    )


logger = init_logger(__name__)

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

EMBODIMENT_TAG_TO_PROJECTOR_INDEX = {
    # Pretrain embodiment ids
    "oxe_droid_relative_eef_relative_joint": 24,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    # Posttrain embodiment ids
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "unitree_g1_sonic": 11,
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
}

QWEN3_VL_2B_PROCESSOR = "Qwen/Qwen3-VL-2B-Instruct"


def build_processor(model_name: str, transformers_loading_kwargs: dict) -> ProcessorMixin:
    try:
        from transformers import Qwen3VLProcessor
    except ImportError as exc:
        raise ImportError(
            "GR00T-N1.7 requires transformers>=4.57.1 for Qwen3VLProcessor "
            "(the repo's declared floor is 4.56.0). Please upgrade transformers."
        ) from exc
    if model_name == "nvidia/Cosmos-Reason2-2B":
        # Cosmos-Reason2-2B lacks a Qwen3VLProcessor; fall back to upstream artifacts.
        logger.warning_once(
            "Substituting processor from %s because %s does not ship one. "
            "If you fine-tune Cosmos-Reason2-2B's tokenizer/image processor, "
            "load the processor explicitly instead of relying on this fallback.",
            QWEN3_VL_2B_PROCESSOR,
            model_name,
        )
        model_name = QWEN3_VL_2B_PROCESSOR
    return Qwen3VLProcessor.from_pretrained(model_name, **transformers_loading_kwargs)


class Gr00tN1d7DataCollator:
    def __init__(
        self,
        model_name: str,
        model_type: str = "qwen",
        transformers_loading_kwargs: dict | None = None,
    ):
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {}
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name

    def __call__(self, features: list[dict[str, Any]]) -> BatchFeature:
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                text_list = []
                image_inputs = []
                for v in values:
                    curr_text_list = [v["text"]]

                    text_list += curr_text_list
                    curr_image_inputs = v["images"]
                    image_inputs += curr_image_inputs

                vlm_inputs = self.processor(
                    text=text_list,
                    images=image_inputs,
                    return_tensors="pt",
                    padding=True,
                )
                for k, v in vlm_inputs.items():
                    batch[k] = v
            elif key in (
                "pixel_values",
                "image_grid_thw",
                "attention_mask",
                "input_ids",
            ):
                raise Exception("Not implemented")
            else:
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    def __str__(self):
        return f"Gr00tN1d7DataCollator(model_name={self.model_name}, model_type={self.model_type})"


class Gr00tN1d7Processor(ProcessorMixin):
    data_collator_class = Gr00tN1d7DataCollator

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: (dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None) = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        image_crop_size: list[int] = None,
        image_target_size: list[int] = None,
        shortest_image_edge: int = 256,
        crop_fraction: float = 0.95,
        random_rotation_angle: int | None = None,
        color_jitter_params: dict[str, float] | None = None,
        formalize_language: bool = True,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        model_type: str = "qwen",
        max_state_dim: int = 29,
        max_action_dim: int = 29,
        max_action_horizon: int = 50,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
        embodiment_id_mapping: dict[str, int] | None = None,
        transformers_loading_kwargs: dict | None = None,
        exclude_state: bool = False,
        # Normalization
        use_mean_std: bool = False,
        **kwargs,  # absorb deprecated training-only keys from saved processor_config.json
    ):
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}
        if kwargs:
            logger.debug("Gr00tN1d7Processor: ignoring unknown keys: %s", list(kwargs))
        self.modality_configs = parse_modality_configs(modality_configs)

        # Initialize StateActionProcessor for state/action normalization
        self.state_action_processor = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=apply_sincos_state_encoding,
            use_relative_action=use_relative_action,
        )

        self.use_percentiles = use_percentiles
        self.use_mean_std = use_mean_std
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        self.exclude_state = exclude_state

        self.formalize_language = formalize_language
        self.model_name = model_name
        self.model_type = model_type

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon

        self.image_crop_size = image_crop_size
        self.image_target_size = image_target_size
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        self.processor.tokenizer.padding_side = "left"
        self.embodiment_id_mapping = embodiment_id_mapping or EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        for k, v in EMBODIMENT_TAG_TO_PROJECTOR_INDEX.items():
            if k not in self.embodiment_id_mapping:
                self.embodiment_id_mapping[k] = v
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction

        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}

        # Eval/inference image transform
        self.eval_image_transform = _build_eval_image_transform(
            image_target_size,
            image_crop_size,
        )
        self._collator = self.data_collator_class(
            model_name=model_name,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    @property
    def collator(self):
        return self._collator

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """Set dataset statistics for normalization."""
        for key in statistics:
            if key not in self.statistics or override:
                if override:
                    logger.info("Overriding statistics for %s", key)
                self.statistics[key] = deepcopy(statistics[key])
            else:
                logger.debug("Embodiment tag %s already in statistics, skipping update", key)

        self.state_action_processor.set_statistics(statistics, override=override)

        # Compute action dimensions for convenience
        self.action_dim = {}
        for embodiment_tag in self.state_action_processor.statistics:
            self.action_dim[embodiment_tag] = self.state_action_processor.get_action_dim(embodiment_tag)

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ):
        """Undo action normalization and convert relative actions to absolute."""
        # Split concatenated action into joint groups
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][key]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        # Use StateActionProcessor to unnormalize and convert to absolute
        return self.state_action_processor.unapply_action(out_dict, embodiment_tag.value, state=state)

    def _apply_vlm_processing(self, images: np.ndarray, language: str) -> BatchFeature:
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": language},
                ],
            }
        ]

        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        return {
            "vlm_content": {
                "text": text,
                "images": pil_images,
                "conversation": conversation,
            }
        }

    def __call__(
        self,
        messages: list[dict[str, Any]],
    ):
        assert len(messages) == 1
        content = messages[0]["content"]
        embodiment_tag = content.embodiment
        action_data = content.actions
        state_data = content.states

        norm_state_dict, normalized_actions = self.state_action_processor.apply(
            state=state_data,
            action=action_data,
            embodiment_tag=embodiment_tag.value,
        )

        if normalized_actions:
            action_keys = self.modality_configs[embodiment_tag.value]["action"].modality_keys
            normalized_actions = torch.cat(
                [torch.from_numpy(normalized_actions[key]) for key in action_keys],
                dim=-1,
            )  # (t, d)
            action_dim = normalized_actions.shape[1]
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        normalized_actions.shape[0],
                        self.max_action_dim - normalized_actions.shape[1],
                    ),
                ],
                dim=-1,
            )  # (t, max_action_dim)
            action_horizon = normalized_actions.shape[0]
            assert action_horizon <= self.max_action_horizon, (
                f"Action sequence length {action_horizon} exceeds max_action_horizon"
                f" {self.max_action_horizon}. Increase model config action_horizon to"
                f" >= {action_horizon}."
            )
            normalized_actions = torch.cat(
                [
                    normalized_actions,
                    torch.zeros(
                        self.max_action_horizon - normalized_actions.shape[0],
                        self.max_action_dim,
                    ),
                ],
                dim=0,
            )  # (max_action_horizon, max_action_dim)
            action_mask = torch.ones_like(normalized_actions)
            action_mask[action_horizon:] = 0
            action_mask[:, action_dim:] = 0
        else:
            normalized_actions = None
            action_mask = None

        state_keys = self.modality_configs[embodiment_tag.value]["state"].modality_keys
        exclude_state = self.exclude_state or getattr(
            self.modality_configs[embodiment_tag.value]["state"], "exclude_state", False
        )
        if exclude_state:
            normalized_states = torch.cat(
                [torch.from_numpy(np.zeros_like(state_data[key])) for key in state_keys], dim=-1
            )
        else:
            normalized_states = torch.cat([torch.from_numpy(norm_state_dict[key]) for key in state_keys], dim=-1)
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(
                    normalized_states.shape[0],
                    self.max_state_dim - normalized_states.shape[1],
                ),
            ],
            dim=-1,
        )

        image_transform = self.eval_image_transform
        image_keys = self.modality_configs[embodiment_tag.value]["video"].modality_keys

        if self.formalize_language:
            language = content.text.lower()
            language = re.sub(r"[^\w\s]", "", language)
        else:
            language = content.text

        vlm_inputs = self._get_vlm_inputs(
            image_keys=image_keys,
            images=content.images,
            masks=content.masks,
            image_transform=image_transform,
            language=language,
        )

        transformed_inputs = {
            "state": normalized_states.to(torch.get_default_dtype()),
        }
        if normalized_actions is not None:
            transformed_inputs["action"] = normalized_actions.to(torch.get_default_dtype())
        # Add VLM inputs
        transformed_inputs.update(vlm_inputs)
        if action_mask is not None:
            transformed_inputs["action_mask"] = action_mask
        transformed_inputs["embodiment_id"] = self.embodiment_id_mapping[embodiment_tag.value]
        return transformed_inputs

    def _get_vlm_inputs(
        self,
        image_keys: list[str],
        images: list[Image.Image],
        masks: dict[str, list[np.ndarray]] | None,
        image_transform: transforms.Compose,
        language: str,
    ):
        temporal_stacked_images = {}

        if masks is not None:
            raise ValueError("Mask-based transforms are not supported at inference.")
        for view in image_keys:
            assert view in images, f"{view} not in {images}"
            temporal_stacked_images[view] = torch.stack([image_transform(img) for img in images[view]])  # (T, C, H, W)

        for k, v in temporal_stacked_images.items():
            assert isinstance(k, str), f"{k} is not a string"
            assert isinstance(v, torch.Tensor), f"{v} is not a torch tensor"
            assert v.ndim == 4, f"{v} is not a 4D tensor"
            assert v.dtype == torch.uint8, f"{v} is not a uint8 tensor"
            assert v.shape[1] == 3, f"{v} is not a 3 channel tensor"

        stacked_images = (
            torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1).flatten(0, 1).numpy()
        )  # (T*V, C, H, W), processor expects numpy array

        vlm_inputs = self._apply_vlm_processing(stacked_images, language)
        return vlm_inputs

    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        main_config_file = save_directory / "processor_config.json"
        statistics_file = save_directory / "statistics.json"
        embodiment_id_file = save_directory / "embodiment_id.json"

        config = {
            "processor_class": self.__class__.__name__,
            "processor_kwargs": {
                "modality_configs": to_json_serializable(self.modality_configs),
                "image_crop_size": self.image_crop_size,
                "image_target_size": self.image_target_size,
                "random_rotation_angle": self.random_rotation_angle,
                "color_jitter_params": self.color_jitter_params,
                "shortest_image_edge": self.shortest_image_edge,
                "crop_fraction": self.crop_fraction,
                "model_name": self.model_name,
                "model_type": self.model_type,
                "formalize_language": self.formalize_language,
                "max_state_dim": self.max_state_dim,
                "max_action_dim": self.max_action_dim,
                "max_action_horizon": self.max_action_horizon,
                "use_percentiles": self.use_percentiles,
                "use_mean_std": self.use_mean_std,
                "clip_outliers": self.clip_outliers,
                "apply_sincos_state_encoding": self.apply_sincos_state_encoding,
                "use_relative_action": self.use_relative_action,
                "exclude_state": self.exclude_state,
            },
        }
        with open(main_config_file, "w") as f:
            json.dump(config, f, indent=2)
        with open(statistics_file, "w") as f:
            json.dump(
                to_json_serializable(self.state_action_processor.statistics),
                f,
                indent=2,
            )
        with open(embodiment_id_file, "w") as f:
            json.dump(self.embodiment_id_mapping, f, indent=2)
        return [main_config_file, statistics_file, embodiment_id_file]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        transformers_loading_kwargs = kwargs.pop("transformers_loading_kwargs", {"trust_remote_code": True})
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config_file = pretrained_model_name_or_path / "processor_config.json"
        statistics_file = pretrained_model_name_or_path / "statistics.json"
        embodiment_id_file = pretrained_model_name_or_path / "embodiment_id.json"
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if not is_local:
            config_file = Path(cached_file(pretrained_model_name_or_path, "processor_config.json"))
            statistics_file = Path(cached_file(pretrained_model_name_or_path, "statistics.json"))
            embodiment_id_file = Path(cached_file(pretrained_model_name_or_path, "embodiment_id.json"))

        with open(config_file) as f:
            config = json.load(f)
        with open(statistics_file) as f:
            statistics = json.load(f)
        if embodiment_id_file.exists():
            with open(embodiment_id_file) as f:
                embodiment_id_mapping = json.load(f)
        else:
            embodiment_id_mapping = None
        processor_kwargs = config["processor_kwargs"]
        processor_kwargs["statistics"] = statistics
        processor_kwargs["embodiment_id_mapping"] = embodiment_id_mapping

        # Backfill missing fields from older checkpoints.
        processor_kwargs.setdefault("model_name", "nvidia/Cosmos-Reason2-2B")
        processor_kwargs.setdefault("model_type", "qwen")
        processor_kwargs.setdefault("clip_outliers", True)

        # Directly override other processor kwargs
        if kwargs:
            # Override modality configs while keeping pretrained embodiment configs
            modality_configs = kwargs.pop("modality_configs", {})
            for embodiment_tag, modality_config in modality_configs.items():
                processor_kwargs["modality_configs"][embodiment_tag] = modality_config
            override_keys = [
                "random_rotation_angle",
                "color_jitter_params",
                "use_relative_action",
                "exclude_state",
                "use_mean_std",
                "model_name",
                "model_type",
                "max_action_horizon",
                "max_state_dim",
                "max_action_dim",
            ]
            for key in override_keys:
                if key in kwargs:
                    override = kwargs.pop(key)
                    if override is not None:
                        processor_kwargs[key] = override
        return cls(**processor_kwargs, transformers_loading_kwargs=transformers_loading_kwargs)


# transformers >= 5.5 requires the config *class* (not a model_type string) as the
# first arg to AutoProcessor.register — it does `key.__module__` internally. Passing
# the string "Gr00tN1d7" raises AttributeError: 'str' object has no attribute
# '__module__'. Match the AutoModel.register(Gr00tN1d7Config, ...) call in
# modeling/gr00t_n1d7.py.
AutoProcessor.register(Gr00tN1d7Config, Gr00tN1d7Processor)
