# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Mapping
from dataclasses import field

from pydantic import TypeAdapter
from vllm.config.utils import config

from vllm_omni.watermarking import WATERMARKER_REGISTRY

ALGORITHM_KEY = "algorithm"
TEXT_MODALITY = "text"
WATERMARK_CONFIG_EXAMPLE = '{"strict": false, "modalities": {"<modality>": {"algorithm": "<algorithm>"}}}'


@config
class WatermarkConfig:
    modalities: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    # If true, fails requests whose outputs can't be watermarked in post. Note that
    # this is only for multimodal outputs, since text watermarking happens at sampling
    # time.
    strict: bool = False

    @classmethod
    def from_dict(cls, raw_config: Mapping[str, object]) -> "WatermarkConfig":
        """Build a config from its JSON form, rejecting misplaced modality keys."""
        # Check for common errors, e.g., not nesting by modality, since this is what vLLM does
        has_misplaced_modality_config = ALGORITHM_KEY in raw_config or any(
            modality in raw_config for modality in WATERMARKER_REGISTRY
        )
        if has_misplaced_modality_config:
            raise ValueError(f"watermark algorithms are configured per modality; expected {WATERMARK_CONFIG_EXAMPLE}")
        return TypeAdapter(cls).validate_python(raw_config)

    def __post_init__(self) -> None:
        for modality, modality_config in self.modalities.items():
            if modality == TEXT_MODALITY:
                # Text watermarking is done at sampling time and implemented in vLLM, so we don't register
                # in the watermarking registry for Omni, which is primarily post-processing on mm outputs
                continue
            registered_algorithms = WATERMARKER_REGISTRY.get(modality)
            if registered_algorithms is None:
                supported_modalities = ", ".join(sorted((*WATERMARKER_REGISTRY, TEXT_MODALITY)))
                raise ValueError(f"unsupported watermark modality {modality}; supported: {supported_modalities}")
            algorithm = modality_config.get(ALGORITHM_KEY)
            if not isinstance(algorithm, str) or algorithm not in registered_algorithms:
                valid_algorithms = ", ".join(sorted(registered_algorithms))
                raise ValueError(
                    f"unsupported watermark algorithm {algorithm} for {modality}; supported: {valid_algorithms}"
                )
