# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from dataclasses import fields

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest

# TODO configure logging properly
logging.basicConfig(level=logging.INFO)

logger = init_logger(__name__)


def prepare_requests(prompt: str | list[str], **kwargs):
    field_names = {f.name for f in fields(OmniDiffusionRequest)}

    init_kwargs = {"prompt": prompt}

    for key, value in kwargs.items():
        if key in field_names:
            init_kwargs[key] = value

    return OmniDiffusionRequest(**init_kwargs)


class OmniDiffusion:
    """
    It is the main class to interact with vLLM-Omni diffusion models.
    It acts as a high-level interface that prepares requests and
    delegates the actual diffusion process to the DiffusionEngine.

    You can pass either an `OmniDiffusionConfig` via `od_config`, or
    pass kwargs such as `model="Qwen/Qwen-Image"`,
    which will be forwarded to `OmniDiffusionConfig.from_kwargs`.
    """

    def __init__(self, od_config: OmniDiffusionConfig | None = None, **kwargs):
        if od_config is None:
            od_config = OmniDiffusionConfig.from_kwargs(**kwargs)
        elif isinstance(od_config, dict):
            od_config = OmniDiffusionConfig.from_kwargs(**od_config)

        self.od_config = od_config

        # Diffusers-style models expose `model_index.json` with `_class_name`.
        # Bagel models (and other non-diffusers) typically expose `config.json`.
        try:
            config_dict = get_hf_file_to_dict(
                "model_index.json",
                od_config.model,
            )
            od_config.model_class_name = config_dict.get("_class_name", None)
            od_config.update_multimodal_support()

            tf_config_dict = get_hf_file_to_dict(
                "transformer/config.json",
                od_config.model,
            )
            od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
        except (AttributeError, OSError, ValueError):
            cfg = get_hf_file_to_dict("config.json", od_config.model)
            if cfg is None:
                raise ValueError(f"Could not find config.json or model_index.json for model {od_config.model}")

            model_type = cfg.get("model_type")
            architectures = cfg.get("architectures") or []
            if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                od_config.model_class_name = "BagelPipeline"
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()
            else:
                raise

        self.engine: DiffusionEngine = DiffusionEngine.make_engine(od_config)

    def generate(
        self,
        prompt: str | list[str],
        **kwargs,
    ):
        prompts = []
        if isinstance(prompt, str):
            prompts.append(prompt)
        elif isinstance(prompt, list):
            prompts.extend(prompt)
        else:
            raise ValueError("Prompt must be a string or a list of strings")

        requests: list[OmniDiffusionRequest] = []

        # Check if request_id is provided in kwargs
        request_id = kwargs.get("request_id")

        for i, p in enumerate(prompts):
            req_kwargs = kwargs.copy()
            if request_id is None:
                # Generate default ID consistent with OmniLLM: "{i}_{uuid}"
                req_kwargs["request_id"] = f"{i}"

            requests.append(
                prepare_requests(
                    p,
                    **req_kwargs,
                )
            )
        logger.info(f"Prepared {len(requests)} requests for generation.")
        return self._run_engine(requests)

    def _run_engine(self, requests: list[OmniDiffusionRequest]):
        return self.engine.step(requests)

    def close(self) -> None:
        self.engine.close()

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
