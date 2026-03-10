# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import uuid
from collections.abc import Sequence

from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType
from vllm_omni.outputs import OmniRequestOutput


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
        # Capture stage info from kwargs before they might be filtered out
        stage_id = kwargs.get("stage_id")
        engine_input_source = kwargs.get("engine_input_source")
        cfg_kv_collect_func = kwargs.pop("cfg_kv_collect_func", None)

        if od_config is None:
            od_config = OmniDiffusionConfig.from_kwargs(**kwargs)
        elif isinstance(od_config, dict):
            # If config is dict, check it too (priority to kwargs if both exist)
            if stage_id is None:
                stage_id = od_config.get("stage_id")
            if engine_input_source is None:
                engine_input_source = od_config.get("engine_input_source")
            od_config = OmniDiffusionConfig.from_kwargs(**od_config)

        self.od_config = od_config

        # Inject stage info into omni_kv_config if present
        if stage_id is not None:
            self.od_config.omni_kv_config.setdefault("stage_id", stage_id)
        if engine_input_source is not None:
            self.od_config.omni_kv_config.setdefault("engine_input_source", engine_input_source)

        # Detect model class and load config
        # Diffusers-style models expose `model_index.json` with `_class_name`.
        # Non-diffusers models (e.g. Bagel, NextStep, GLM-Image) only have `config.json`,
        # so we fall back to reading that and mapping model_type manually.
        model_path = od_config.model

        def _load_local_json(filename: str) -> dict | None:
            if not model_path or not os.path.isdir(model_path):
                return None
            path = os.path.join(model_path, filename)
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        local_model_index = _load_local_json("model_index.json")
        local_config = _load_local_json("config.json")

        if local_model_index is not None:
            if od_config.model_class_name is None:
                od_config.model_class_name = local_model_index.get("_class_name", None)
            od_config.update_multimodal_support()
            tf_config_path = os.path.join(model_path, "transformer", "config.json")
            if os.path.exists(tf_config_path):
                with open(tf_config_path, "r", encoding="utf-8") as f:
                    tf_config_dict = json.load(f)
                od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
            else:
                od_config.tf_model_config = TransformerConfig()
        elif local_config is not None:
            cfg = local_config
            model_type = cfg.get("model_type")
            architectures = cfg.get("architectures") or []
            pipeline_class = None
            if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                pipeline_class = "BagelPipeline"
            elif model_type == "nextstep":
                if od_config.model_class_name is None:
                    pipeline_class = "NextStep11Pipeline"
            elif model_type == "glm-image" or "GlmImageForConditionalGeneration" in architectures:
                pipeline_class = "GlmImagePipeline"
            elif architectures and len(architectures) == 1:
                pipeline_class = architectures[0]

            if pipeline_class is None:
                raise ValueError(f"Unknown model type: {model_type}, architectures: {architectures}")

            if od_config.model_class_name is None:
                od_config.model_class_name = pipeline_class
            od_config.tf_model_config = TransformerConfig()
            od_config.update_multimodal_support()
        else:
            # Local checkpoint dir without config files: allow custom pipeline via model_class_name.
            if model_path and os.path.isdir(model_path):
                if od_config.model_class_name is None:
                    raise ValueError(
                        f"Local model dir {od_config.model} has no config.json or model_index.json. "
                        "Please provide model_class_name explicitly."
                    )
                od_config.tf_model_config = TransformerConfig()
                od_config.update_multimodal_support()
            else:
                try:
                    config_dict = get_hf_file_to_dict(
                        "model_index.json",
                        od_config.model,
                    )
                    if config_dict is not None:
                        if od_config.model_class_name is None:
                            od_config.model_class_name = config_dict.get("_class_name", None)
                        od_config.update_multimodal_support()

                        tf_config_dict = get_hf_file_to_dict(
                            "transformer/config.json",
                            od_config.model,
                        )
                        od_config.tf_model_config = TransformerConfig.from_dict(tf_config_dict)
                    else:
                        raise FileNotFoundError("model_index.json not found")
                except (AttributeError, OSError, ValueError, FileNotFoundError):
                    cfg = get_hf_file_to_dict("config.json", od_config.model)
                    if cfg is None:
                        if od_config.model_class_name is not None:
                            od_config.tf_model_config = TransformerConfig()
                            od_config.update_multimodal_support()
                        else:
                            raise ValueError(
                                f"Could not find config.json or model_index.json for model {od_config.model}"
                            )
                    else:
                        # Map model_type or architecture to pipeline class
                        model_type = cfg.get("model_type")
                        architectures = cfg.get("architectures") or []
                        pipeline_class = None
                        if model_type == "bagel" or "BagelForConditionalGeneration" in architectures:
                            pipeline_class = "BagelPipeline"
                        elif model_type == "nextstep":
                            if od_config.model_class_name is None:
                                pipeline_class = "NextStep11Pipeline"
                        elif model_type == "glm-image" or "GlmImageForConditionalGeneration" in architectures:
                            pipeline_class = "GlmImagePipeline"
                        elif architectures and len(architectures) == 1:
                            pipeline_class = architectures[0]

                        if pipeline_class is None:
                            raise ValueError(f"Unknown model type: {model_type}, architectures: {architectures}")

                        if od_config.model_class_name is None:
                            od_config.model_class_name = pipeline_class
                        od_config.tf_model_config = TransformerConfig()
                        od_config.update_multimodal_support()

        if cfg_kv_collect_func is not None:
            od_config.cfg_kv_collect_func = cfg_kv_collect_func

        self.engine: DiffusionEngine = DiffusionEngine.make_engine(od_config)

    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params: OmniDiffusionSamplingParams,
        request_ids: list[str] = [],
    ) -> list[OmniRequestOutput]:
        if isinstance(prompts, (str, dict)):
            prompts = [prompts]
        else:
            prompts = list(prompts)

        # Check if request_id is provided in kwargs
        if len(request_ids) < len(prompts):
            request_ids.extend(f"{i + len(request_ids)}_{uuid.uuid4()}" for i in range(len(prompts) - len(request_ids)))

        request = OmniDiffusionRequest(prompts, sampling_params, request_ids)
        return self._run_engine(request)

    def _run_engine(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        return self.engine.step(request)

    def close(self) -> None:
        self.engine.close()

    def __del__(self):  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def start_profile(self, trace_filename: str | None = None) -> None:
        """Start profiling for the diffusion model.

        Args:
            trace_filename: Optional base filename for trace files.
                           If None, a timestamp-based name will be generated.
        """
        if hasattr(self, "engine") and self.engine:
            self.engine.start_profile(trace_filename)
        else:
            raise RuntimeError("Diffusion engine not initialized")

    def stop_profile(self) -> dict:
        """Stop profiling and return profiling results.

        Returns:
            Dictionary containing paths to trace and table files.
        """
        if hasattr(self, "engine") and self.engine:
            return self.engine.stop_profile()
        else:
            raise RuntimeError("Diffusion engine not initialized")
