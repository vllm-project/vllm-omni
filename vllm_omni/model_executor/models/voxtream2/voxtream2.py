"""Voxtream2 1-stage wrapper.

This registers one Omni model class and lets the Voxtream package own its
generation loop. The current scope is intentionally limited to final audio
output; 2-stage Mimi codec streaming would require scheduler/orchestrator
changes and is left out here.
"""

from __future__ import annotations

import json
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch._inductor.config
import torch.nn as nn
from vllm.config import VllmConfig

from vllm_omni.logger import init_logger
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.voxtream2.voxtream2_import_utils import (
    ensure_voxtream_available,
    resolve_voxtream_file,
)

ensure_voxtream_available()

from voxtream.config import SpeechGeneratorConfig  # noqa: E402
from voxtream.generator import SpeechGenerator  # noqa: E402
from voxtream.utils.generator import interpolate_speaking_rate_params  # noqa: E402

logger = init_logger(__name__)


class Voxtream2ForConditionalGeneration(nn.Module):
    input_modalities = "audio"

    def __init__(
        self,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
        config: SpeechGeneratorConfig | None = None,
        compile: bool = False,
    ):
        torch.set_float32_matmul_precision("medium")
        torch._inductor.config.fx_graph_cache = True
        super().__init__()
        del prefix

        self.vllm_config = vllm_config
        self.have_multimodal_outputs = True
        if config is None:
            if vllm_config is None:
                raise ValueError(
                    "Either `config` or `vllm_config` must be provided for Voxtream2ForConditionalGeneration."
                )
            hf_config = vllm_config.model_config.hf_config
            cfg_rel = getattr(hf_config, "generator_config_path", "configs/generator.json")
            cfg_path = resolve_voxtream_file(cfg_rel)
            with open(cfg_path, encoding="utf-8") as f:
                cfg_dict = json.load(f)
            config = SpeechGeneratorConfig(**cfg_dict)

        self.config = config
        self.speech_generator = SpeechGenerator(config, compile=compile)
        self.logger = self.speech_generator.logger

        # Register Voxtream modules as submodules for vLLM device/weight
        # accounting, while keeping the generation logic in SpeechGenerator.
        self.model = self.speech_generator.model
        self.mimi = self.speech_generator.mimi
        self._speaking_rate_config_cache: tuple[str, dict[str, Any]] | None = None

    @staticmethod
    def get_dummy_runtime_additional_information(num_reqs: int) -> list[dict[str, object]]:
        # Used by vLLM warmup/profile dummy forward passes.
        return [{"_voxtream2_dummy": True} for _ in range(max(int(num_reqs), 1))]

    def _make_empty_omni_output(self, num_items: int = 1) -> OmniOutput:
        sample_rate = torch.tensor(int(self.config.mimi_sr), dtype=torch.int32)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": [torch.zeros((0,), dtype=torch.float32) for _ in range(max(num_items, 1))],
                "sr": [sample_rate for _ in range(max(num_items, 1))],
            },
        )

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_audio_path: Path,
        text: str | Generator[str, None, None],
        target_spk_rate_cnt: list[int] = None,
        spk_rate_weight: float = None,
        cfg_gamma: float = None,
        enhance_prompt: bool = None,
        apply_vad: bool = None,
        yield_codec_tokens: bool = False,
    ) -> Generator[tuple[np.ndarray, float], None, None]:
        if yield_codec_tokens:
            raise NotImplementedError(
                "Voxtream2 SpeechGenerator wrapper currently supports "
                "1-stage audio output only; 2-stage latent codec streaming is disabled."
            )

        yield from self.speech_generator.generate_stream(
            prompt_audio_path=prompt_audio_path,
            text=text,
            target_spk_rate_cnt=target_spk_rate_cnt,
            spk_rate_weight=spk_rate_weight,
            cfg_gamma=cfg_gamma,
            enhance_prompt=enhance_prompt,
            apply_vad=apply_vad,
        )

    @staticmethod
    def _extract_first(value: Any, default: Any = None) -> Any:
        if isinstance(value, list):
            return value[0] if value else default
        return value if value is not None else default

    def _load_speaking_rate_config(self, path_override: Any = None) -> dict[str, Any]:
        if self.vllm_config is None:
            cfg_rel = "configs/speaking_rate.json"
        else:
            hf_config = self.vllm_config.model_config.hf_config
            cfg_rel = getattr(hf_config, "speaking_rate_config_path", "configs/speaking_rate.json")

        raw_path = self._extract_first(path_override, None)
        if raw_path is not None:
            cfg_rel = str(raw_path)

        cached = self._speaking_rate_config_cache
        if cached is not None and cached[0] == cfg_rel:
            return cached[1]

        cfg_path = resolve_voxtream_file(cfg_rel)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        self._speaking_rate_config_cache = (cfg_rel, cfg)
        return cfg

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del input_ids, positions, intermediate_tensors, inputs_embeds

        info_list = runtime_additional_information
        if info_list is None:
            maybe_info = kwargs.get("additional_information")
            if isinstance(maybe_info, list):
                info_list = maybe_info
            elif isinstance(maybe_info, dict):
                info_list = [maybe_info]
            else:
                info_list = []

        if not info_list:
            raise ValueError(
                "Voxtream2 1-stage inference requires runtime additional information with text and ref audio path."
            )

        # vLLM warmup/profile path injects model-specific dummy runtime metadata.
        if all(isinstance(info, dict) and bool(info.get("_voxtream2_dummy", False)) for info in info_list):
            return self._make_empty_omni_output(len(info_list))

        outputs: list[torch.Tensor] = []
        output_audio_codes = (
            self.vllm_config is not None
            and getattr(self.vllm_config.model_config, "engine_output_type", None) == "latent"
        )
        if output_audio_codes:
            raise NotImplementedError(
                "Voxtream2 now wraps SpeechGenerator for 1-stage "
                "audio output only. Use voxtream2_1stage.yaml; 2-stage latent "
                "codec output is intentionally disabled."
            )

        srs: list[torch.Tensor] = []
        sample_rate = torch.tensor(int(self.config.mimi_sr), dtype=torch.int32)

        for info in info_list:
            if not isinstance(info, dict):
                continue

            text = self._extract_first(info.get("text"), "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Voxtream2 requires non-empty `text` in additional information.")

            ref_audio_path = self._extract_first(info.get("ref_audio_path"), None)
            if ref_audio_path is None:
                ref_audio_path = self._extract_first(info.get("ref_audio"), None)
            if ref_audio_path is None:
                ref_audio_path = self._extract_first(info.get("prompt_audio_path"), None)

            if not isinstance(ref_audio_path, str) or not ref_audio_path:
                raise ValueError(
                    "Voxtream2 requires `ref_audio_path` (or `ref_audio` / "
                    "`prompt_audio_path`) in additional information."
                )

            # Optional speaking-rate control. This intentionally only
            # supports the original one-shot path from voxtream.run: load
            # voxtream/configs/speaking_rate.json, interpolate the target rate,
            # and pass the resulting knobs into SpeechGenerator.generate_stream.
            target_spk_rate_cnt = None
            spk_rate_weight = None
            cfg_gamma = self._extract_first(info.get("cfg_gamma"), None)
            spk_rate = self._extract_first(info.get("spk_rate"), None)
            if spk_rate is None:
                spk_rate = self._extract_first(info.get("speaking_rate"), None)
            if spk_rate is not None:
                speaking_rate_config = self._load_speaking_rate_config(info.get("speaking_rate_config_path"))
                target_spk_rate_cnt, spk_rate_weight, cfg_gamma = interpolate_speaking_rate_params(
                    speaking_rate_config,
                    float(spk_rate),
                    logger=self.logger,
                )

            frames: list[torch.Tensor] = []
            for audio_frame, _gen_time in self.generate_stream(
                prompt_audio_path=Path(ref_audio_path),
                text=text,
                target_spk_rate_cnt=target_spk_rate_cnt,
                spk_rate_weight=spk_rate_weight,
                cfg_gamma=cfg_gamma,
                yield_codec_tokens=False,
            ):
                frame_tensor = torch.from_numpy(audio_frame).to(dtype=torch.float32).reshape(-1)
                frames.append(frame_tensor)

            if frames:
                outputs.append(torch.cat(frames, dim=-1))
            else:
                outputs.append(torch.zeros((0,), dtype=torch.float32))
            srs.append(sample_rate)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": outputs, "sr": srs},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        del kwargs
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        raise TypeError(f"Voxtream2 expected OmniOutput, got {type(model_outputs)}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        # Voxtream2 weights are loaded internally in __init__. Report all
        # parameter names so vLLM's strict checkpoint check can pass.
        return {name for name, _ in self.named_parameters()}
