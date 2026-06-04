from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class MingTTSPreparedStep:
    request_ids: list[str]
    runtime_infos: list[dict]
    inputs_embeds: torch.Tensor
    latents_by_request: list[list[Any]] | None = None
    audios: list[Any] | None = None
    sample_rate: int | None = None


@dataclass(slots=True)
class MingTTSStageStepRunner:
    max_batch_size: int = 8
    last_reject_reason: str | None = None

    def _reject(self, reason: str) -> bool:
        self.last_reject_reason = reason
        return False

    def supports_batch(self, *, runner: Any, runtime_infos: list[dict], is_graph_capturing: bool) -> bool:
        if is_graph_capturing:
            return self._reject("graph_capture")
        if len(runtime_infos) <= 1 or len(runtime_infos) > self.max_batch_size:
            return self._reject("batch_size")
        model_config = getattr(getattr(runner, "vllm_config", None), "model_config", None)
        stage = getattr(model_config, "model_stage", None)
        arch = getattr(model_config, "model_arch", None)
        if stage not in {"ming_tts", "ming_flash_omni_tts"} and arch != "MingFlashOmniTalkerForConditionalGeneration":
            return self._reject(f"wrong_stage:{stage}:{arch}")
        model_name = type(getattr(runner, "model", None)).__name__
        if model_name != "MingFlashOmniTalkerForConditionalGeneration":
            return self._reject("wrong_model")
        first = runtime_infos[0] or {}
        keys = (
            "ming_task",
            "prompt",
            "instruction",
            "cfg",
            "sigma",
            "temperature",
            "max_steps",
            "max_decode_steps",
            "max_text_length",
            "use_static_cache",
            "stream_decode",
            "use_zero_spk_emb",
        )
        for info in runtime_infos:
            info = info or {}
            if info.get("prompt_wav_lat") is not None or info.get("prompt_wav_emb") is not None:
                return self._reject("prompt_wav")
            if info.get("voice_name") is not None or info.get("spk_emb") is not None:
                return self._reject("voice_or_spk")
            for key in keys:
                if info.get(key) != first.get(key):
                    return self._reject("param_mismatch")
        self.last_reject_reason = None
        return True

    def prepare_step(
        self, *, request_ids: list[str], runtime_infos: list[dict], inputs_embeds: torch.Tensor
    ) -> MingTTSPreparedStep:
        return MingTTSPreparedStep(list(request_ids), list(runtime_infos), inputs_embeds)

    def run_step(self, *, prepared: MingTTSPreparedStep, runner: Any) -> None:
        first = prepared.runtime_infos[0] if prepared.runtime_infos else {}
        prepared.latents_by_request = runner.model.audio_generator.generate_latents_batch(
            prepared.inputs_embeds,
            max_steps=int(first.get("max_steps", first.get("max_decode_steps", 200))),
            cfg=first.get("cfg"),
            sigma=float(first.get("sigma", 0.25)),
            temperature=float(first.get("temperature", 0.0)),
            use_static_cache=bool(first.get("use_static_cache", True)),
        )
        decode = getattr(runner.model, "decode_batch_latents_for_runner", None)
        if decode is not None:
            prepared.audios, prepared.sample_rate = decode(
                prepared.latents_by_request,
                stream_decode=bool(first.get("stream_decode", True)),
            )

    def commit_step(self, *, prepared: MingTTSPreparedStep, runner: Any) -> None:
        if prepared.audios is None:
            raise RuntimeError("run_step must decode audios before commit_step")
        sr = prepared.sample_rate or 24000
        for req_id, audio in zip(prepared.request_ids, prepared.audios, strict=True):
            req_buffer = runner.model_intermediate_buffer.setdefault(req_id, {})
            req_buffer["audio"] = {"audio": audio, "sr": sr}
