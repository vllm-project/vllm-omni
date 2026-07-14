"""
VLLM wrapper for single-stage Miso TTS model.
This integrates the official Miso TTS Generator into vllm-omni.
"""
import logging
import threading
from typing import Any, Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig

from .miso_tts_single_stage import MisoTTSSingleStage, load_miso_single_stage, Segment

logger = logging.getLogger(__name__)

DEFAULT_MISO_TTS_REPO_ID = "MisoLabs/MisoTTS"
_DEFAULT_TEMPERATURE = 0.9
_DEFAULT_TOPK = 50
_DEFAULT_MAX_FRAMES = 125


class MisoTTSSingleStageForVLLM(nn.Module):
    """VLLM wrapper for single-stage Miso TTS - bypasses vLLM generation framework."""
    
    requires_raw_input_tokens = False  # We don't use vLLM's token generation
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_path = vllm_config.model_config.model
        self._model: MisoTTSSingleStage | None = None
        self._device: torch.device | None = None
        self._lock = threading.Lock()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        with self._lock:
            if self._model is not None:
                return None
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device
            dtype = (
                torch.bfloat16
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
                if device.type == "cuda"
                else torch.float32
            )
            path = self.model_path or DEFAULT_MISO_TTS_REPO_ID
            self._model = load_miso_single_stage(path, device, dtype)
        for _ in weights:
            pass
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict]:
        return [{"text": "hello", "speaker": 0, "_is_dummy": True}] * num_reqs

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        
        infos = runtime_additional_information or [{}]
        
        # Process each request using official Generator
        audios = []
        for info in infos:
            text = str(info.get("text", "") or "").strip()
            if not text:
                audios.append(torch.zeros(24000, device=self._device))  # 1 second of silence
                continue
            
            speaker = int(info.get("speaker", 0))
            max_frames = int(info.get("max_generation_frames", _DEFAULT_MAX_FRAMES))
            max_audio_length_ms = max_frames * 80
            temperature = float(info.get("temperature", _DEFAULT_TEMPERATURE))
            topk = int(info.get("topk", _DEFAULT_TOPK))
            
            # Parse context if provided
            ctx = info.get("context", None)
            context = []
            if ctx is not None:
                if not isinstance(ctx, list):
                    ctx = [ctx]
                for seg in ctx:
                    if isinstance(seg, dict):
                        context.append(Segment(
                            speaker=int(seg.get("speaker", 0)),
                            text=str(seg.get("text", "")),
                            audio=torch.tensor(seg.get("audio", []), dtype=torch.float32, device=self._device)
                        ))
            
            try:
                audio = self._model.generate(
                    text=text,
                    speaker=speaker,
                    context=context,
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=temperature,
                    topk=topk,
                )
                audios.append(audio)
            except Exception as e:
                audios.append(torch.zeros(24000, device=self._device))
        
        # Return audio directly - bypass vLLM's token generation
        return {"audio": audios}
