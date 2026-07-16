"""SoulX preprocess pipeline."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.soulx_singer.modules.preprocess.stack import SoulXPreprocessStack
from vllm_omni.diffusion.models.soulx_singer.modules.preprocess.utils import load_mono_audio, resample_mono
from vllm_omni.diffusion.models.soulx_singer.preprocess.metadata_utils import (
    SegmentMetadata,
    _merge_group,
    convert_metadata,
)
from vllm_omni.diffusion.models.soulx_singer.utils import (
    load_config,
    load_wav,
    preprocess_weight_paths,
    resolve_preprocess_weights_root,
    resolve_soulx_kind,
)

logger = init_logger(__name__)


def build_svs_prompt_meta(
    preprocess: SoulXPreprocessPipeline,
    *,
    prompt_audio: str | tuple[np.ndarray, int],
    prompt_meta_raw: dict[str, Any],
    metadata_processor,
    vocal_sep: bool | None = None,
) -> dict[str, Any]:
    """Build SVS prompt_meta; in-memory prompt wav is trimmed to mel2note length."""
    if isinstance(prompt_audio, str):
        return metadata_processor.process(prompt_meta_raw, prompt_audio)
    prompt_meta = metadata_processor.process(prompt_meta_raw, None)
    prompt_vocal, prompt_sr = preprocess._extract_vocal(prompt_audio, vocal_sep=vocal_sep)
    if prompt_sr != metadata_processor.sample_rate:
        prompt_vocal = resample_mono(
            prompt_vocal,
            orig_sr=prompt_sr,
            target_sr=metadata_processor.sample_rate,
        )
    max_samples = prompt_meta["mel2note"].shape[1] * metadata_processor.hop_size
    segment_wav = np.asarray(prompt_vocal[:max_samples], dtype=np.float32)
    prompt_meta["wav"] = torch.from_numpy(segment_wav).unsqueeze(0).float().to(metadata_processor.device)
    return prompt_meta


class SoulXPreprocessPipeline(nn.Module, SupportAudioInput, SupportsComponentDiscovery):
    """Lazy-loaded preprocess stack integrated with vLLM-Omni diffusion lifecycle."""

    support_audio_input: ClassVar[bool] = True
    weights_sources: ClassVar[tuple] = ()
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = [
        "stack.vocal_sep",
        "stack.lyric",
        "stack.rosvot",
    ]
    _vae_modules: ClassVar[list[str]] = []
    _resident_modules: ClassVar[list[str]] = ["stack.rmvpe"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
        vocal_sep: bool = True,
        midi_transcribe: bool = True,
        max_merge_duration_ms: int = 60000,
        verbose: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        extra_args = extra_args or {}
        self.kind = resolve_soulx_kind(od_config.model)
        hf_config = load_config(str(Path(od_config.model) / "config.yaml"))
        audio_config = hf_config.audio
        if self.kind == "svc" and midi_transcribe:
            logger.warning("SVC model (config.json): midi_transcribe is ignored; lyric/MIDI transcription is SVS-only.")
            midi_transcribe = False

        self.vocal_sep = vocal_sep
        self.midi_transcribe = midi_transcribe
        self.max_merge_duration_ms = max_merge_duration_ms
        self.verbose = verbose
        self.target_sr = int(audio_config.sample_rate)
        self.hop_size = int(audio_config.hop_size)
        self._metadata_processor = None
        self._weights_root: Path | None = None
        self.stack: SoulXPreprocessStack | None = None
        self._ensure_stack(extra_args)

    def _ensure_stack(self, extra_args: dict[str, Any] | None = None) -> None:
        extra_args = extra_args or {}
        weights_root = resolve_preprocess_weights_root(self.od_config)
        if self.stack is not None and self._weights_root == weights_root:
            return
        weights = preprocess_weight_paths(weights_root)
        device = str(get_local_device())
        self.stack = SoulXPreprocessStack(
            weights,
            device,
            target_sr=self.target_sr,
            hop_size=self.hop_size,
            verbose=self.verbose,
        )
        self._weights_root = weights_root

    def forward(self, req) -> DiffusionOutput:
        from vllm_omni.diffusion.models.soulx_singer.preprocess.payload import (
            SOULX_PREPROCESSED_KEY,
        )
        from vllm_omni.diffusion.models.soulx_singer.preprocess.pre_process import (
            build_metadata_processor,
            build_preprocess_payload,
            build_warmup_payload,
            is_warmup_request,
        )
        from vllm_omni.inputs.data import OmniTextPrompt

        extra_args = dict(getattr(req.sampling_params, "extra_args", None) or {})
        prompt = OmniTextPrompt(prompt=req.prompts[0]) if isinstance(req.prompts[0], str) else req.prompts[0]
        info = prompt.get("additional_information") or {}
        if isinstance(info, dict):
            extra_args.update(dict(info.get("extra_args") or {}))

        self._ensure_stack(extra_args)
        if self._metadata_processor is None:
            self._metadata_processor = build_metadata_processor(self.od_config)

        device = get_local_device()
        if is_warmup_request(req):
            payload = build_warmup_payload(
                self.kind,
                metadata_processor=self._metadata_processor,
                device=device,
                sample_rate=self.target_sr,
            )
        else:
            payload = build_preprocess_payload(
                self.kind,
                prompt=prompt,
                extra_args=extra_args,
                preprocess=self,
                metadata_processor=self._metadata_processor,
                device=device,
                sample_rate=self.target_sr,
            )

        return DiffusionOutput(
            output={
                "payload": {},
                "metadata": {SOULX_PREPROCESSED_KEY: payload},
            },
            to_cpu=True,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        self._ensure_stack({})
        return set()

    def _extract_vocal(
        self,
        audio: str | tuple[np.ndarray, int],
        *,
        vocal_sep: bool | None = None,
    ) -> tuple[np.ndarray, int]:
        if self.stack is None:
            self._ensure_stack({})
        use_sep = self.vocal_sep if vocal_sep is None else vocal_sep
        if use_sep:
            return self.stack.extract_vocal(audio)
        return load_mono_audio(audio)

    def extract_f0(self, vocal: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.stack is None:
            self._ensure_stack({})
        return self.stack.extract_f0(vocal, sample_rate)

    def iter_svs_metadata(
        self,
        audio_source: str | tuple[np.ndarray, int],
        *,
        language: str = "Mandarin",
        vocal_sep: bool | None = None,
        max_merge_duration_ms: int | None = None,
    ):
        """Yield merged SVS metadata dicts one segment at a time."""
        if isinstance(audio_source, tuple):
            vocal, sample_rate = audio_source
            origin_wav_fn = ""
        else:
            vocal, sample_rate = self._extract_vocal(str(audio_source), vocal_sep=vocal_sep)
            origin_wav_fn = str(audio_source)

        with tempfile.TemporaryDirectory(prefix="soulx_preprocess_") as work_dir:
            work = Path(work_dir)
            vocal_path = work / "vocal.wav"
            sf.write(vocal_path, vocal, sample_rate)

            vocal_f0_path = str(vocal_path).replace(".wav", "_f0.npy")
            vocal_f0 = self.stack.extract_f0(str(vocal_path), sample_rate=sample_rate, f0_path=vocal_f0_path)

            if not self.midi_transcribe:
                end_ms = int(len(vocal) / sample_rate * 1000)
                item = SegmentMetadata(
                    item_name="full",
                    wav_fn=str(vocal_path),
                    language=language,
                    start_time_ms=0,
                    end_time_ms=end_ms,
                    note_text=["<SP>"],
                    note_dur=[end_ms / 1000.0],
                    note_pitch=[0],
                    note_type=[1],
                )
                yield convert_metadata(item)
                return

            segments = self.stack.ensure_segmenter().forward(
                vocal,
                sample_rate,
                vocal_f0,
                base_name="vocal",
                origin_wav_fn=origin_wav_fn,
                verbose=self.verbose,
            )
            cut_dir = work / "cut_wavs"
            cut_dir.mkdir(parents=True, exist_ok=True)
            lyric = self.stack.ensure_lyric()
            rosvot = self.stack.ensure_rosvot()

            long_cut_dir = work / "long_cut_wavs"
            max_dur = max_merge_duration_ms or self.max_merge_duration_ms
            max_gap_ms = 10000
            current_group: list[Any] = []
            current_len = 0
            prev_end = -1

            def _flush_group() -> dict[str, Any]:
                merged_item = _merge_group(vocal, sample_rate, current_group, long_cut_dir)
                merged_f0_path = merged_item.wav_fn.replace(".wav", "_f0.npy")
                self.stack.extract_f0(merged_item.wav_fn, sample_rate=sample_rate, f0_path=merged_f0_path)
                return convert_metadata(merged_item)

            for seg in segments:
                seg_key = seg["item_name"]
                seg_wav_path = cut_dir / f"{seg_key}.wav"
                sf.write(seg_wav_path, seg["wav"], seg["sample_rate"])
                seg_f0_path = str(seg_wav_path).replace(".wav", "_f0.npy")
                self.stack.extract_f0(str(seg_wav_path), sample_rate=seg["sample_rate"], f0_path=seg_f0_path)

                words, durs = lyric.forward(
                    str(seg_wav_path),
                    language,
                    sample_rate=seg["sample_rate"],
                )
                seg_item = {
                    "item_name": seg_key,
                    "wav_fn": str(seg_wav_path),
                    "start_time_ms": seg["start_time_ms"],
                    "end_time_ms": seg["end_time_ms"],
                    "origin_wav_fn": origin_wav_fn or str(vocal_path),
                    "words": words,
                    "word_durs": durs,
                    "language": language,
                }
                meta_item = rosvot.transcribe(seg_item, segment_info=seg_item, verbose=self.verbose)

                start_time = int(meta_item.get("start_time_ms", seg["start_time_ms"]))
                end_time = int(meta_item.get("end_time_ms", seg["end_time_ms"]))
                if current_group and (
                    start_time - prev_end > max_gap_ms or current_len + end_time - start_time > max_dur
                ):
                    yield _flush_group()
                    current_group = []
                    current_len = 0

                current_group.append(meta_item)
                current_len += end_time - start_time
                prev_end = end_time

            if current_group:
                yield _flush_group()

    @staticmethod
    def build_svs_payload_from_paths(
        *,
        prompt_metadata_path: str,
        target_metadata_path: str,
        audio_path: str,
        metadata_processor,
    ) -> dict[str, Any]:
        with open(prompt_metadata_path, encoding="utf-8") as f:
            prompt_metadata = json.load(f)
        with open(target_metadata_path, encoding="utf-8") as f:
            target_metadata = json.load(f)
        if not prompt_metadata:
            raise ValueError("prompt_metadata is empty")
        if not target_metadata:
            raise ValueError("target_metadata is empty")

        prompt_meta = metadata_processor.process(prompt_metadata[0], audio_path)
        return {
            "kind": "svs",
            "prompt_meta": prompt_meta,
            "target_meta_list": target_metadata,
        }

    def build_svs_payload_from_audio(
        self,
        *,
        prompt_audio: str | tuple[np.ndarray, int],
        target_audio: str | tuple[np.ndarray, int],
        metadata_processor,
        language: str = "Mandarin",
        vocal_sep: bool | None = None,
        prompt_vocal_sep: bool | None = None,
        target_vocal_sep: bool | None = None,
        prompt_max_merge_duration_ms: int | None = None,
        target_max_merge_duration_ms: int | None = None,
    ) -> dict[str, Any]:
        p_sep = prompt_vocal_sep if prompt_vocal_sep is not None else vocal_sep
        t_sep = target_vocal_sep if target_vocal_sep is not None else vocal_sep
        p_merge = (
            prompt_max_merge_duration_ms if prompt_max_merge_duration_ms is not None else self.max_merge_duration_ms
        )
        t_merge = (
            target_max_merge_duration_ms if target_max_merge_duration_ms is not None else self.max_merge_duration_ms
        )
        prompt_list = list(
            self.iter_svs_metadata(
                prompt_audio,
                language=language,
                vocal_sep=p_sep,
                max_merge_duration_ms=p_merge,
            )
        )
        target_list = list(
            self.iter_svs_metadata(
                target_audio,
                language=language,
                vocal_sep=t_sep,
                max_merge_duration_ms=t_merge,
            )
        )
        if not prompt_list or not target_list:
            raise ValueError("SVS preprocess produced empty metadata")

        # Prompt wav is trimmed to mel2note length, not the metadata time window.
        prompt_meta = build_svs_prompt_meta(
            self,
            prompt_audio=prompt_audio,
            prompt_meta_raw=prompt_list[0],
            metadata_processor=metadata_processor,
            vocal_sep=vocal_sep,
        )

        return {
            "kind": "svs",
            "prompt_meta": prompt_meta,
            "target_meta_list": target_list,
        }

    @staticmethod
    def build_svc_payload_from_paths(
        *,
        prompt_wav_path: str,
        target_wav_path: str,
        prompt_f0_path: str,
        target_f0_path: str,
        sample_rate: int,
        device: torch.device | str,
    ) -> dict[str, Any]:
        prompt_wav = load_wav(prompt_wav_path, sample_rate).to(device)
        target_wav = load_wav(target_wav_path, sample_rate).to(device)
        prompt_f0 = torch.from_numpy(np.load(prompt_f0_path)).unsqueeze(0).to(device)
        target_f0 = torch.from_numpy(np.load(target_f0_path)).unsqueeze(0).to(device)
        return {
            "kind": "svc",
            "prompt_wav": prompt_wav,
            "target_wav": target_wav,
            "prompt_f0": prompt_f0,
            "target_f0": target_f0,
        }

    def build_svc_payload_from_audio(
        self,
        *,
        prompt_audio: str | tuple[np.ndarray, int],
        target_audio: str | tuple[np.ndarray, int],
        sample_rate: int,
        device: torch.device | str,
        vocal_sep: bool | None = None,
    ) -> dict[str, Any]:
        prompt_vocal, prompt_sr = self._extract_vocal(prompt_audio, vocal_sep=vocal_sep)
        target_vocal, target_sr = self._extract_vocal(target_audio, vocal_sep=vocal_sep)

        if prompt_sr != sample_rate:
            prompt_vocal = resample_mono(prompt_vocal, orig_sr=prompt_sr, target_sr=sample_rate)
        if target_sr != sample_rate:
            target_vocal = resample_mono(target_vocal, orig_sr=target_sr, target_sr=sample_rate)

        prompt_f0 = self.extract_f0(prompt_vocal, sample_rate)
        target_f0 = self.extract_f0(target_vocal, sample_rate)

        return {
            "kind": "svc",
            "prompt_wav": torch.from_numpy(prompt_vocal).unsqueeze(0).to(device),
            "target_wav": torch.from_numpy(target_vocal).unsqueeze(0).to(device),
            "prompt_f0": torch.from_numpy(prompt_f0).unsqueeze(0).to(device),
            "target_f0": torch.from_numpy(target_f0).unsqueeze(0).to(device),
        }
