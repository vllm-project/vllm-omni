"""SoulX-Singer SVS (score-driven) pipeline implementation."""

from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.soulx_singer.modules import (
    CFMDecoder,
    ConvNeXtV2Block,
)
from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_base import (
    FlowMatchingAudioPipeline,
    convert_soulx_audio_output_to_numpy,
)
from vllm_omni.diffusion.models.soulx_singer.preprocess.payload import (
    SOULX_PREPROCESSED_KEY,
    get_soulx_preprocessed_payload,
    has_precomputed,
)
from vllm_omni.diffusion.models.soulx_singer.preprocess.pre_process import (
    attach_preprocess_for_diffusion_request,
    build_metadata_processor,
    is_warmup_request,
    normalize_svs_control_extra_args,
    resolve_preprocess_audio,
)
from vllm_omni.diffusion.models.soulx_singer.utils import (
    f0_to_coarse,
    resolve_pitch_shift,
    validate_soulx_extra_args,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = init_logger(__name__)


def get_soulxsinger_pre_process_func(od_config: OmniDiffusionConfig):
    """Validate/load SVS preprocess payload for single-stage or stage-1 DiT."""
    metadata_processor = build_metadata_processor(od_config)
    _pipeline = None

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        nonlocal _pipeline

        extra_args = validate_soulx_extra_args(
            "svs",
            dict(getattr(request.sampling_params, "extra_args", None) or {}),
        )

        # Inline build: when no warmup/no precomputed paths/no IPC payload,
        # build the preprocess payload directly from audio file paths.
        if not (is_warmup_request(request) or has_precomputed(extra_args, "svs")):
            prompt = request.prompt
            if not isinstance(prompt, str) and not get_soulx_preprocessed_payload(prompt):  # type: ignore[arg-type]
                prompt_audio, target_audio = resolve_preprocess_audio(prompt, extra_args)  # type: ignore[arg-type]
                if prompt_audio is not None and target_audio is not None:
                    if _pipeline is None:
                        from vllm_omni.diffusion.models.soulx_singer.modules.preprocess.pipeline import (
                            SoulXPreprocessPipeline,
                        )

                        _pipeline = SoulXPreprocessPipeline(
                            od_config=od_config,
                            vocal_sep=bool(extra_args.get("vocal_sep", False)),
                            verbose=bool(extra_args.get("preprocess_verbose", False)),
                            extra_args=extra_args,
                        )
                    payload = _pipeline.build_svs_payload_from_audio(
                        prompt_audio=prompt_audio,
                        target_audio=target_audio,
                        metadata_processor=metadata_processor,
                        language=str(extra_args.get("language", "Mandarin")),
                        vocal_sep=extra_args.get("vocal_sep"),
                        prompt_vocal_sep=extra_args.get("prompt_vocal_sep"),
                        target_vocal_sep=extra_args.get("target_vocal_sep"),
                        prompt_max_merge_duration_ms=extra_args.get("prompt_max_merge_duration"),
                        target_max_merge_duration_ms=extra_args.get("target_max_merge_duration"),
                    )
                    payload.setdefault("kind", "svs")
                    prompt.setdefault("additional_information", {})[SOULX_PREPROCESSED_KEY] = payload  # type: ignore[union-attr, assignment]

        return attach_preprocess_for_diffusion_request(
            request,
            kind="svs",
            metadata_processor=metadata_processor,
        )

    return pre_process_func


def get_soulxsinger_post_process_func(od_config: OmniDiffusionConfig):
    """Convert pipeline audio tensor output for offline consumers."""

    def post_process_func(audio: torch.Tensor):
        return convert_soulx_audio_output_to_numpy(audio)

    return post_process_func


def _expand_states(h: torch.Tensor, mel2token: torch.Tensor) -> torch.Tensor:
    if mel2token.max() > h.size(1) - 1:
        logger.warning(
            "mel2token.max() (%s) is greater than h.size(1) - 1 (%s); clamping.",
            mel2token.max(),
            h.size(1) - 1,
        )
        mel2token = torch.clamp(mel2token, 0, h.size(1) - 1)
    mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
    return torch.gather(h, 1, mel2token_)


class PipelineSoulXSingerSVS(FlowMatchingAudioPipeline):
    """Pipeline for the SoulX-Singer model (SoulX-Singer)."""

    _encoder_modules: ClassVar[list[str]] = [
        "mel",
        "f0_encoder",
        "preflow",
        "note_text_encoder",
        "note_pitch_encoder",
        "note_type_encoder",
    ]

    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "prompt_metadata_path",
            "target_metadata_path",
            "audio_path",
            "prompt_audio",
            "target_audio",
            "language",
            "vocal_sep",
            "max_merge_duration",
            "preprocess_weights_dir",
            "preprocess_verbose",
            "control",
            "auto_shift",
            "pitch_shift",
        }
    )

    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset({"f0_shift"})

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.metadata_processor = build_metadata_processor(od_config)

        self.f0_encoder = nn.Embedding(self.f0_bin, self.f0_dim)
        self.preflow = nn.Sequential(
            *[
                ConvNeXtV2Block(
                    self.text_dim,
                    self.text_dim * 2,
                )
                for _ in range(self.encoder_config.num_layers)
            ]
        )

        self.cfm_decoder = CFMDecoder(self.flow_matching_config)

        self.mel, self.vocoder = self._build_fp32_audio_modules(self.audio_config)

        self.note_text_encoder = nn.Embedding(self.vocab_size, self.text_dim)
        self.note_pitch_encoder = nn.Embedding(256, self.pitch_dim)
        self.note_type_encoder = nn.Embedding(256, self.type_dim)

        self._setup_soulx_profiler()

    def _encode_condition(
        self,
        *,
        note_pitch: torch.Tensor,
        note_type: torch.Tensor,
        note_text: torch.Tensor,
        mel2note: torch.Tensor,
        f0_coarse: torch.Tensor,
    ) -> torch.Tensor:
        features = (
            self.note_pitch_encoder(note_pitch) + self.note_type_encoder(note_type) + self.note_text_encoder(note_text)
        )
        features = self.preflow(features)
        features = _expand_states(features, mel2note)
        features = features + self.f0_encoder(f0_coarse)
        return self._to_trunk_dtype(features)[0]

    def _infer_svs_segment(
        self,
        prompt_meta: dict,
        target_meta: dict,
        *,
        pitch_shift: int,
        num_inference_steps: int,
        guidance_scale: float,
        prompt_mel: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        prompt_note_text = prompt_meta["phoneme"]
        prompt_mel2note = prompt_meta["mel2note"]
        prompt_note_type = prompt_meta["note_type"]
        prompt_note_pitch = prompt_meta["note_pitch"]
        prompt_f0 = prompt_meta["f0"]

        target_note_text = target_meta["phoneme"]
        target_mel2note = target_meta["mel2note"]
        target_note_type = target_meta["note_type"]
        target_note_pitch = target_meta["note_pitch"]
        target_f0 = target_meta["f0"]

        if target_f0 is None or prompt_f0 is None:
            target_f0 = torch.zeros_like(target_mel2note).float()
            prompt_f0 = torch.zeros_like(prompt_mel2note).float()
        if target_note_pitch is None or prompt_note_pitch is None:
            target_note_pitch = torch.zeros_like(target_note_type).int()
            prompt_note_pitch = torch.zeros_like(prompt_note_type).int()

        len_prompt_note = prompt_note_pitch.shape[1]
        len_prompt_mel = prompt_f0.shape[1] if prompt_f0 is not None else prompt_mel2note.shape[1]

        note_pitch = torch.cat([prompt_note_pitch, target_note_pitch], dim=1)
        note_text = torch.cat([prompt_note_text, target_note_text], dim=1)
        note_type = torch.cat([prompt_note_type, target_note_type], dim=1)
        # Target note indices follow prompt notes in the concatenated score.
        mel2note = torch.cat([prompt_mel2note, target_mel2note + len_prompt_note], dim=1)

        # pitch_shift is semitones; each coarse F0 bin is 20 cents (×5 bins per semitone).
        f0_coarse_prompt = f0_to_coarse(prompt_f0)
        f0_coarse_target = f0_to_coarse(target_f0, f0_shift=pitch_shift * 5)
        f0_coarse = torch.cat([f0_coarse_prompt, f0_coarse_target], dim=1)

        note_pitch = note_pitch.clone()
        note_pitch[note_pitch > 0] = note_pitch[note_pitch > 0] + pitch_shift
        note_pitch = torch.clamp(note_pitch, 0, 255)

        if prompt_mel is None:
            prompt_wav = prompt_meta["wav"]
            with self._stage_timer("mel"):
                prompt_mel = self._mel_from_wav(prompt_wav)

        if prompt_mel.shape[1] > len_prompt_mel:
            prompt_mel = prompt_mel[:, :len_prompt_mel, :]
        elif prompt_mel.shape[1] < len_prompt_mel:
            logger.warning(
                "prompt_mel length %s is shorter than metadata frames %s; padding mel.",
                prompt_mel.shape[1],
                len_prompt_mel,
            )
            prompt_mel = F.pad(prompt_mel, (0, 0, 0, len_prompt_mel - prompt_mel.shape[1]))

        with self._stage_timer("cond_encode"):
            cond = self._encode_condition(
                note_pitch=note_pitch,
                note_type=note_type,
                note_text=note_text,
                mel2note=mel2note,
                f0_coarse=f0_coarse,
            )

        with self._stage_timer("cfm"):
            generated_mel = self._run_flow_matching_loop(
                prompt_mel,
                cond,
                num_inference_steps,
                guidance_scale,
                generator=generator,
            )
        with self._stage_timer("vocoder"):
            return self._mel_to_audio(generated_mel)

    def infer_svs_batch(
        self,
        payload: dict[str, Any],
        *,
        extra_args: dict[str, Any],
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Pure batch (non-streaming): run diffusion for each target segment in order and concat."""
        prompt_meta_raw = payload.get("prompt_meta")
        target_meta_list = payload["target_meta_list"]
        control = str(extra_args.get("control", "score"))

        prompt_meta = None
        prompt_mel = None
        if prompt_meta_raw:
            prompt_meta = self._ensure_processed_meta(prompt_meta_raw)
            prompt_meta = self._apply_control_mode(prompt_meta, control)
            if prompt_meta and prompt_meta.get("wav") is not None:
                prompt_mel = self._mel_from_wav(prompt_meta["wav"])

        pieces: list[torch.Tensor] = []
        pitch_shift = 0

        for idx, target_raw in enumerate(target_meta_list):
            target_meta = self._ensure_processed_meta(target_raw)
            target_meta = self._apply_control_mode(target_meta, control)

            if idx == 0:
                pitch_shift = resolve_pitch_shift(
                    auto_shift=bool(extra_args.get("auto_shift", True)),
                    manual_shift=int(extra_args.get("pitch_shift", 0)),
                    prompt_f0=prompt_meta["f0"] if prompt_meta else None,
                    target_f0=target_meta["f0"],
                    prompt_note_pitch=prompt_meta["note_pitch"] if prompt_meta else None,
                    target_note_pitch=target_meta["note_pitch"],
                )

            seg = self._infer_svs_segment(
                prompt_meta or {},
                target_meta,
                pitch_shift=pitch_shift,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                prompt_mel=prompt_mel,
                generator=generator,
            )
            pieces.append(seg.squeeze())

        if not pieces:
            return torch.zeros(1, 0, device=self.device, dtype=torch.float32), pitch_shift

        full = torch.cat(pieces, dim=-1)
        return full.unsqueeze(0), pitch_shift

    @staticmethod
    def _apply_control_mode(meta: dict, control: str) -> dict:
        meta["note_pitch"] = meta["note_pitch"] if control == "score" else None
        meta["f0"] = meta["f0"] if control == "melody" else None
        return meta

    def _ensure_processed_meta(self, meta: dict) -> dict:
        if isinstance(meta.get("phoneme"), torch.Tensor):
            return meta
        return self.metadata_processor.process(meta, None)

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        return self._forward_batch_from_request(
            req,
            kind="svs",
            metadata_key="f0_shift",
            infer_batch_fn=self.infer_svs_batch,
            prepare_extra_args=lambda extra_args, _sampling: normalize_svs_control_extra_args(extra_args),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        self._load_soulx_checkpoint("model.pt")
        return {name for name, _ in self.named_parameters()}
