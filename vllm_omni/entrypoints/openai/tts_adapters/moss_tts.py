# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS-TTS serving adapters (Nano + full family).

Both variants share the same build/validate flow (``_build_moss_tts_params``
handles each); they are registered under distinct model-type names.
"""

import asyncio
import threading
from typing import TYPE_CHECKING, Any, cast

from transformers import AutoModel, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from vllm.inputs import tokens_input
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    OutputPolicy,
    PreparedRequest,
    apply_max_new_tokens,
    conditioning_cache_salt,
)
from vllm_omni.model_executor.models.moss_tts.realtime_prompt import build_realtime_prompt

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)


class _MossTTSAdapterBase(ARTTSAdapter):
    accumulate_nonstreaming: bool = False

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._moss_variant = None if self.name == "moss_tts_nano" else self._detect_moss_variant()
        self._moss_processor_cache = None
        self._moss_realtime_components_lock = threading.Lock()

    @property
    def engine_client(self):
        return self.ctx.engine_client

    @property
    def uploaded_speakers(self):
        return self.ctx.server.uploaded_speakers

    @property
    def _speaker_cache(self):
        return self.ctx.server._speaker_cache

    async def _resolve_ref_audio(self, ref_audio: str):
        if self._moss_variant is None:  # Nano sends lists through engine IPC.
            return await self.ctx.server._resolve_ref_audio(ref_audio)
        return await self.ctx.server._resolve_ref_audio_array(ref_audio)

    def _voice_created_at(self, voice: str) -> int:
        return self.ctx.server._voice_created_at(voice)

    def _get_resolved_ref_audio_artifact_key(self, cache_key: str):
        return self.ctx.server._get_resolved_ref_audio_artifact_key(cache_key)

    def _get_moss_ref_encoder(self):
        """Lazily build the per-server MOSS reference-audio encoder once."""
        cached = getattr(self, "_moss_ref_encoder", None)
        if cached is not None:
            return cached
        from vllm_omni.model_executor.models.moss_tts.reference_encoder import build_reference_encoder

        # The variant's encode geometry (n_vq, working sample rate) is derived
        # inside the model package; this layer only supplies the processor and
        # the process-wide speaker cache.
        encoder = build_reference_encoder(
            self._get_moss_realtime_components()[2] if self._moss_variant == "realtime" else self._get_moss_processor(),
            variant=cast(str, self._moss_variant),
            speaker_cache=self._speaker_cache,
        )
        self._moss_ref_encoder = encoder
        return encoder

    async def _encode_moss_references(
        self,
        request: "OpenAICreateSpeechRequest",
        *,
        has_inline_ref_audio: bool,
        two_speaker: bool,
    ) -> tuple[list, dict[int, str]]:
        """Encode the request's reference clip(s) into MOSS RVQ code tensors.

        Reference encoding + speaker caching lives in the model package
        (moss_tts.reference_encoder), mirroring Fish Speech / CosyVoice3 /
        Qwen3-TTS which keep reference handling with the model rather than in
        this shared serving file. This method only resolves the serving-side
        inputs the model helper needs: the generic audio resolver/artifact-key
        lookups and whether the request may use the named-voice cache.

        Returns ``(codes_per_speaker, resolve_keys)`` where ``resolve_keys``
        maps the reference slot (0 = ref_audio, 1 = ref_audio_2) to its
        content-aware resolve key, for salting the KV prefix cache.
        """
        from vllm_omni.model_executor.models.moss_tts.reference_encoder import encode_request_references

        # Named-voice caching is only valid for uploaded speakers without an
        # inline ref_audio: ``request.voice`` plus a file/URL would otherwise
        # key on (name, created_at=0) and skip the content-aware resolve.
        raw_voice = getattr(request, "voice", None)
        raw_voice = raw_voice.strip() if isinstance(raw_voice, str) else ""
        voice_lower = raw_voice.lower()
        use_named_voice = bool(voice_lower) and voice_lower in self.uploaded_speakers and not has_inline_ref_audio
        voice = voice_lower if use_named_voice else ""
        voice_created = self._voice_created_at(voice) if voice else 0

        return await encode_request_references(
            self._get_moss_ref_encoder(),
            cast(str, request.ref_audio),
            request.ref_audio_2 if two_speaker else None,
            resolve_ref_audio=self._resolve_ref_audio,
            get_artifact_key=self._get_resolved_ref_audio_artifact_key,
            voice_name=voice or None,
            voice_created_at=voice_created,
        )

    def _detect_moss_variant(self) -> str:
        """Sub-classify a ``moss_tts``-stage server into the actual MOSS-TTS
        variant family (tts, ttsd, sound_effect, voice_generator, realtime).

        Detection key is the HF repo path / model_name; matches
        ``_try_resolve_omni_model_type`` in entrypoints/utils.py so users get
        consistent behaviour no matter how they launched the server (--model
        OpenMOSS-Team/MOSS-TTSD-v1.0 vs --deploy-config moss_ttsd.yaml).
        """
        try:
            name = (self.engine_client.model_config.model or "").lower().replace("-", "").replace("_", "")
        except Exception:
            name = ""
        if "realtime" in name:
            return "realtime"
        if "local" in name:
            return "local"
        if "ttsd" in name:
            return "ttsd"
        if "soundeffect" in name:
            return "sound_effect"
        if "voicegenerator" in name:
            return "voice_generator"
        return "tts"

    def _codec_stage_devices(self) -> str | None:
        """Read the runtime ``devices`` entry of this pipeline's codec stage.

        The codec (code2wav) stage is the final stage of every MOSS pipeline
        (``moss_tts_local_codec`` / ``moss_tts_codec``); prefer a stage whose
        ``model_stage`` mentions "codec", else anchor on the last stage. The
        runtime mapping accepts the resolved ``runtime_config`` object, the
        typed ``runtime_config``.
        """
        stages = list(getattr(self.engine_client, "stage_configs", None) or ())
        if not stages:
            return None
        codec_stages = [s for s in stages if "codec" in str(getattr(s, "model_stage", "") or "")]
        anchor = codec_stages[-1] if codec_stages else stages[-1]
        for attr in ("runtime_config",):
            runtime_cfg = getattr(anchor, attr, None)
            if runtime_cfg is None:
                continue
            devices = (
                runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "devices", None)
            )
            if devices is not None and str(devices).strip():
                return str(devices)
        devices = getattr(anchor, "devices", None)
        return str(devices) if devices is not None and str(devices).strip() else None

    def _resolve_ref_encoder_device(self):
        """Pick the device of the API-process reference-audio encoder.

        The encoder follows the code2wav stage onto its first GPU: the stage
        worker remaps its ``devices`` entry through ``CUDA_VISIBLE_DEVICES``
        and sees it as logical index 0, while replica launch restores the
        parent environment, so the same index names the same physical GPU in
        this process. When the stage pins no devices the stage worker and the
        encoder both default to the first visible GPU. Falls back to CPU when
        CUDA is unavailable or the stage device is not parseable.
        """
        import torch  # local to avoid pulling torch at module import time

        if not torch.cuda.is_available():
            return torch.device("cpu")
        devices_str = self._codec_stage_devices()
        if devices_str is None:
            # No explicit pinning: the stage worker lands on the first visible
            # GPU (vLLM picks cuda:0 of the un-remapped environment).
            return torch.device("cuda", 0)
        first = devices_str.split(",")[0].strip()
        if not first:
            return torch.device("cuda", 0)
        if first == "cpu":
            return torch.device("cpu")
        if not first.isdigit():
            logger.warning("MOSS ref encoder: codec stage devices %r not parseable; keeping CPU", devices_str)
            return torch.device("cpu")
        index = int(first)
        device_count = torch.accelerator.device_count()
        if index >= device_count:
            logger.warning(
                "MOSS ref encoder: codec stage device cuda:%d out of range (%d visible); keeping CPU",
                index,
                device_count,
            )
            return torch.device("cpu")
        return torch.device("cuda", index)

    def _get_moss_processor(self):
        """Lazily load the upstream MOSS-TTS processor once per server.

        Cached on ``self._moss_processor_cache``. The processor owns its own
        audio_tokenizer (~1.6 B params) used for per-request reference-audio
        encoding; it is placed on the code2wav stage's GPU (see
        ``_resolve_ref_encoder_device``) so cold encodes run on device instead
        of paying a CPU forward per request. Note this memory is charged to
        the API process, outside each stage's ``gpu_memory_utilization``
        budget — keep some headroom on the codec GPU.
        """
        cached = getattr(self, "_moss_processor_cache", None)
        if cached is not None:
            return cached
        from transformers import AutoProcessor

        model_id = self.engine_client.model_config.model
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(proc, "audio_tokenizer"):
            device = self._resolve_ref_encoder_device()
            proc.audio_tokenizer = proc.audio_tokenizer.to(device).eval()
            logger.info("MOSS reference-audio encoder (audio_tokenizer) placed on %s", device)
        self._moss_processor_cache = proc
        return proc

    def _get_moss_realtime_components(self):
        with self._moss_realtime_components_lock:
            cached = getattr(self, "_moss_realtime_components", None)
            if cached is not None:
                return cached

            model_id = self.engine_client.model_config.model
            processor_cls = get_class_from_dynamic_module(
                "processing_mossttsrealtime.MossTTSRealtimeProcessor",
                model_id,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            processor = processor_cls(tokenizer=tokenizer)
            hf_config = self.engine_client.model_config.hf_config
            codec_path = str(
                getattr(
                    hf_config,
                    "codec_model_name_or_path",
                    getattr(
                        hf_config,
                        "audio_tokenizer_name_or_path",
                        "OpenMOSS-Team/MOSS-Audio-Tokenizer",
                    ),
                )
            )
            codec = AutoModel.from_pretrained(
                codec_path,
                trust_remote_code=True,
            )
            # Reference encoding runs in the API process; follow the codec
            # (code2wav) stage's GPU like the processor path above.
            codec = codec.to(self._resolve_ref_encoder_device()).eval()
            cached = (tokenizer, processor, codec)
            self._moss_realtime_components = cached
            return cached

    async def _build_moss_tts_params(
        self,
        request: "OpenAICreateSpeechRequest",
        *,
        has_inline_ref_audio: bool = False,
    ) -> dict[str, Any]:
        """Build the talker prompt + ``additional_information`` payload for any
        MOSS-TTS-family request (nano + 5 full variants).

        For the legacy ``moss_tts_nano`` model_type, keeps the original nano
        contract (``{text, mode=voice_clone, prompt_audio_array}``); the
        caller still uses a ``[1]`` placeholder prompt.

        For the full MOSS-TTS family (``MossTTSDelay*`` / ``MossTTSRealtime*``)
        we **call the upstream processor server-side** to produce the unified
        ``(text_ids, audio_codes)`` shape the talker actually consumes — same
        flow as ``examples/.../moss_tts/end2end.py:_build_unified_codes``.
        Returns ``{prompt_token_ids: list[int], codes.ref: torch.LongTensor,
        max_new_frames, ...}``. The caller treats ``prompt_token_ids`` as the
        prompt and forwards the rest as ``additional_information``.
        """
        import torch  # local to avoid pulling torch at module import time

        v = self._moss_variant

        params: dict[str, Any]
        # ---- Legacy nano path (unchanged) ----
        if v is None:  # moss_tts_nano
            params = {
                "text": [request.input or ""],
                "mode": ["voice_clone"],
            }
            if request.max_new_tokens is not None:
                params["max_new_frames"] = [request.max_new_tokens]
            wav_list, sr, cache_key = await self._resolve_ref_audio(cast(str, request.ref_audio))
            params["prompt_audio_array"] = [[wav_list, sr]]
            params["ref_audio_cache_key"] = cache_key
            return params

        if v == "realtime":
            tokenizer, processor, _ = await asyncio.to_thread(self._get_moss_realtime_components)
            references, realtime_resolve_keys = await self._encode_moss_references(
                request,
                has_inline_ref_audio=has_inline_ref_audio,
                two_speaker=False,
            )
            params = build_realtime_prompt(tokenizer, processor, request.input or "", references[0])
            if request.max_new_tokens is not None:
                params["max_new_frames"] = [request.max_new_tokens]
            if 0 in realtime_resolve_keys:
                params["ref_audio_cache_key"] = realtime_resolve_keys[0]
            return params

        # ---- MossTTSDelay family (tts/ttsd/sound_effect/voice_generator)
        # and MOSS-TTS-Local-Transformer-v1.5: call the upstream processor
        # server-side to produce unified codes. Local-v1.5 ships its own
        # AutoProcessor (processor_config.json + processing_moss_tts.py) and
        # reuses this exact build_user_message/encode_audios_from_wav path in
        # the offline example (examples/.../moss_tts/end2end.py:
        # _build_unified_codes) -- it is NOT in the same boat as Realtime
        # (no processor_config.json there), so it must not fall back to the
        # prompt_audio_array path above (which the talker's preprocess()
        # never reads -- info_dict["codes"]["ref"] is the only thing it
        # consumes, so skipping this path silently drops all voice-clone
        # conditioning and produces unconditioned/garbage audio online). ----
        proc = self._get_moss_processor()

        user_kwargs: dict[str, Any] = {"text": request.input or ""}
        resolve_keys: dict[int, str] = {}
        if v in ("tts", "local", "ttsd"):
            user_kwargs["reference"], resolve_keys = await self._encode_moss_references(
                request,
                has_inline_ref_audio=has_inline_ref_audio,
                two_speaker=(v == "ttsd"),
            )
        elif v == "sound_effect":
            user_kwargs["text"] = request.input or ""  # may be empty
            user_kwargs["ambient_sound"] = request.ambient_sound or ""
            if request.duration_seconds is not None:
                user_kwargs["tokens"] = max(1, int(float(request.duration_seconds) * 12.5))
            elif request.max_new_tokens is not None:
                user_kwargs["tokens"] = int(request.max_new_tokens)
        elif v == "voice_generator":
            user_kwargs["instruction"] = request.instructions or ""

        # Optional language tag for the spoken-text variants. MOSS-TTS-v1.5's
        # headline improvement is multilingual synthesis when the language is
        # given (build_user_message(..., language=...)); 1.0 ignores it
        # gracefully. Sound-effect output is non-verbal, so skip it there.
        if v in ("tts", "local", "ttsd", "voice_generator") and getattr(request, "language", None):
            user_kwargs["language"] = request.language

        # Build the unified-codes prompt: (L, 1+n_vq) where col 0 is text/special
        # tokens and cols 1..n_vq are the delay-pattern audio code grid (mostly
        # audio_pad_code outside the reference block).
        if v == "local" and request.ref_text and request.ref_text.strip():
            reference = user_kwargs.pop("reference")
            user_kwargs["text"] = request.ref_text.strip() + " " + (request.input or "")
            user_msg = proc.build_user_message(**user_kwargs)
            assistant_msg = proc.build_assistant_message(audio_codes_list=reference)
            batch = proc(conversations=[[user_msg, assistant_msg]], mode="continuation")
        else:
            user_msg = proc.build_user_message(**user_kwargs)
            batch = proc(conversations=[[user_msg]], mode="generation")
        unified = batch["input_ids"][0]  # torch.LongTensor (L, 1+n_vq)
        text_ids: list[int] = unified[:, 0].tolist()
        audio_codes: torch.Tensor = unified[:, 1:].contiguous().to(torch.int64)

        params = {
            "prompt_token_ids": text_ids,
            "codes": {"ref": audio_codes},
        }
        if request.max_new_tokens is not None:
            params["max_new_frames"] = [request.max_new_tokens]
        if 0 in resolve_keys:
            params["ref_audio_cache_key"] = resolve_keys[0]
        if 1 in resolve_keys:
            params["ref_audio_2_cache_key"] = resolve_keys[1]
        return params

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Validate any MOSS-TTS-family request (nano + 5 full variants).

        Dispatches by ``self._moss_variant``:
          - ``tts``/``realtime``: require ``ref_audio`` (voice cloning).
          - ``ttsd``: require ``ref_audio`` (speaker 1); ``ref_audio_2``
            optional (defaults to the same ref for both speakers).
          - ``sound_effect``: require ``ambient_sound`` (no ref_audio).
          - ``voice_generator``: require ``instructions`` (no ref_audio).
          - For the legacy moss_tts_nano model_type the variant is None and
            we fall through to the original nano contract (ref_audio only).
        """
        server = self.ctx.server
        err = server._apply_uploaded_speaker(request)
        if err:
            return err

        if not request.input or not request.input.strip():
            # SoundEffect can legitimately have empty input (just ambient_sound).
            if self._moss_variant != "sound_effect":
                return "Input text cannot be empty"

        v = self._moss_variant
        if v in (None, "tts", "realtime", "local"):
            if request.ref_audio is None:
                label = (
                    "MOSS-TTS-Nano"
                    if v is None
                    else (
                        "MOSS-TTS-Realtime"
                        if v == "realtime"
                        else ("MOSS-TTS-Local-Transformer" if v == "local" else "MOSS-TTS")
                    )
                )
                return f"{label} requires 'ref_audio' (reference audio for voice cloning)."
            return server._validate_ref_audio_format(request.ref_audio)

        if v == "ttsd":
            if request.ref_audio is None:
                return "MOSS-TTSD requires 'ref_audio' (speaker 1 reference)."
            fmt_err = server._validate_ref_audio_format(request.ref_audio)
            if fmt_err:
                return fmt_err
            if request.ref_audio_2 is not None:
                return server._validate_ref_audio_format(request.ref_audio_2)
            return None

        if v == "sound_effect":
            if not request.ambient_sound or not request.ambient_sound.strip():
                return (
                    "MOSS-SoundEffect requires 'ambient_sound' (natural language "
                    "description of the sound effect to synthesise)."
                )
            return None

        if v == "voice_generator":
            if not request.instructions or not request.instructions.strip():
                return (
                    "MOSS-VoiceGenerator requires 'instructions' (natural language "
                    "voice description, e.g. 'a warm female voice with an American accent')."
                )
            return None

        return None  # unreachable

    async def build(
        self, request: "OpenAICreateSpeechRequest", sampling_params_list: list, has_inline_ref_audio: bool
    ) -> PreparedRequest:
        server = self.ctx.server
        tts_params = await self._build_moss_tts_params(request, has_inline_ref_audio=has_inline_ref_audio)
        if request.voice:
            voice_lower = request.voice.lower()
            if voice_lower in server.uploaded_speakers and not has_inline_ref_audio:
                tts_params["voice_name"] = [voice_lower]
                tts_params["voice_created_at"] = [server._voice_created_at(voice_lower)]
        # MOSS samples internally from additional_information. build() runs
        # before the shared path applies request.seed to SamplingParams.
        seed = request.seed
        if seed is None and sampling_params_list:
            seed = getattr(sampling_params_list[0], "seed", None)
        if seed is not None:
            tts_params["seed"] = [int(seed)]
        if isinstance(tts_params.get("prompt_token_ids"), list):
            prompt_token_ids = tts_params.pop("prompt_token_ids")
            prompt = tokens_input(prompt_token_ids=prompt_token_ids)
        else:
            prompt = tokens_input(prompt_token_ids=[1])
        prompt["additional_information"] = tts_params
        prompt["cache_salt"] = conditioning_cache_salt(request, tts_params)
        return PreparedRequest(
            prompt=prompt,
            tts_params=tts_params,
            model_type=self.name,
            output_policy=OutputPolicy(accumulate_nonstreaming=self.accumulate_nonstreaming),
        )

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)


@register_tts_adapter
class MossTTSNanoAdapter(_MossTTSAdapterBase):
    stage_keys = frozenset({"moss_tts_nano"})
    name = "moss_tts_nano"
    accumulate_nonstreaming = True


@register_tts_adapter
class MossTTSAdapter(_MossTTSAdapterBase):
    stage_keys = frozenset({"moss_tts", "moss_tts_codec", "moss_tts_local", "moss_tts_local_codec"})
    name = "moss_tts"
