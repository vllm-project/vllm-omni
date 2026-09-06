# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Base contract for per-model TTS serving adapters.

This package factors the per-model ``if self._tts_model_type == ...`` dispatch
in ``serving_speech.py`` into one adapter class per model. Each adapter owns its
model's request normalization, validation, prompt/param building, sampling
overrides, and output policy, so adding a model means writing one adapter file
instead of editing the shared serving module in ~10 scattered places.

See the RFC for the full design (issue #4327).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from vllm_omni.entrypoints.openai.tts_adapters.capabilities import load_codec_frame_rate, load_supported_speakers

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

DEFAULT_TTS_LANGUAGES = frozenset(
    {
        "Auto",
        "Chinese",
        "English",
        "Japanese",
        "Korean",
        "German",
        "French",
        "Russian",
        "Portuguese",
        "Spanish",
        "Italian",
    }
)

_conditioning_cache_salt_fn: "Callable[..., str] | None" = None


def conditioning_cache_salt(request: "OpenAICreateSpeechRequest", tts_params: dict) -> str:
    """Return the conditioning cache salt for ``request`` + ``tts_params``.

    Lazily imports and caches ``serving_speech._conditioning_cache_salt`` on first
    use: the import is deferred to break the adapters<->serving_speech import
    cycle, and cached so it resolves once instead of on every ``build()`` call.
    """
    global _conditioning_cache_salt_fn
    if _conditioning_cache_salt_fn is None:
        from vllm_omni.entrypoints.openai.serving_speech import _conditioning_cache_salt

        _conditioning_cache_salt_fn = _conditioning_cache_salt
    return _conditioning_cache_salt_fn(request, tts_params)


def apply_max_new_tokens(
    sampling_params_list: list,
    request: "OpenAICreateSpeechRequest",
) -> list:
    """Apply a request-level ``max_new_tokens`` limit."""
    if request.max_new_tokens is None:
        return sampling_params_list

    import copy

    sampling_params_list = copy.deepcopy(sampling_params_list)
    sampling_params_list[0].max_tokens = request.max_new_tokens
    return sampling_params_list


class TTSGenerationError(RuntimeError):
    """A completed TTS generation that must not be returned as valid audio."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class OutputPolicy:
    """How the orchestrator aggregates engine output for a model.

    ``accumulate_nonstreaming`` enables MOSS-style cross-step accumulation in the
    non-streaming path. Streaming cumulative/delta semantics stay engine-side,
    keyed by request id, and are unaffected by this flag.
    """

    accumulate_nonstreaming: bool = False


@dataclass
class PreparedRequest:
    """Everything the generic orchestrator needs to call ``<engine>.generate()``.

    The fields mirror what ``_prepare_speech_generation`` assembled inline:
    ``prompt`` is the engine prompt dict, ``tts_params`` the per-model parameter
    dict, ``model_type`` the discriminator used for logging, and
    ``output_policy`` controls non-streaming aggregation.
    """

    prompt: dict[str, Any]
    tts_params: dict[str, Any] = field(default_factory=dict)
    model_type: str = "generic"
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)
    #: Cross-cutting per-request state the orchestrator still owns (e.g. the
    #: Qwen3-TTS ref-audio warmup artifact key tracked after ``generate()``).
    warmup_artifact_key: str | None = None


@dataclass
class SpeechServingContext:
    """Shared state and helpers an adapter may use.

    During the incremental migration this holds a back-reference to the owning
    ``OmniOpenAIServingSpeech`` instance (``server``) so adapters can reuse the
    existing, battle-tested helper implementations (ref-audio resolution,
    uploaded-speaker handling, prompt-length estimation, speaker storage)
    without relocating them. Follow-up PRs may narrow this to explicit fields as
    more models migrate. Exactly one of ``engine_client`` / ``diffusion_engine``
    is set, matching the deployment's serving backend.
    """

    server: Any
    engine_client: Any | None = None
    diffusion_engine: Any | None = None


@dataclass(frozen=True)
class TTSCapabilities:
    precomputed_speakers: dict[str, dict[str, Any]] = field(default_factory=dict)
    supported_speakers: frozenset[str] = frozenset()
    supported_languages: frozenset[str] = DEFAULT_TTS_LANGUAGES
    codec_frame_rate: float | None = None


class TTSModelAdapter(ABC):
    """Mandatory base class for a TTS model served via ``/v1/audio/speech``.

    One concrete subclass per model, registered by stage key. The serving layer
    resolves exactly one adapter at startup and dispatches all per-model
    decisions to it. Adapters reuse shared helpers through ``ctx`` rather than
    re-implementing them.
    """

    #: Stable discriminator string (the model-type from detection); registry key.
    name: ClassVar[str]
    #: Engine ``model_stage`` key(s) this model uses. Load-bearing: the serving
    #: layer discovers the TTS stage and resolves the model type from these.
    stage_keys: ClassVar[frozenset[str]] = frozenset()
    #: Engine ``model_arch`` value(s) that identify this model type. Checked
    #: before ``stage_keys`` in :meth:`matches`. Only needed by models whose
    #: ``model_stage`` alone is ambiguous or absent.
    model_archs: ClassVar[frozenset[str]] = frozenset()
    #: Set when the model has no dedicated ``model_stage`` value and its AR
    #: entry stage must be discovered by ``model_archs`` instead (Ming dense).
    arch_identifies_entry_stage: ClassVar[bool] = False
    #: Detection order; lower runs first. Only set this to break a genuine
    #: overlap with another adapter — see ``tests/.../test_tts_detection.py``,
    #: which fails if two same-priority detectors can match the same input.
    detect_priority: ClassVar[int] = 100
    #: Serving backend: ``"ar"`` (engine_client) or ``"diffusion"``.
    backend: ClassVar[str] = "ar"
    #: Whether streaming entrypoints must retain terminal metrics for validation.
    validates_generation: ClassVar[bool] = False
    #: Whether the model consumes ``request.speed`` in its native parameters.
    native_speed_control: ClassVar[bool] = False
    #: Target sample rates validated for this adapter's output path. An empty
    #: set means that the adapter does not expose per-request resampling.
    supported_output_sample_rates: ClassVar[frozenset[int]] = frozenset()

    max_new_tokens_min = 1

    max_new_tokens_max = 4096

    def __init__(self, ctx: SpeechServingContext) -> None:
        self.ctx = ctx
        self.capabilities = TTSCapabilities()

    @classmethod
    def matches(cls, model_stage: str | None, model_arch: str | None) -> bool:
        """Whether a deployed stage with this ``model_stage``/``model_arch``
        is served by this adapter.

        Architecture wins over stage key, so a model that declares both is
        recognized even when deployed under a stage key it shares with another
        model. Override only for match rules that are not a set membership
        test (see ``covo_audio``).
        """
        if model_arch is not None and model_arch in cls.model_archs:
            return True
        return model_stage is not None and model_stage in cls.stage_keys

    @classmethod
    def stage_serves_speech(cls, model_stage: str | None, all_stage_keys: frozenset[str]) -> bool:
        """Whether a matched stage really accepts ``/v1/audio/speech`` in *this*
        deployment.

        Lets a model that is speech-capable only in some topologies say so
        (Audex: its omni thinker is text-final unless the speech decoder is
        deployed alongside). Default: always.
        """
        return True

    def normalize(self, request: "OpenAICreateSpeechRequest") -> None:
        """In-place request normalization/mutation (e.g. infer task type,
        lowercase voice). Default: no-op."""

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        """Return an error string, or ``None`` if the request is valid.

        Should be free of new side effects beyond what ``normalize`` did.
        """
        return None

    @abstractmethod
    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        """Build the engine prompt + tts_params for this request.

        ``sampling_params_list`` is passed read-only for models (e.g. MOSS) that
        fold the resolved seed into ``additional_information`` at build time.

        ``has_inline_ref_audio`` is captured by the orchestrator *before*
        ``validate()`` runs, because ``_apply_uploaded_speaker`` (invoked inside
        several adapters' ``validate``) sets ``request.ref_audio`` in place.
        Recomputing it here would misclassify uploaded voices as inline and drop
        the ``voice_name`` / ``voice_created_at`` metadata.
        """

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: "OpenAICreateSpeechRequest",
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        """Apply model-specific sampling mutations.

        The orchestrator guarantees the call order:
        stream-coercion -> extra_params -> THIS -> seed. Default: identity.
        """
        return sampling_params_list

    def validate_generation(
        self,
        tts_params: Mapping[str, object],
        *,
        stage0_finish_reason: str | None,
        output_tokens: int,
    ) -> None:
        """Reject a completed generation that is invalid for this model.

        Adapters that override this hook must also set
        :attr:`validates_generation` so streaming entrypoints retain the
        terminal metrics needed by the validation without charging that cost
        to unrelated TTS models.
        """

    async def warmup(self) -> None:
        return

    def validate_tts_embedding_dim(self, emb_dim: int) -> str | None:
        return None

    def load_capabilities(self) -> TTSCapabilities:
        self.capabilities = TTSCapabilities(
            precomputed_speakers=self._load_precomputed_speakers(),
            supported_speakers=frozenset(self._load_supported_speakers()),
            supported_languages=self._load_supported_languages(),
            codec_frame_rate=self._load_codec_frame_rate(),
        )
        return self.capabilities

    def _load_precomputed_speakers(self) -> dict[str, dict[str, Any]]:
        return {}

    def _load_supported_speakers(self) -> set[str]:
        # Preserve the legacy default path, which reads talker_config.
        return load_supported_speakers(self.ctx.engine_client)

    def _load_supported_languages(self) -> frozenset[str]:
        return DEFAULT_TTS_LANGUAGES

    def _load_codec_frame_rate(self) -> float | None:
        return load_codec_frame_rate(self.ctx.engine_client)


class ARTTSAdapter(TTSModelAdapter):
    """Adapter for models served through the AR ``engine_client`` path.

    Covers pure-AR codec models as well as AR-base-LM + diffusion-side-computation
    hybrids (e.g. VoxCPM2, Ming) whose internal diffusion is invisible to the
    serving layer.
    """

    backend: ClassVar[str] = "ar"


class DiffusionTTSAdapter(TTSModelAdapter):
    """Adapter for pure-diffusion pipelines served through the diffusion engine.

    In scope today: OmniVoice. Bridges to the diffusion pipeline parameter
    contract (#3572) when present; see ``extra_body_params``.
    """

    backend: ClassVar[str] = "diffusion"

    #: Backing diffusion pipeline class (for EXTRA_BODY_PARAMS lookup).
    pipeline_cls: ClassVar[type | None] = None

    @classmethod
    def extra_body_params(cls) -> frozenset[str]:
        """Fallback-safe access to the pipeline's declared body params.

        Returns the pipeline's ``EXTRA_BODY_PARAMS`` if the #3572 contract is
        present, else an empty frozenset (the adapter then uses its own inline
        parameter logic).
        """
        params = getattr(cls.pipeline_cls, "EXTRA_BODY_PARAMS", None)
        return frozenset(params) if params is not None else frozenset()


# Re-exported here to avoid import cycles at call sites.
__all__ = [
    "ARTTSAdapter",
    "DiffusionTTSAdapter",
    "OutputPolicy",
    "PreparedRequest",
    "SpeechServingContext",
    "TTSGenerationError",
    "TTSModelAdapter",
]
