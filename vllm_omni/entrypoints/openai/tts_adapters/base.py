# SPDX-License-Identifier: Apache-2.0
"""Base contract for per-model TTS serving adapters.

This package factors the per-model ``if self._tts_model_type == ...`` dispatch
in ``serving_speech.py`` into one adapter class per model. Each adapter owns its
model's request normalization, validation, prompt/param building, sampling
overrides, and output policy, so adding a model means writing one adapter file
instead of editing the shared serving module in ~10 scattered places.

See the RFC for the full design (issue #4327). This is the foundation landed in
the first migration PR; Qwen3-TTS is the first model routed through it while the
remaining models stay on the legacy path until individually migrated.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


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


class TTSModelAdapter(ABC):
    """Mandatory base class for a TTS model served via ``/v1/audio/speech``.

    One concrete subclass per model, registered by stage key. The serving layer
    resolves exactly one adapter at startup and dispatches all per-model
    decisions to it. Adapters reuse shared helpers through ``ctx`` rather than
    re-implementing them.
    """

    #: Stable discriminator string (the model-type from detection); registry key.
    name: ClassVar[str]
    #: Engine ``model_stage`` key(s) this model uses, for documentation only.
    stage_keys: ClassVar[frozenset[str]] = frozenset()
    #: Serving backend: ``"ar"`` (engine_client) or ``"diffusion"``.
    backend: ClassVar[str] = "ar"

    max_new_tokens_min = 1

    max_new_tokens_max = 4096

    def __init__(self, ctx: SpeechServingContext) -> None:
        self.ctx = ctx

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
    ) -> list:
        """Apply model-specific sampling mutations.

        The orchestrator guarantees the call order:
        stream-coercion -> extra_params -> THIS -> seed. Default: identity.
        """
        return sampling_params_list


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
    "TTSModelAdapter",
]
