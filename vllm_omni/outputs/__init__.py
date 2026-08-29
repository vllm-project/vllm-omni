from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.v1.outputs import ModelRunnerOutput

from vllm_omni.inputs.data import OmniPromptType


@dataclass
class OmniConnectorOutput:
    """Communication results from Model Runner to Scheduler.

    Carries transfer readiness signals so the Scheduler can make scheduling
    decisions without ever calling connector.put()/get() directly.

    Attributes:
        chunk_ready_req_ids: Request IDs with newly arrived chunks this cycle.
        chunk_finished_req_ids: Request IDs whose final chunk has arrived.
        request_metadata: Lightweight scheduling metadata keyed by request ID
            (e.g. next_stage_prompt_len, code_predictor_codes, left_context_size).
            Full payloads are owned by the Model Runner's local cache.
        kv_sent_req_ids: Request IDs whose KV cache was successfully sent.
        stage_recv_req_ids: Request IDs that received batch stage inputs.
        has_pending_kv_work: True if the mixin has pending, active, or
            completed KV transfers that the scheduler should account for.
    """

    chunk_ready_req_ids: set[str] = field(default_factory=set)
    chunk_finished_req_ids: set[str] = field(default_factory=set)
    request_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    kv_sent_req_ids: list[str] = field(default_factory=list)
    stage_recv_req_ids: set[str] = field(default_factory=set)
    has_pending_kv_work: bool = False


@dataclass
class OmniModelRunnerOutput(ModelRunnerOutput):
    """Model runner output for omni models.

    Extends the base ModelRunnerOutput with support for multimodal outputs
    that may be produced by non-autoregressive stages.

    Attributes:
        multimodal_outputs: Optional per-request list of client-facing multimodal
            output dicts, indexed by req_index.
        inter_stage_outputs: Optional per-request list of inter-stage payload dicts
            for connector transport (``save_async`` / full_payload).  Not forwarded
            to the orchestrator output processor.
    """

    multimodal_outputs: list[dict[str, object]] | None = None
    inter_stage_outputs: list[dict[str, object]] | None = None
    # IDs of requests whose KV cache has been extracted from GPU/NPU to CPU.
    # The Scheduler can safely free the block tables for these requests.
    kv_extracted_req_ids: list[str] | None = None
    omni_connector_output: OmniConnectorOutput | None = None

    @classmethod
    def with_kv_conn_output_only(cls, kv_connector_output: Any) -> "OmniModelRunnerOutput":
        return cls(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            kv_connector_output=kv_connector_output,
        )


# RequestOutput attributes that hold generation content.  When wrapping a
# stage output, these are copied onto the OmniRequestOutput itself so that
# the object IS the RequestOutput instead of holding one.
_REQUEST_OUTPUT_CONTENT_ATTRS = (
    "request_id",
    "prompt",
    "prompt_token_ids",
    "prompt_logprobs",
    "outputs",
    "finished",
    "lora_request",
    "encoder_prompt",
    "encoder_prompt_token_ids",
    "num_cached_tokens",
    "num_cache_creation_tokens",
    "kv_transfer_params",
    "ec_transfer_params",
)

# Omni-specific content copied when the wrapped stage output is itself an
# OmniRequestOutput (e.g. a diffusion stage inside a multi-stage pipeline).
_OMNI_CONTENT_ATTRS = (
    "images",
    "latents",
    "trajectory_latents",
    "trajectory_timesteps",
    "trajectory_log_probs",
    "trajectory_decoded",
    "_multimodal_output",
    "_custom_output",
)


@dataclass
class OmniRequestOutput(RequestOutput):
    """Unified request output for both pipeline stages and diffusion models.

    Extends vLLM's ``RequestOutput`` so that omni outputs can flow directly
    through vLLM serving codepaths (which expect ``prompt_token_ids``,
    ``outputs``, etc. as real attributes).  The inherited fields store the
    LLM generation content; omni-specific fields store pipeline/diffusion
    extras.

    Note: ``RequestOutput`` is a plain class (not a dataclass), so all of its
    attributes are redeclared below as dataclass fields with defaults — the
    dataclass-generated ``__init__`` replaces ``RequestOutput.__init__`` and
    must set them itself.

    This class handles outputs from:
    1. Multi-stage LLM pipelines (with stage_id, final_output_type, and the
       inherited RequestOutput fields carrying the stage's generation content)
    2. Diffusion models (with images, prompt, metrics)

    Attributes:
        request_id: Unique identifier for this request
        finished: Whether generation is complete
        stage_id: Identifier of the stage that produced this output (pipeline mode)
        replica_id: Identifier of the stage replica that produced this output
        final_output_type: Type of output ("text", "image", "audio", "latents")
        images: List of generated PIL images (diffusion mode)
        prompt: The prompt used for generation
        latents: Optional tensor of latent representations (diffusion mode)
        metrics: Generation metrics. A plain dict for omni outputs; may carry
            vLLM's request stats object when copied from a raw RequestOutput.
    """

    # --- Inherited RequestOutput attributes (redeclared as fields) ---
    request_id: str = ""
    prompt: OmniPromptType | None = None
    prompt_token_ids: list[int] | None = None
    prompt_logprobs: Any = None
    outputs: list[CompletionOutput] = field(default_factory=list)
    finished: bool = True
    metrics: Any = field(default_factory=dict)
    lora_request: Any = None
    encoder_prompt: str | None = None
    encoder_prompt_token_ids: list[int] | None = None
    num_cached_tokens: int | None = None
    num_cache_creation_tokens: int | None = None
    kv_transfer_params: dict[str, Any] | None = None
    ec_transfer_params: dict[str, Any] | None = None

    # --- Pipeline stage fields ---
    stage_id: int | None = None
    replica_id: int | None = None
    final_output_type: str = "text"

    # --- Diffusion model fields ---
    images: list[Image.Image] = field(default_factory=list)
    latents: torch.Tensor | None = None
    trajectory_latents: torch.Tensor | None = None
    trajectory_timesteps: torch.Tensor | None = None
    trajectory_log_probs: torch.Tensor | None = None
    trajectory_decoded: list | None = None
    _multimodal_output: dict[str, Any] = field(default_factory=dict)
    _custom_output: dict[str, Any] = field(default_factory=dict)

    # profiling data
    stage_durations: dict[str, float] = field(default_factory=dict)

    # memory usage info
    peak_memory_mb: float = 0.0

    # error handling
    error: str | None = None
    error_status_code: int | None = None
    error_type: str | None = None

    def _copy_content_from(self, source: RequestOutput) -> None:
        """Copy generation content from a stage output into this object.

        *source* should be a vLLM ``RequestOutput`` or a subclass such as
        ``OmniRequestOutput``.  Control fields (``stage_id``,
        ``final_output_type``, ``stage_durations``, ``peak_memory_mb``,
        ``error``, ...) are intentionally not copied.
        """
        # RequestOutput attributes — guaranteed on any RequestOutput.
        # Only copy attributes that actually exist on the source, so that
        # duck-typed mocks can omit optional fields without overwriting the
        # dataclass defaults (e.g. ``outputs=[]``, ``finished=True``).
        for name in _REQUEST_OUTPUT_CONTENT_ATTRS:
            if hasattr(source, name):
                setattr(self, name, getattr(source, name))

        # Omni-specific fields — only present when the source is itself an
        # OmniRequestOutput (e.g. a diffusion stage inside a pipeline).
        if isinstance(source, OmniRequestOutput):
            for name in _OMNI_CONTENT_ATTRS:
                if hasattr(source, name):
                    setattr(self, name, getattr(source, name))

        # Propagate multimodal_output from duck-typed sources (e.g. test mocks
        # that set multimodal_output directly on a non-OmniRequestOutput).
        if not isinstance(source, OmniRequestOutput):
            src_mm = getattr(source, "multimodal_output", None)
            if src_mm:
                self._multimodal_output = src_mm

    @classmethod
    def from_stage_output(cls, source: RequestOutput, **kwargs: Any) -> "OmniRequestOutput":
        """Create an OmniRequestOutput from a stage's raw output.

        Copies generation content (``outputs``, ``prompt``, ``prompt_token_ids``,
        ``finished``, ``images``, ``latents``, etc.) from *source* onto the
        returned object.  *source* may be a vLLM ``RequestOutput``, another
        ``OmniRequestOutput`` (which inherits from ``RequestOutput``).

        This is the **preferred** way to construct an ``OmniRequestOutput``
        that wraps a stage result.

        Args:
            source: The stage output whose content is copied onto the new object.
            **kwargs: Passed through to the dataclass constructor (``request_id``,
                ``stage_id``, ``final_output_type``, ``metrics``,
                ``stage_durations``, ``peak_memory_mb``, ``finished``, etc.).
                Typed as ``Any`` because the exact set of valid keys is the
                dataclass field list, which is validated by ``cls(**kwargs)``
                at call time.

        Returns:
            A new ``OmniRequestOutput`` with the stage's content flattened onto it.
        """
        obj = cls(**kwargs)
        obj._copy_content_from(source)
        return obj

    @classmethod
    def from_error(
        cls,
        request_id: str,
        error_message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
    ) -> "OmniRequestOutput":
        """Create a terminal error output.

        Args:
            request_id: Request identifier
            error_message: Human-readable error description

        Returns:
            OmniRequestOutput with ``finished=True`` and the ``error`` field set.
        """
        return cls(
            request_id=request_id,
            finished=True,
            error=error_message,
            error_status_code=status_code,
            error_type=error_type,
        )

    @classmethod
    def from_diffusion(
        cls,
        request_id: str,
        images: list[Image.Image],
        prompt: OmniPromptType | None = None,
        metrics: dict[str, Any] | None = None,
        latents: torch.Tensor | None = None,
        trajectory_latents: torch.Tensor | None = None,
        trajectory_timesteps: torch.Tensor | None = None,
        trajectory_log_probs: torch.Tensor | None = None,
        trajectory_decoded: list | None = None,
        multimodal_output: dict[str, Any] | None = None,
        custom_output: dict[str, Any] | None = None,
        final_output_type: str = "image",
        stage_durations: dict[str, float] | None = None,
        peak_memory_mb: float = 0.0,
        finished: bool = True,
    ) -> "OmniRequestOutput":
        """Create output from diffusion model.

        Args:
            request_id: Request identifier
            images: Generated images
            prompt: The prompt used
            metrics: Generation metrics
            latents: Optional latent tensors
            trajectory_latents: Optional stacked trajectory latent tensors
            trajectory_timesteps: Optional stacked trajectory timestep tensors
            trajectory_log_probs: Optional stacked trajectory log-probability tensors
            trajectory_decoded: Optional list of decoded trajectory images
            multimodal_output: Optional multimodal output dict
            custom_output: Optional custom output dict (e.g. prompt embeds)
            stage_durations: Optional stage durations (execution time of each stage) dict
            peak_memory_mb: Peak memory usage in MB

        Returns:
            OmniRequestOutput configured for diffusion mode
        """
        return cls(
            request_id=request_id,
            final_output_type=final_output_type,
            images=images,
            prompt=prompt,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            trajectory_log_probs=trajectory_log_probs,
            trajectory_decoded=trajectory_decoded,
            metrics=metrics or {},
            _multimodal_output=multimodal_output or {},
            _custom_output=custom_output or {},
            stage_durations=stage_durations or {},
            peak_memory_mb=peak_memory_mb,
            finished=finished,
        )

    @property
    def multimodal_output(self) -> Any:
        """Return the multimodal output payload.

        Checks completion outputs first (where multimodal_output is attached
        by AR stages), then the local _multimodal_output field.

        Returns either a MultimodalPayload (Phase 3+) or a plain dict (legacy).
        """
        for output in self.outputs:
            if mm := getattr(output, "multimodal_output", None):
                return mm
        return self._multimodal_output

    @property
    def custom_output(self) -> dict[str, Any]:
        """Return custom output data from diffusion pipelines."""
        return self._custom_output

    @custom_output.setter
    def custom_output(self, value: dict[str, Any]) -> None:
        self._custom_output = value

    @property
    def num_images(self) -> int:
        """Return the number of generated images."""
        return len(self.images)

    @property
    def is_diffusion_output(self) -> bool:
        """Check if this is a diffusion model output."""
        return len(self.images) > 0 or self.final_output_type == "image"

    @property
    def is_pipeline_output(self) -> bool:
        """Check if this is a pipeline stage output."""
        return self.stage_id is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "request_id": self.request_id,
            "finished": self.finished,
            "final_output_type": self.final_output_type,
        }

        if self.is_diffusion_output:
            result.update(
                {
                    "num_images": self.num_images,
                    "prompt": self.prompt,
                    "metrics": self.metrics,
                }
            )

        if self.is_pipeline_output:
            result.update(
                {
                    "stage_id": self.stage_id,
                }
            )

        return result

    def __repr__(self) -> str:
        """Custom repr to properly show image count instead of image objects."""
        # For images, show count instead of full list
        images_repr = f"[{len(self.images)} PIL Images]" if self.images else "[]"

        # Build repr string
        parts = [
            f"request_id={self.request_id!r}",
            f"finished={self.finished}",
            f"stage_id={self.stage_id}",
            f"final_output_type={self.final_output_type!r}",
            f"prompt_token_ids={self.prompt_token_ids}",
            f"outputs={self.outputs}",
            f"images={images_repr}",
            f"prompt={self.prompt!r}",
            f"latents={self.latents}",
            f"metrics={self.metrics}",
            f"multimodal_output={self._multimodal_output}",
            f"custom_output={self._custom_output}",
            f"stage_durations={self.stage_durations}",
            f"peak_memory_mb={self.peak_memory_mb}",
        ]

        return f"OmniRequestOutput({', '.join(parts)})"
