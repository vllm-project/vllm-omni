import copy
import pprint
from dataclasses import asdict, dataclass, field
from typing import Any, TypeAlias

from vllm import PromptType, SamplingParams

from vllm_omni.lora.request import LoRARequest

try:
    from typing import NotRequired
except ImportError:
    # Python < 3.11: use typing_extensions
    from typing_extensions import NotRequired

import torch
from vllm.inputs.data import EmbedsPrompt, TextPrompt, TokenInputs, TokensPrompt


class OmniTextPrompt(TextPrompt):
    """Text prompt with optional embeddings and additional information.

    Extends TextPrompt to support prompt embeddings and additional
    information payloads for direct transfer between pipeline stages.

    Attributes:
        prompt_embeds: Optional tensor containing prompt embeddings
        additional_information: Optional dictionary containing additional
            information (tensors or lists) to pass along with the prompt
    """

    negative_prompt: NotRequired[str]
    prompt_embeds: NotRequired[torch.Tensor]
    negative_prompt_embeds: NotRequired[torch.Tensor]
    additional_information: NotRequired[dict[str, Any]]


class OmniTokensPrompt(TokensPrompt):
    """Tokens prompt with optional embeddings and additional information.

    Extends TokensPrompt to support prompt embeddings and additional
    information payloads for direct transfer between pipeline stages.

    Attributes:
        prompt_embeds: Optional tensor containing prompt embeddings
        additional_information: Optional dictionary containing additional
            information (tensors or lists) to pass along with the prompt
    """

    negative_prompt: NotRequired[str]
    prompt_embeds: NotRequired[torch.Tensor]
    negative_prompt_embeds: NotRequired[list[torch.Tensor] | None]
    """The embeddings of the prompt."""

    # New: optional additional information dictionary
    # Values may be torch.Tensor or list
    additional_information: NotRequired[dict[str, Any]]


class OmniTokenInputs(TokenInputs):
    """Token inputs with optional embeddings and additional information.

    Extends TokenInputs to support prompt embeddings and additional
    information payloads for direct transfer between pipeline stages.

    Attributes:
        prompt_embeds: Optional tensor containing prompt embeddings
            aligned with token IDs
        additional_information: Optional dictionary containing additional
            information (tensors or lists) to pass along with the inputs
    """

    # New: optional prompt embeddings aligned with token ids
    negative_prompt: NotRequired[str]
    prompt_embeds: NotRequired[torch.Tensor]
    negative_prompt_embeds: NotRequired[list[torch.Tensor] | None]

    # New: optional additional information dictionary
    # Values may be torch.Tensor or list
    additional_information: NotRequired[dict[str, Any]]


class OmniEmbedsPrompt(EmbedsPrompt):
    """Embeddings prompt with optional additional information.

    Extends EmbedsPrompt to support additional information payloads
    for direct transfer between pipeline stages.

    Attributes:
        prompt_embeds: Optional tensor containing prompt embeddings
        additional_information: Optional dictionary containing additional
            information (tensors or lists) to pass along with the prompt
    """

    # New: optional prompt embeddings aligned with token ids
    prompt_embeds: NotRequired[torch.Tensor]
    negative_prompt_embeds: NotRequired[list[torch.Tensor] | None]

    # New: optional additional information dictionary
    # Values may be torch.Tensor or list
    additional_information: NotRequired[dict[str, Any]]


# Must ensure that all additional prompt types are inherited from vLLM prompt types
# Because TypedDict doesn't support isinstance and are dict. Cannot distinguish them in runtime.
# Inheritance ensure that there are only additional fields but not removing fields--safe to route to LLM.generate()
OmniPromptType: TypeAlias = PromptType | OmniTextPrompt | OmniTokensPrompt | OmniEmbedsPrompt


def token_inputs_omni(
    prompt_token_ids: list[int],
    prompt: str | None = None,
    cache_salt: str | None = None,
    prompt_embeds: torch.Tensor | None = None,
    additional_information: dict[str, Any] | None = None,
) -> OmniTokenInputs:
    """Construct token inputs with optional embeddings and metadata.

    Creates an OmniTokenInputs object with token IDs and optional
    embeddings and additional information for pipeline stage transfer.

    Args:
        prompt_token_ids: List of token IDs for the prompt
        prompt: Optional prompt string
        cache_salt: Optional cache salt for prefix caching
        prompt_embeds: Optional tensor containing prompt embeddings
        additional_information: Optional dictionary containing additional
            information (tensors or lists)

    Returns:
        OmniTokenInputs instance with the provided data
    """
    inputs = OmniTokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt
    if prompt_embeds is not None:
        inputs["prompt_embeds"] = prompt_embeds
    if additional_information is not None:
        inputs["additional_information"] = additional_information

    return inputs


@dataclass
class OmniDiffusionSamplingParams:
    """
    The collection of sampling parameters passed to diffusion pipelines.

    This dataclass contains all information needed during the diffusion pipeline
    execution, allowing methods to update specific components without needing
    to manage numerous individual parameters.
    """

    # Additional text-related parameters
    max_sequence_length: int | None = None
    prompt_template: dict[str, Any] | None = None
    do_classifier_free_guidance: bool = False

    # Batch info
    num_outputs_per_prompt: int = 1
    seed: int | None = None
    generator: torch.Generator | list[torch.Generator] | None = None

    # layered info
    layers: int = 4

    # cfg info
    cfg_normalize: bool = False

    # caption language
    use_en_prompt: bool = False

    # different bucket in (640, 1024) to determine the condition and output resolution
    resolution: int = 640

    # Tracking if embeddings are already processed
    is_prompt_processed: bool = False

    # Latent tensors
    latents: torch.Tensor | None = None
    raw_latent_shape: torch.Tensor | None = None
    noise_pred: torch.Tensor | None = None
    image_latent: torch.Tensor | None = None

    # Latent dimensions
    height_latents: list[int] | int | None = None
    width_latents: list[int] | int | None = None
    num_frames: int = 1  # Default for image models
    num_frames_round_down: bool = False  # Whether to round down num_frames if it's not divisible by num_gpus

    # Original dimensions (before VAE scaling)
    height: int | None = None
    width: int | None = None
    fps: int | None = None
    height_not_provided: bool = False
    width_not_provided: bool = False

    # Timesteps
    timesteps: torch.Tensor | None = None
    timestep: torch.Tensor | float | int | None = None
    step_index: int | None = None
    boundary_ratio: float | None = None

    # Scheduler parameters
    num_inference_steps: int = 50
    guidance_scale: float = 0.0
    guidance_scale_provided: bool = False
    guidance_scale_2: float | None = None
    guidance_rescale: float = 0.0
    eta: float = 0.0
    sigmas: list[float] | None = None

    true_cfg_scale: float | None = None  # qwen-image specific now

    n_tokens: int | None = None
    extra_step_kwargs: dict[str, Any] = field(default_factory=dict)

    # [Omni] KV Cache Transfer, for bagel model now
    past_key_values: Any | None = None  # Injected KV Cache
    kv_metadata: dict[str, Any] | None = None  # Metadata for KV Cache (e.g., kv_lens, ropes)
    need_kv_receive: bool = True  # Flag to indicate if this request expects KV transfer

    # Component modules
    modules: dict[str, Any] = field(default_factory=dict)

    return_trajectory_latents: bool = False
    return_trajectory_decoded: bool = False
    trajectory_timesteps: list[torch.Tensor] | None = None
    trajectory_latents: torch.Tensor | None = None

    # Extra parameters that might be needed by specific pipeline implementations
    extra_args: dict[str, Any] = field(default_factory=dict)

    # Misc
    save_output: bool = True
    return_frames: bool = False

    # LoRA
    lora_request: LoRARequest | None = None
    lora_scale: float = 1.0

    # STA parameters
    STA_param: list | None = None
    is_cfg_negative: bool = False
    mask_search_final_result_pos: list[list] | None = None
    mask_search_final_result_neg: list[list] | None = None

    # VSA parameters
    VSA_sparsity: float = 0.0
    # perf_logger: PerformanceLogger | None = None

    # stage logging
    # logging_info: PipelineLoggingInfo = field(default_factory=PipelineLoggingInfo)

    # profile
    profile: bool = False
    num_profiled_timesteps: int = 8

    # debugging
    debug: bool = False

    # results
    output: torch.Tensor | None = None

    @property
    def batch_size(self):
        # This class is changed to only represent a single prompt request
        # Only adjust batch size for number of videos per prompt
        return self.num_outputs_per_prompt

    def __str__(self):
        return pprint.pformat(asdict(self), indent=2, width=120)

    def clone(self) -> "OmniDiffusionSamplingParams":
        return copy.deepcopy(self)


OmniSamplingParams: TypeAlias = SamplingParams | OmniDiffusionSamplingParams
