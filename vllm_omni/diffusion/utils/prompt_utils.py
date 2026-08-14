import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest


def do_prompt_upscaling(req: OmniDiffusionRequest) -> bool:
    """Check whether prompt upscaling should run for this request."""
    # NOTE: This may get more complex in the future since some models
    # Use their text encoder directly, while others have an externally
    # loaded component. Since only Ernie Image does this for now
    # though, we currently don't distinguish at load time to avoid having
    # too many obscure flags.
    value = req.sampling_params.extra_args.get("use_prompt_upscaling")
    if not isinstance(value, bool) and value is not None:
        raise TypeError(f"use_prompt_upscaling must be a bool, got {type(value).__name__}")
    return bool(value)


def validate_prompt_sequence_lengths(
    attention_mask: torch.Tensor,
    *,
    max_sequence_length: int,
    supported_max_sequence_length: int,
    prompt_name: str = "prompt",
    length_offset: int = 0,
    baseline_attention_mask: torch.Tensor | None = None,
    error_context: str,
) -> None:
    sequence_lengths = attention_mask.sum(dim=1)
    if baseline_attention_mask is not None:
        # Some callers need to validate only the user-controlled portion of a
        # templated prompt. In those cases we subtract the fully-tokenized
        # template baseline instead of only removing a fixed prefix length,
        # because the template may also contribute a suffix or image markers.
        baseline_lengths = baseline_attention_mask.sum(dim=1)
        if baseline_lengths.shape[0] == 1 and sequence_lengths.shape[0] > 1:
            baseline_lengths = baseline_lengths.expand(sequence_lengths.shape[0])
        sequence_lengths = sequence_lengths - baseline_lengths
    if length_offset:
        sequence_lengths = sequence_lengths - length_offset
    sequence_lengths = torch.clamp(sequence_lengths, min=0)
    too_long = torch.nonzero(sequence_lengths > max_sequence_length, as_tuple=False)
    if too_long.numel() == 0:
        return

    batch_idx = int(too_long[0].item())
    actual_length = int(sequence_lengths[batch_idx].item())
    prompt_ref = f"`{prompt_name}` at batch index {batch_idx}" if attention_mask.shape[0] > 1 else f"`{prompt_name}`"
    raise ValueError(
        f"{prompt_ref} is too long {error_context}: got {actual_length} tokens, but "
        f"`max_sequence_length` is {max_sequence_length}. Shorten the prompt or increase "
        f"`max_sequence_length` up to {supported_max_sequence_length}."
    )
