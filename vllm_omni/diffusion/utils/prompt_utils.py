import torch

#: Keys a diffusion prompt may use to carry pre-tokenized prompt ids.
#: ``OmniCustomPrompt`` declares ``prompt_ids``; ``prompt_token_ids`` is accepted
#: as an alias because the executor's emptiness check and the existing
#: HunyuanImage3 writer both use that spelling.
PROMPT_ID_KEYS = ("prompt_ids", "prompt_token_ids")
NEGATIVE_PROMPT_ID_KEYS = ("negative_prompt_ids", "negative_prompt_token_ids")


def _single_prompt_ids(ids: object, *, key: str) -> list[int]:
    """Unwrap the single prompt a diffusion request carries and cast it to ``int``."""
    if not isinstance(ids, (list, tuple)) or not ids:
        raise ValueError(f"`{key}` must be a non-empty list or tuple of token ids, got {ids!r}")
    first = ids[0]
    if isinstance(first, (list, tuple)):
        if len(ids) != 1:
            raise ValueError(f"`{key}` holds {len(ids)} prompts, but a diffusion request carries a single prompt")
        ids = first
    return [int(token_id) for token_id in ids]


def _pre_tokenized_ids(prompt: object, *, keys: tuple[str, ...]) -> list[int] | None:
    if not isinstance(prompt, dict):
        return None
    for key in keys:
        ids = prompt.get(key)
        if ids:
            return _single_prompt_ids(ids, key=key)
    return None


def pre_tokenized_prompt_ids(prompt: object) -> list[int] | None:
    """Return the pre-tokenized prompt ids of ``prompt``, if it carries any.

    ``OmniCustomPrompt`` exists so that a caller who has already tokenized the
    prompt can stop the pipeline from tokenizing it again. Pipelines that
    support that use these ids verbatim, so the caller owns the exact token
    sequence, wrappers included.
    """
    return _pre_tokenized_ids(prompt, keys=PROMPT_ID_KEYS)


def pre_tokenized_negative_prompt_ids(prompt: object) -> list[int] | None:
    """Return the pre-tokenized negative prompt ids of ``prompt``, if it carries any."""
    return _pre_tokenized_ids(prompt, keys=NEGATIVE_PROMPT_ID_KEYS)


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
