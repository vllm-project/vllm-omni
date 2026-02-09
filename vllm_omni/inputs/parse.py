from vllm.inputs.parse import (
    ParsedEmbedsPrompt,
    ParsedSingletonPrompt,
    ParsedStrPrompt,
    ParsedTextPrompt,
    ParsedTokensPrompt,
    SingletonPrompt,
)


def parse_singleton_prompt_omni(prompt: SingletonPrompt) -> ParsedSingletonPrompt:
    """Parse a singleton prompt into a typed parsed prompt.

    Handles omni-specific prompt types including tokens prompts with
    embeddings and additional information. Supports string, text,
    tokens, and embeddings prompts.

    Args:
        prompt: Singleton prompt to parse. Can be a string, TextPrompt,
            TokensPrompt (with optional prompt_embeds and additional_information),
            or EmbedsPrompt.

    Returns:
        ParsedSingletonPrompt containing the parsed prompt with type information

    Raises:
        TypeError: If the prompt type is not supported
    """
    if isinstance(prompt, str):
        return ParsedStrPrompt(type="str", content=prompt)
    elif isinstance(prompt, dict):
        # Type ignores are because mypy does not correctly infer the TypedDicts
        # Pyright does succeed.
        # Priority tokens: When both tokens and embeds exist, keep both and
        # follow the tokens path
        if "prompt_token_ids" in prompt:
            return ParsedTokensPrompt(type="tokens", content=prompt)  # type: ignore[typeddict-item]
        elif "prompt_embeds" in prompt:
            return ParsedEmbedsPrompt(type="embeds", content=prompt)  # type: ignore[typeddict-item]
        elif "prompt" in prompt:
            return ParsedTextPrompt(type="text", content=prompt)
    raise TypeError("inputs must be a string, TextPrompt, TokensPrompt, or EmbedsPrompt")
