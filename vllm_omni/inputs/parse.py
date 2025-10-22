from vllm.inputs.parse import ParsedSingletonPrompt, ParsedStrPrompt, ParsedEmbedsPrompt, ParsedTokensPrompt, ParsedTextPrompt
from vllm.inputs.data import SingletonPrompt


def parse_singleton_prompt_omni(prompt: SingletonPrompt) -> ParsedSingletonPrompt:
    if isinstance(prompt, str):
        return ParsedStrPrompt(type="str", content=prompt)
    elif isinstance(prompt, dict):
        # Type ignores are because mypy does not correctly infer the TypedDicts
        # Pyright does succeed.
        # 优先 tokens：当 tokens 与 embeds 同在时，保留两者并走 tokens 路径
        if "prompt_token_ids" in prompt:
            return ParsedTokensPrompt(
                type="tokens", content=prompt)  # type: ignore[typeddict-item]
        elif "prompt_embeds" in prompt:
            return ParsedEmbedsPrompt(
                type="embeds", content=prompt)  # type: ignore[typeddict-item]
        elif "prompt" in prompt:
            return ParsedTextPrompt(type="text", content=prompt)
    raise TypeError(
        "inputs must be a string, TextPrompt, TokensPrompt, or EmbedsPrompt")