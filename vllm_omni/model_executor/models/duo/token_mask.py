# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-specific logits constraints and penalties."""


def forbidden_token_ids(tokenizer):
    allowed = {tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<wait>")}
    allowed.update(tokenizer.convert_tokens_to_ids(t) for t in tokenizer.all_special_tokens if t.startswith("<emo_"))
    manifest = getattr(tokenizer, "duplex_protocol_manifest", {})
    if manifest.get("training_options", {}).get("explicit_speak_token", False):
        allowed.add(tokenizer.convert_tokens_to_ids("<speak>"))
    ids = sorted(set(tokenizer.all_special_ids) - allowed)
    if any(type(i) is not int or not 0 <= i < len(tokenizer) for i in ids):
        raise ValueError("Invalid forbidden token vocabulary")
    return ids


def apply_forbidden_mask(logits, indices, vocab_size):
    if logits is None:
        return None
    if logits.shape[-1] < vocab_size or indices.device != logits.device:
        raise ValueError("Forbidden mask requires gathered vocabulary on the model device")
    logits.index_fill_(-1, indices, -float("inf"))
    # Match range(len(tokenizer)) allowlist, including any TP padded tail.
    logits[..., vocab_size:] = -float("inf")
    return logits


def normalize_action_logits(logits, last_tokens, config):
    """Commit the reference greedy action without generating discarded tails.

    Invalid/empty first actions become WAIT, rather than choosing the best
    of a restricted action vocabulary. WAIT ends after its own forward;
    SPEAK keeps the original logits until EOS or the reference body cap.
    """
    import torch

    count, limit, wait, speak, eos = config.unbind(-1)
    first = count == 0
    best = logits.argmax(-1)
    fallback = first & (best != wait) & (best != speak)
    stop = ((count == 1) & (last_tokens == wait)) | ((count >= limit) & (count > 0))
    chosen = torch.where(fallback, wait, eos)
    forced = fallback | stop
    logits.masked_fill_(forced[:, None], -float("inf"))
    rows = torch.arange(logits.shape[0], device=logits.device)
    logits[rows, chosen] = torch.where(forced, 0.0, logits[rows, chosen])


def assistant_bigram_pairs(blocks, punctuation_ids, user_ids):
    """Collect neighboring assistant tokens without crossing silent blocks."""
    pairs = set()
    previous = ()
    for tokens in blocks:
        if previous and tokens:
            pairs.add((previous[-1], tokens[0]))
        pairs.update(zip(tokens, tokens[1:]))
        previous = tokens
    return sorted(
        (prefix, token)
        for prefix, token in pairs
        if prefix not in punctuation_ids and token not in punctuation_ids and token not in user_ids
    )


def apply_assistant_bigram_penalty(logits, last_tokens, pairs, penalties):
    """Lower a candidate only when it repeats a recent assistant bigram."""
    rows, prefixes, targets = pairs.unbind(-1)
    amounts = penalties * (last_tokens[rows] == prefixes)
    logits.index_put_((rows, targets), -amounts.to(logits.dtype), accumulate=True)
