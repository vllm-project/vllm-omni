# Adapted from https://github.com/lucidrains/muse-maskgit-pytorch

import math
from functools import partial

import torch
import torch.nn.functional as F


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t, generator=None):
    noise = torch.zeros_like(t).uniform_(0, 1, generator=generator)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, generator=None):
    return (
        (t / max(temperature, 1e-10)) + gumbel_noise(t, generator=generator)
    ).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    confidence = log(probs) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def linear_schedule(t):
    mask_ratio = 1 - t
    mask_ratio = mask_ratio.clamp(min=1e-6, max=1.0)
    return mask_ratio


def pow(t, method):
    exponent = float(method.replace("pow", ""))
    mask_ratio = 1.0 - t**exponent
    mask_ratio = mask_ratio.clamp(min=1e-6, max=1.0)
    return mask_ratio


def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-6):
    for item in [t, start, end, tau]:
        item = torch.tensor(item) if not torch.is_tensor(item) else item

    # A gamma function based on sigmoid function.
    v_start = torch.sigmoid(torch.tensor(start / tau))
    v_end = torch.sigmoid(torch.tensor(end / tau))
    output = torch.sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clip(output, clip_min, 1.0)


def get_mask_schedule(method, **schedule_kwargs):
    if method == "cosine":
        return cosine_schedule
    elif method == "linear":
        return linear_schedule
    elif "pow" in method:
        return partial(pow, method=method)
    elif method == "sigmoid":
        return partial(sigmoid_schedule, **schedule_kwargs)
    else:
        raise ValueError("Unknown schedule method: {}".format(method))


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits
