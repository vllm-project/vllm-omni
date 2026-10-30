# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Synchronous, request-local DRVEXO Thinker -> Talker transfer."""

import torch

from vllm_omni.inputs.data import OmniTokensPrompt


def thinker2talker_full_payload(transfer_manager, pooling_output, request):
    del transfer_manager
    hidden = pooling_output.get("hidden") if isinstance(pooling_output, dict) else None
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or hidden.shape[-1] != 4096:
        raise ValueError("DRVEXO Thinker must provide accumulated 4096-dim hidden states")
    prompt = list(getattr(request, "prompt_token_ids", []) or [])
    output = list(getattr(request, "output_token_ids", []) or [])
    # Each forward produces the state of the input token, not of the token
    # sampled from its logits. Do not include the prompt's final prediction row.
    # With EOS termination, rows after prompt correspond to all spoken tokens.
    memory = hidden[len(prompt) : len(prompt) + len(output)].detach().cpu().contiguous()
    if len(memory) == 0 or not torch.isfinite(memory).all():
        raise ValueError("DRVEXO generated no finite assistant-token memory")
    return {"hidden_states": {"output": memory}, "meta": {"finished": torch.tensor(True, dtype=torch.bool)}}


def thinker2talker_token_only(source_outputs, prompt=None, requires_multimodal_data=False):
    del prompt, requires_multimodal_data
    inputs = []
    for source in source_outputs:
        if not source.finished:
            continue
        output = source.outputs[0]
        text = getattr(output, "cumulative_text", None) or output.text
        if not isinstance(text, str) or not text.strip():
            raise ValueError("DRVEXO cannot speak an empty assistant reply")
        generated_ids = list(getattr(output, "cumulative_token_ids", None) or getattr(output, "token_ids", None) or [])
        # Native Qwen3-TTS Base, Auto language, x-vector-only streaming prefill:
        # 3 role + (3 codec prefix + 1 speaker + 2 codec tags - 1) + 1 text.
        # Remaining text is consumed by the native Talker decode preprocessor.
        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * 9,
                model_intermediate_buffer={
                    "text": [text],
                    "task_type": ["Base"],
                    "language": ["Auto"],
                    "x_vector_only_mode": [True],
                    "non_streaming_mode": [False],
                    "drvexo_generated_token_ids": generated_ids,
                },
            )
        )
    return inputs
