# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Complete native Talker input for application-authorized fixed replies."""


def fixed_reply_prompt(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 512:
        raise ValueError("Fixed speech requires 1–512 characters")
    # Same native Chinese x-vector-only initial prompt layout as the generated
    # lexical path; no Thinker-generated tokens or hidden states are fabricated.
    return {
        "prompt_token_ids": [0] * 10,
        "model_intermediate_buffer": {
            "text": [text],
            "task_type": ["Base"],
            "language": ["Chinese"],
            "x_vector_only_mode": [True],
            "non_streaming_mode": [False],
            "drvexo_fixed_reply": True,
        },
    }
