# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
"""Stage input processors for Ming-flash-omni-2.0 multi-stage pipeline."""

from __future__ import annotations

from typing import Any

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def _validate_stage_inputs(stage_list, engine_input_source):
    """Validate stage inputs and return the source engine outputs."""
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build talker stage inputs from thinker stage outputs.

    Extracts the generated text from thinker output and constructs
    a talker input prompt with the text and any speaker/instruction info
    from the original request.
    """
    source_outputs = _validate_stage_inputs(stage_list, engine_input_source)

    if not isinstance(prompt, list):
        prompt = [prompt]

    talker_inputs: list[OmniTokensPrompt] = []
    for i, source_output in enumerate(source_outputs):
        output = source_output.outputs[0]

        # Get the generated text from thinker
        generated_text = output.text if hasattr(output, "text") and output.text else ""

        # Extract additional information from the original prompt
        original_prompt = prompt[i] if i < len(prompt) else None
        additional_info = {}
        if original_prompt is not None and hasattr(original_prompt, "additional_information"):
            additional_info = original_prompt.additional_information or {}

        # spk_emb can arrive serialised as a plain list from JSON requests;
        # the talker's spk_head wants a torch tensor.
        spk_emb = additional_info.get("spk_emb", None)
        if isinstance(spk_emb, list) and spk_emb and not hasattr(spk_emb[0], "device"):
            import torch

            spk_emb = torch.tensor(spk_emb, dtype=torch.float32).unsqueeze(0)

        # Omni speech path mirrors upstream `omni_audio_generation`:
        # - `prompt` is hardcoded, `instruction` is forced to None,
        #   cfg/sigma/temperature inherit the `tts_job` defaults (the
        #   upstream API does NOT expose these knobs).
        # - Voice cloning is preset-only via `voice_name` (default
        #   'DB30'); `get_prompt_emb` is called with
        #   `use_spk_emb=True, use_zero_spk_emb=False`, so when no
        #   preset resolves upstream simply passes `spk_emb=None`
        #   through to `tts_job` rather than substituting a zero
        #   vector.
        # The bridge only plumbs the request-specific fields; the
        # talker `forward()` enforces the per-task defaults from
        # `ming_task="omni"` so any stray caller overrides are ignored.
        # Voice presets are resolved by voice_name in the talker's
        # forward() from its registered_prompts cache.
        talker_info = {
            "ming_task": "omni",
            "text": generated_text,
            "spk_emb": spk_emb,
            "voice_name": additional_info.get("voice_name", "DB30"),
            "prompt_text": additional_info.get("prompt_text", None),
            "prompt_wav_lat": additional_info.get("prompt_wav_lat", None),
            "prompt_wav_emb": additional_info.get("prompt_wav_emb", None),
            "max_text_length": additional_info.get("max_text_length", 50),
        }

        # Use dummy token IDs (talker builds its own embeddings from text)
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=talker_info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs
