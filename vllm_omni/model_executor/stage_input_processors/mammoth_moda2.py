"""Stage input processor for MammothModa2 (AR -> DiT)."""

from collections.abc import Mapping
from typing import Any

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def _as_dict(prompt: Any) -> dict[str, Any]:
    """Coerce an original-stage prompt to a dict.

    It may arrive as a dict, a NamedTuple/object, or a bare string depending on
    the calling flow (the shared text_to_image example vs the bespoke script).
    """
    if isinstance(prompt, dict):
        return prompt
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def _coerce_dim(value: Any, default: int) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    return iv if iv > 0 else default


def ar2dit(
    source_outputs: list[Any],
    prompts: OmniTokensPrompt | TextPrompt | list[OmniTokensPrompt | TextPrompt] | None = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Convert AR stage outputs to DiT stage inputs."""
    ar_outputs = source_outputs

    # The shared text_to_image example forwards a single prompt (not a list); normalize
    # so a lone dict isn't iterated as its keys. Mirrors glm_image.ar2diffusion.
    if not isinstance(prompts, list):
        prompts = [prompts] if prompts is not None else [{}]

    dit_inputs: list[OmniTokensPrompt] = []
    for i, ar_output in enumerate(ar_outputs):
        prompt_dict = _as_dict(prompts[i] if i < len(prompts) else {})
        addi_info = prompt_dict.get("additional_information") or {}
        mm_kwargs = prompt_dict.get("mm_processor_kwargs") or {}

        # Image size: prefer mm_processor_kwargs target_h/target_w (set by the serving
        # layer), fall back to additional_information, then a 1024 default.
        image_height = _coerce_dim(
            mm_kwargs.get("target_h"),
            _coerce_dim((addi_info.get("image_height") or [None])[0], 1024),
        )
        image_width = _coerce_dim(
            mm_kwargs.get("target_w"),
            _coerce_dim((addi_info.get("image_width") or [None])[0], 1024),
        )

        # Sampling knobs arrive on the DiT stage via extra_body -> extra_args; these are
        # defensive fallbacks (defaults mirror the former bespoke script's argparse).
        text_guidance_scale = (addi_info.get("text_guidance_scale") or [9.0])[0]
        cfg_range = addi_info.get("cfg_range") or [0.0, 1.0]
        num_inference_steps = (addi_info.get("num_inference_steps") or [50])[0]

        prompt_token_ids = ar_output.prompt_token_ids
        # exclude the last token because it has no corresponding hidden state
        completion_output = ar_output.outputs[0]
        gen_token_ids = completion_output.cumulative_token_ids[:-1]
        full_token_ids = prompt_token_ids + gen_token_ids

        mm_output = getattr(completion_output, "multimodal_output", None)
        if not isinstance(mm_output, Mapping) or "latent" not in mm_output:
            raise ValueError(
                "AR stage output missing latent multimodal output. "
                f"request_id={getattr(ar_output, 'request_id', None)}, "
                f"completion_has_mm={hasattr(completion_output, 'multimodal_output')}"
            )
        full_hidden_states = mm_output["latent"]
        hidden_total = int(full_hidden_states.shape[0])
        assert hidden_total == len(prompt_token_ids) + len(gen_token_ids), (
            f"Hidden states length mismatch: expected {len(prompt_token_ids) + len(gen_token_ids)}, got {hidden_total}"
        )

        # The text/image condition split is performed in the DiT pipeline, which sources
        # the distinguishing token ids (gen_vocab_start_index, vision placeholder ids)
        # from the model config. Pass through the raw AR hidden states + token ids and
        # the question/answer boundary so the pipeline can reconstruct the masks.
        additional_information = {
            # float32 so the tensor crosses the stage boundary (the serializer uses
            # numpy, which has no bf16); the DiT re-casts to the model dtype.
            "full_hidden_states": full_hidden_states.float().contiguous(),
            "full_token_ids": full_token_ids,
            "answer_start_index": [len(prompt_token_ids)],
            "image_height": [int(image_height)],
            "image_width": [int(image_width)],
            "text_guidance_scale": [float(text_guidance_scale)],
            "cfg_range": [float(cfg_range[0]), float(cfg_range[1])],
            "num_inference_steps": [int(num_inference_steps)],
        }

        dit_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return dit_inputs
