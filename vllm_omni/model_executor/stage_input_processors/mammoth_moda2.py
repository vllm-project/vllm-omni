"""Stage input processor scaffold for MammothModa2 (AR -> DiT)."""

from typing import Any

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def ar2dit(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Placeholder processor bridging AR outputs to DiT inputs.

    This will be filled with logic that extracts generated ids/hidden states
    from the AR stage and packages them for diffusion decoding.
    """
    raise NotImplementedError("ar2dit is not implemented for MammothModa2 scaffold.")
