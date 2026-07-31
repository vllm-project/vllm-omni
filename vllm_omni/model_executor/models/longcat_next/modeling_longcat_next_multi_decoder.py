"""LongCat-Next combined decoder stage: audio codes -> waveform, OR visual
codes -> RGB image, from one process.

Mirrors the reference's own ``PostProcessor.decode_multi`` (postprocessor.py),
which loads both decoders once and dispatches on ``gen_image``/``gen_audio``
flags computed from the prompt's trigger token -- a single conditional
dispatch in one process, not a chain of separate services.

This exists because vllm-omni's stage orchestrator only supports a strict
``src_stage_id + 1`` hop: ``_forward_to_next_stage`` always forwards the
just-finished stage's own output to the next stage, regardless of each
stage's declared ``input_sources``. In a 3-stage ``thinker -> image_decoder
-> audio_decoder`` chain, the audio decoder therefore receives the image
decoder's output (never the thinker's), so it can never see real audio
codes -- audio silently breaks unconditionally, not just when the wrong
modality was generated. Collapsing both decoders into one stage sidesteps
this at the root: ``thinker(0) -> multi_decoder(1)`` is a 2-stage chain, so
stage 1 unambiguously receives stage 0's real output every time, regardless
of which modality talker_mtp actually produced.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .modeling_longcat_next_audio_decoder import LongcatNextAudioDecoder
from .modeling_longcat_next_image_decoder import LongcatNextImageDecoder

logger = init_logger(__name__)


def _retag_model_outputs(output: OmniOutput, key: str) -> OmniOutput:
    """Rename the sub-decoder's generic ``"model_outputs"`` key to the
    real modality (``"image"``/``"audio"``) before it leaves this stage.

    MultimodalPayload.from_raw (outputs/mm_outputs.py) remaps a producer's
    "model_outputs" key to the STAGE's static engine_output_type -- correct
    for a single-modality stage, wrong here, since this stage's
    engine_output_type is one fixed string ("audio") but a given response
    may actually be an image. There is a per-output override path
    (EngineCoreOutput.output_type, checked in output_processor.py:501) but
    nothing in the codebase ever sets it -- it's dead for every model today.
    Renaming the key here instead is the smaller, fully local fix: from_raw
    only remaps keys literally equal to "model_outputs" (or "hidden"), so a
    correctly-named key survives untouched regardless of the stage's static
    label. Confirmed against the *other* half of this exact bug, fixed the
    same way at the call site: the proven 2-stage audio-only pipeline had to
    read the waveform back under key "audio", not "model_outputs" -- this
    mirrors that, just decided per-response instead of being statically
    true for the whole stage.
    """
    if not isinstance(output.multimodal_outputs, dict) or "model_outputs" not in output.multimodal_outputs:
        return output
    retagged = dict(output.multimodal_outputs)
    retagged[key] = retagged.pop("model_outputs")
    return output._replace(multimodal_outputs=retagged)


class LongcatNextMultiDecoder(nn.Module):
    """Composes the image and audio decoders as submodules and dispatches
    per request on whichever of ``visual_token_ids``/``audio_token_ids``
    talker_mtp actually populated. The two are mutually exclusive per the
    reference's own state machine (``GEN_IMAGE_STAGE``/``GEN_AUDIO_STAGE``
    in output_processor.py), so at most one branch ever does real work for
    a given request -- this is a straight dispatch, not a merge.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.prefix = prefix

        # Each sub-decoder lazily loads its own weight subtree on its own
        # first use (_ensure_weights, unchanged from the standalone
        # classes), so constructing both here -- before either has decoded
        # anything -- is as cheap as constructing either alone. Real weight
        # I/O only happens once, on whichever branch first actually runs.
        self.image_decoder = LongcatNextImageDecoder(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "image_decoder")
        )
        self.audio_decoder = LongcatNextAudioDecoder(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "audio_decoder")
        )

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.image_decoder.embed_input_ids(input_ids, **kwargs)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> None:
        return None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        model_intermediate_buffer = (
            kwargs.get("model_intermediate_buffer")
            or kwargs.get("runtime_additional_information")
            or {}
        )
        if isinstance(model_intermediate_buffer, dict):
            additional_info = next(
                (info for info in model_intermediate_buffer.values() if isinstance(info, dict)),
                {},
            )
        else:
            additional_info = next(
                (info for info in model_intermediate_buffer if isinstance(info, dict)),
                {},
            )

        has_visual = bool(additional_info.get("visual_token_ids"))
        has_audio = bool(additional_info.get("audio_token_ids"))

        if has_visual and has_audio:
            # Not expected from this checkpoint's own state machine, but
            # cheaper to log and pick one deterministically than to crash.
            logger.warning(
                "Both visual_token_ids and audio_token_ids present -- the "
                "reference's state machine treats image-gen and audio-gen "
                "as mutually exclusive per request; decoding image only."
            )

        if has_visual:
            out = self.image_decoder.forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
            return _retag_model_outputs(out, "image")
        if has_audio:
            out = self.audio_decoder.forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )
            return _retag_model_outputs(out, "audio")

        logger.warning("No visual_token_ids or audio_token_ids provided for multi decoder")
        return OmniOutput(text_hidden_states=None, multimodal_outputs=None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Both sub-decoders self-load their own weight subtrees lazily on
        # first decode (see their own load_weights); the engine-side loader
        # has nothing to place here either.
        consumed = {name for name, _ in weights}
        return consumed | {name for name, _ in self.named_parameters()}
