"""LongCat-Next combined decoder stage: audio codes -> waveform, OR visual
codes -> RGB image, from one process.

Mirrors the reference's PostProcessor.decode_multi, dispatching on which of
gen_image/gen_audio the trigger token set. Collapsing both decoders into one
2-stage-pipeline stage avoids a 3-stage thinker->image->audio chain, where
the orchestrator's strict src_stage_id+1 forwarding would give the audio
decoder the image decoder's output instead of the thinker's.
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
    """Rename the sub-decoder's generic "model_outputs" key to the real
    modality ("image"/"audio") before it leaves this stage.

    MultimodalPayload.from_raw remaps a "model_outputs" key to the stage's
    static engine_output_type, which is wrong here since this stage's
    engine_output_type is fixed ("audio") but a response may be an image.
    Renaming the key per-response instead sidesteps that remap entirely.
    """
    if not isinstance(output.multimodal_outputs, dict) or "model_outputs" not in output.multimodal_outputs:
        return output
    retagged = dict(output.multimodal_outputs)
    retagged[key] = retagged.pop("model_outputs")
    return output._replace(multimodal_outputs=retagged)


class LongcatNextMultiDecoder(nn.Module):
    """Composes the image and audio decoders as submodules, dispatching per
    request on whichever of visual_token_ids/audio_token_ids talker_mtp
    populated. The two are mutually exclusive per the reference's own state
    machine, so at most one branch ever does real work."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.prefix = prefix

        # Each sub-decoder lazily loads its own weights on first use, so
        # constructing both here is as cheap as constructing either alone.
        self.image_decoder = LongcatNextImageDecoder(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "image_decoder")
        )
        self.audio_decoder = LongcatNextAudioDecoder(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "audio_decoder")
        )

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.image_decoder.embed_input_ids(input_ids, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
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
            kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or {}
        )
        if isinstance(model_intermediate_buffer, dict):
            info_dicts = [info for info in model_intermediate_buffer.values() if isinstance(info, dict)]
        else:
            info_dicts = [info for info in model_intermediate_buffer if isinstance(info, dict)]
        if len(info_dicts) > 1:
            logger.warning(
                "LongcatNextMultiDecoder got %d requests in one batch; only the "
                "first is decoded (max_num_seqs should be 1 for this stage).",
                len(info_dicts),
            )
        additional_info = info_dicts[0] if info_dicts else {}

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
            out = self.image_decoder.forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
            return _retag_model_outputs(out, "image")
        if has_audio:
            out = self.audio_decoder.forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
            return _retag_model_outputs(out, "audio")

        logger.warning("No visual_token_ids or audio_token_ids provided for multi decoder")
        return OmniOutput(text_hidden_states=None, multimodal_outputs=None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Both sub-decoders self-load lazily on first decode; the engine-side
        # loader has nothing to place here.
        consumed = {name for name, _ in weights}
        return consumed | {name for name, _ in self.named_parameters()}
