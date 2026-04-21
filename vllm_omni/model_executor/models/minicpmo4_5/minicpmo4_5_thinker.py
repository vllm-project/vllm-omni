from vllm.model_executor.models.minicpmo import (
    MiniCPMO4_5,
    MiniCPMODummyInputsBuilder,
    MiniCPMOMultiModalProcessor,
    MiniCPMOProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMOMultiModalProcessor,
    info=MiniCPMOProcessingInfo,
    dummy_inputs=MiniCPMODummyInputsBuilder,
)
class MiniCPMO4_5ThinkerForConditionalGeneration(MiniCPMO4_5):
    """Thinker-only MiniCPM-o 4.5 model."""

    pass
