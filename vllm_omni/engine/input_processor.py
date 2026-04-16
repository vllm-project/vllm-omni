"""OmniInputProcessor: extends vLLM InputProcessor with OmniInputPreprocessor."""

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.renderers import BaseRenderer
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.inputs.preprocess import OmniInputPreprocessor


class OmniInputProcessor(InputProcessor):
    """InputProcessor for omni models.

    Extends the base vLLM InputProcessor by replacing the default
    InputPreprocessor with OmniInputPreprocessor, which handles
    omni-specific input types (prompt embeddings, additional information).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        super().__init__(vllm_config, renderer, mm_registry=mm_registry)
        # Replace the base InputPreprocessor with OmniInputPreprocessor
        self.input_preprocessor = OmniInputPreprocessor(
            vllm_config,
            renderer=self.renderer,
            mm_registry=mm_registry,
        )
