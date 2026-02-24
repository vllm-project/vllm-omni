"""Unit tests for OmniProcessingInterface and OmniProcessingMixin."""

import pytest
import torch

from vllm_omni.model_executor.omni_process_mixin import (
    OmniModelCapability,
    OmniProcessingMixin,
    has_postprocess,
    has_preprocess,
)

# =============================================================================
# Mock: Stage-dependent model (mirrors real Omni models)
# =============================================================================


class MockOmniModel(OmniProcessingMixin):
    def __init__(self, model_stage: str = "thinker"):
        # Allow OmniProcessingMixin to initialize its capabilities
        super().__init__()
        self.model_stage = model_stage

        if model_stage == "thinker":
            # Thinker stage: no pre/postprocess
            pass
        elif model_stage == "talker":
            # Talker stage: has both pre and postprocess
            self.set_custom_preprocess(self._talker_preprocess)
            self.set_custom_postprocess(self._talker_postprocess)
        elif model_stage == "code2wav":
            # Code2wav: only postprocess
            self.set_custom_postprocess(self._code2wav_postprocess)

    def _talker_preprocess(self, input_ids, input_embeds, **kwargs):
        return input_ids, input_embeds * 2, {"stage": "talker", **kwargs}

    def _talker_postprocess(self, model_output, **kwargs):
        return {"processed_by": "talker"}

    def _code2wav_postprocess(self, model_output, **kwargs):
        return {"audio_samples": model_output.shape[0]}


@pytest.mark.parametrize(
    "stage,expect_preprocess,expect_postprocess",
    [
        ("thinker", False, False),
        ("talker", True, True),
        ("code2wav", False, True),
    ],
)
def test_stage_dependent_capabilities(stage, expect_preprocess, expect_postprocess):
    """Different stages register different capabilities."""
    model = MockOmniModel(model_stage=stage)

    assert model.has_capability(OmniModelCapability.PREPROCESS) == expect_preprocess
    assert model.has_capability(OmniModelCapability.POSTPROCESS) == expect_postprocess


def test_preprocess_executes_custom_function():
    """Preprocess calls the registered function with correct args."""
    model = MockOmniModel(model_stage="talker")

    ids = torch.tensor([1, 2, 3])
    embeds = torch.ones(3, 4)

    new_ids, new_embeds, update = model.preprocess(ids, embeds, extra="value")

    assert torch.equal(new_ids, ids)
    assert torch.equal(new_embeds, embeds * 2)
    assert update == {"stage": "talker", "extra": "value"}


def test_preprocess_raises_when_not_configured():
    """Preprocess raises NotImplementedError when capability not registered."""
    model = MockOmniModel(model_stage="thinker")

    with pytest.raises(NotImplementedError):
        model.preprocess(torch.tensor([1]), torch.ones(1, 4))
    with pytest.raises(NotImplementedError):
        model.postprocess(torch.randn(2, 3))


@pytest.mark.parametrize(
    "stage, expect_preprocess, expect_postprocess",
    [
        ("thinker", False, False),
        ("talker", True, True),
        ("code2wav", False, True),
    ],
)
def test_backwards_compat_with_legacy_model(stage, expect_preprocess, expect_postprocess):
    """Helpers fallback to boolean flags for legacy models."""

    class LegacyOmniModel:
        def __init__(self, model_stage: str = "thinker"):
            self.model_stage = model_stage
            self.has_preprocess = False
            self.has_postprocess = False

            if model_stage == "thinker":
                # Thinker stage: no pre/postprocess
                pass
            elif model_stage == "talker":
                self.preprocess = self._talker_preprocess
                self.postprocess = self._talker_postprocess
                self.has_preprocess = True
                self.has_postprocess = True
            elif model_stage == "code2wav":
                self.postprocess = self._code2wav_postprocess
                self.has_postprocess = True

        def _talker_preprocess(self, input_ids, input_embeds, **kwargs):
            return input_ids, input_embeds * 2, {"stage": "talker", **kwargs}

        def _talker_postprocess(self, model_output, **kwargs):
            return {"processed_by": "talker"}

        def _code2wav_postprocess(self, model_output, **kwargs):
            return {"audio_samples": model_output.shape[0]}

    legacy = LegacyOmniModel(model_stage=stage)
    new_model = MockOmniModel(model_stage=stage)
    assert has_preprocess(legacy) is expect_preprocess is has_preprocess(new_model)
    assert has_postprocess(legacy) is expect_postprocess is has_postprocess(new_model)


def test_different_subclasses_have_isolated_capabilities():
    """Each subclass gets its own capability set."""

    class StageA(OmniProcessingMixin):
        def __init__(self):
            super().__init__()
            self.register_capability(OmniModelCapability.PREPROCESS)

    class StageB(OmniProcessingMixin):
        def __init__(self):
            super().__init__()
            self.register_capability(OmniModelCapability.POSTPROCESS)

    a, b = StageA(), StageB()

    assert a.has_capability(OmniModelCapability.PREPROCESS)
    assert not a.has_capability(OmniModelCapability.POSTPROCESS)
    assert b.has_capability(OmniModelCapability.POSTPROCESS)
    assert not b.has_capability(OmniModelCapability.PREPROCESS)
