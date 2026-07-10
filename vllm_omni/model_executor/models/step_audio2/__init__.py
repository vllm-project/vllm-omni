__all__ = ["StepAudio2ForConditionalGeneration"]


def __getattr__(name: str):
    if name == "StepAudio2ForConditionalGeneration":
        from .step_audio2 import StepAudio2ForConditionalGeneration

        return StepAudio2ForConditionalGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
