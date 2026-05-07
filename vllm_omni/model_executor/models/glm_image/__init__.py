def __getattr__(name: str):
    """Lazy import to avoid importing transformers.models.glm_image at module init.

    The AR model depends on ``transformers.models.glm_image`` which is only
    available when the model weights are present.  Importing it eagerly in
    ``__init__.py`` breaks the pipeline registry lookup for environments that
    don't have this custom transformers extension installed.
    """
    if name == "GlmImageForConditionalGeneration":
        from .glm_image_ar import GlmImageForConditionalGeneration

        return GlmImageForConditionalGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GlmImageForConditionalGeneration"]
