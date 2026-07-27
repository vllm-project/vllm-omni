from pathlib import Path

from tests.helpers.tiny_model import build_tiny_from_configs

TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"


def tiny_flux2_klein_builder() -> str:
    """Build a tiny Flux2Klein model from vendored configs."""
    return build_tiny_from_configs(
        "Flux2KleinPipeline", "black-forest-labs/FLUX.2-klein-4B", TINY_CONFIGS_DIR / "Flux2KleinPipeline"
    )


def tiny_ltx2_builder() -> str:
    """Build a tiny LTX2 model from vendored configs."""
    return build_tiny_from_configs("LTX2Pipeline", "Lightricks/LTX-2", TINY_CONFIGS_DIR / "LTX2Pipeline")
