"""
Root pytest entrypoint for the vLLM-Omni test suite.

- `tests/conftest.py` stays thin: plugin registration + compatibility re-exports.
- Importable utilities live under `tests/helpers/`.
- Fixtures live under `tests/helpers/fixtures/` and are loaded via `pytest_plugins`.
"""

from __future__ import annotations

pytest_plugins = (
    "tests.helpers.fixtures.env",
    "tests.helpers.fixtures.log",
    "tests.helpers.fixtures.run_args",
    "tests.helpers.fixtures.runtime",
)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Marker for Buildkite log folding before pytest summary lines.
    terminalreporter.write_sep("-", "Result Summary")


# Backward-compatible re-exports.
# (Many tests still import from `tests.conftest`; migrate these imports to `tests.helpers.*` over time.)
from tests.helpers.assertions import (  # noqa: F401,E402
    assert_audio_speech_response,
    assert_diffusion_response,
    assert_image_diffusion_response,
    assert_image_valid,
    assert_omni_response,
    assert_video_diffusion_response,
    assert_video_valid,
)
from tests.helpers.media import (  # noqa: F401,E402
    convert_audio_bytes_to_text,
    convert_audio_file_to_text,
    cosine_similarity_text,
    decode_b64_image,
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
)
from tests.helpers.stage_config import (  # noqa: F401,E402
    dummy_messages_from_mix_data,
    modify_stage_config,
)

# Lazy: importing `tests.helpers.runtime` at conftest load runs before session
# autouse fixtures and can scramble vLLM/vllm_omni init order.
_RUNTIME_EXPORT_NAMES = (
    "DiffusionResponse",
    "OmniResponse",
    "OmniRunner",
    "OmniRunnerHandler",
    "OmniServer",
    "OmniServerParams",
    "OmniServerStageCli",
    "OpenAIClientHandler",
)


def __getattr__(name: str):
    if name in _RUNTIME_EXPORT_NAMES:
        import tests.helpers.runtime as _runtime

        return getattr(_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_RUNTIME_EXPORT_NAMES})
