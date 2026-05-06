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
    "tests.helpers.fixtures.speaker_cache",
)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # Marker for Buildkite log folding before pytest summary lines.
    terminalreporter.write_line("--- Running Summary")


# Backward-compatible lazy re-exports.
# (Many tests still import from `tests.conftest`; migrate these imports to `tests.helpers.*` over time.)
# Keep these lazy so conftest import does not trigger heavy helper dependencies.
_ASSERTION_EXPORT_NAMES = (
    "assert_audio_speech_response",
    "assert_diffusion_response",
    "assert_image_diffusion_response",
    "assert_image_valid",
    "assert_omni_response",
    "assert_video_diffusion_response",
    "assert_video_valid",
)
_MEDIA_EXPORT_NAMES = (
    "convert_audio_bytes_to_text",
    "convert_audio_file_to_text",
    "cosine_similarity_text",
    "decode_b64_image",
    "generate_synthetic_audio",
    "generate_synthetic_image",
    "generate_synthetic_video",
)
_STAGE_CONFIG_EXPORT_NAMES = ("modify_stage_config",)
_RUNTIME_EXPORT_NAMES = (
    "DiffusionResponse",
    "OmniResponse",
    "OmniRunner",
    "OmniRunnerHandler",
    "OmniServer",
    "OmniServerParams",
    "OmniServerStageCli",
    "OpenAIClientHandler",
    "dummy_messages_from_mix_data",
)
_LAZY_EXPORT_MODULES = {
    **{name: "tests.helpers.assertions" for name in _ASSERTION_EXPORT_NAMES},
    **{name: "tests.helpers.media" for name in _MEDIA_EXPORT_NAMES},
    **{name: "tests.helpers.stage_config" for name in _STAGE_CONFIG_EXPORT_NAMES},
    **{name: "tests.helpers.runtime" for name in _RUNTIME_EXPORT_NAMES},
}
