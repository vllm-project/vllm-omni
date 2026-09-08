"""Fixtures for the reused Whisper transcription worker."""

import sys

import pytest


@pytest.fixture(autouse=True, scope="module")
def release_audio_transcriber_after_module():
    """Free the transcription worker's device memory once a test module is done.

    A backstop. The server and runner fixtures already release the worker when
    each of their instances tears down, which is what keeps Whisper off the
    device while the next parametrized server initializes. This covers the
    remaining case: a module that transcribes without using those fixtures.

    Reached through ``sys.modules`` because ``tests/conftest.py`` deliberately
    keeps ``tests.helpers.media`` lazy: a module that never transcribed should
    not pay to import it.
    """
    yield
    media = sys.modules.get("tests.helpers.media")
    if media is not None:
        media.release_audio_transcriber()
