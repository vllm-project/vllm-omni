import pytest

from vllm_omni.experimental.fullduplex.engine.intermediate import get_stream_request_key

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_get_stream_request_key_requires_stable_identifier():
    with pytest.raises(ValueError, match="stable request id"):
        get_stream_request_key({"ids": {"tts": [1, 2, 3]}})


def test_get_stream_request_key_accepts_global_request_id():
    assert get_stream_request_key({"global_request_id": ["duplex-sid-stage0"]}) == "duplex-sid-stage0"
