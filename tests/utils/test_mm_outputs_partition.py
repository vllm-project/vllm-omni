import torch

from vllm_omni.utils.mm_outputs import partition_flat_payload, partition_payload_list


def test_partition_thinker_latent_payload():
    payload = {
        "hidden_states.layer_0": torch.zeros(2, 4),
        "hidden_states.layer_24": torch.zeros(2, 8),
        "embed.tts_bos": [torch.zeros(1, 1, 4)],
    }
    inter, client = partition_flat_payload(payload)
    assert inter == payload
    assert client == {}


def test_partition_talker_intermediate_codes():
    payload = {
        "codes.audio": torch.zeros(3, 2),
        "hidden": torch.zeros(3, 16),
    }
    inter, client = partition_flat_payload(payload)
    assert inter == payload
    assert client == {}


def test_partition_code2wav_client_audio():
    payload = {
        "model_outputs": torch.zeros(1, 2400),
        "sr": torch.tensor(24000, dtype=torch.int32),
    }
    inter, client = partition_flat_payload(payload)
    assert inter == {}
    assert client == payload


def test_partition_payload_list_preserves_request_alignment():
    payloads = [
        {"hidden_states.layer_0": torch.zeros(1, 2)},
        {"model_outputs": torch.zeros(1, 10)},
    ]
    inter_list, client_list = partition_payload_list(payloads)
    assert inter_list == [payloads[0], None]
    assert client_list == [None, payloads[1]]
