import torch

from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information


def test_serialize_additional_information():
    info = {
        "tensor_f32": torch.rand(3, dtype=torch.float32),
        "tensor_f64": torch.rand(3, 3, dtype=torch.float64),
        "list": [1, 2, 3],
        "scalar": 1,
    }

    payload = serialize_additional_information(info)

    assert payload.entries["tensor_f32"].tensor_dtype == "float32"
    assert payload.entries["tensor_f32"].tensor_shape == [3]
    assert payload.entries["tensor_f32"].tensor_data == info["tensor_f32"].numpy().tobytes()

    assert payload.entries["tensor_f64"].tensor_dtype == "float64"
    assert payload.entries["tensor_f64"].tensor_shape == [3, 3]
    assert payload.entries["tensor_f64"].tensor_data == info["tensor_f64"].numpy().tobytes()

    assert payload.entries["list"].list_data == info["list"]
    assert payload.entries["scalar"].scalar_data == info["scalar"]

    deserialized = deserialize_additional_information(payload)

    # compare tensors separately first
    assert torch.equal(deserialized["tensor_f32"], info["tensor_f32"])
    assert torch.equal(deserialized["tensor_f64"], info["tensor_f64"])
    del deserialized["tensor_f32"], info["tensor_f32"]
    del deserialized["tensor_f64"], info["tensor_f64"]

    assert deserialized == info
