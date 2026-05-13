"""Tests for data_entry_keys."""

import msgspec
import pytest
import torch

from vllm_omni.data_entry_keys import (
    CodesStruct,
    EmbeddingsStruct,
    HiddenStatesStruct,
    IdsStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
    deserialize_payload,
    flatten_payload,
    serialize_payload,
    to_dict,
    to_struct,
    unflatten_payload,
)
from vllm_omni.engine import AdditionalInformationPayload


class TestOmniPayloadStruct:
    """Runtime-validated mirror of OmniPayload (msgspec.Struct)."""

    def test_to_struct_validates_dict(self):
        d = {"meta": {"left_context_size": 25, "finished": torch.tensor(False)}}
        s = to_struct(d)
        assert s.meta.left_context_size == 25

    def test_to_struct_rejects_legacy_flat_top_level(self):
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            to_struct({"code_predictor_codes": torch.zeros(3, 8)})

    def test_to_struct_rejects_legacy_flat_meta_field(self):
        # `left_context_size` at top level (legacy) instead of under `meta`
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            to_struct({"left_context_size": 25})

    def test_to_struct_rejects_typo_in_subkey(self):
        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            to_struct({"meta": {"finisheed": True}})

    def test_to_struct_rejects_wrong_type(self):
        with pytest.raises(msgspec.ValidationError, match="Expected"):
            to_struct({"meta": {"left_context_size": "not_an_int"}})

    def test_round_trip_dict_struct_dict(self):
        original = {
            "meta": {"left_context_size": 7, "finished": torch.tensor(True)},
            "codes": {"audio": torch.zeros(2, 8)},
            "hidden_states": {"output": torch.zeros(4, 16)},
        }
        s = to_struct(original)
        d = to_dict(s)
        assert sorted(d.keys()) == sorted(original.keys())
        for top, sub in original.items():
            assert sorted(d[top].keys()) == sorted(sub.keys())

    def test_to_dict_drops_unset_fields(self):
        s = OmniPayloadStruct(meta=MetaStruct(left_context_size=10))
        d = to_dict(s)
        assert d == {"meta": {"left_context_size": 10}}

    def test_struct_with_all_categories(self):
        d = {
            "hidden_states": {"output": torch.zeros(1)},
            "embed": {"prefill": torch.zeros(1), "tts_bos": torch.zeros(1)},
            "ids": {"all": [1, 2], "prompt": [1]},
            "codes": {"audio": torch.zeros(1)},
            "meta": {"left_context_size": 3, "num_processed_tokens": 7},
        }
        s = to_struct(d)
        assert isinstance(s.hidden_states, HiddenStatesStruct)
        assert isinstance(s.embed, EmbeddingsStruct)
        assert isinstance(s.ids, IdsStruct)
        assert isinstance(s.codes, CodesStruct)
        assert isinstance(s.meta, MetaStruct)
        assert s.ids.all == [1, 2]
        assert s.meta.num_processed_tokens == 7


class TestValidatePayload:
    def test_raises_on_unknown_top_level(self):
        from vllm_omni.data_entry_keys import validate_payload

        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            validate_payload({"code_predictor_codes": torch.zeros(3, 8)}, context="test_boundary")

    def test_raises_on_unknown_sub_key(self):
        from vllm_omni.data_entry_keys import validate_payload

        with pytest.raises(msgspec.ValidationError, match="unknown field"):
            validate_payload({"meta": {"finisheed": True}})

    def test_none_is_ok(self):
        from vllm_omni.data_entry_keys import validate_payload

        validate_payload(None)  # should not raise

    def test_valid_payload_passes(self):
        from vllm_omni.data_entry_keys import validate_payload

        validate_payload({"meta": {"left_context_size": 5}})

    def test_context_in_error_message(self):
        from vllm_omni.data_entry_keys import validate_payload

        with pytest.raises(msgspec.ValidationError, match="my_call_site"):
            validate_payload({"bad": 1}, context="my_call_site")


class TestWireEquivalenceStructVsDict:
    """Producer return-side migration invariant: encoding an ``OmniPayloadStruct``
    via the connector serializer must decode to the same payload as encoding
    the equivalent ``to_dict(struct)``.

    Guards against regressions where the wire format diverges between the two
    paths (e.g. msgspec adds a struct tag, or ``to_dict`` drops a non-default
    sub-field that ``omit_defaults`` retains).
    """

    @staticmethod
    def _round_trip(obj):
        from vllm_omni.distributed.omni_connectors.utils.serialization import (
            OmniMsgpackDecoder,
            OmniMsgpackEncoder,
        )

        return OmniMsgpackDecoder().decode(OmniMsgpackEncoder().encode(obj))

    @staticmethod
    def _assert_decoded_equal(a, b):
        if isinstance(a, dict):
            assert isinstance(b, dict)
            assert sorted(a.keys()) == sorted(b.keys())
            for k in a:
                TestWireEquivalenceStructVsDict._assert_decoded_equal(a[k], b[k])
        elif isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor)
            assert a.dtype == b.dtype
            assert a.shape == b.shape
            assert torch.equal(a, b)
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                TestWireEquivalenceStructVsDict._assert_decoded_equal(x, y)
        else:
            assert a == b

    def test_basic_payload(self):
        struct = OmniPayloadStruct(
            codes=CodesStruct(audio=torch.tensor([1, 2, 3], dtype=torch.long)),
            meta=MetaStruct(left_context_size=10, finished=torch.tensor(True, dtype=torch.bool)),
        )
        self._assert_decoded_equal(self._round_trip(struct), self._round_trip(to_dict(struct)))

    def test_nested_sub_structs(self):
        # Exercises depth-2 sub-struct encoding (embed.*) which was the case that
        # exposed schema drift in the #1829 migration.
        struct = OmniPayloadStruct(
            codes=CodesStruct(audio=torch.tensor([5, 6], dtype=torch.long)),
            meta=MetaStruct(finished=torch.tensor(False, dtype=torch.bool)),
            embed=EmbeddingsStruct(
                speech_token=torch.tensor([[1, 2, 3]]),
                speech_feat=torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
                embedding=torch.tensor([[0.5, 0.6]]),
            ),
        )
        self._assert_decoded_equal(self._round_trip(struct), self._round_trip(to_dict(struct)))


class TestFlattenPayload:
    def test_basic_nested_to_dotted(self):
        nested = {
            "codes": {"audio": torch.tensor([1.0])},
            "meta": {"finished": torch.tensor(True, dtype=torch.bool), "left_context_size": 5},
        }
        flat = flatten_payload(nested)
        assert torch.equal(flat["codes.audio"], torch.tensor([1.0]))
        assert flat["meta.finished"].item() is True
        assert flat["meta.left_context_size"] == 5
        assert "codes" not in flat
        assert "meta" not in flat

    def test_top_level_keys_preserved(self):
        nested = {
            "latent": torch.tensor([9.0]),
            "generated_len": 42,
        }
        flat = flatten_payload(nested)
        assert torch.equal(flat["latent"], torch.tensor([9.0]))
        assert flat["generated_len"] == 42

    def test_hidden_states_layers_expanded(self):
        nested = {
            "hidden_states": {
                "output": torch.tensor([1.0]),
                "layers": {
                    0: torch.tensor([2.0]),
                    24: torch.tensor([3.0]),
                },
            },
        }
        flat = flatten_payload(nested)
        assert torch.equal(flat["hidden_states.output"], torch.tensor([1.0]))
        assert torch.equal(flat["hidden_states.layer_0"], torch.tensor([2.0]))
        assert torch.equal(flat["hidden_states.layer_24"], torch.tensor([3.0]))
        assert "hidden_states.layers" not in flat

    def test_mixed_nested_and_top_level(self):
        nested: OmniPayload = {
            "codes": {"audio": torch.tensor([1.0])},
            "latent": torch.tensor([2.0]),
            "meta": {"finished": torch.tensor(False, dtype=torch.bool)},
        }
        flat = flatten_payload(nested)
        assert set(flat.keys()) == {"codes.audio", "latent", "meta.finished"}


class TestUnflattenPayload:
    def test_basic_dotted_to_nested(self):
        flat = {
            "codes.audio": torch.tensor([1.0]),
            "meta.finished": torch.tensor(True, dtype=torch.bool),
            "meta.left_context_size": 5,
        }
        nested = unflatten_payload(flat)
        assert torch.equal(nested["codes"]["audio"], torch.tensor([1.0]))
        assert nested["meta"]["finished"].item() is True
        assert nested["meta"]["left_context_size"] == 5

    def test_top_level_keys_preserved(self):
        flat = {"latent": torch.tensor([9.0]), "generated_len": 42}
        nested = unflatten_payload(flat)
        assert torch.equal(nested["latent"], torch.tensor([9.0]))
        assert nested["generated_len"] == 42

    def test_hidden_states_layers_collected(self):
        flat = {
            "hidden_states.output": torch.tensor([1.0]),
            "hidden_states.layer_0": torch.tensor([2.0]),
            "hidden_states.layer_24": torch.tensor([3.0]),
        }
        nested = unflatten_payload(flat)
        assert torch.equal(nested["hidden_states"]["output"], torch.tensor([1.0]))
        assert torch.equal(nested["hidden_states"]["layers"][0], torch.tensor([2.0]))
        assert torch.equal(nested["hidden_states"]["layers"][24], torch.tensor([3.0]))


class TestFlattenUnflattenRoundTrip:
    def test_round_trip_simple(self):
        original: OmniPayload = {
            "codes": {"audio": torch.tensor([1.0, 2.0])},
            "meta": {"finished": torch.tensor(True, dtype=torch.bool), "left_context_size": 10},
            "ids": {"prompt": [1, 2, 3]},
            "latent": torch.tensor([5.0]),
        }
        restored = unflatten_payload(flatten_payload(original))
        assert torch.equal(restored["codes"]["audio"], original["codes"]["audio"])
        assert restored["meta"]["finished"].item() is True
        assert restored["meta"]["left_context_size"] == 10
        assert restored["ids"]["prompt"] == [1, 2, 3]
        assert torch.equal(restored["latent"], original["latent"])

    def test_round_trip_with_layers(self):
        original = {
            "hidden_states": {
                "output": torch.tensor([1.0]),
                "layers": {0: torch.tensor([2.0]), 24: torch.tensor([3.0])},
            },
        }
        restored = unflatten_payload(flatten_payload(original))
        assert torch.equal(restored["hidden_states"]["output"], torch.tensor([1.0]))
        assert torch.equal(restored["hidden_states"]["layers"][0], torch.tensor([2.0]))
        assert torch.equal(restored["hidden_states"]["layers"][24], torch.tensor([3.0]))

    def test_round_trip_all_categories(self):
        original: OmniPayload = {
            "hidden_states": {"output": torch.tensor([1.0]), "last": torch.tensor([2.0])},
            "embed": {"prefill": torch.tensor([3.0]), "tts_bos": torch.tensor([4.0])},
            "codes": {"audio": torch.tensor([5.0]), "ref": torch.tensor([6.0])},
            "ids": {"all": [1, 2], "prompt": [3, 4]},
            "meta": {"finished": torch.tensor(False, dtype=torch.bool), "ar_width": 8},
        }
        restored = unflatten_payload(flatten_payload(original))
        assert torch.equal(restored["hidden_states"]["output"], torch.tensor([1.0]))
        assert torch.equal(restored["hidden_states"]["last"], torch.tensor([2.0]))
        assert torch.equal(restored["embed"]["prefill"], torch.tensor([3.0]))
        assert torch.equal(restored["embed"]["tts_bos"], torch.tensor([4.0]))
        assert torch.equal(restored["codes"]["audio"], torch.tensor([5.0]))
        assert torch.equal(restored["codes"]["ref"], torch.tensor([6.0]))
        assert restored["ids"]["all"] == [1, 2]
        assert restored["ids"]["prompt"] == [3, 4]
        assert restored["meta"]["finished"].item() is False
        assert restored["meta"]["ar_width"] == 8


class TestSerializeDeserializePayload:
    def test_tensor_round_trip(self):
        original: OmniPayload = {
            "hidden_states": {"output": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        }
        wire = serialize_payload(original)
        assert isinstance(wire, AdditionalInformationPayload)
        restored = deserialize_payload(wire)
        assert torch.equal(restored["hidden_states"]["output"], original["hidden_states"]["output"])

    def test_list_round_trip(self):
        original: OmniPayload = {
            "ids": {"prompt": [10, 20, 30]},
        }
        wire = serialize_payload(original)
        restored = deserialize_payload(wire)
        assert restored["ids"]["prompt"] == [10, 20, 30]

    def test_finished_tensor_round_trip(self):
        original: OmniPayload = {
            "meta": {"finished": torch.tensor(True, dtype=torch.bool), "left_context_size": 5},
        }
        wire = serialize_payload(original)
        restored = deserialize_payload(wire)
        assert isinstance(restored["meta"]["finished"], torch.Tensor)
        assert restored["meta"]["finished"].dtype == torch.bool
        assert restored["meta"]["finished"].item() is True
        assert restored["meta"]["left_context_size"] == 5

    def test_mixed_types_round_trip(self):
        original: OmniPayload = {
            "hidden_states": {"output": torch.tensor([1.0, 2.0])},
            "ids": {"all": [1, 2, 3]},
            "meta": {"finished": torch.tensor(False, dtype=torch.bool), "ar_width": 4},
            "codes": {"audio": torch.tensor([3.0])},
        }
        wire = serialize_payload(original)
        restored = deserialize_payload(wire)
        assert torch.equal(restored["hidden_states"]["output"], original["hidden_states"]["output"])
        assert restored["ids"]["all"] == [1, 2, 3]
        assert restored["meta"]["finished"].item() is False
        assert restored["meta"]["ar_width"] == 4
        assert torch.equal(restored["codes"]["audio"], original["codes"]["audio"])

    def test_hidden_states_layers_round_trip(self):
        original = {
            "hidden_states": {
                "output": torch.tensor([1.0]),
                "layers": {0: torch.tensor([2.0]), 24: torch.tensor([3.0])},
            },
        }
        wire = serialize_payload(original)
        restored = deserialize_payload(wire)
        assert torch.equal(restored["hidden_states"]["output"], torch.tensor([1.0]))
        assert torch.equal(restored["hidden_states"]["layers"][0], torch.tensor([2.0]))
        assert torch.equal(restored["hidden_states"]["layers"][24], torch.tensor([3.0]))

    def test_tensor_dtype_preserved(self):
        # bfloat16 excluded: numpy() doesn't support it; callers must cast before serializing.
        for dtype in [torch.float16, torch.float32, torch.int64, torch.int32, torch.bool]:
            original: OmniPayload = {"codes": {"audio": torch.tensor([1], dtype=dtype)}}
            wire = serialize_payload(original)
            restored = deserialize_payload(wire)
            assert restored["codes"]["audio"].dtype == dtype, f"dtype mismatch for {dtype}"

    def test_tensor_shape_preserved(self):
        t = torch.randn(3, 4, 5)
        original: OmniPayload = {"hidden_states": {"output": t}}
        wire = serialize_payload(original)
        restored = deserialize_payload(wire)
        assert restored["hidden_states"]["output"].shape == (3, 4, 5)
        assert torch.allclose(restored["hidden_states"]["output"], t)

    def test_empty_payload_returns_none(self):
        assert serialize_payload({}) is None

    def test_none_values_skipped(self):
        original: OmniPayload = {"meta": {"finished": None}}
        wire = serialize_payload(original)
        assert wire is None
