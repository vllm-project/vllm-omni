# MiniCPM-o 4.5 tensor handoff

Status: inline host TensorEnvelope implemented and statically validated. Native
NPU-handle transport, A3 measurements, and complete quality evidence remain
pending.

## Audited boundary

The competition deploy uses SharedMemoryConnector. That connector serializes
tensors through the host and exposes no NPU P2P or device-handle lifetime
contract. Mooncake/Mori raw connectors exist in the repository but are not the
official deploy topology. This change therefore does not claim device-resident
or zero-copy transfer.

The concrete baseline overhead is a large Thinker-to-Talker hidden tensor being
converted to a nested Python list. Talker-to-Code2Wav already emits a tensor and
does not use that list path.

## Protocol

Thinker-to-Talker now keeps hidden conditioning as a contiguous FP32 host tensor
and stores a version-1 envelope beside it. Control metadata and payload remain
separate:

- request/session/epoch/chunk identity;
- shape, dtype, and source device descriptor;
- handle kind and payload path;
- the tensor at the canonical hidden_states.tts payload path.

`model_intermediate_buffer` is typed as `dict[str, Any]` at the EngineCore IPC
boundary, so the sender explicitly converts nested tensors to a separate
versioned dtype/shape/bytes wire envelope and Stage Core restores owned CPU
tensors before handing the request to the model runner. This is host
serialization, not a device handle or zero-copy path.

Talker validates request identity, version, payload path, shape, and dtype
before consuming a known inline handle. A legacy list remains accepted. Unknown
future handle kinds preserve the already-materialized payload so a connector
can add a verified NPU-native implementation without changing Talker math.

Reference audio also remains a host tensor instead of a Python float list.

## Activation and rollback

The tensor representation defaults on in Python and requires no deploy YAML.
Set VLLM_OMNI_MINICPMO45_TENSOR_HANDOFF=0 and cleanly restart to restore the
legacy hidden/ref list materialization.
Invalid switch values fail at request preparation instead of silently choosing
an unintended path.

## Promotion gate

- Parent/candidate token, hidden, codec, shape, dtype, and audio equality.
- Timeline/profile confirms reduced bridge serialization/CPU overhead.
- Official Chinese c=1 follows the RTF/TTFP promotion gates with TTFT guarded.
- Peak HBM grows no more than 5%; all requests and audio streams succeed.
- Daily-Omni, Video-MME, ASV, and WER complete gates pass.

Native NPU handles require a separate capability-gated implementation with
explicit producer/consumer ownership, release, abort, epoch, stream/event, and
failure semantics. They must not be enabled merely because a connector class
advertises generic raw-byte support.
