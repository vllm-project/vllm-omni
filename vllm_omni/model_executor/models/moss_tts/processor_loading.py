# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Loading the processor that MOSS-TTS checkpoints ship as remote code."""

from typing import Any

from transformers.dynamic_module_utils import get_class_from_dynamic_module

# MOSS-TTS, MOSS-TTS-v1.5, MOSS-TTSD, MOSS-SoundEffect and MOSS-VoiceGenerator.
MOSS_DELAY_PROCESSOR = "processing_moss_tts.MossTTSDelayProcessor"
# MOSS-TTS-Local-Transformer-v1.5.
MOSS_LOCAL_PROCESSOR = "processing_moss_tts.MossTTSLocalProcessor"


def load_moss_processor(model: str, class_reference: str) -> Any:
    """Load a MOSS-TTS checkpoint's own processor class instead of using ``AutoProcessor``.

    ``AutoProcessor.from_pretrained`` (Transformers 5.18 and later) always adds
    a resolved ``revision`` to the kwargs it hands to a remote-code processor.
    The checkpoints' ``from_pretrained`` forwards every leftover kwarg to
    ``ProcessorMixin.__init__``, which raises ``TypeError: Unexpected keyword
    argument revision``. Calling the checkpoint's loader with nothing but
    ``trust_remote_code`` avoids that. This is a workaround: it can go once the
    checkpoints stop forwarding loader kwargs to ``cls(...)``.

    ``model`` is a Hub id or a local checkpoint directory, and ``class_reference``
    is ``MOSS_DELAY_PROCESSOR`` or ``MOSS_LOCAL_PROCESSOR``. The result is an
    instance of that remote-code ``ProcessorMixin`` subclass. Its API is defined
    by the checkpoint, so the return type is ``Any``.
    """
    processor_cls = get_class_from_dynamic_module(class_reference, model)
    return processor_cls.from_pretrained(model, trust_remote_code=True)
