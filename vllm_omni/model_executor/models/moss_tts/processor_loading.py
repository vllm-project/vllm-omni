# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Loading the processor that MOSS-TTS checkpoints ship as remote code."""

import json
from typing import Any

from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import PROCESSOR_NAME, cached_file

_MOSS_PROCESSOR_CLASSES = {
    "processing_moss_tts.MossTTSDelayProcessor",
    "processing_moss_tts.MossTTSLocalProcessor",
}


def load_moss_processor(model: str) -> Any:
    """Load the checkpoint's processor, bypassing AutoProcessor for MOSS Delay and Local.

    Transformers 5.18+ passes a resolved ``revision`` to remote processors.
    The affected MOSS loaders forward it to ``ProcessorMixin.__init__``, which
    rejects it. Use the class declared in the checkpoint: Local v1.0 and v1.5
    declare different classes, and local directories may have arbitrary names.
    """
    config_path = cached_file(model, PROCESSOR_NAME, _raise_exceptions_for_missing_entries=False)
    if config_path is not None:
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        class_reference = (config.get("auto_map") or {}).get("AutoProcessor")
        if isinstance(class_reference, str) and class_reference in _MOSS_PROCESSOR_CLASSES:
            processor_cls = get_class_from_dynamic_module(class_reference, model)
            return processor_cls.from_pretrained(model, trust_remote_code=True)

    return AutoProcessor.from_pretrained(model, trust_remote_code=True)
