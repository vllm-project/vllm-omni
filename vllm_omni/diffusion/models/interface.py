# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import (
    ClassVar,
    Protocol,
    runtime_checkable,
)


@runtime_checkable
class SupportImageInput(Protocol):
    support_image_input: ClassVar[bool] = True


@runtime_checkable
class SupportAudioOutput(Protocol):
    support_audio_output: ClassVar[bool] = True
