from typing import Literal

from pydantic import Field

from vllm.entrypoints.openai.protocol import ChatCompletionResponseChoice


class OmniChatCompletionResponseChoice(ChatCompletionResponseChoice):
    choice_output_type: Literal["text", "audio"] = Field(
        default="text", description="The type of the choice output."
    )
