# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import time

from fastapi import Request
from pydantic import ValidationError
from vllm.entrypoints.openai.chat_completion.protocol import (
    BatchChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    UsageInfo,
)
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

logger = init_logger(__name__)


class OmniOpenAIServingChatBatch(OmniOpenAIServingChat):
    @staticmethod
    def _maybe_collapse_choices(choices: list[ChatCompletionResponseChoice]):
        """Collapse content + audio choices into one choice.

        NOTE: This is a hack for models that produce separate text and
        audio choices to ensure we have 1:1 mapping between number of
        inputs in the batch and responses.
        """
        content_choice, audio_choice = None, None
        if len(choices) == 1:
            return choices[0]
        if len(choices) != 2:
            raise ValueError(f"Expected 1 or 2 choices to collapse, got {len(choices)}. ")

        for choice in choices:
            if choice.message.audio:
                if audio_choice is not None:
                    raise ValueError("Multiple audio choices cannot be set in one completion message")
                audio_choice = choice
            else:
                if content_choice is not None:
                    raise ValueError("Multiple content choices cannot be set in one completion message")
                content_choice = choice

        if content_choice is None or audio_choice is None:
            raise ValueError(
                "Expected one content and one audio choice, but got: "
                f"content={content_choice is not None}, audio={audio_choice is not None}"
            )
        content_choice.message.audio = audio_choice.message.audio
        return content_choice

    @staticmethod
    def _get_subrequest_ids(base_id, request: BatchChatCompletionRequest) -> list[str]:
        """Get the request ID for each entry in the batch request to be resubmitted to chat completions."""
        return [f"{base_id}-idx-{idx}" for idx in range(len(request.messages))]

    async def create_batch_chat_completion(
        self,
        request: BatchChatCompletionRequest,
        raw_request: Request,
    ) -> ChatCompletionResponse | ErrorResponse:
        """Given a request, submit each request to chat completions & collect the results."""
        model = ""
        enabled_streaming = False
        chat_requests: list[ChatCompletionRequest] = []
        choices: list[ChatCompletionResponseChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Get the base request from the raw_request before maybe mutating the header
        base_id = OmniOpenAIServingChatBatch._base_request_id(
            raw_request,
            request.request_id,
        )
        batch_req_id = f"chatcmpl-batch-{base_id}"
        sub_req_ids = self._get_subrequest_ids(base_id, request)

        # Remove the raw request header if it exists now that we have subreq IDs,
        # since this takes priority over the object request_ids
        if raw_request.headers.get("X-Request-Id"):
            copy_headers = raw_request.headers.mutablecopy()
            del copy_headers["X-Request-Id"]
            raw_request._headers = copy_headers

        for idx, msg in enumerate(request.messages):
            # Update the current request ID to avoid subrequest collisions
            request.request_id = sub_req_ids[idx]
            try:
                chat_cmp_request = request.to_chat_completion_request(msg)
            except ValidationError as e:
                return self._create_error_response(
                    f"Message {idx} could not be converted to a chat completion request: {e}",
                )
            # Streaming isn't supported for batch chat completions,
            # so always set to False & warn if it was requested.
            enabled_streaming |= chat_cmp_request.stream
            chat_cmp_request.stream = False
            chat_requests.append(chat_cmp_request)

        if enabled_streaming:
            logger.warning("Streaming is not supported for batched chat completions; ignoring stream=True.")

        # Submit each chat completion request as a task, then gather results.
        # TODO (Alex): optimize this
        tasks = [asyncio.create_task(self.create_chat_completion(c, raw_request)) for c in chat_requests]
        try:
            results = await asyncio.gather(*tasks)
        finally:
            # Ensure we cancel remaining tasks if needed, e.g., early exit due to bad behavior
            for t in tasks:
                if not t.done():
                    t.cancel()

        for i, resp in enumerate(results):
            if isinstance(resp, ErrorResponse):
                return resp
            completion: ChatCompletionResponse = resp
            model = completion.model
            # FIXME (Alex): We should probably handle this in chat completions,
            # not here, but we need to ensure streaming is properly handled
            try:
                collapsed_choice = self._maybe_collapse_choices(completion.choices)
            except ValueError as e:
                return self._create_error_response(
                    f"Failed to collapse choices with error: {e}",
                )
            collapsed_choice.index = i
            choices.append(collapsed_choice)
            if completion.usage:
                total_prompt_tokens += completion.usage.prompt_tokens
                total_completion_tokens += completion.usage.completion_tokens

        usage = UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        )

        return ChatCompletionResponse(
            id=batch_req_id,
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
        )
