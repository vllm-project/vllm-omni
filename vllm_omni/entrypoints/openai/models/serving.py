# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI models serving helpers.

This module mirrors upstream's OpenAI models area for helpers that back
``/v1/models`` behavior."""

from vllm.entrypoints.openai.engine.protocol import ModelCard, ModelList, ModelPermission
from vllm.entrypoints.openai.models.protocol import BaseModelPath


class _DiffusionServingModels:
    """Minimal OpenAIServingModels implementation for diffusion-only servers.

    vLLM's /v1/models route expects `app.state.openai_serving_models` to expose
    `show_available_models()`. In pure diffusion mode we don't initialize the
    full OpenAIServingModels (it depends on LLM-specific processors), so we
    provide a lightweight fallback.
    """

    class _NullModelConfig:
        def __getattr__(self, name):
            return None

    class _Unsupported:
        def __init__(self, name: str):
            self.name = name

        def __call__(self, *args, **kwargs):
            raise NotImplementedError(f"{self.name} is not supported in diffusion mode")

        def __getattr__(self, attr):
            raise NotImplementedError(f"{self.name}.{attr} is not supported in diffusion mode")

    def __init__(self, base_model_paths: list[BaseModelPath]) -> None:
        self._base_model_paths = base_model_paths
        self.model_config = self._NullModelConfig()

    @property
    def base_model_paths(self) -> list[BaseModelPath]:
        return self._base_model_paths

    def is_base_model(self, model_name: str) -> bool:
        return any(p.name == model_name for p in self._base_model_paths)

    def __getattr__(self, name):
        return self._Unsupported(name)

    async def show_available_models(self) -> ModelList:
        return ModelList(
            data=[
                ModelCard(
                    id=base_model.name,
                    root=base_model.model_path,
                    permission=[ModelPermission()],
                )
                for base_model in self._base_model_paths
            ]
        )
