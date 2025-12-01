from collections.abc import Callable
import torch

class PreprocessMixin:
    """
    Mixin class for all stages in the Omni model.
    """
    def set_preprocess(self, preprocess_fn: Callable) -> None:
        """
        Set a preprocess function for the stage.
        Args:
            preprocess_fn: The preprocess function to register.
        """
        self.preprocess = preprocess_fn

    def preprocess(self,
                input_ids: torch.Tensor,
                inputs_embeds: torch.Tensor,
                **input_dict: object
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Process the input_ids and inputs_embeds for the given input_dict.
        Returns the processed input_ids, inputs_embeds, and the input_dict.

        If the stage don't applicable, return the original input_ids, inputs_embeds, and an empty dict.
        """
        return input_ids, inputs_embeds, {}
