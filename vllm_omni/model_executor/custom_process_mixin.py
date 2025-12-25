from collections.abc import Callable

import torch


class CustomProcessMixin:
    """
    Mixin class for all stages in the Omni model.
    """

    def set_custom_preprocess(self, preprocess_fn: Callable) -> None:
        """
        Set a preprocess function for the stage.
        Args:
            preprocess_fn: The preprocess function to register.
        """
        self.preprocess = preprocess_fn

    def set_custom_postprocess(self, postprocess_fn: Callable) -> None:
        """
        Set a postprocess function for the stage.
        Args:
            postprocess_fn: The postprocess function to register.
        """
        self.postprocess = postprocess_fn

    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **input_dict: object
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Process the input_ids and input_embeds for the given input_dict.
        Returns the processed input_ids, input_embeds, and the input_dict.
        If the stage don't applicable, return the original input_ids, input_embeds, and an empty dict.
        """
        raise NotImplementedError("Preprocess is not implemented for this stage.")

    def postprocess(self, model_output, **info_dict: object):
        """
        Postprocess the model output.
        Returns the postprocessed model output and the save dictionary.
        Args:
            model_output: The model output to postprocess.
        """
        raise NotImplementedError("Postprocess is not implemented for this stage.")
