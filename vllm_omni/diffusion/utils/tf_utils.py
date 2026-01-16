import inspect
from typing import Any

from vllm_omni.diffusion.data import TransformerConfig


def get_transformer_config_kwargs(
    tf_model_config: TransformerConfig, model_class: type[Any] | None = None
) -> dict[str, Any]:
    """
    This function extracts parameters from a TransformerConfig instance and filters out internal
    diffusers metadata keys (those starting with '_') that should not be passed to model initialization.
    Also filters out parameters that are not accepted by the model's __init__ method (e.g., pooled_projection_dim
    for QwenImageTransformer2DModel).

    This uses inspect.signature to dynamically detect accepted parameters, making it general for any model class.
    Similar to how diffusers' @register_to_config decorator works.

    Args:
        tf_model_config: TransformerConfig instance containing model parameters
        model_class: Optional model class to inspect for accepted __init__ parameters.
                   If None, all non-internal parameters are returned (backward compatibility).

    Returns:
        dict: Filtered dictionary of parameters suitable for transformer model initialization
    """
    # Extract transformer config parameters, filtering out internal diffusers metadata
    # TransformerConfig stores params in a 'params' dict, and we need to exclude
    # internal keys like '_class_name' and '_diffusers_version'
    tf_config_params = tf_model_config.to_dict()

    # Filter out internal diffusers metadata keys that start with '_'
    filtered_params = {k: v for k, v in tf_config_params.items() if not k.startswith("_")}

    # If model_class is provided, use inspect.signature to get accepted parameters
    if model_class is not None:
        try:
            # Get the signature of the model's __init__ method
            sig = inspect.signature(model_class.__init__)
            # Get all parameter names (excluding 'self' and special parameters)
            accepted_params = {
                name
                for name, param in sig.parameters.items()
                if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD  # Exclude **kwargs
            }

            # Filter to only include parameters that are in the model's signature
            filtered_params = {k: v for k, v in filtered_params.items() if k in accepted_params}
        except (TypeError, AttributeError):
            # If inspection fails, fall back to returning all non-internal params
            # This maintains backward compatibility
            pass

    return filtered_params
