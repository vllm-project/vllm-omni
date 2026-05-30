# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GLM-Image model __init__.py lazy import pattern.

The ``__init__.py`` uses ``__getattr__`` for lazy loading to avoid importing
``transformers.models.glm_image`` at module init, which may not be available
in all environments.
"""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestLazyImport:
    """Test the __getattr__ lazy import pattern in glm_image __init__.py."""

    def test_getattr_exists_and_is_callable(self):
        """Verify __getattr__ exists and is callable."""
        import vllm_omni.model_executor.models.glm_image as glm_image_pkg

        assert hasattr(glm_image_pkg, "__getattr__")
        assert callable(glm_image_pkg.__getattr__)

    def test_getattr_returns_class_for_known_attribute(self):
        """Verify __getattr__ returns GlmImageForConditionalGeneration for known attribute."""
        import vllm_omni.model_executor.models.glm_image as glm_image_pkg

        # Call __getattr__ directly to test the lazy import logic
        result = glm_image_pkg.__getattr__("GlmImageForConditionalGeneration")

        # Verify we get a class (the actual GlmImageForConditionalGeneration)
        assert result is not None
        assert isinstance(result, type)
        assert result.__name__ == "GlmImageForConditionalGeneration"

    def test_getattr_raises_for_unknown_attribute(self):
        """Verify __getattr__ raises AttributeError for unknown attributes."""
        import vllm_omni.model_executor.models.glm_image as glm_image_pkg

        # Test unknown attribute via __getattr__ directly
        with pytest.raises(AttributeError, match="has no attribute"):
            glm_image_pkg.__getattr__("UnknownClass")

    def test___all___exports_correct_symbols(self):
        """Verify __all__ contains the expected exported symbols."""
        import vllm_omni.model_executor.models.glm_image as glm_image_pkg

        assert hasattr(glm_image_pkg, "__all__")
        assert "GlmImageForConditionalGeneration" in glm_image_pkg.__all__


class TestLazyImportDoesNotImportTransformersAtInit:
    """Verify that importing the package does not eagerly load transformers."""

    def test_glm_image_module_has_getattr(self):
        """Test that the module has __getattr__ for lazy loading."""
        import vllm_omni.model_executor.models.glm_image as glm_image_pkg

        # The module should have __getattr__
        assert hasattr(glm_image_pkg, "__getattr__")
        assert callable(glm_image_pkg.__getattr__)

        # And __all__
        assert hasattr(glm_image_pkg, "__all__")

    def test_module_has_proper_structure(self):
        """Test that the module has proper Python module structure."""
        import vllm_omni.model_executor.models.glm_image as glm_image_pkg

        # Should be a module
        assert hasattr(glm_image_pkg, "__name__")
        assert glm_image_pkg.__name__ == "vllm_omni.model_executor.models.glm_image"

        # Should have __file__ (even if it's a package)
        assert hasattr(glm_image_pkg, "__file__")
