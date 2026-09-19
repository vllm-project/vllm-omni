import pytest

from vllm_omni.model_executor.models.omnivoice.prompt_utils import validate_instruction

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestOmniVoiceInstructions:
    """Test OmniVoice instructions validation"""

    def test_english_instruction(self):
        """Test valid english instruction"""
        instructions = "female, young adult, australian accent"
        warning = validate_instruction(instructions)
        assert warning is None

    def test_chinese_instruction(self) -> None:
        """Test valid chinese instruction"""
        instructions = "男， 河南话"
        warning = validate_instruction(instructions)
        assert warning is None

    def test_free_form_instruction(self) -> None:
        """Test free form instruction rejected"""
        instructions = "happy little scottish boy"
        warning = validate_instruction(instructions)
        assert warning.startswith("Unsupported instruct items found")

    def test_conflicting_instruction(self) -> None:
        """Test conflicting instructs from the same category"""
        instructions = "male, teenager, middle-aged"
        warning = validate_instruction(instructions)
        assert warning.startswith("Conflicting instruct items within the same category")

    def test_non_existent_instruction_english(self) -> None:
        """Test non existent instruct rejected"""
        instructions = "male, english accent"
        warning = validate_instruction(instructions)
        assert warning.startswith("Unsupported instruct items found")

    def test_non_existent_instruction_chinese(self) -> None:
        """Test non existent instruct rejected"""
        instructions = "男生"
        warning = validate_instruction(instructions)
        assert warning.startswith("Unsupported instruct items found")

    def test_mixed_dialect_accent_instruction(self) -> None:
        """Test conflicting chinese dialect and english accent is handled"""
        instructions = "male, american accent, 河南话"
        warning = validate_instruction(instructions)
        assert warning.startswith("Cannot mix Chinese dialect and English accent")
