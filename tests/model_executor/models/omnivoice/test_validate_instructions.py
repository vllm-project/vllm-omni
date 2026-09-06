import pytest

from vllm_omni.model_executor.models.omnivoice.prompt_utils import validate_instruction

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestOmniVoiceInstructions:
    """Test OmniVoice instructions validation"""

    def test_english_instruction(self):
        """Test valid english instruction"""
        instructions = "female, young adult, australian accent"
        validate_instruction(instructions)

    def test_chinese_instruction(self) -> None:
        """Test valid chinese instruction"""
        instructions = "男， 河南话"
        validate_instruction(instructions)

    def test_free_form_instruction(self) -> None:
        """Test free form instruction raises warning"""
        instructions = "happy little scottish boy"

        with pytest.raises(SyntaxWarning, match="Unsupported instruct items found"):
            validate_instruction(instructions)

    def test_conflicting_instruction(self) -> None:
        """Test conflicting instructs from the same category"""
        instructions = "male, teenager, middle-aged"

        with pytest.raises(SyntaxWarning, match="Conflicting instruct items within the same category"):
            validate_instruction(instructions)

    def test_non_existent_instruction_english(self) -> None:
        """Test non existent instruct raises warning"""
        instructions = "male, english accent"

        with pytest.raises(SyntaxWarning, match="Unsupported instruct items found"):
            validate_instruction(instructions)

    def test_non_existent_instruction_chinese(self) -> None:
        """Test non existent instruct rejected"""
        # TODO ask a native speaker to provide and validate this case
        return

    def test_mixed_dialect_accent_instruction(self) -> None:
        """Test conflicting chinese dialect and english accent is handled"""
        instructions = "male, american accent, 河南话"

        with pytest.raises(SyntaxWarning, match="Cannot mix Chinese dialect and English accent"):
            validate_instruction(instructions)
