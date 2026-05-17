from __future__ import annotations

from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer

try:
    from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
except ImportError:

    class Gemma4Proposer:
        """Compatibility placeholder for vLLM wheels without Gemma 4 proposer."""

        pass
