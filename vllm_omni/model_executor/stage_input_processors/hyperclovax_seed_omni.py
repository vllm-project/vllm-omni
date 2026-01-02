from typing import List, Union
import torch
from vllm_omni.inputs.data import OmniTokensPrompt

def thinker2code2wav(
    stage_list,
    engine_input_source,
    prompt=None,
    requires_multimodal_data: bool = False,
):
    """
    Process output from Thinker (LLM) stage and prepare input for Code2Wav stage.
    Assumes Thinker output contains audio tokens that need to be decoded.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    
    thinker_outputs = stage_list[source_stage_id].engine_outputs
    code2wav_inputs = []
    
    # Iterate over batch
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        # Get generated tokens
        token_ids = output.token_ids
        
        # Filter for audio tokens if they are mixed with text.
        # For now, we assume ALL generated tokens are passed to code2wav 
        # or that there's a specific range/mask. 
        # In a real implementation, we'd filter based on token IDs (e.g. > specific ID).
        audio_tokens = token_ids 
        
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_tokens,
                multi_modal_data=None # Audio generation usually doesn't need MM input again
            )
        )
        
    return code2wav_inputs
