import enum
import time
from typing import Optional, Dict, Any
from vllm.v1.request import Request as vLLMRequest

class OmniRequest(vLLMRequest):
    # pass