import enum
import time
from typing import Optional, Dict, Any


Class Request:
    # initialize request
    def __init__(self, request_id: str, request_type: str, request_data: Dict[str, Any]):
        self.request_id = request_id
        self.request_type = request_type
        self.request_data = request_data
        self.request_time = time.time()
        self.request_status = RequestStatus.PENDING
        self.request_result = None
        self.request_error = None
        self.request_time = time.time()