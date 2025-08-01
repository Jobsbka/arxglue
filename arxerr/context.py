"""
Enhanced context implementations
"""
from typing import Any
from .mixins import TracingMixin, LoggingMixin

class ErrorContext(TracingMixin, LoggingMixin):
    """
    Full-featured execution context with tracing and logging
    Compatible with ArxGLUE systems
    """
    def __init__(self, input_data: Any):
        super().__init__()
        self.input = input_data
        self.output = None
        self.state = {}