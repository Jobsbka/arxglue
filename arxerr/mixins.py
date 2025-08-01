import time
from typing import Any, Dict, List

class TracingMixin:
    """Mixin for execution tracing capabilities"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace: List[Dict[str, Any]] = []
        self._current_depth = 0

    def trace_call(self, component_name: str, input_data: Any):
        """Record component execution start"""
        self.trace.append({
            "component": component_name,
            "depth": self._current_depth,
            "input": input_data,
            "start_time": time.time(),
            "end_time": None,
            "success": None,
            "error": None
        })
        self._current_depth += 1

    def trace_return(self, output_data: Any = None):
        """Record successful component completion"""
        if self.trace:
            self._current_depth -= 1
            self.trace[-1].update({
                "output": output_data,
                "end_time": time.time(),
                "success": True
            })

    def trace_error(self, error: Exception):
        """Record component execution error"""
        if self.trace:
            self._current_depth -= 1
            self.trace[-1].update({
                "end_time": time.time(),
                "success": False,
                "error": str(error)
            })

class LoggingMixin:
    """Mixin for execution logging capabilities"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logs: List[str] = []
    
    def log(self, message: str, level: str = "INFO"):
        """Add log entry to context"""
        self.logs.append(f"[{level}] {message}")