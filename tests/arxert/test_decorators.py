import pytest
from arxerr.decorators import traced, logged

def test_traced_decorator_success():
    class Context:
        def __init__(self):
            self.input = "test input"
            self.trace = []
            self._current_depth = 0
            
        def trace_call(self, name, input_data):
            self.trace.append({"name": name})
            
        def trace_return(self, output_data):
            self.trace[0]["output"] = output_data
    
    @traced
    def test_component(ctx):
        return "test output"
    
    ctx = Context()
    result = test_component(ctx)
    
    assert result == "test output"
    assert len(ctx.trace) == 1
    assert ctx.trace[0]["name"] == "test_component"  # Исправлено
    assert ctx.trace[0]["output"] == "test output"

def test_traced_decorator_error():
    class Context:
        def __init__(self):
            self.input = "test input"
            self.trace = []
            self._current_depth = 0
            
        def trace_call(self, name, input_data):
            self.trace.append({"name": name})
            
        def trace_error(self, error):
            self.trace[0]["error"] = str(error)
    
    @traced
    def test_component(ctx):
        raise ValueError("Test error")
    
    ctx = Context()
    with pytest.raises(ValueError):
        test_component(ctx)
    
    assert len(ctx.trace) == 1
    assert ctx.trace[0]["name"] == "test_component"  # Исправлено
    assert ctx.trace[0]["error"] == "Test error"

def test_logged_decorator_success():
    class Context:
        def __init__(self):
            self.input = "test input"
            self.logs = []
            
        def log(self, message, level="INFO"):
            self.logs.append(f"[{level}] {message}")
    
    @logged
    def test_component(ctx):
        return "test output"
    
    ctx = Context()
    result = test_component(ctx)
    
    assert result == "test output"
    assert len(ctx.logs) == 2
    assert "Entering test_component" in ctx.logs[0]
    assert "Completed test_component" in ctx.logs[1]

def test_logged_decorator_error():
    class Context:
        def __init__(self):
            self.input = "test input"
            self.logs = []
            
        def log(self, message, level="INFO"):
            self.logs.append(f"[{level}] {message}")
    
    @logged
    def test_component(ctx):
        raise RuntimeError("Test error")
    
    ctx = Context()
    with pytest.raises(RuntimeError):
        test_component(ctx)
    
    assert len(ctx.logs) == 2
    assert "Entering test_component" in ctx.logs[0]
    assert "Error in test_component" in ctx.logs[1]