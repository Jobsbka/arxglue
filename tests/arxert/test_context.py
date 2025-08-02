from arxerr.context import ErrorContext

def test_error_context_basic():
    ctx = ErrorContext("input data")
    
    assert ctx.input == "input data"
    assert ctx.output is None
    assert ctx.state == {}
    assert ctx.logs == []
    assert ctx.trace == []

def test_error_context_logging():
    ctx = ErrorContext("input data")
    ctx.log("Test message")
    
    assert len(ctx.logs) == 1
    assert "Test message" in ctx.logs[0]

def test_error_context_tracing():
    ctx = ErrorContext("input data")
    ctx.trace_call("Component", "input")
    ctx.trace_return("output")
    
    assert len(ctx.trace) == 1
    assert ctx.trace[0]["component"] == "Component"
    assert ctx.trace[0]["input"] == "input"
    assert ctx.trace[0]["output"] == "output"