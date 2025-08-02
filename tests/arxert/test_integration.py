import pytest
from arxerr.context import ErrorContext
from arxerr.decorators import traced, logged

@traced
@logged
def sample_component(ctx):
    ctx.output = ctx.input * 2
    return ctx.output

@traced
@logged
def error_component(ctx):
    raise ValueError("Calculation error")

def test_component_integration():
    ctx = ErrorContext(5)
    result = sample_component(ctx)
    
    assert result == 10
    assert ctx.output == 10
    
    # Проверяем логи
    assert len(ctx.logs) == 2
    assert "Entering sample_component" in ctx.logs[0]
    assert "Completed sample_component" in ctx.logs[1]
    
    # Проверяем трассировку
    assert len(ctx.trace) == 1
    trace_entry = ctx.trace[0]
    assert trace_entry["component"] == "sample_component"
    assert trace_entry["input"] == 5
    assert trace_entry["output"] == 10
    assert trace_entry["success"] is True

def test_error_handling_integration():
    ctx = ErrorContext(5)
    
    with pytest.raises(ValueError):
        error_component(ctx)
    
    # Проверяем логи
    assert len(ctx.logs) == 2
    assert "Entering error_component" in ctx.logs[0]
    assert "Error in error_component" in ctx.logs[1]
    
    # Проверяем трассировку
    assert len(ctx.trace) == 1
    trace_entry = ctx.trace[0]
    assert trace_entry["component"] == "error_component"
    assert "Calculation error" in trace_entry["error"]
    assert trace_entry["success"] is False