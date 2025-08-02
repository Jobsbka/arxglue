import pytest
from arxerr.mixins import TracingMixin, LoggingMixin

class TestTracingMixin:
    def test_tracing_call(self):
        class Context(TracingMixin):
            pass
        
        ctx = Context()
        ctx.trace_call("TestComponent", {"data": 123})
        
        assert len(ctx.trace) == 1
        entry = ctx.trace[0]
        assert entry["component"] == "TestComponent"
        assert entry["input"] == {"data": 123}
        assert entry["depth"] == 0
        assert entry["start_time"] is not None
        assert entry["end_time"] is None

    def test_tracing_return(self):
        class Context(TracingMixin):
            pass
        
        ctx = Context()
        ctx.trace_call("TestComponent", {"data": 123})
        ctx.trace_return({"result": 456})
        
        entry = ctx.trace[0]
        assert entry["output"] == {"result": 456}
        assert entry["end_time"] is not None
        assert entry["success"] is True

    def test_tracing_error(self):
        class Context(TracingMixin):
            pass
        
        ctx = Context()
        ctx.trace_call("TestComponent", {"data": 123})
        ctx.trace_error(ValueError("Test error"))
        
        entry = ctx.trace[0]
        assert entry["error"] == "Test error"
        assert entry["success"] is False

class TestLoggingMixin:
    def test_logging(self):
        class Context(LoggingMixin):
            pass
        
        ctx = Context()
        ctx.log("Info message")
        ctx.log("Error message", "ERROR")
        
        assert len(ctx.logs) == 2
        assert ctx.logs[0] == "[INFO] Info message"
        assert ctx.logs[1] == "[ERROR] Error message"