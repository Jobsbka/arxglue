from arxglue import Component
from arxerr import ErrorContext, traced, logged
import pytest

class DataLoader(Component):
    @traced
    @logged
    def __call__(self, ctx):
        ctx.output = [1, 2, 3]

class DataProcessor(Component):
    @traced
    @logged
    def __call__(self, ctx):
        if ctx.input is None:
            raise ValueError("Input data is missing")
        ctx.output = [x * 2 for x in ctx.input]

class DataSaver(Component):
    @traced
    @logged
    def __call__(self, ctx):
        if ctx.input is None:
            raise ValueError("Input data is missing")
        ctx.state["result"] = ctx.input

def test_arxglue_integration():
    loader = DataLoader()
    processor = DataProcessor()
    saver = DataSaver()
    
    ctx = ErrorContext(None)
    
    # Выполним компоненты
    loader(ctx)
    assert ctx.output == [1, 2, 3]
    
    # Передадим данные следующему компоненту
    ctx.input = ctx.output
    ctx.output = None
    
    processor(ctx)
    assert ctx.output == [2, 4, 6]
    
    ctx.input = ctx.output
    ctx.output = None
    
    saver(ctx)
    assert ctx.state["result"] == [2, 4, 6]
    
    # Проверим логи
    assert len(ctx.logs) >= 6  # 2 сообщения на компонент
    
    # Проверим, что логи содержат нужные сообщения
    assert any("Entering __call__" in log for log in ctx.logs)
    assert any("Completed __call__" in log for log in ctx.logs)
    
    # Проверим трассировку
    assert len(ctx.trace) == 3
    assert ctx.trace[0]["component"] == "__call__"
    assert ctx.trace[0]["input"] is None
    assert ctx.trace[0]["output"] == [1, 2, 3]
    
    assert ctx.trace[1]["component"] == "__call__"
    assert ctx.trace[1]["input"] == [1, 2, 3]
    assert ctx.trace[1]["output"] == [2, 4, 6]
    
    assert ctx.trace[2]["component"] == "__call__"
    assert ctx.trace[2]["input"] == [2, 4, 6]