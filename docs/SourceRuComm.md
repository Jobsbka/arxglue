"""
GLUE - Минималистичный интерфейс композиции
Version: 1.0
"""

from typing import Any, Callable, Optional, Union

# 1. Ядро: только фундаментальные примитивы
Component = Callable[[Any], Any]  # Любой вызываемый объект

def connect(
    source: Union[Component, tuple[Component, ...]], 
    target: Union[Component, tuple[Component, ...]],
    transformer: Optional[Callable[[Any], Any]] = None
) -> tuple:
    """Декларация связи между компонентами. Возвращает описатель."""
    return (source, target, transformer)

# 2. Контекст как протокол (полностью опционален)
class ContextProtocol:
    input: Any
    output: Optional[Any]
    state: dict
    
    def __init__(self, input_data: Any):
        self.input = input_data
        self.output = None
        self.state = {}

# 3. Минималистичный исполнитель (опциональный пример)
def execute_linear(
    components: list[Component], 
    input_data: Any
) -> Any:
    """Последовательное выполнение без связей (пример)"""
    result = input_data
    for comp in components:
        result = comp(result)
    return result
