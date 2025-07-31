"""
GlueCore - Minimalistic Component Composition Interface
Version: 3.1
"""

from typing import Any, Callable, Optional, Union

# 1. Core primitives
Component = Callable[[Any], Any]  # Any callable is a component

def connect(
    source: Union[Component, tuple[Component, ...]], 
    target: Union[Component, tuple[Component, ...]],
    transformer: Optional[Callable[[Any], Any]] = None
) -> tuple:
    """
    Declares a connection between components
    
    :param source: Source component(s)
    :param target: Target component(s)
    :param transformer: Optional data transformation function
    :return: Connection descriptor tuple
    """
    return (source, target, transformer)

# 2. Optional Context Protocol
class ContextProtocol:
    """
    Optional execution context protocol
    Usage:
        class MyContext(ContextProtocol):
            ...
    """
    input: Any
    output: Optional[Any]
    state: dict
    
    def __init__(self, input_data: Any):
        self.input = input_data
        self.output = None
        self.state = {}

# 3. Optional linear executor
def execute_linear(
    components: list[Component], 
    input_data: Any
) -> Any:
    """
    Sequential component execution (example)
    
    :param components: List of components to execute
    :param input_data: Input data
    :return: Processing result
    """
    result = input_data
    for comp in components:
        result = comp(result)
    return result