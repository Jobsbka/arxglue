"""
ArxGLUE Core - Minimalistic Component Composition Implementation
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