"""
ArxGLUE - Minimalistic Component Composition Interface
Version: 1.0
"""


from .core import Component, connect, ContextProtocol
from .utils import execute_linear, flatten_connections, component


__all__ = [
    'Component',
    'connect',
    'ContextProtocol',
    'execute_linear',
    'flatten_connections',
    'component'
]