#NOT USE THIS MODULE IN YOUR PROJECTS THIS MODULE WILL DELETED IN NEXT REV JUST FOR TESTS!!!

from typing import Any, Callable, List, Tuple

def flatten_connections(connections: list) -> List[Tuple]:
    """
    Flattens group connections into 1:1 connections
    
    :param connections: List of connection descriptors
    :return: Flat list of (source, target, transformer) tuples
    """
    flattened = []
    for conn in connections:
        sources = conn[0] if isinstance(conn[0], tuple) else (conn[0],)
        targets = conn[1] if isinstance(conn[1], tuple) else (conn[1],)
        
        for src in sources:
            for tgt in targets:
                flattened.append((src, tgt, conn[2]))
    return flattened

def component(func: Callable) -> Callable:
    """
    Component decorator (optional)
    
    :param func: Component function
    :return: Marked component function
    """
    func._is_arxglue_component = True
    return func

# Перенесён из core.py
def execute_linear(
    components: list[Callable[[Any], Any]], 
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