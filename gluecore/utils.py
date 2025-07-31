from typing import List, Tuple, Callable

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
    func._is_gluecore_component = True
    return func