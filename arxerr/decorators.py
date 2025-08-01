import functools
import time
from typing import Callable, Any

def traced(component: Callable) -> Callable:
    """Decorator for automatic execution tracing"""
    @functools.wraps(component)
    def wrapper(*args, **kwargs):
        # Ищем контекст в аргументах
        ctx = None
        for arg in args:
            if hasattr(arg, 'trace') and hasattr(arg, 'trace_call'):
                ctx = arg
                break
        
        if not ctx:
            return component(*args, **kwargs)
        
        # Всегда используем имя функции/компонента
        component_name = component.__name__
        
        ctx.trace_call(component_name, ctx.input)
        
        try:
            result = component(*args, **kwargs)
            output = result if result is not None else getattr(ctx, 'output', None)
            
            ctx.trace_return(output)
            return result
        except Exception as e:
            ctx.trace_error(e)
            raise
    return wrapper

def logged(component: Callable) -> Callable:
    """Decorator for automatic execution logging"""
    @functools.wraps(component)
    def wrapper(*args, **kwargs):
        # Ищем контекст в аргументах
        ctx = None
        for arg in args:
            if hasattr(arg, 'logs') and hasattr(arg, 'log'):
                ctx = arg
                break
        
        if not ctx:
            return component(*args, **kwargs)
        
        # Всегда используем имя функции/компонента
        component_name = component.__name__
        
        ctx.log(f"Entering {component_name} with input: {ctx.input}")
        
        start_time = time.time()
        try:
            result = component(*args, **kwargs)
            duration = time.time() - start_time
            output = result if result is not None else getattr(ctx, 'output', None)
            
            ctx.log(f"Completed {component_name} in {duration:.4f}s. Output: {output}")
            return result
        except Exception as e:
            ctx.log(f"Error in {component_name}: {str(e)}", "ERROR")
            raise
    return wrapper