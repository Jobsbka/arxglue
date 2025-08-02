import unittest
import asyncio
from arxglue import connect, execute_linear, ContextProtocol
from arxglue.utils import flatten_connections
import sys

class TestLibraryIntegrations(unittest.TestCase):
    def test_requests_integration(self):
        """Интеграция с библиотекой requests"""
        try:
            import requests
        except ImportError:
            self.skipTest("Библиотека requests не установлена")
        
        # Создаем компоненты на основе requests
        def fetch_data(url):
            response = requests.get(url)
            return response.json()
        
        def process_data(data):
            return {
                'status': 'success',
                'data': {k: v for k, v in data.items() if k != 'headers'}
            }
        
        # Создаем конвейер
        pipeline = [fetch_data, process_data]
        
        # Выполняем с тестовым URL
        result = execute_linear(pipeline, 'https://httpbin.org/get?test=arxglue')
        
        # Проверяем результат
        self.assertEqual(result['status'], 'success')
        self.assertIn('args', result['data'])
        self.assertEqual(result['data']['args'], {'test': 'arxglue'})
    
    def test_pandas_integration(self):
        """Интеграция с библиотекой pandas"""
        try:
            import pandas as pd
            import numpy as np
        except ImportError:
            self.skipTest("Библиотека pandas не установлена")
        
        # Создаем DataFrame-компонент
        def create_dataframe(_):
            return pd.DataFrame({
                'A': [1, 2, 3],
                'B': [4, 5, 6]
            })
        
        # Компонент обработки данных
        def process_data(df):
            return df.assign(C=df.A + df.B)
        
        # Компонент анализа
        def analyze_data(df):
            return {
                'mean_A': df.A.mean(),
                'sum_C': df.C.sum()
            }
        
        # Создаем конвейер
        pipeline = [create_dataframe, process_data, analyze_data]
        result = execute_linear(pipeline, None)
        
        # Проверяем результат
        self.assertEqual(result['mean_A'], 2.0)
        self.assertEqual(result['sum_C'], 21)
    
    def test_numpy_integration(self):
        """Интеграция с библиотекой numpy"""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("Библиотека numpy не установлена")
        
        # Компоненты обработки массивов
        def create_array(size):
            return np.random.rand(size)
        
        def transform_array(arr):
            return arr * 100
        
        def analyze_array(arr):
            return {
                'mean': np.mean(arr),
                'max': np.max(arr),
                'min': np.min(arr)
            }
        
        # Создаем конвейер
        pipeline = [create_array, transform_array, analyze_array]
        result = execute_linear(pipeline, 1000)
        
        # Проверяем результат с реалистичными допусками
        self.assertAlmostEqual(result['mean'], 50, delta=2.0)
        self.assertGreaterEqual(result['min'], 0)
        self.assertLessEqual(result['max'], 100)
        self.assertLess(result['min'], 5.0)  # Минимум обычно <5
        self.assertGreater(result['max'], 95.0)  # Максимум обычно >95
    
    def test_async_execution(self):
        """Интеграция с asyncio"""
        # Асинхронные компоненты
        async def async_fetch(url):
            await asyncio.sleep(0.01)
            return f"Data from {url}"
        
        async def async_process(data):
            await asyncio.sleep(0.01)
            return data.upper()
        
        async def async_save(data):
            await asyncio.sleep(0.01)
            return f"SAVED: {data}"
        
        # Создаем асинхронный исполнитель
        async def async_execute_linear(components, input_data):
            result = input_data
            for comp in components:
                result = await comp(result)
            return result
        
        # Создаем конвейер
        pipeline = [async_fetch, async_process, async_save]
        
        # Выполняем асинхронно
        result = asyncio.run(async_execute_linear(pipeline, "https://example.com"))
        
        # Проверяем результат
        self.assertEqual(result, "SAVED: DATA FROM HTTPS://EXAMPLE.COM")
    
    def test_async_with_context(self):
        """Асинхронный контекст выполнения"""
        class AsyncContext(ContextProtocol):
            def __init__(self, input_data):
                super().__init__(input_data)
                self.log = []
            
            async def add_log(self, message):
                await asyncio.sleep(0.001)
                self.log.append(message)
        
        # Асинхронный компонент с контекстом
        async def async_component1(ctx: AsyncContext):
            await ctx.add_log("Component 1 start")
            ctx.output = ctx.input + "_processed1"
            await ctx.add_log("Component 1 end")
            return ctx
        
        async def async_component2(ctx: AsyncContext):
            await ctx.add_log("Component 2 start")
            ctx.output = ctx.output.upper()
            await ctx.add_log("Component 2 end")
            return ctx
        
        # Асинхронный исполнитель для контекста
        async def async_context_executor(components, context):
            ctx = context
            for comp in components:
                ctx = await comp(ctx)
            return ctx
        
        # Создаем и выполняем конвейер
        ctx = AsyncContext("test_data")
        result = asyncio.run(async_context_executor(
            [async_component1, async_component2], 
            ctx
        ))
        
        # Проверяем результат
        self.assertEqual(result.output, "TEST_DATA_PROCESSED1")
        self.assertEqual(result.log, [
            "Component 1 start",
            "Component 1 end",
            "Component 2 start",
            "Component 2 end"
        ])
    
    def test_celery_integration(self):
        """Интеграция с Celery для распределенных задач"""
        try:
            from celery import Celery
        except ImportError:
            self.skipTest("Библиотека Celery не установлена")
        
        # Тестовое Celery-приложение с корректным бэкендом
        app = Celery('test_app', broker='memory://', backend='cache+memory://')
        
        # Создаем Celery-задачи как компоненты
        @app.task
        def celery_add(a, b):
            return a + b
        
        @app.task
        def celery_multiply(x, factor=2):
            return x * factor
        
        # Компонент-адаптер для Celery
        def celery_adapter(task, *args, **kwargs):
            def wrapper(input_data):
                # Передаем входные данные как первый аргумент
                return task.apply(args=(input_data, *args), kwargs=kwargs).get()
            return wrapper
        
        # Создаем конвейер
        pipeline = [
            celery_adapter(celery_add, 10),      # input + 10
            celery_adapter(celery_multiply, factor=3)  # (input + 10) * 3
        ]
        
        # Выполняем
        result = execute_linear(pipeline, 5)
        
        # Проверяем результат
        self.assertEqual(result, (5 + 10) * 3)  # 45

if __name__ == "__main__":
    unittest.main()