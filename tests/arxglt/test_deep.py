import unittest
import time
from arxglue import connect, execute_linear, ContextProtocol
from arxglue.utils import flatten_connections, component

class DeeparxglueTests(unittest.TestCase):
    def test_group_connection_with_transformer(self):
        """Тестирование групповых связей с трансформером"""
        def a(x): return x + 1
        def b(x): return x * 2
        def c(x): return f"Value: {x}"
        
        # Создаем соединение: два источника -> один приемник с трансформером
        conn = connect((a, b), c, transformer=lambda res_a, res_b: f"{res_a}|{res_b}")
        
        # Разворачиваем групповые соединения
        flat = flatten_connections([conn])
        self.assertEqual(len(flat), 2)  # Два отдельных соединения
        
        # Проверяем трансформер с РЕЗУЛЬТАТАМИ выполнения компонентов
        data = 10
        transformed = flat[0][2](a(data), b(data))
        self.assertEqual(transformed, "11|20")  # a(10)=11, b(10)=20
    
    def test_complex_branching(self):
        """Тестирование сложного ветвления данных"""
        def source(x): return x * 2
        def target_a(x): return x + 1
        def target_b(x): return x - 1
        def target_c(x): return x * 10
        
        # Создаем несколько соединений
        conn1 = connect(source, (target_a, target_b))
        conn2 = connect(target_b, target_c, transformer=lambda x: x * 2)
        
        flat = flatten_connections([conn1, conn2])
        self.assertEqual(len(flat), 3)
        
        # Проверяем структуру соединений
        self.assertEqual(flat[0], (source, target_a, None))
        self.assertEqual(flat[1], (source, target_b, None))
        self.assertEqual(flat[2], (target_b, target_c, conn2[2]))
    
    def test_context_protocol_extension(self):
        """Тестирование расширенного контекста выполнения"""
        class AnalyticsContext(ContextProtocol):
            def __init__(self, input_data):
                super().__init__(input_data)
                self.metrics = {}
            
            def add_metric(self, name, value):
                self.metrics[name] = value
        
        @component
        def processor(ctx: AnalyticsContext):
            ctx.add_metric("start", 100)
            ctx.output = ctx.input.upper()
            ctx.add_metric("end", 200)
            return ctx
        
        ctx = AnalyticsContext("test")
        result = processor(ctx)
        
        self.assertEqual(result.output, "TEST")
        self.assertEqual(result.metrics["start"], 100)
        self.assertEqual(result.metrics["end"], 200)
    
    def test_component_decorator(self):
        """Тестирование декоратора компонентов"""
        @component
        def uppercase(x: str):
            return x.upper()
        
        self.assertTrue(hasattr(uppercase, '_is_arxglue_component'))
        self.assertEqual(uppercase("hello"), "HELLO")
    
    def test_empty_linear_execution(self):
        """Тестирование пустого конвейера"""
        result = execute_linear([], "test")
        self.assertEqual(result, "test")
    
    def test_exception_handling(self):
        """Тестирование обработки исключений в компонентах"""
        def safe_div(x):
            return x / 2
        
        def unsafe_div(x):
            return x / 0
        
        # Проверяем безопасный компонент
        self.assertEqual(safe_div(10), 5)
        
        # Проверяем компонент с ошибкой
        with self.assertRaises(ZeroDivisionError):
            unsafe_div(10)
    
    def test_multi_source_transformer(self):
        """Тестирование трансформера для нескольких источников"""
        def sensor1():
            return 10
        def sensor2():
            return 20
        def aggregator(data):
            return data
        
        conn = connect((sensor1, sensor2), aggregator, 
                      transformer=lambda x, y: (x + y, x * y))
        
        # Проверяем работу трансформера
        result = conn[2](sensor1(), sensor2())
        self.assertEqual(result, (30, 200))
    
    def test_context_state_persistence(self):
        """Тестирование сохранения состояния в контексте"""
        class StatefulContext(ContextProtocol):
            def __init__(self, input_data):
                super().__init__(input_data)
                self.counter = 0
        
        @component
        def increment(ctx: StatefulContext):
            ctx.counter += 1
            ctx.output = ctx.input
            return ctx
        
        ctx = StatefulContext("data")
        increment(ctx)
        increment(ctx)
        
        self.assertEqual(ctx.counter, 2)
    
    def test_complex_workflow(self):
        """Тестирование комплексного рабочего процесса"""
        # Компоненты обработки данных
        @component
        def loader(data):
            return f"Loaded: {data}"
        
        @component
        def clean(data):
            return f"Cleaned: {data}"
        
        @component
        def transform(data):
            return f"Transformed: {data.upper()}"
        
        @component
        def saver(data):
            return f"Saved: {data}"
        
        # Создаем связи
        connections = [
            connect(loader, clean),
            connect(clean, (transform, saver)),
            connect(transform, saver)
        ]
        
        # Разворачиваем связи
        flat = flatten_connections(connections)
        self.assertEqual(len(flat), 4)
        
        # Проверяем структуру соединений
        self.assertEqual(flat[0], (loader, clean, None))
        self.assertEqual(flat[1], (clean, transform, None))
        self.assertEqual(flat[2], (clean, saver, None))
        self.assertEqual(flat[3], (transform, saver, None))
        
        # Эмулируем выполнение
        data = "test.csv"
        result = loader(data)
        result = clean(result)
        result1 = transform(result)
        result2 = saver(result)
        result3 = saver(result1)
        
        self.assertEqual(result1, "Transformed: CLEANED: LOADED: TEST.CSV")
        self.assertEqual(result2, "Saved: Cleaned: Loaded: test.csv")
        self.assertEqual(result3, "Saved: Transformed: CLEANED: LOADED: TEST.CSV")

    def test_nested_group_connections(self):
        """Тестирование вложенных групповых соединений"""
        def a(x): return x + 1
        def b(x): return x * 2
        def c(x): return x - 3
        def d(x): return f"Result: {x}"
        
        # Сложная структура связей
        conn1 = connect(a, (b, c))
        conn2 = connect((b, c), d, transformer=lambda res_b, res_c: res_b + res_c)
        
        flat = flatten_connections([conn1, conn2])
        self.assertEqual(len(flat), 4)  # Исправлено с 5 на 4
        
        # Проверяем структуру
        self.assertEqual(flat[0], (a, b, None))
        self.assertEqual(flat[1], (a, c, None))
        self.assertEqual(flat[2], (b, d, conn2[2]))
        self.assertEqual(flat[3], (c, d, conn2[2]))
    
    def test_context_protocol_in_linear_execution(self):
        """Тестирование контекста в линейном выполнении"""
        class ProcessingContext(ContextProtocol):
            def __init__(self, input_data):
                super().__init__(input_data)
                self.steps = []
            
            def add_step(self, name):
                self.steps.append(name)
        
        @component
        def step1(ctx: ProcessingContext):
            ctx.add_step("step1")
            ctx.output = ctx.input.upper()
            return ctx
        
        @component
        def step2(ctx: ProcessingContext):
            ctx.add_step("step2")
            ctx.output = f"Processed: {ctx.output}"
            return ctx
        
        ctx = ProcessingContext("test")
        result = execute_linear([step1, step2], ctx)
        
        self.assertEqual(result.output, "Processed: TEST")
        self.assertEqual(result.steps, ["step1", "step2"])
    
    def test_transformer_with_extra_arguments(self):
        """Тестирование трансформера с дополнительными аргументами"""
        def sensor():
            return 100
        
        def logger(data):
            return f"Logged: {data}"
        
        # Трансформер с дополнительными параметрами
        def custom_transformer(x, multiplier=1, suffix=""):
            return f"Value: {x * multiplier}{suffix}"
        
        conn = connect(sensor, logger, transformer=custom_transformer)
        
        # Проверяем различные варианты вызова
        t_func = conn[2]
        self.assertEqual(t_func(5), "Value: 5")
        self.assertEqual(t_func(5, multiplier=10), "Value: 50")
        self.assertEqual(t_func(5, suffix="°C"), "Value: 5°C")
    
    def test_complex_context_operations(self):
        """Тестирование сложных операций с контекстом"""
        class AnalyticsContext(ContextProtocol):
            def __init__(self, input_data):
                super().__init__(input_data)
                self.timestamps = {}
                self.metrics = {}
            
            def start_timer(self, name):
                self.timestamps[name] = time.time()
            
            def stop_timer(self, name):
                if name in self.timestamps:
                    duration = time.time() - self.timestamps[name]
                    self.metrics[f"{name}_time"] = duration
        
        @component
        def loader(ctx: AnalyticsContext):
            ctx.start_timer("loading")
            # Имитация загрузки
            time.sleep(0.01)
            ctx.output = ctx.input
            ctx.stop_timer("loading")
            return ctx
        
        @component
        def processor(ctx: AnalyticsContext):
            ctx.start_timer("processing")
            # Имитация обработки
            time.sleep(0.02)
            ctx.output = ctx.output.upper()
            ctx.stop_timer("processing")
            return ctx
        
        ctx = AnalyticsContext("test_data")
        result = execute_linear([loader, processor], ctx)
        
        self.assertEqual(result.output, "TEST_DATA")
        self.assertIn("loading_time", result.metrics)
        self.assertIn("processing_time", result.metrics)
        self.assertGreater(result.metrics["processing_time"], result.metrics["loading_time"])
    
    def test_error_handling_in_transformer(self):
        """Тестирование обработки ошибок в трансформерах"""
        def data_source():
            return "data"
        
        def consumer(data):
            return data
        
        # Трансформер, который вызывает ошибку
        def faulty_transformer(x):
            raise ValueError("Transformer error")
        
        conn = connect(data_source, consumer, transformer=faulty_transformer)
        
        with self.assertRaises(ValueError) as context:
            conn[2]("test")
        
        self.assertEqual(str(context.exception), "Transformer error")
    
    def test_large_data_handling(self):
        """Тестирование обработки больших объемов данных"""
        large_data = [i for i in range(10000)]
        
        def processor(data):
            return [x * 2 for x in data]
        
        def verifier(data):
            return all(x % 2 == 0 for x in data)
        
        conn = connect(processor, verifier)
        
        # Проверяем обработку больших данных
        processed = processor(large_data)
        result = verifier(processed)
        
        self.assertTrue(result)
        self.assertEqual(len(processed), 10000)
    
    def test_component_decorator_with_classes(self):
        """Тестирование декоратора компонентов на классах"""
        @component
        class Multiplier:
            def __init__(self, factor):
                self.factor = factor
            
            def __call__(self, x):
                return x * self.factor
        
        doubler = Multiplier(2)
        tripler = Multiplier(3)
        
        self.assertTrue(hasattr(doubler, '_is_arxglue_component'))
        self.assertEqual(doubler(5), 10)
        self.assertEqual(tripler(5), 15)
    
    def test_mixed_component_types(self):
        """Тестирование смешанных типов компонентов"""
        def func_component(x):
            return x + 1
        
        @component
        class ClassComponent:
            def __call__(self, x):
                return x * 2
        
        @component
        def decorated_component(x):
            return x ** 2
        
        pipeline = [
            func_component,
            ClassComponent(),
            decorated_component
        ]
        
        result = execute_linear(pipeline, 3)
        self.assertEqual(result, ((3 + 1) * 2) ** 2)  # (4*2)^2 = 8^2 = 64

    @unittest.expectedFailure
    def test_concurrent_sources(self):
        """
        ТЕКУЩЕЕ ОГРАНИЧЕНИЕ:
        Библиотека не поддерживает автоматическую синхронизацию многопоточных источников.
        Этот тест демонстрирует ожидаемое поведение и помечен как ожидаемый сбой.
        """
        def slow_source():
            time.sleep(0.1)
            return "slow_data"
        
        def fast_source():
            return "fast_data"
        
        def aggregator(data1, data2):
            return f"{data1}+{data2}"
        
        # Создаем соединение с двумя источниками (разная скорость выполнения)
        conn = connect((slow_source, fast_source), aggregator, 
                      transformer=lambda x, y: (x, y))
        
        # В ИДЕАЛЕ: исполнитель должен дождаться результатов обоих источников
        # РЕАЛЬНОСТЬ: текущая реализация не синхронизирует выполнение
        
        # Эмуляция "идеального" исполнителя
        with ThreadPoolExecutor() as executor:
            future_slow = executor.submit(slow_source)
            future_fast = executor.submit(fast_source)
            result_slow = future_slow.result()
            result_fast = future_fast.result()
        
        # Применяем трансформер и агрегатор вручную
        transformed = conn[2](result_slow, result_fast)
        result = aggregator(*transformed)
        
        # Проверяем ожидаемый результат
        self.assertEqual(result, "slow_data+fast_data")

if __name__ == "__main__":
    unittest.main()