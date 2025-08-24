"""
GluePaddle - Модульная система для PaddlePaddle
с поддержкой портов, валидацией и компиляцией графов
"""

import paddle
from typing import Any, Callable, Optional, Dict, List, Tuple, Union, Set, Type
import hashlib
import inspect
from collections import defaultdict
from dataclasses import dataclass
import math
from enum import Enum
import warnings
import json
import time
from contextlib import contextmanager

class PaddleAggregationStrategy(Enum):
    """Стратегии агрегации входных данных для портов"""
    CONCAT = "concat"  # Конкатенация по последнему измерению
    SUM = "sum"        # Поэлементное суммирование
    MEAN = "mean"      # Поэлементное усреднение
    MAX = "max"        # Поэлементный максимум
    STACK = "stack"    # Создание нового измерения
    CUSTOM = "custom"  # Пользовательская агрегация

class PaddlePortType(Enum):
    """Типы данных для портов"""
    TENSOR = "tensor"
    SCALAR = "scalar"
    SEQUENCE = "sequence"
    DICT = "dict"
    ANY = "any"

@dataclass
class PaddlePortSpec:
    """Спецификация порта с расширенной валидацией"""
    name: str
    type: PaddlePortType = PaddlePortType.TENSOR
    shape: Optional[Tuple[Optional[int]]] = None  # None для любых размерностей
    dtype: Optional[paddle.dtype] = None
    required: bool = True
    aggregation: PaddleAggregationStrategy = PaddleAggregationStrategy.CONCAT
    custom_aggregator: Optional[Callable[[List[Any]], Any]] = None
    
    def is_compatible_with(self, other: 'PaddlePortSpec') -> bool:
        """Проверяет совместимость с другим портом"""
        # Проверка типов
        if self.type != other.type and self.type != PaddlePortType.ANY and other.type != PaddlePortType.ANY:
            return False
        
        # Проверка dtype
        if self.dtype is not None and other.dtype is not None and self.dtype != other.dtype:
            return False
        
        # Проверка формы (если обе спецификации имеют форму)
        if self.shape is not None and other.shape is not None:
            if len(self.shape) != len(other.shape):
                return False
            
            for dim_self, dim_other in zip(self.shape, other.shape):
                if dim_self is not None and dim_other is not None and dim_self != dim_other:
                    return False
        
        return True
    
    def validate_value(self, value: Any) -> bool:
        """Валидирует значение по спецификации порта"""
        try:
            # Проверка типа
            if self.type == PaddlePortType.TENSOR and not isinstance(value, paddle.Tensor):
                return False
            elif self.type == PaddlePortType.SCALAR and not isinstance(value, (int, float, paddle.Tensor)):
                return False
            elif self.type == PaddlePortType.SEQUENCE and not isinstance(value, (list, tuple)):
                return False
            elif self.type == PaddlePortType.DICT and not isinstance(value, dict):
                return False
            
            # Проверка dtype для тензоров
            if isinstance(value, paddle.Tensor) and self.dtype is not None and value.dtype != self.dtype:
                return False
            
            # Проверка формы для тензоров
            if isinstance(value, paddle.Tensor) and self.shape is not None:
                if len(value.shape) != len(self.shape):
                    return False
                
                for dim_value, dim_spec in zip(value.shape, self.shape):
                    if dim_spec is not None and dim_value != dim_spec:
                        return False
            
            return True
        except:
            return False

class PaddleComponent(paddle.nn.Layer):
    """Базовый компонент системы с поддержкой портов"""
    
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name or self.__class__.__name__
        self.input_ports: Dict[str, PaddlePortSpec] = self._define_input_ports()
        self.output_ports: Dict[str, PaddlePortSpec] = self._define_output_ports()
        self._compiled = False
        
    def _define_input_ports(self) -> Dict[str, PaddlePortSpec]:
        """Определяет входные порты компонента (переопределяется в подклассах)"""
        return {'default': PaddlePortSpec('default')}
    
    def _define_output_ports(self) -> Dict[str, PaddlePortSpec]:
        """Определяет выходные порты компонента (переопределяется в подклассах)"""
        return {'default': PaddlePortSpec('default')}
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Прямой проход с поддержкой множественных входов/выходов"""
        raise NotImplementedError("Components must implement forward method")
    
    def validate_ports(self) -> List[str]:
        """Проверка корректности определения портов"""
        errors = []
    
        # Проверяем, что все обязательные порты имеют спецификации
        for port_name, port_spec in self.input_ports.items():
            if port_spec.required and port_spec.type == PaddlePortType.ANY:
                warnings.warn(f"Input port {port_name} is required but has type ANY")
    
        # Проверяем сигнатуру forward метода только если не используется **inputs
        try:
            sig = inspect.signature(self.forward)
            forward_params = list(sig.parameters.keys())
        
            # Если метод использует **kwargs или **inputs, пропускаем проверку параметров
            has_var_keyword = any(
                param.kind == param.VAR_KEYWORD 
                for param in sig.parameters.values()
            )
        
            if not has_var_keyword:
                # Проверяем, что все входные порты есть в параметрах forward
                for port_name in self.input_ports:
                    if port_name not in forward_params and port_name != 'default':
                        errors.append(f"Input port {port_name} not found in forward method parameters")
        except:
            pass  # Пропускаем проверку, если невозможно получить сигнатуру
    
        return errors
    
    def compile(self):
        """Компиляция компонента (если нужна)"""
        self._compiled = True
        
    def reset(self):
        """Сброс состояния компонента (переопределяется в подклассах)"""
        pass
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, inputs={list(self.input_ports.keys())}, outputs={list(self.output_ports.keys())})"

@dataclass
class PaddleConnection:
    """Соединение между портами компонентов"""
    source: str  # format: "component_name.port_name"
    target: str  # format: "component_name.port_name"
    transformer: Optional[Callable] = None
    delay: int = 0  # Для рекуррентных соединений
    
    def __post_init__(self):
        if '.' not in self.source:
            self.source = f"{self.source}.default"
        if '.' not in self.target:
            self.target = f"{self.target}.default"
    
    @property
    def source_component(self) -> str:
        return self.source.split('.')[0]
    
    @property
    def source_port(self) -> str:
        return self.source.split('.')[1]
    
    @property
    def target_component(self) -> str:
        return self.target.split('.')[0]
    
    @property
    def target_port(self) -> str:
        return self.target.split('.')[1]
    
    def is_recurrent(self) -> bool:
        """Проверяет, является ли соединение рекуррентным"""
        return self.delay > 0
    
    def __repr__(self):
        delay_str = f", delay={self.delay}" if self.delay > 0 else ""
        return f"PaddleConnection({self.source} -> {self.target}{delay_str})"

class PaddleSharedMemory:
    """Общая память для быстрого обмена данными между компонентами"""
    
    def __init__(self):
        self._storage = {}
        self._access_times = {}
        self._max_size = 1000  # Максимальное количество элементов в памяти
        
    def set(self, key: str, value: Any):
        """Устанавливает значение в общей памяти"""
        self._storage[key] = value
        self._access_times[key] = time.time()
        self._cleanup()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Получает значение из общей памяти"""
        value = self._storage.get(key, default)
        if key in self._storage:
            self._access_times[key] = time.time()
        return value
    
    def delete(self, key: str):
        """Удаляет значение из общей памяти"""
        if key in self._storage:
            del self._storage[key]
            del self._access_times[key]
    
    def clear(self):
        """Очищает всю общую память"""
        self._storage.clear()
        self._access_times.clear()
    
    def _cleanup(self):
        """Очищает старые элементы при превышении лимита"""
        if len(self._storage) > self._max_size:
            # Удаляем самые старые элементы
            sorted_keys = sorted(self._access_times.keys(), key=lambda k: self._access_times[k])
            for key in sorted_keys[:len(self._storage) - self._max_size]:
                self.delete(key)

class GluePaddle(paddle.nn.Layer):
    """Модульная система для построения нейросетевых архитектур на PaddlePaddle"""
    
    def __init__(self):
        super().__init__()
        self.components: Dict[str, PaddleComponent] = {}
        self.connections: List[PaddleConnection] = []
        self.execution_order: List[str] = []
        self._compiled = False
        self._connection_map: Dict[str, List[PaddleConnection]] = defaultdict(list)
        self._recurrent_connections: List[PaddleConnection] = []
        self._state_buffers: Dict[str, Any] = {}
        self._input_map: Dict[str, List[PaddleConnection]] = defaultdict(list)
        self.shared_memory = PaddleSharedMemory()
        self._performance_stats: Dict[str, List[float]] = defaultdict(list)
        
    def register_component(self, component: PaddleComponent):
        """Регистрирует компонент в системе с валидацией"""
        if component.name in self.components:
            raise ValueError(f"Component {component.name} already registered")
        
        # Валидируем порты компонента
        errors = component.validate_ports()
        if errors:
            raise ValueError(f"Component {component.name} has port errors: {errors}")
        
        self.components[component.name] = component
        self.add_sublayer(component.name, component)
        
    def add_connection(self, source: Union[str, Tuple[str, str]], 
                      target: Union[str, Tuple[str, str]],
                      transformer: Optional[Callable] = None,
                      delay: int = 0):
        """Добавляет соединение между портами компонентов с валидации"""
        if isinstance(source, tuple):
            source_str = f"{source[0]}.{source[1]}"
        else:
            source_str = source
            
        if isinstance(target, tuple):
            target_str = f"{target[0]}.{target[1]}"
        else:
            target_str = target
            
        connection = PaddleConnection(source_str, target_str, transformer, delay)
        
        # Проверяем существование компонентов и портов
        self._validate_connection(connection)
        
        # Проверяем совместимость портов
        self._validate_port_compatibility(connection)
        
        self.connections.append(connection)
        
        # Добавляем в карту соединений для быстрого доступа
        if connection.is_recurrent():
            self._recurrent_connections.append(connection)
        else:
            self._connection_map[connection.target].append(connection)
            self._input_map[connection.target].append(connection)
    
    def _validate_connection(self, connection: PaddleConnection):
        """Валидирует соединение на существование компонентов и портов"""
        # Проверяем существование компонентов
        if (connection.source_component != 'input' and 
            connection.source_component not in self.components):
            raise ValueError(f"Source component {connection.source_component} not found")
            
        if (connection.target_component != 'output' and 
            connection.target_component not in self.components):
            raise ValueError(f"Target component {connection.target_component} not found")
            
        # Проверяем существование портов
        if (connection.source_component != 'input' and 
            connection.source_port not in self.components[connection.source_component].output_ports):
            raise ValueError(f"Source port {connection.source_port} not found in component {connection.source_component}")
            
        if (connection.target_component != 'output' and 
            connection.target_port not in self.components[connection.target_component].input_ports):
            raise ValueError(f"Target port {connection.target_port} not found in component {connection.target_component}")
    
    def _validate_port_compatibility(self, connection: PaddleConnection):
        """Проверяет совместимость портов для соединения с учетом агрегации"""
        if connection.source_component == 'input' or connection.target_component == 'output':
            return  # Пропускаем проверку для входов/выходов системы

        source_comp = self.components[connection.source_component]
        target_comp = self.components[connection.target_component]

        source_port = source_comp.output_ports[connection.source_port]
        target_port = target_comp.input_ports[connection.target_port]

        # Для агрегируемых портов откладываем проверку до компиляции
        if target_port.aggregation != PaddleAggregationStrategy.CUSTOM:
            # Проверяем только базовую совместимость типов
            if source_port.type != target_port.type and source_port.type != PaddlePortType.ANY and target_port.type != PaddlePortType.ANY:
                raise ValueError(
                    f"Port type mismatch: {connection.source} ({source_port.type}) -> "
                    f"{connection.target} ({target_port.type})"
                )
            return

        # Для неагрегируемых портов выполняем полную проверку
        if not source_port.is_compatible_with(target_port):
            raise ValueError(
                f"Port compatibility error: {connection.source} ({source_port.type}, "
                f"{source_port.shape}, {source_port.dtype}) -> {connection.target} "
                f"({target_port.type}, {target_port.shape}, {target_port.dtype})"
            )

    def _validate_aggregation_compatibility(self, target_component: str, target_port: str):
        """Проверяет совместимость всех входов для агрегируемого порта"""
        component = self.components[target_component]
        port_spec = component.input_ports[target_port]
    
        # Получаем все соединения к целевому порту
        target_connections = [
            conn for conn in self.connections 
            if conn.target_component == target_component and conn.target_port == target_port
        ]
    
        if len(target_connections) <= 1:
            return  # Для одиночных соединений нет необходимости проверять агрегацию
    
        # Собираем спецификации всех исходных портов
        source_specs = []
        for conn in target_connections:
            if conn.source_component == 'input':
                # Для входных портов создаем временную спецификацию
                source_specs.append(PaddlePortSpec('input', PaddlePortType.ANY))
            else:
                source_comp = self.components[conn.source_component]
                source_specs.append(source_comp.output_ports[conn.source_port])
    
        # Проверяем совместимость в зависимости от стратегии агрегации
        if port_spec.aggregation == PaddleAggregationStrategy.CONCAT:
            # Для конкатенации проверяем, что все размерности кроме последней совпадают
            first_shape = source_specs[0].shape
            if first_shape is None:
                return  # Не можем проверить
            
            for spec in source_specs[1:]:
                if spec.shape is None:
                    continue
                
                if len(spec.shape) != len(first_shape):
                    raise ValueError(f"Shape rank mismatch for aggregation: "
                                   f"{len(spec.shape)} vs {len(first_shape)}")
            
                for i in range(len(first_shape) - 1):
                    if (spec.shape[i] != first_shape[i] and 
                        spec.shape[i] is not None and 
                        first_shape[i] is not None):
                        raise ValueError(f"Shape mismatch for aggregation at dimension {i}: "
                                       f"{spec.shape[i]} vs {first_shape[i]}")
    
        elif port_spec.aggregation in [PaddleAggregationStrategy.SUM, PaddleAggregationStrategy.MEAN, 
                                      PaddleAggregationStrategy.MAX, PaddleAggregationStrategy.STACK]:
            # Для этих стратегий все формы должны совпадать
            first_shape = source_specs[0].shape
            for spec in source_specs[1:]:
                if not spec.is_compatible_with(source_specs[0]):
                    raise ValueError(f"Shape mismatch for {port_spec.aggregation.value} aggregation: "
                                   f"{spec.shape} vs {first_shape}")

    def _precompile_connections(self):
        """Предварительная компиляция карты соединений для оптимизации производительности"""
        self._input_map.clear()
        for conn in self.connections:
            self._input_map[conn.target].append(conn)
    
    def compile(self):
        """Компилирует граф выполнения с расширенной валидацией"""
        # Валидируем все компоненты
        for comp_name, component in self.components.items():
            errors = component.validate_ports()
            if errors:
                raise ValueError(f"Component {comp_name} has validation errors: {errors}")
    
        # Предварительная компиляция соединений
        self._precompile_connections()
    
        # Строим граф зависимостей на уровне компонентов
        dependency_graph = defaultdict(list)
        in_degree = defaultdict(int)
    
        # Добавляем все компоненты в граф
        for comp_name in self.components:
            dependency_graph[comp_name] = []
            in_degree[comp_name] = 0
    
        # Строим зависимости на основе соединений
        for conn in self.connections:
            if conn.is_recurrent():
                continue  # Пропускаем рекуррентные соединения для топологической сортировки
            
            if (conn.source_component != 'input' and 
                conn.target_component != 'output' and
                conn.source_component != conn.target_component):
            
                dependency_graph[conn.source_component].append(conn.target_component)
                in_degree[conn.target_component] += 1
    
        # Топологическая сортировка (Kahn's algorithm)
        queue = [comp for comp in in_degree if in_degree[comp] == 0]
        execution_order = []
    
        while queue:
            node = queue.pop(0)
            execution_order.append(node)
        
            for neighbor in dependency_graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                
        if len(execution_order) != len(self.components):
            # Проверяем наличие циклов (кроме рекуррентных)
            cyclic_components = set(self.components.keys()) - set(execution_order)
            if cyclic_components:
                raise ValueError(f"Graph contains cycles involving components: {cyclic_components}")
        
        self.execution_order = execution_order
    
        # Проверяем совместимость агрегации для всех портов
        for comp_name, component in self.components.items():
            for port_name, port_spec in component.input_ports.items():
                self._validate_aggregation_compatibility(comp_name, port_name)
    
        # Компилируем все компоненты
        for component in self.components.values():
            component.compile()
    
        self._compiled = True
    
        return self
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Прямой проход через всю систему с поддержкой рекуррентных соединений"""
        if not self._compiled:
            self.compile()
            
        # Словарь для хранения промежуточных результатов по портам
        results = {'input': inputs}
        
        # Инициализируем буферы состояний для рекуррентных соединений
        self._init_state_buffers()
        
        # Выполняем компоненты в порядке топологической сортировки
        for comp_name in self.execution_order:
            component = self.components[comp_name]
            
            # Собираем входные данные для каждого порта компонента
            component_inputs = self._gather_inputs(comp_name, results)
            
            # Выполняем компонент
            try:
                start_time = time.time()
                component_outputs = component(component_inputs)
                end_time = time.time()
                
                # Сохраняем метрики производительности
                self._performance_stats[comp_name].append(end_time - start_time)
                
                results[comp_name] = component_outputs
            except Exception as e:
                raise RuntimeError(f"Error executing component {comp_name}: {str(e)}")
            
        # Обрабатываем рекуррентные соединения (обновляем состояния)
        self._update_recurrent_states(results)
            
        # Собираем выходные данные
        output_data = self._gather_outputs(results)
        
        return output_data
    
    def _gather_inputs(self, comp_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Собирает входные данные для компонента"""
        component = self.components[comp_name]
        component_inputs = {}
    
        for port_name, port_spec in component.input_ports.items():
            port_inputs = []
            target_key = f"{comp_name}.{port_name}"
        
            # Ищем все соединения, ведущие к этому порту
            for conn in self._input_map.get(target_key, []):
                # Получаем данные из источника
                source_data = self._get_source_data(conn, results)
            
                # Пропускаем None значения (например, для рекуррентных соединений в начале)
                if source_data is None:
                    continue
                
                # Применяем трансформер если есть
                if conn.transformer:
                    try:
                        source_data = self._apply_transformer(conn, source_data)
                    except Exception as e:
                        self._log_transformer_error(conn, e)
                        raise
            
                # Для агрегируемых портов откладываем валидацию до после агрегации
                if port_spec.aggregation == PaddleAggregationStrategy.CUSTOM or len(self._input_map.get(target_key, [])) <= 1:
                    # Валидируем данные по спецификации порта
                    if not port_spec.validate_value(source_data):
                        raise ValueError(
                            f"Data validation failed for {conn.source} -> {conn.target}: "
                            f"value {type(source_data)} doesn't match port spec"
                        )
            
                port_inputs.append(source_data)
        
            # Обрабатываем собранные входы для порта
            if not port_inputs:
                if port_spec.required:
                    raise ValueError(f"No connections to required port {comp_name}.{port_name}")
                continue
        
            # Применяем стратегию агрегации
            if len(port_inputs) == 1:
                component_inputs[port_name] = port_inputs[0]
            else:
                component_inputs[port_name] = self._aggregate_inputs(port_inputs, port_spec)
                # После агрегации валидируем результат для агрегируемых портов
                if port_spec.aggregation != PaddleAggregationStrategy.CUSTOM:
                    if not port_spec.validate_value(component_inputs[port_name]):
                        raise ValueError(
                            f"Data validation failed after aggregation for port {comp_name}.{port_name}: "
                            f"value {type(component_inputs[port_name])} doesn't match port spec"
                        )
    
        return component_inputs   

    def _apply_transformer(self, connection: PaddleConnection, data: Any) -> Any:
        """Применяет трансформер к данным с обработкой ошибок"""
        try:
            return connection.transformer(data)
        except Exception as e:
            error_msg = f"Transformer error in connection {connection}: {str(e)}"
            if hasattr(self, '_log_transformer_error'):
                self._log_transformer_error(connection, e)
            raise RuntimeError(error_msg)
    
    def _log_transformer_error(self, connection: PaddleConnection, error: Exception):
        """Логирует ошибку трансформера"""
        warnings.warn(f"Transformer error in connection {connection}: {str(error)}")
    
    def _aggregate_inputs(self, inputs: List[Any], port_spec: PaddlePortSpec) -> Any:
        """Агрегирует входные данные согласно стратегии агрегации"""
        if port_spec.aggregation == PaddleAggregationStrategy.CONCAT:
            return paddle.concat(inputs, axis=-1)
        elif port_spec.aggregation == PaddleAggregationStrategy.SUM:
            return sum(inputs)
        elif port_spec.aggregation == PaddleAggregationStrategy.MEAN:
            return sum(inputs) / len(inputs)
        elif port_spec.aggregation == PaddleAggregationStrategy.MAX:
            return paddle.stack(inputs).max(axis=0)
        elif port_spec.aggregation == PaddleAggregationStrategy.STACK:
            return paddle.stack(inputs)
        elif port_spec.aggregation == PaddleAggregationStrategy.CUSTOM and port_spec.custom_aggregator:
            return port_spec.custom_aggregator(inputs)
        else:
            # По умолчанию конкатенируем
            return paddle.concat(inputs, axis=-1)
    
    def _get_source_data(self, connection: PaddleConnection, results: Dict[str, Any]) -> Any:
        """Получает данные из источника соединения"""
        if connection.source_component == 'input':
            source_data = results['input'].get(connection.source_port, None)
            if source_data is None:
                raise ValueError(f"Input port {connection.source_port} not provided")
        else:
            # Для рекуррентных соединений, если компонент еще не выполнен, используем буфер состояний
            if connection.is_recurrent() and connection.source_component not in results:
                buffer_key = f"{connection.source}->{connection.target}"
                if buffer_key in self._state_buffers and self._state_buffers[buffer_key]:
                    # Берем последнее состояние из буферов
                    return self._state_buffers[buffer_key][-1]
                else:
                    # Если буфер пуст, возвращаем None (должно обрабатываться вызывающим кодом)
                    return None
        
            source_data = results.get(connection.source_component, {}).get(connection.source_port, None)
            if source_data is None:
                raise ValueError(f"Source port {connection.source_component}.{connection.source_port} not found")
    
        return source_data
    
    def _init_state_buffers(self):
        """Инициализирует буферы состояний для рекуррентных соединений"""
        for conn in self._recurrent_connections:
            buffer_key = f"{conn.source}->{conn.target}"
            if buffer_key not in self._state_buffers:
                self._state_buffers[buffer_key] = []
    
    def _update_recurrent_states(self, results: Dict[str, Any]):
        """Обновляет состояния для рекуррентных соединений"""
        for conn in self._recurrent_connections:
            buffer_key = f"{conn.source}->{conn.target}"
            source_data = self._get_source_data(conn, results)
            
            # Добавляем текущее состояние в буфер
            self._state_buffers[buffer_key].append(source_data)
            
            # Ограничиваем размер буфера для предотвращения утечек памяти
            if len(self._state_buffers[buffer_key]) > conn.delay * 2:
                self._state_buffers[buffer_key] = self._state_buffers[buffer_key][-conn.delay:]
    
    def _gather_outputs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Собирает выходные данные системы"""
        output_data = {}
        
        for conn in self.connections:
            if conn.target_component == 'output':
                source_data = self._get_source_data(conn, results)
                
                if conn.transformer:
                    try:
                        source_data = self._apply_transformer(conn, source_data)
                    except Exception as e:
                        self._log_transformer_error(conn, e)
                        raise
                
                output_data[conn.target_port] = source_data
        
        return output_data
    
    def reset(self):
        """Сброс состояния системы и всех компонентов"""
        for component in self.components.values():
            if hasattr(component, 'reset'):
                component.reset()
        self._state_buffers.clear()
        self.shared_memory.clear()
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Возвращает статистику производительности компонентов"""
        stats = {}
        for comp_name, times in self._performance_stats.items():
            if times:
                stats[comp_name] = {
                    'calls': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times)
                }
        return stats
    
    def get_connection_graph(self) -> Dict:
        """Возвращает граф соединений для визуализации"""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Добавляем специальные узлы input/output
        graph['nodes'].append({'id': 'input', 'type': 'input'})
        graph['nodes'].append({'id': 'output', 'type': 'output'})
        
        # Добавляем компоненты как узлы
        for comp_name, component in self.components.items():
            graph['nodes'].append({
                'id': comp_name,
                'type': 'component',
                'inputs': list(component.input_ports.keys()),
                'outputs': list(component.output_ports.keys())
            })
        
        # Добавляем соединения как ребра
        for conn in self.connections:
            graph['edges'].append({
                'source': conn.source,
                'target': conn.target,
                'recurrent': conn.is_recurrent(),
                'delay': conn.delay
            })
        
        return graph
    
    def visualize(self, output_file: str = None):
        """Визуализирует граф соединений"""
        try:
            import graphviz
            dot = graphviz.Digraph()
            
            # Добавляем узлы
            dot.node('input', 'Input', shape='ellipse', color='green')
            dot.node('output', 'Output', shape='ellipse', color='red')
            
            for comp_name, component in self.components.items():
                label = f"{comp_name}|{{{'|'.join(component.input_ports.keys())}}}|{{{'|'.join(component.output_ports.keys())}}}"
                dot.node(comp_name, f"{{{label}}}", shape='record')
            
            # Добавляем ребра
            for conn in self.connections:
                if conn.is_recurrent():
                    dot.edge(conn.source, conn.target, label=f"delay={conn.delay}", color='blue', style='dashed')
                else:
                    dot.edge(conn.source, conn.target)
            
            if output_file:
                dot.render(output_file, format='png', cleanup=True)
            
            return dot
        except ImportError:
            warnings.warn("Install graphviz for visualization: pip install graphviz")
            # Fallback to text visualization
            graph = self.get_connection_graph()
            print("Graph visualization:")
            print(f"Nodes: {[node['id'] for node in graph['nodes']]}")
            for edge in graph['edges']:
                rec_str = " (recurrent)" if edge['recurrent'] else ""
                print(f"  {edge['source']} -> {edge['target']}{rec_str}")
            return None

    def save(self, filepath: str):
        """Сохраняет конфигурацию системы в файл"""
        config = {
            'components': {},
            'connections': []
        }
        
        # Сохраняем информацию о компонентах
        for comp_name, component in self.components.items():
            config['components'][comp_name] = {
                'type': component.__class__.__name__,
                'input_ports': {name: {'type': port.type.value, 
                                      'shape': port.shape,
                                      'dtype': str(port.dtype) if port.dtype else None,
                                      'required': port.required,
                                      'aggregation': port.aggregation.value}
                               for name, port in component.input_ports.items()},
                'output_ports': {name: {'type': port.type.value, 
                                       'shape': port.shape,
                                       'dtype': str(port.dtype) if port.dtype else None}
                                for name, port in component.output_ports.items()}
            }
        
        # Сохраняем соединения
        for conn in self.connections:
            config['connections'].append({
                'source': conn.source,
                'target': conn.target,
                'delay': conn.delay
            })
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, component_registry: Dict[str, Type[PaddleComponent]]):
        """Загружает конфигурацию системы из файла"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        system = cls()
        
        # Создаем и регистрируем компоненты
        for comp_name, comp_config in config['components'].items():
            comp_type = comp_config['type']
            if comp_type not in component_registry:
                raise ValueError(f"Component type {comp_type} not found in registry")
            
            component = component_registry[comp_type]()
            component.name = comp_name
            
            # Восстанавливаем порты (упрощенная версия)
            system.register_component(component)
        
        # Восстанавливаем соединения
        for conn_config in config['connections']:
            system.add_connection(
                conn_config['source'],
                conn_config['target'],
                delay=conn_config.get('delay', 0)
            )
        
        return system

    def to_distributed(self, device_map: Dict[str, str]):
        """Распределяет компоненты по устройствам"""
        for comp_name, device in device_map.items():
            if comp_name in self.components:
                # PaddlePaddle автоматически размещает операции на устройстве
                # через paddle.set_device() перед созданием модели
                pass
    
    def to_inference_model(self, filepath: str, sample_input: Dict[str, Any]):
        """Экспортирует систему для инференса"""
        # Сохраняем модель для инференса
        paddle.jit.save(self, filepath, input_spec=[sample_input])

# Базовые компоненты для PaddlePaddle
class PaddleLinearComponent(PaddleComponent):
    """Линейный компонент с расширенной спецификацией портов"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 activation: str = 'relu', name: str = None):
        super().__init__(name)
        self.linear = paddle.nn.Linear(input_dim, output_dim)
        self.activation = self._get_activation(activation)
        
        # Переопределяем спецификацию портов
        self.input_ports = {
            'default': PaddlePortSpec(
                'default', 
                type=PaddlePortType.TENSOR,
                shape=(None, input_dim),
                dtype=paddle.float32
            )
        }
        self.output_ports = {
            'default': PaddlePortSpec(
                'default',
                type=PaddlePortType.TENSOR,
                shape=(None, output_dim),
                dtype=paddle.float32
            )
        }
        
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return paddle.nn.ReLU()
        elif activation == 'sigmoid':
            return paddle.nn.Sigmoid()
        elif activation == 'tanh':
            return paddle.nn.Tanh()
        else:
            return paddle.nn.Identity()
            
    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        return {'default': self.activation(self.linear(inputs['default']))}

class PaddleMultiHeadAttentionComponent(PaddleComponent):
    """Компонент multi-head attention с расширенной спецификацией портов"""
    
    def __init__(self, d_model: int, num_heads: int, name: str = None):
        super().__init__(name)
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Определяем порты с точными спецификациями
        self.input_ports = {
            'query': PaddlePortSpec(
                'query', 
                type=PaddlePortType.TENSOR,
                shape=(None, None, d_model),
                dtype=paddle.float32,
                required=True
            ),
            'key': PaddlePortSpec(
                'key',
                type=PaddlePortType.TENSOR, 
                shape=(None, None, d_model),
                dtype=paddle.float32,
                required=True
            ),
            'value': PaddlePortSpec(
                'value',
                type=PaddlePortType.TENSOR,
                shape=(None, None, d_model), 
                dtype=paddle.float32,
                required=True
            ),
            'mask': PaddlePortSpec(
                'mask',
                type=PaddlePortType.TENSOR,
                shape=(None, None, None),
                dtype=paddle.bool,
                required=False
            )
        }
        self.output_ports = {
            'output': PaddlePortSpec(
                'output',
                type=PaddlePortType.TENSOR,
                shape=(None, None, d_model),
                dtype=paddle.float32
            )
        }
        
        # Инициализируем слои
        self.q_linear = paddle.nn.Linear(d_model, d_model)
        self.k_linear = paddle.nn.Linear(d_model, d_model)
        self.v_linear = paddle.nn.Linear(d_model, d_model)
        self.out_linear = paddle.nn.Linear(d_model, d_model)
        
    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        query = inputs['query']
        key = inputs['key']
        value = inputs['value']
        mask = inputs.get('mask', None)
        
        batch_size = query.shape[0]
        d_k = self.d_model // self.num_heads
        
        # Linear transformations and split into heads
        q = self.q_linear(query).reshape([batch_size, -1, self.num_heads, d_k]).transpose([0, 2, 1, 3])
        k = self.k_linear(key).reshape([batch_size, -1, self.num_heads, d_k]).transpose([0, 2, 1, 3])
        v = self.v_linear(value).reshape([batch_size, -1, self.num_heads, d_k]).transpose([0, 2, 1, 3])
        
        # Calculate attention scores
        scores = paddle.matmul(q, k.transpose([0, 1, 3, 2])) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask.astype(scores.dtype) - 1) * 1e9
        
        # Apply softmax to get attention weights
        attn_weights = paddle.nn.functional.softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_output = paddle.matmul(attn_weights, v)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([batch_size, -1, self.d_model])
        return {'output': self.out_linear(attn_output)}

# Пример использования
def paddle_example_usage():
    """Пример использования библиотеки GluePaddle"""
    
    # Создаем систему
    system = GluePaddle()
    
    # Создаем и регистрируем компоненты
    attention = PaddleMultiHeadAttentionComponent(d_model=512, num_heads=8, name="attention")
    linear = PaddleLinearComponent(input_dim=512, output_dim=256, name="linear")
    
    system.register_component(attention)
    system.register_component(linear)
    
    # Добавляем соединения с указанием портов
    system.add_connection(('input', 'query'), ('attention', 'query'))
    system.add_connection(('input', 'key'), ('attention', 'key')) 
    system.add_connection(('input', 'value'), ('attention', 'value'))
    system.add_connection(('attention', 'output'), ('linear', 'default'))
    system.add_connection(('linear', 'default'), ('output', 'result'))
    
    # Визуализируем граф
    system.visualize()
    
    # Компилируем и выполняем
    system.compile()
    
    # Входные данные для каждого порта
    inputs = {
        'query': paddle.randn([1, 10, 512]),
        'key': paddle.randn([1, 10, 512]),
        'value': paddle.randn([1, 10, 512])
    }
    
    # Прямой проход
    outputs = system(inputs)
    print(f"Output shape: {outputs['result'].shape}")

if __name__ == "__main__":
    paddle_example_usage()