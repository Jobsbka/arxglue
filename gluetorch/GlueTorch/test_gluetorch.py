import torch
import pytest
import sys
import os
from typing import Any, Callable, Optional, Dict, List, Tuple, Union, Set, Type

# Добавляем путь к модулю
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gluetorch import (
    GlueTorch, Component, LinearComponent, MultiHeadAttentionComponent,
    PortSpec, PortType, AggregationStrategy, Connection, SharedMemory,
    MemoryComponent
)

def test_port_spec_validation():
    """Тестирование валидации спецификаций портов"""
    # Создаем спецификацию для тензора определенной формы
    spec = PortSpec('test', PortType.TENSOR, shape=(None, 10), dtype=torch.float32)
    
    # Valid cases
    valid_tensor = torch.randn(5, 10)
    assert spec.validate_value(valid_tensor) == True
    
    # Invalid cases
    invalid_shape = torch.randn(5, 5)  # Wrong shape
    assert spec.validate_value(invalid_shape) == False
    
    invalid_type = [1, 2, 3]  # Not a tensor
    assert spec.validate_value(invalid_type) == False

def test_linear_component():
    """Тестирование линейного компонента"""
    comp = LinearComponent(input_dim=10, output_dim=5, name='linear')
    
    # Test forward pass
    input_tensor = torch.randn(2, 10)
    output = comp(default=input_tensor)
    
    assert 'default' in output
    assert output['default'].shape == (2, 5)

def test_attention_component():
    """Тестирование компонента внимания"""
    comp = MultiHeadAttentionComponent(d_model=12, num_heads=3, name='attention')
    
    # Test forward pass
    query = torch.randn(2, 5, 12)
    key = torch.randn(2, 5, 12)
    value = torch.randn(2, 5, 12)
    
    output = comp(query=query, key=key, value=value)
    
    assert 'output' in output
    assert output['output'].shape == (2, 5, 12)

def test_memory_component():
    """Тестирование компонента памяти"""
    comp = MemoryComponent(keys=['state'], name='memory')
    
    # Test write and read
    input_tensor = torch.randn(3, 5)
    output = comp(write_state=input_tensor)
    
    assert 'read_state' in output
    assert torch.allclose(output['read_state'], input_tensor)

def test_basic_connection():
    """Тестирование соединения двух компонентов"""
    system = GlueTorch()
    
    # Create components
    linear1 = LinearComponent(input_dim=10, output_dim=8, name='linear1')
    linear2 = LinearComponent(input_dim=8, output_dim=5, name='linear2')
    
    # Register components
    system.register_component(linear1)
    system.register_component(linear2)
    
    # Add connections
    system.add_connection(('input', 'data'), ('linear1', 'default'))
    system.add_connection(('linear1', 'default'), ('linear2', 'default'))
    system.add_connection(('linear2', 'default'), ('output', 'result'))
    
    # Compile system
    system.compile()
    
    # Test forward pass
    input_data = torch.randn(3, 10)
    result = system(data=input_data)
    
    assert 'result' in result
    assert result['result'].shape == (3, 5)

def test_shared_memory():
    """Тестирование общей памяти"""
    memory = SharedMemory()
    
    # Test set and get
    test_data = torch.randn(5, 3)
    memory.set('key1', test_data)
    
    retrieved = memory.get('key1')
    assert torch.allclose(retrieved, test_data)
    
    # Test nonexistent key
    assert memory.get('nonexistent') is None
    assert memory.get('nonexistent', 'default') == 'default'

def test_multiple_connections():
    """Тестирование множественных соединений к одному порту"""
    system = GlueTorch()
    
    # Create components
    linear1 = LinearComponent(input_dim=5, output_dim=3, name='linear1')
    linear2 = LinearComponent(input_dim=5, output_dim=3, name='linear2')
    
    # Создаем компонент с портом, поддерживающим агрегацию CONCAT
    class AggregatingComponent(Component):
        def __init__(self, name: str = None):
            super().__init__(name)
            # Порт с агрегацией CONCAT - ожидаем 6 features после конкатенации
            self.input_ports = {
                'default': PortSpec(
                    'default', 
                    PortType.TENSOR, 
                    shape=(None, 6),  # После конкатенации 3+3=6
                    aggregation=AggregationStrategy.CONCAT
                )
            }
            self.output_ports = {
                'default': PortSpec('default', PortType.TENSOR, shape=(None, 6))
            }
            
        def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
            return {'default': default}
    
    aggregator = AggregatingComponent(name='aggregator')
    linear3 = LinearComponent(input_dim=6, output_dim=4, name='linear3')
    
    # Register components
    system.register_component(linear1)
    system.register_component(linear2)
    system.register_component(aggregator)
    system.register_component(linear3)
    
    # Add connections (оба выхода идут к aggregator)
    system.add_connection(('input', 'data1'), ('linear1', 'default'))
    system.add_connection(('input', 'data2'), ('linear2', 'default'))
    system.add_connection(('linear1', 'default'), ('aggregator', 'default'))
    system.add_connection(('linear2', 'default'), ('aggregator', 'default'))
    system.add_connection(('aggregator', 'default'), ('linear3', 'default'))
    system.add_connection(('linear3', 'default'), ('output', 'result'))
    
    # Compile system - здесь произойдет проверка агрегации
    system.compile()
    
    # Test forward pass
    input1 = torch.randn(3, 5)
    input2 = torch.randn(3, 5)
    result = system(data1=input1, data2=input2)
    
    assert 'result' in result
    assert result['result'].shape == (3, 4)

def test_recurrent_connection():
    """Тестирование рекуррентных соединений"""
    system = GlueTorch()
    
    # Create components with correct dimensions
    rnn_cell = LinearComponent(input_dim=5, output_dim=5, name='rnn_cell')
    memory = MemoryComponent(keys=['hidden'], name='memory')
    
    # Изменяем стратегию агрегации на SUM для порта default
    rnn_cell.input_ports['default'].aggregation = AggregationStrategy.SUM
    
    # Register components
    system.register_component(rnn_cell)
    system.register_component(memory)
    
    # Add connections with delay for recurrence
    system.add_connection(('input', 'x'), ('rnn_cell', 'default'))
    system.add_connection(('memory', 'read_hidden'), ('rnn_cell', 'default'), delay=1)
    system.add_connection(('rnn_cell', 'default'), ('memory', 'write_hidden'))
    system.add_connection(('rnn_cell', 'default'), ('output', 'y'))
    
    # Compile system
    system.compile()
    
    # Test forward pass with sequence
    sequence = [torch.randn(2, 5) for _ in range(3)]
    outputs = []
    
    for step, x in enumerate(sequence):
        result = system(x=x)
        outputs.append(result['y'])
        
        # Reset memory between independent sequences
        if step == len(sequence) - 1:
            system.reset()
    
    assert len(outputs) == 3
    assert all(out.shape == (2, 5) for out in outputs)

def test_integration_simple_ai():
    """Интеграционный тест: простая нейросеть для классификации"""
    system = GlueTorch()
    
    # Create components for a simple classifier
    encoder = LinearComponent(input_dim=20, output_dim=10, name='encoder')
    classifier = LinearComponent(input_dim=10, output_dim=3, name='classifier')
    
    # Register components
    system.register_component(encoder)
    system.register_component(classifier)
    
    # Add connections
    system.add_connection(('input', 'features'), ('encoder', 'default'))
    system.add_connection(('encoder', 'default'), ('classifier', 'default'))
    system.add_connection(('classifier', 'default'), ('output', 'logits'))
    
    # Compile system
    system.compile()
    
    # Test with sample data (batch_size=4, features=20)
    features = torch.randn(4, 20)
    result = system(features=features)
    
    assert 'logits' in result
    assert result['logits'].shape == (4, 3)
    
    # Test prediction
    predictions = torch.softmax(result['logits'], dim=1)
    assert predictions.shape == (4, 3)
    assert torch.allclose(predictions.sum(dim=1), torch.ones(4))

def test_aggregation_compatibility():
    """Тестирование проверки совместимости при агрегации"""
    system = GlueTorch()
    
    # Create components with incompatible shapes
    linear1 = LinearComponent(input_dim=5, output_dim=3, name='linear1')  # output: (None, 3)
    linear2 = LinearComponent(input_dim=5, output_dim=4, name='linear2')  # output: (None, 4)
    
    # Компонент с агрегацией CONCAT
    class AggregatingComponent(Component):
        def __init__(self, name: str = None):
            super().__init__(name)
            self.input_ports = {
                'default': PortSpec(
                    'default', 
                    PortType.TENSOR, 
                    shape=(None, 7),  # Ожидаем 7 features (3+4=7)
                    aggregation=AggregationStrategy.CONCAT
                )
            }
            self.output_ports = {
                'default': PortSpec('default', PortType.TENSOR, shape=(None, 7))
            }
            
        def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
            return {'default': default}
    
    aggregator = AggregatingComponent(name='aggregator')
    
    system.register_component(linear1)
    system.register_component(linear2)
    system.register_component(aggregator)
    
    system.add_connection(('input', 'data1'), ('linear1', 'default'))
    system.add_connection(('input', 'data2'), ('linear2', 'default'))
    system.add_connection(('linear1', 'default'), ('aggregator', 'default'))
    system.add_connection(('linear2', 'default'), ('aggregator', 'default'))
    
    # This should work - 3 + 4 = 7
    system.compile()
    
    # Теперь попробуем с несовместимыми формами
    linear3 = LinearComponent(input_dim=5, output_dim=2, name='linear3')  # output: (None, 2)
    system.register_component(linear3)
    system.add_connection(('input', 'data3'), ('linear3', 'default'))
    system.add_connection(('linear3', 'default'), ('aggregator', 'default'))
    
    # This should fail - нельзя конкатенировать (None,3), (None,4) и (None,2)
    # потому что первые размерности должны совпадать
    with pytest.raises(ValueError):
        system.compile()

def test_aggregation_error():
    """Тестирование обработки ошибок при несовместимой агрегации"""
    system = GlueTorch()
    
    # Create components with incompatible shapes for aggregation
    linear1 = LinearComponent(input_dim=5, output_dim=3, name='linear1')  # output: (None, 3)
    linear2 = LinearComponent(input_dim=5, output_dim=4, name='linear2')  # output: (None, 4)
    
    # Компонент с агрегацией CONCAT, но ожидающий другую форму
    class AggregatingComponent(Component):
        def __init__(self, name: str = None):
            super().__init__(name)
            self.input_ports = {
                'default': PortSpec(
                    'default', 
                    PortType.TENSOR, 
                    shape=(None, 5),  # Ожидаем 5 features, но получим 3+4=7
                    aggregation=AggregationStrategy.CONCAT
                )
            }
            self.output_ports = {
                'default': PortSpec('default', PortType.TENSOR, shape=(None, 5))
            }
            
        def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
            return {'default': default}
    
    aggregator = AggregatingComponent(name='aggregator')
    
    system.register_component(linear1)
    system.register_component(linear2)
    system.register_component(aggregator)
    
    system.add_connection(('input', 'data1'), ('linear1', 'default'))
    system.add_connection(('input', 'data2'), ('linear2', 'default'))
    system.add_connection(('linear1', 'default'), ('aggregator', 'default'))
    system.add_connection(('linear2', 'default'), ('aggregator', 'default'))
    
    # This should raise an error due to incompatible aggregation
    with pytest.raises(ValueError):
        system.compile()

def test_error_handling():
    """Тестирование обработки ошибок"""
    system = GlueTorch()
    
    # Create incompatible components
    linear1 = LinearComponent(input_dim=10, output_dim=5, name='linear1')
    linear2 = LinearComponent(input_dim=8, output_dim=3, name='linear2')  # Expects 8 dims
    
    # Изменяем тип агрегации на CUSTOM, который требует строгой проверки
    linear2.input_ports['default'].aggregation = AggregationStrategy.CUSTOM
    
    system.register_component(linear1)
    system.register_component(linear2)
    
    # This should fail - incompatible dimensions
    # Теперь ошибка должна возникать при добавлении соединения, а не при компиляции
    with pytest.raises(ValueError):
        system.add_connection(('linear1', 'default'), ('linear2', 'default'))

if __name__ == "__main__":
    # Run tests
    test_port_spec_validation()
    test_linear_component()
    test_attention_component()
    test_memory_component()
    test_basic_connection()
    test_shared_memory()
    test_multiple_connections()
    test_recurrent_connection()
    test_integration_simple_ai()
    test_error_handling()
    
    print("All tests passed!")