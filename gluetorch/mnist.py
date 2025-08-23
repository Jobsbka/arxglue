import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Dict, List, Tuple, Union, Set, Type
from gluetorch import GlueTorch, Component, PortSpec, PortType, AggregationStrategy, LinearComponent

# Компонент для flattening
class FlattenComponent(Component):
    def __init__(self, name=None):
        super().__init__(name)
        self.input_ports = {
            'default': PortSpec('default', PortType.TENSOR, shape=(None, 1, 28, 28))
        }
        self.output_ports = {
            'default': PortSpec('default', PortType.TENSOR, shape=(None, 784))
        }
        
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'default': torch.flatten(default, start_dim=1)}

# Компонент с Dropout
class DropoutComponent(Component):
    def __init__(self, p=0.5, name=None):
        super().__init__(name)
        self.dropout = nn.Dropout(p)
        self.input_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        self.output_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'default': self.dropout(default)}

# Компонент с ReLU активацией
class ReLUComponent(Component):
    def __init__(self, name=None):
        super().__init__(name)
        self.relu = nn.ReLU()
        self.input_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        self.output_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'default': self.relu(default)}

# Создаем и настраиваем простую сеть
def create_simple_network():
    system = GlueTorch()
    
    # Создаем компоненты
    flatten = FlattenComponent(name='flatten')
    linear1 = LinearComponent(input_dim=784, output_dim=128, name='linear1')
    relu1 = ReLUComponent(name='relu1')
    dropout = DropoutComponent(p=0.5, name='dropout')
    linear2 = LinearComponent(input_dim=128, output_dim=10, name='linear2')
    
    # Регистрируем компоненты
    system.register_component(flatten)
    system.register_component(linear1)
    system.register_component(relu1)
    system.register_component(dropout)
    system.register_component(linear2)
    
    # Создаем соединения
    system.add_connection(('input', 'image'), ('flatten', 'default'))
    system.add_connection(('flatten', 'default'), ('linear1', 'default'))
    system.add_connection(('linear1', 'default'), ('relu1', 'default'))
    system.add_connection(('relu1', 'default'), ('dropout', 'default'))
    system.add_connection(('dropout', 'default'), ('linear2', 'default'))
    system.add_connection(('linear2', 'default'), ('output', 'logits'))
    
    # Компилируем систему
    system.compile()
    
    return system

# Тестируем простую сеть
if __name__ == "__main__":
    simple_net = create_simple_network()
    test_input = torch.randn(32, 1, 28, 28)  # Пакет из 32 изображений MNIST
    output = simple_net(image=test_input)
    print(f"Simple network output shape: {output['logits'].shape}")