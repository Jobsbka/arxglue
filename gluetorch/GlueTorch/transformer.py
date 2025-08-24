import torch
import torch.nn as nn
import math
from typing import Dict, Optional
from gluetorch import GlueTorch, Component, PortSpec, PortType, LinearComponent, MultiHeadAttentionComponent

# Позиционное кодирование
class PositionalEncodingComponent(Component):
    def __init__(self, d_model, max_len=5000, name=None):
        super().__init__(name)
        self.d_model = d_model
        
        # Создаем матрицу позиционного кодирования
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        self.input_ports = {
            'default': PortSpec('default', PortType.TENSOR, shape=(None, None, d_model))
        }
        self.output_ports = {
            'default': PortSpec('default', PortType.TENSOR, shape=(None, None, d_model))
        }
        
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq_len = default.size(1)
        return {'default': default + self.pe[:, :seq_len, :]}

# Нормализация слоя
class LayerNormComponent(Component):
    def __init__(self, normalized_shape, eps=1e-5, name=None):
        super().__init__(name)
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.input_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        self.output_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'default': self.norm(default)}

# Блок Трансформера
class TransformerBlockComponent(Component):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, name=None):
        super().__init__(name)
        self.d_model = d_model
        
        # Создаем компоненты блока
        self.attention = MultiHeadAttentionComponent(d_model, num_heads, name=f'{name}_attention')
        self.norm1 = LayerNormComponent(d_model, name=f'{name}_norm1')
        self.linear1 = LinearComponent(d_model, dim_feedforward, name=f'{name}_linear1')
        self.dropout = nn.Dropout(dropout)
        self.linear2 = LinearComponent(dim_feedforward, d_model, name=f'{name}_linear2')
        self.norm2 = LayerNormComponent(d_model, name=f'{name}_norm2')
        self.relu = nn.ReLU()
        
        # Определяем порты
        self.input_ports = {
            'x': PortSpec('x', PortType.TENSOR, shape=(None, None, d_model)),
            'mask': PortSpec('mask', PortType.TENSOR, shape=(None, None, None), required=False)
        }
        self.output_ports = {
            'output': PortSpec('output', PortType.TENSOR, shape=(None, None, d_model))
        }
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Self-attention
        attn_output = self.attention(query=x, key=x, value=x, mask=mask)['output']
        x = x + attn_output
        x = self.norm1(x)['default']
        
        # Feedforward
        ff_output = self.linear1(x)['default']
        ff_output = self.relu(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)['default']
        
        x = x + ff_output
        x = self.norm2(x)['default']
        
        return {'output': x}

# Полный Трансформер
class TransformerComponent(Component):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes, max_len=5000, name=None):
        super().__init__(name)
        self.d_model = d_model
        
        # Embedding слой
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Позиционное кодирование
        self.pos_encoding = PositionalEncodingComponent(d_model, max_len, name=f'{name}_pos_encoding')
        
        # Блоки Трансформера
        self.blocks = nn.ModuleList([
            TransformerBlockComponent(d_model, num_heads, name=f'{name}_block_{i}')
            for i in range(num_layers)
        ])
        
        # Классификатор
        self.classifier = LinearComponent(d_model, num_classes, name=f'{name}_classifier')
        
        # Определяем порты
        self.input_ports = {
            'input_ids': PortSpec('input_ids', PortType.TENSOR, shape=(None, None)),
            'mask': PortSpec('mask', PortType.TENSOR, shape=(None, None, None), required=False)
        }
        self.output_ports = {
            'logits': PortSpec('logits', PortType.TENSOR, shape=(None, num_classes))
        }
        
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Позиционное кодирование
        x = self.pos_encoding(x)['default']
        
        # Блоки Трансформера
        for block in self.blocks:
            x = block(x=x, mask=mask)['output']
        
        # Усреднение по последовательности (можно заменить на [CLS] токен)
        x = x.mean(dim=1)
        
        # Классификация
        logits = self.classifier(x)['default']
        
        return {'logits': logits}

# Создаем и тестируем Трансформер
def create_transformer():
    system = GlueTorch()
    
    # Создаем и регистрируем компонент Трансформера
    transformer = TransformerComponent(
        vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        num_classes=10,
        name='transformer'
    )
    
    system.register_component(transformer)
    
    # Создаем соединения
    system.add_connection(('input', 'input_ids'), ('transformer', 'input_ids'))
    system.add_connection(('input', 'mask'), ('transformer', 'mask'))
    system.add_connection(('transformer', 'logits'), ('output', 'logits'))
    
    # Компилируем систему
    system.compile()
    
    return system

# Тестируем Трансформер
if __name__ == "__main__":
    transformer_net = create_transformer()
    test_input_ids = torch.randint(0, 10000, (16, 32))  # Пакет из 16 последовательностей по 32 токена
    test_mask = torch.ones(16, 32, 32).bool()  # Маска без маскирования
    output = transformer_net(input_ids=test_input_ids, mask=test_mask)
    print(f"Transformer output shape: {output['logits'].shape}")