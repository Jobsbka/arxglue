import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import requests
import zipfile
import io
from tqdm import tqdm
import warnings
import argparse
import psutil
import GPUtil
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    # Параметры данных
    dataset_name = 'imdb'
    max_seq_length = 64
    vocab_size = 100000  # Уменьшим для более быстрого тестирования
    batch_size = 32
    
    # Параметры модели
    d_model = 256
    nhead = 4
    num_layers = 7
    dim_feedforward = 512
    dropout = 0.1
    
    # Параметры истории (новые настраиваемые параметры)
    history_heads = 2
    history_dropout = 0.1
    
    # Стратегия выбора слоев для истории
    history_layer_strategy = "all"  # "all", "first_last", "custom" "first_mid_prelast"
    custom_history_layers = [0, 3, 5]  # если strategy = "custom"
    
    # Тип агрегации исторических состояний
    history_aggregation = "attention"  # "concat", "sum", "max_pool", "gated"
    
    # Уровень применения исторического внимания
    history_application_level = "output"  # "per_layer"
    
    # Метод объединения с основным потоком
    history_fusion = "residual"  # "gate", "concat"
    
    # Включение различных улучшений
    use_memory_bank = False  # Экспериментальная функция
    use_layerwise_gating = False  # Разные ворота для каждого исторического слоя
    use_learnable_weights = False  # Обучаемые веса для разных исторических слоев
    
    # Параметры обучения
    learning_rate = 5e-4
    num_epochs = 100  # Уменьшим для более быстрого тестирования
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Параметры генерации
    gen_length = 30  # Уменьшим для более быстрого тестирования
    temperature = 0.8
    
    # Визуализация и отладка
    print_interval = 50
    save_models = True
    visualize_attention = False  # Визуализация весов внимания

# ==================== УТИЛИТЫ ДАННЫХ ====================
def download_imdb_data():
    """Загрузка датасета IMDB если он отсутствует"""
    data_dir = "imdb_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    file_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    
    if not os.path.exists(file_path):
        print("Загрузка IMDB датасета...")
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc="Загрузка",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))
            
            print("Распаковка датасета...")
            import tarfile
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=data_dir)
            
            print("Датасет успешно загружен и распакован.")
        except Exception as e:
            print(f"Ошибка при загрузке датасета: {e}")
            print("Создаем демо-данные для тестирования...")
            create_demo_data(data_dir)
    else:
        print("Датасет уже существует.")
    
    return os.path.join(data_dir, "aclImdb")

def create_demo_data(data_dir):
    """Создание демо-данных если загрузка не удалась"""
    imdb_path = os.path.join(data_dir, "aclImdb", "train")
    pos_path = os.path.join(imdb_path, "pos")
    neg_path = os.path.join(imdb_path, "neg")
    
    os.makedirs(pos_path, exist_ok=True)
    os.makedirs(neg_path, exist_ok=True)
    
    # Создаем несколько демо-отзывов
    demo_reviews = [
        ("pos", "This movie was absolutely fantastic! Great acting and storyline."),
        ("pos", "One of the best films I've seen this year. Highly recommend."),
        ("pos", "Brilliant performance by the lead actor. The plot was engaging."),
        ("neg", "Terrible movie. Poor acting and boring storyline."),
        ("neg", "Waste of time. The plot made no sense and the acting was awful."),
        ("neg", "Disappointing film. Expected much more from this director.")
    ]
    
    for i, (label, review) in enumerate(demo_reviews):
        path = pos_path if label == "pos" else neg_path
        with open(os.path.join(path, f"review_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(review)
    
    print("Созданы демо-данные для тестирования.")

def load_imdb_texts(data_path):
    """Загрузка текстов из IMDB"""
    texts = []
    
    for label in ['pos', 'neg']:
        labeled_path = os.path.join(data_path, 'train', label)
        if os.path.exists(labeled_path):
            for file_name in os.listdir(labeled_path):
                if file_name.endswith('.txt'):
                    with open(os.path.join(labeled_path, file_name), 'r', encoding='utf-8') as f:
                        texts.append(f.read())
        else:
            print(f"Предупреждение: путь {labeled_path} не существует.")
    
    # Если текстов нет, создаем несколько демо-текстов
    if len(texts) == 0:
        texts = [
            "This movie was great! I really enjoyed it.",
            "Terrible film. Waste of time.",
            "Amazing acting and plot. Highly recommend.",
            "Poor storyline and bad acting.",
            "One of the best movies I've seen this year.",
            "Disappointing. Expected more from this director."
        ]
    
    return texts

def build_vocab(texts, vocab_size):
    """Построение словаря"""
    word_freq = defaultdict(int)
    for text in texts:
        for word in text.lower().replace('<br />', ' ').split():
            word_freq[word] += 1

    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:vocab_size-4]
    
    for idx, (word, freq) in enumerate(most_common):
        vocab[word] = idx + 4
    
    return vocab

def text_to_sequence(text, vocab, max_length):
    """Преобразование текста в последовательность индексов"""
    tokens = text.lower().replace('<br />', ' ').split()[:max_length-2]  # -2 для sos и eos
    sequence = [vocab['<sos>']] + [vocab.get(token, 1) for token in tokens] + [vocab['<eos>']]
    return torch.tensor(sequence)

def preprocess_data(texts, vocab, max_length):
    """Препроцессинг данных для языкового моделирования"""
    processed_texts = []
    
    for text in texts:
        sequence = text_to_sequence(text, vocab, max_length)
        if len(sequence) > 3:  # Убедимся, что последовательность достаточно длинная
            processed_texts.append(sequence)
    
    return processed_texts

def collate_fn(batch):
    """Функция для объединения примеров в батчи"""
    # Добавляем padding к последовательностям
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    
    # Для языкового моделирования цель - это та же последовательность, но сдвинутая на 1
    # Вход: все токены кроме последнего, цель: все токены кроме первого
    data = padded[:, :-1]
    target = padded[:, 1:]

    if data.size(1) != target.size(1):
        min_len = min(data.size(1), target.size(1))
        data = data[:, :min_len]
        target = target[:, :min_len]

    return data, target

# ==================== ПОЗИЦИОННОЕ КОДИРОВАНИЕ ====================
class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==================== БАЗОВАЯ МОДЕЛЬ ====================
class BaselineGenerator(nn.Module):
    """Базовая генеративная модель на трансформере (только декодер)"""
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Эмбеддинги
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)
        
        # Декодер трансформера
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.num_layers)
        
        # Выходной слой
        self.output_layer = nn.Linear(config.d_model, vocab_size)
        
    def forward(self, tgt, memory=None):
        # Эмбеддинги и позиционное кодирование
        tgt_emb = self.dropout(self.pos_encoding(self.token_embedding(tgt)))
        
        # Создаем маску для декодера
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.config.device)
        
        # Прямой проход через трансформер
        if memory is None:
            # Если memory не предоставлена, используем zeros like tgt_emb
            memory = torch.zeros_like(tgt_emb)
        
        output = self.transformer(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True
        )
        
        # Выходные вероятности
        logits = self.output_layer(output)
        return logits
    
    def generate(self, prompt, max_length, temperature=1.0):
        """Генерация текста по промпту"""
        self.eval()
        with torch.no_grad():
            # Начинаем с промпта
            generated = prompt.clone().unsqueeze(0)
            
            for _ in range(max_length):
                # Прямой проход для получения следующего токена
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Применяем softmax и выбираем следующий токен
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                # Добавляем к сгенерированной последовательности
                generated = torch.cat([generated, next_token], dim=1)
                
                # Если сгенерировали конец последовательности, останавливаемся
                if next_token.item() == 3:  # <eos>
                    break
            
            return generated.squeeze(0)

# ==================== УЛУЧШЕННАЯ МОДЕЛЬ ====================
class EnhancedHistoryAwareGenerator(nn.Module):
    """Улучшенная генеративная модель с историческим вниманием"""
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Эмбеддинги
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)
        
        # Декодерные слои
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # Инициализация механизмов истории
        self._init_history_mechanisms()
        
        # Внешняя память (экспериментальная функция)
        if config.use_memory_bank:
            self.memory_bank = nn.ParameterList()
            self.memory_attention = MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.history_heads,
                dropout=config.history_dropout,
                batch_first=True
            )
        
        # Выходной слой
        self.output_layer = nn.Linear(config.d_model, vocab_size)
        
        # Визуализация внимания
        self.attention_weights = None
    
    def _init_history_mechanisms(self):
        """Инициализация механизмов работы с истории"""
        config = self.config
        
        # Механизм исторического внимания
        if config.history_aggregation == "attention":
            self.history_attention = MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.history_heads,
                dropout=config.history_dropout,
                batch_first=True
            )
        
        # Нормализация и dropout для истории
        self.history_norm = nn.LayerNorm(config.d_model)
        self.history_dropout = nn.Dropout(config.history_dropout)
        
        # Механизмы слияния
        if config.history_fusion == "gate":
            if config.use_layerwise_gating:
                # Отдельные ворота для каждого исторического слоя
                self.history_gates = nn.ModuleList([
                    nn.Linear(config.d_model * 2, config.d_model)
                    for _ in range(config.num_layers)
                ])
            else:
                # Общие ворота
                self.history_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.sigmoid = nn.Sigmoid()
        
        elif config.history_fusion == "concat":
            self.history_projection = nn.Linear(config.d_model * 2, config.d_model)
        
        # Обучаемые веса для исторических слоев
        if config.use_learnable_weights:
            self.layer_weights = nn.Parameter(torch.ones(config.num_layers))
        
        # Проекционные слои для разных типов агрегации
        if config.history_aggregation in ["concat", "sum", "max_pool"]:
            self.history_proj = nn.Linear(config.d_model, config.d_model)
    
    def _select_history_layers(self, layer_idx):
        """Выбор слоев для истории на основе стратегии"""
        config = self.config
        
        if config.history_layer_strategy == "all":
            return True
        elif config.history_layer_strategy == "first_mid_prelast":
            mid_layer = config.num_layers // 2
            prelast_layer = config.num_layers - 2
            return layer_idx == 0 or layer_idx == mid_layer or layer_idx == prelast_layer
        elif config.history_layer_strategy == "first_last":
            return layer_idx == 0 or layer_idx == config.num_layers - 1
        elif config.history_layer_strategy == "custom":
            return layer_idx in config.custom_history_layers
        elif config.history_layer_strategy == "skip_even":
            return layer_idx % 2 == 0
        elif config.history_layer_strategy == "skip_odd":
            return layer_idx % 2 == 1
        
        return False
    
    def _apply_history_aggregation(self, history_states, current_state):
        """Применение выбранного метода агрегации исторических состояний"""
        config = self.config
        
        if config.history_aggregation == "attention":
            # Объединяем исторические состояния
            history = torch.stack(history_states, dim=2)
            batch_size, seq_len, history_len, d_model = history.shape
            
            # Изменяем форму для внимания
            history_flat = history.reshape(batch_size * seq_len, history_len, d_model)
            current_flat = current_state.reshape(batch_size * seq_len, 1, d_model)
            
            # Применяем внимание к истории
            attn_output, attn_weights = self.history_attention(
                query=current_flat,
                key=history_flat,
                value=history_flat
            )
            
            # Сохраняем веса для визуализации
            if config.visualize_attention:
                self.attention_weights = attn_weights.detach().cpu().numpy()
            
            attn_output = attn_output.reshape(batch_size, seq_len, d_model)
            return attn_output
        
        elif config.history_aggregation == "concat":
            # Конкатенация всех исторических состояний
            concat_states = torch.cat(history_states, dim=-1)
            return self.history_proj(concat_states)
        
        elif config.history_aggregation == "sum":
            # Суммирование исторических состояний
            sum_states = torch.stack(history_states, dim=0).sum(dim=0)
            return self.history_proj(sum_states)
        
        elif config.history_aggregation == "max_pool":
            # Max-pooling по историческим состояниям
            stacked_states = torch.stack(history_states, dim=0)
            max_states, _ = stacked_states.max(dim=0)
            return self.history_proj(max_states)
        
        elif config.history_aggregation == "gated":
            # Гейтированная агрегация
            weighted_states = []
            for i, state in enumerate(history_states):
                if self.config.use_learnable_weights:
                    weight = torch.sigmoid(self.layer_weights[i])
                else:
                    weight = 1.0 / len(history_states)
                weighted_states.append(state * weight)
            
            return torch.stack(weighted_states, dim=0).sum(dim=0)
        
        return None
    
    def _apply_history_fusion(self, current_state, history_output):
        """Применение выбранного метода слияния с историей"""
        config = self.config
        
        if config.history_fusion == "residual":
            return current_state + self.history_dropout(history_output)
        
        elif config.history_fusion == "gate":
            combined = torch.cat([current_state, history_output], dim=-1)
            if config.use_layerwise_gating:
                # Используем разные ворота для каждого слоя
                gate = self.sigmoid(self.history_gates[self.current_layer](combined))
            else:
                gate = self.sigmoid(self.history_gate(combined))
            return gate * current_state + (1 - gate) * history_output
        
        elif config.history_fusion == "concat":
            combined = torch.cat([current_state, history_output], dim=-1)
            return self.history_projection(combined)
        
        return current_state
    
    def forward(self, tgt, memory=None, use_history=True):
        # Эмбеддинги и позиционное кодирование
        tgt_emb = self.dropout(self.pos_encoding(self.token_embedding(tgt)))
        
        # Создаем маску для декодера
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.config.device)
        
        if memory is None:
            memory = torch.zeros_like(tgt_emb)
        
        # Собираем историю состояний
        history_states = []
        x = tgt_emb
        
        # Прямой проход через слои декодера
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, tgt_mask=tgt_mask, tgt_is_causal=True)
            
            # Сохраняем текущий слой для layerwise_gating
            self.current_layer = i
            
            # Сохраняем состояния выбранных слоев
            if use_history and self._select_history_layers(i):
                history_states.append(x)
                
                # Применяем историческое внимание на уровне каждого слоя
                if self.config.history_application_level == "per_layer" and len(history_states) > 1:
                    history_output = self._apply_history_aggregation(history_states[:-1], x)
                    if history_output is not None:
                        x = self._apply_history_fusion(x, history_output)
        
        # Применяем внимание к истории на выходном уровне
        if use_history and len(history_states) > 0 and self.config.history_application_level == "output":
            history_output = self._apply_history_aggregation(history_states, x)
            if history_output is not None:
                x = self._apply_history_fusion(x, history_output)
                x = self.history_norm(x)
        
        # Работа с внешней памятью (экспериментальная функция)
        if use_history and self.config.use_memory_bank and len(self.memory_bank) > 0:
            memory_output, _ = self.memory_attention(x, self.memory_bank[-1], self.memory_bank[-1])
            x = x + self.history_dropout(memory_output)
        
        # Выходные вероятности
        logits = self.output_layer(x)
        return logits
    
    def generate(self, prompt, max_length, temperature=1.0):
        """Генерация текста по промпту с использованием истории"""
        self.eval()
        with torch.no_grad():
            # Начинаем с промпта
            generated = prompt.clone().unsqueeze(0)
            
            # Инициализируем память для хранения состояний
            if self.config.use_memory_bank:
                self.memory_bank = nn.ParameterList()
            
            for i in range(max_length):
                # Прямой проход для получения следующего токена
                logits = self.forward(generated, use_history=True)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Применяем softmax и выбираем следующий токен
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                # Добавляем к сгенерированной последовательности
                generated = torch.cat([generated, next_token], dim=1)
                
                # Сохраняем состояния во внешнюю память
                if self.config.use_memory_bank and i % 5 == 0:  # Сохраняем каждые 5 шагов
                    with torch.no_grad():
                        # Для simplicity, просто сохраняем последнее состояние
                        self.memory_bank.append(nn.Parameter(generated.detach()))
                
                # Если сгенерировали конец последовательности, останавливаемся
                if next_token.item() == 3:  # <eos>
                    break
            
            return generated.squeeze(0)

# ==================== УТИЛИТЫ ОБУЧЕНИЯ ====================
def count_parameters(model):
    """Подсчет количества параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_usage():
    """Получение информации об использовании памяти"""
    memory_info = {}
    
    # CPU память
    memory_info['cpu_memory'] = psutil.virtual_memory().percent
    
    # GPU память (если доступно)
    if torch.cuda.is_available():
        memory_info['gpu_memory'] = torch.cuda.memory_allocated() / 1024**3  # в GB
        memory_info['gpu_memory_max'] = torch.cuda.max_memory_allocated() / 1024**3  # в GB
    else:
        memory_info['gpu_memory'] = 0
        memory_info['gpu_memory_max'] = 0
    
    return memory_info

def train_epoch(model, dataloader, criterion, optimizer, config, epoch, model_name):
    """Обучение на одной эпохе"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(config.device), target.to(config.device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Вычисляем потерю
        loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()
        
        if batch_idx % config.print_interval == 0:
            perplexity = np.exp(loss.item())
            print(f'{model_name} - Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Perplexity: {perplexity:.2f}')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def generate_sample(model, vocab, inv_vocab, config, prompt_text):
    """Генерация примера текста"""
    # Преобразуем промпт в последовательность индексов
    prompt_tokens = prompt_text.lower().split()
    prompt_indices = [vocab.get(token, 1) for token in prompt_tokens]
    prompt_indices = [vocab['<sos>']] + prompt_indices
    
    prompt_tensor = torch.tensor(prompt_indices).to(config.device)
    
    # Генерируем продолжение
    generated = model.generate(prompt_tensor, config.gen_length, config.temperature)
    
    # Преобразуем обратно в текст
    generated_tokens = []
    for idx in generated.cpu().numpy():
        if idx == vocab['<eos>']:
            break
        if idx not in [vocab['<sos>'], vocab['<pad>'], vocab['<unk>']]:
            generated_tokens.append(inv_vocab.get(idx, '<unk>'))
    
    return ' '.join(generated_tokens)

# ==================== УТИЛИТЫ ДЛЯ ВИЗУАЛИЗАЦИИ ====================
def visualize_attention_weights(attention_weights, layer_names, save_path):
    """Визуализация весов внимания"""
    if attention_weights is None:
        return
    
    plt.figure(figsize=(12, 8))
    
    # attention_weights shape: [batch*seq, history_len, 1]
    attn = attention_weights.reshape(-1, attention_weights.shape[1])
    avg_attn = attn.mean(axis=0)
    
    plt.bar(range(len(avg_attn)), avg_attn)
    plt.xticks(range(len(avg_attn)), layer_names)
    plt.xlabel("Исторические слои")
    plt.ylabel("Средний вес внимания")
    plt.title("Распределение внимания по историческим слоям")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_training_results(baseline_history, history_model_history, save_path):
    """Визуализация результатов обучения"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if baseline_history:
        plt.plot(baseline_history['train_perplexity'], label='Baseline')
    if history_model_history:
        plt.plot(history_model_history['train_perplexity'], label='History Model')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.title('Training Perplexity')
    
    plt.subplot(1, 2, 2)
    if baseline_history:
        plt.plot(baseline_history['train_loss'], label='Baseline')
    if history_model_history:
        plt.plot(history_model_history['train_loss'], label='History Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==================== РЕЖИМЫ РАБОТЫ ====================
def competitive_mode(config):
    """Соревновательный режим: обучение обеих моделей и сравнение"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== РЕЖИМ: СОРЕВНОВАТЕЛЬНЫЙ (запуск: {timestamp}) ===")
    
    # Загрузка данных
    data_path = download_imdb_data()
    texts = load_imdb_texts(data_path)
    
    # Построение словаря
    print("Построение словаря...")
    vocab = build_vocab(texts, config.vocab_size)
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_path = f"vocab_с{timestamp}.pkl"
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Словарь сохранен как: {vocab_path}")
    
    # Предобработка данных
    print("Предобработка данных...")
    processed_texts = preprocess_data(texts[:1000], vocab, config.max_seq_length)  # Используем только часть данных
    
    # Создание DataLoader
    train_loader = DataLoader(
        processed_texts, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Инициализация моделей
    baseline_model = BaselineGenerator(config, len(vocab)).to(config.device)
    history_model = EnhancedHistoryAwareGenerator(config, len(vocab)).to(config.device)
    
    # Подсчет параметров
    baseline_params = count_parameters(baseline_model)
    history_params = count_parameters(history_model)
    
    print(f"Параметры Baseline модели: {baseline_params:,}")
    print(f"Параметры History модели: {history_params:,}")
    print(f"Разница: {history_params - baseline_params:,} параметров")
    
    # Функция потерь и оптимизаторы
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Игнорируем padding
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=config.learning_rate)
    history_optimizer = torch.optim.Adam(history_model.parameters(), lr=config.learning_rate)
    
    # Обучение моделей
    print("Начало обучения...")
    
    baseline_history = {
        'train_loss': [], 'train_perplexity': [],
        'memory_usage': [], 'training_time': 0
    }
    
    history_model_history = {
        'train_loss': [], 'train_perplexity': [],
        'memory_usage': [], 'training_time': 0
    }
    
    # Обучение Baseline модели
    print("\n=== ОБУЧЕНИЕ BASELINE МОДЕЛИ ===")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        train_loss, train_ppl = train_epoch(
            baseline_model, train_loader, criterion, baseline_optimizer, config, epoch, "Baseline"
        )
        
        baseline_history['train_loss'].append(train_loss)
        baseline_history['train_perplexity'].append(train_ppl)
        baseline_history['memory_usage'].append(get_memory_usage())
        
        print(f"Baseline - Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
    
    baseline_history['training_time'] = time.time() - start_time
    print(f"Baseline обучение заняло: {baseline_history['training_time']:.2f} секунд")
    
    # Обучение History модели
    print("\n=== ОБУЧЕНИЕ HISTORY МОДЕЛИ ===")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        train_loss, train_ppl = train_epoch(
            history_model, train_loader, criterion, history_optimizer, config, epoch, "History"
        )
        
        history_model_history['train_loss'].append(train_loss)
        history_model_history['train_perplexity'].append(train_ppl)
        history_model_history['memory_usage'].append(get_memory_usage())
        
        print(f"History - Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
    
    history_model_history['training_time'] = time.time() - start_time
    print(f"History обучение заняло: {history_model_history['training_time']:.2f} секунд")
    
    # Генерация примеров
    print("\n=== ГЕНЕРАЦИЯ ПРИМЕРОВ ТЕКСТА ===")
    prompts = [
        "How are you today?",
        "i really liked",
        "the acting was",
        "the story is",
        "this movie is"
    ]
    
    baseline_results = []
    history_results = []
    
    print("\n=== BASELINE MODEL ===")
    for prompt in prompts:
        generated = generate_sample(baseline_model, vocab, inv_vocab, config, prompt)
        baseline_results.append((prompt, generated))
        print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
    
    print("\n=== HISTORY MODEL ===")
    for prompt in prompts:
        generated = generate_sample(history_model, vocab, inv_vocab, config, prompt)
        history_results.append((prompt, generated))
        print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
    
    # Визуализация весов внимания
    if config.visualize_attention and hasattr(history_model, 'attention_weights'):
        layer_names = []
        for i in range(config.num_layers):
            if history_model._select_history_layers(i):
                layer_names.append(f"Layer {i}")
        
        attention_path = f"attention_weights_{timestamp}.png"
        visualize_attention_weights(
            history_model.attention_weights, 
            layer_names,
            attention_path
        )
        print(f"Визуализация внимания сохранена как: {attention_path}")
    
    # Визуализация результатов
    training_plot_path = f"training_history_{timestamp}.png"
    visualize_training_results(baseline_history, history_model_history, training_plot_path)
    print(f"График обучения сохранен как: {training_plot_path}")
    
    # Сохранение моделей
    if config.save_models:
        baseline_path = f'baseline_generator_{timestamp}.pth'
        history_path = f'history_generator_{timestamp}.pth'
        
        torch.save(baseline_model.state_dict(), baseline_path)
        torch.save(history_model.state_dict(), history_path)
        print(f"Модели сохранены как '{baseline_path}' и '{history_path}'")
    
    # Сравнение финальных результатов
    print("\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ===")
    if baseline_history['train_perplexity']:
        print(f"Baseline Model - Final Train Perplexity: {baseline_history['train_perplexity'][-1]:.2f}")
    if history_model_history['train_perplexity']:
        print(f"History Model - Final Train Perplexity: {history_model_history['train_perplexity'][-1]:.2f}")
    
    if baseline_history['train_perplexity'] and history_model_history['train_perplexity']:
        improvement = baseline_history['train_perplexity'][-1] - history_model_history['train_perplexity'][-1]
        print(f"Improvement: {improvement:.2f}")
    
    # Аналитика использования ресурсов
    print("\n=== АНАЛИТИКА ИСПОЛЬЗОВАНИЯ РЕСУРСОВ ===")
    if baseline_history['memory_usage']:
        avg_cpu_memory = np.mean([m['cpu_memory'] for m in baseline_history['memory_usage']])
        avg_gpu_memory = np.mean([m['gpu_memory'] for m in baseline_history['memory_usage']])
        max_gpu_memory = np.max([m['gpu_memory_max'] for m in baseline_history['memory_usage']])
        
        print(f"Baseline - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%")
        print(f"Baseline - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB")
        print(f"Baseline - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB")
        print(f"Baseline - Время обучения: {baseline_history['training_time']:.2f} секунд")
    
    if history_model_history['memory_usage']:
        avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
        avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
        max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
        
        print(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%")
        print(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB")
        print(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB")
        print(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд")
    
    # Сохранение результатов в файл
    results_path = f"results_{timestamp}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ===\n")
        f.write(f"Дата и время: {timestamp}\n")
        f.write(f"Режим: Соревновательный\n\n")
        
        f.write("=== МОДЕЛИ ===\n")
        f.write(f"Baseline параметров: {baseline_params:,}\n")
        f.write(f"History параметров: {history_params:,}\n")
        f.write(f"Разница: {history_params - baseline_params:,} параметров\n\n")
        
        f.write("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===\n")
        if baseline_history['train_perplexity']:
            f.write(f"Baseline - Финальная перплексия: {baseline_history['train_perplexity'][-1]:.2f}\n")
        if history_model_history['train_perplexity']:
            f.write(f"History - Финальная перплексия: {history_model_history['train_perplexity'][-1]:.2f}\n")
        
        if baseline_history['train_perplexity'] and history_model_history['train_perplexity']:
            improvement = baseline_history['train_perplexity'][-1] - history_model_history['train_perplexity'][-1]
            f.write(f"Улучшение: {improvement:.2f}\n")
        
        f.write("\n=== РЕСУРСЫ ===\n")
        if baseline_history['memory_usage']:
            avg_cpu_memory = np.mean([m['cpu_memory'] for m in baseline_history['memory_usage']])
            avg_gpu_memory = np.mean([m['gpu_memory'] for m in baseline_history['memory_usage']])
            max_gpu_memory = np.max([m['gpu_memory_max'] for m in baseline_history['memory_usage']])
            
            f.write(f"Baseline - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%\n")
            f.write(f"Baseline - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB\n")
            f.write(f"Baseline - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB\n")
            f.write(f"Baseline - Время обучения: {baseline_history['training_time']:.2f} секунд\n")
        
        if history_model_history['memory_usage']:
            avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
            avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
            max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
            
            f.write(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%\n")
            f.write(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB\n")
            f.write(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB\n")
            f.write(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд\n")
        
        f.write("\n=== ГЕНЕРАЦИЯ ===\n")
        f.write("Baseline Model:\n")
        for prompt, generated in baseline_results:
            f.write(f"Prompt: '{prompt}' -> Generated: '{generated}'\n")
        
        f.write("\nHistory Model:\n")
        for prompt, generated in history_results:
            f.write(f"Prompt: '{prompt}' -> Generated: '{generated}'\n")
    
    print(f"Полные результаты сохранены в: {results_path}")

def history_only_mode(config):
    """Режим только для History модели"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== РЕЖИМ: ТОЛЬКО HISTORY (запуск: {timestamp}) ===")
    
    # Загрузка данных
    data_path = download_imdb_data()
    texts = load_imdb_texts(data_path)
    
    # Построение словаря
    print("Построение словаря...")
    vocab = build_vocab(texts, config.vocab_size)
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_path = f"vocab_h{timestamp}.pkl"
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Словарь сохранен как: {vocab_path}")
    
    # Предобработка данных
    print("Предобработка данных...")
    processed_texts = preprocess_data(texts[:1000], vocab, config.max_seq_length)  # Используем только часть данных
    
    # Создание DataLoader
    train_loader = DataLoader(
        processed_texts, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Инициализация History модели
    history_model = EnhancedHistoryAwareGenerator(config, len(vocab)).to(config.device)
    
    # Подсчет параметров
    history_params = count_parameters(history_model)
    print(f"Параметры History модели: {history_params:,}")
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Игнорируем padding
    history_optimizer = torch.optim.Adam(history_model.parameters(), lr=config.learning_rate)
    
    # Обучение History модели
    print("\n=== ОБУЧЕНИЕ HISTORY МОДЕЛИ ===")
    
    history_model_history = {
        'train_loss': [], 'train_perplexity': [],
        'memory_usage': [], 'training_time': 0
    }
    
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        train_loss, train_ppl = train_epoch(
            history_model, train_loader, criterion, history_optimizer, config, epoch, "History"
        )
        
        history_model_history['train_loss'].append(train_loss)
        history_model_history['train_perplexity'].append(train_ppl)
        history_model_history['memory_usage'].append(get_memory_usage())
        
        print(f"History - Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
    
    history_model_history['training_time'] = time.time() - start_time
    print(f"History обучение заняло: {history_model_history['training_time']:.2f} секунд")
    
    # Генерация примеров
    print("\n=== ГЕНЕРАЦИЯ ПРИМЕРОВ ТЕКСТА ===")
    prompts = [
        "How are you?",
        "i really liked",
        "the acting was",
        "the story is"
    ]
    
    history_results = []
    
    print("\n=== HISTORY MODEL ===")
    for prompt in prompts:
        generated = generate_sample(history_model, vocab, inv_vocab, config, prompt)
        history_results.append((prompt, generated))
        print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
    
    # Визуализация весов внимания
    if config.visualize_attention and hasattr(history_model, 'attention_weights'):
        layer_names = []
        for i in range(config.num_layers):
            if history_model._select_history_layers(i):
                layer_names.append(f"Layer {i}")
        
        attention_path = f"history_attention_weights_{timestamp}.png"
        visualize_attention_weights(
            history_model.attention_weights, 
            layer_names,
            attention_path
        )
        print(f"Визуализация внимания сохранена как: {attention_path}")
    
    # Визуализация результатов
    training_plot_path = f"history_training_{timestamp}.png"
    visualize_training_results(None, history_model_history, training_plot_path)
    print(f"График обучения сохранен как: {training_plot_path}")
    
    # Сохранение модели
    if config.save_models:
        history_path = f'history_generator_{timestamp}.pth'
        torch.save(history_model.state_dict(), history_path)
        print(f"Модель сохранена как '{history_path}'")
    
    # Аналитика использования ресурсов
    print("\n=== АНАЛИТИКА ИСПОЛЬЗОВАНИЯ РЕСУРСОВ ===")
    if history_model_history['memory_usage']:
        avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
        avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
        max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
        
        print(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%")
        print(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB")
        print(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB")
        print(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд")
    
    # Сохранение результатов в файл
    results_path = f"history_results_{timestamp}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ===\n")
        f.write(f"Дата и время: {timestamp}\n")
        f.write(f"Режим: Только History\n\n")
        
        f.write("=== МОДЕЛЬ ===\n")
        f.write(f"History параметров: {history_params:,}\n\n")
        
        f.write("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===\n")
        if history_model_history['train_perplexity']:
            f.write(f"History - Финальная перплексия: {history_model_history['train_perplexity'][-1]:.2f}\n")
        
        f.write("\n=== РЕСУРСЫ ===\n")
        if history_model_history['memory_usage']:
            avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
            avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
            max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
            
            f.write(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%\n")
            f.write(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB\n")
            f.write(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB\n")
            f.write(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд\n")
        
        f.write("\n=== ГЕНЕРАЦИЯ ===\n")
        f.write("History Model:\n")
        for prompt, generated in history_results:
            f.write(f"Prompt: '{prompt}' -> Generated: '{generated}'\n")
    
    print(f"Полные результаты сохранены в: {results_path}")

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    parser = argparse.ArgumentParser(description='HistoryGPT - Генеративная модель с историческим вниманием')
    parser.add_argument('--mode', type=str, default='competitive', 
                       choices=['competitive', 'history'],
                       help='Режим работы: competitive (обе модели) или history (только history модель)')
    
    args = parser.parse_args()
    
    config = Config()
    print(f"Используется устройство: {config.device}")
    
    if args.mode == 'competitive':
        competitive_mode(config)
    elif args.mode == 'history':
        history_only_mode(config)

if __name__ == "__main__":
    main()