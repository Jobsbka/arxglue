# hatgpt.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import math
import os

# Конфигурация модели
class GPTConfig:
    vocab_size = 1000
    d_model = 256
    n_heads = 8
    d_ff = 1024
    n_layers = 4
    max_len = 512  # Увеличим длину контекста для кода
    dropout = 0.1
    batch_size = 2
    use_hia = True

config = GPTConfig()

# Регистр для хранения исторических данных
class HistoryRegistry:
    def __init__(self):
        self.embeddings = []
        self.self_attentions = []
        self.ffns = []

    def clear(self):
        for attr in self.__dict__.values():
            if isinstance(attr, list):
                attr.clear()

    def get_historical_kv(self, max_history_tokens=256):
        all_tensors = self.embeddings + self.self_attentions + self.ffns
        if not all_tensors:
            return None
        
        historical_context = torch.cat(all_tensors, dim=1)
        
        if historical_context.size(1) > max_history_tokens:
            historical_context = historical_context[:, -max_history_tokens:, :]
            
        return historical_context

# Улучшенный токенизатор для кода
class CodeTokenizer:
    def __init__(self):
        # Базовые токены
        self.vocab = {}
        self.reverse_vocab = {}
        
        # Специальные токены
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        
        # Заполняем словарь
        self._build_vocab()
        
    def _build_vocab(self):
        # Добавляем специальные токены
        self.vocab = self.special_tokens.copy()
        self.reverse_vocab = {v: k for k, v in self.special_tokens.items()}
        
        # Добавляем ASCII символы
        idx = len(self.vocab)
        for i in range(32, 127):  # Печатные ASCII символы
            char = chr(i)
            self.vocab[char] = idx
            self.reverse_vocab[idx] = char
            idx += 1
            
        # Добавляем кириллические символы
        for i in range(1040, 1104):  # Кириллические символы
            char = chr(i)
            self.vocab[char] = idx
            self.reverse_vocab[idx] = char
            idx += 1
            
        # Добавляем дополнительные символы, часто используемые в коде
        extra_chars = ['\t', '\n', '\r', ' ', ' ', '→', '←', '↑', '↓', '§', '©', '®', '™', '°', '±', '×', '÷', 'π', '∞', '√', '∆', '∑', '∫']
        for char in extra_chars:
            if char not in self.vocab:
                self.vocab[char] = idx
                self.reverse_vocab[idx] = char
                idx += 1
    
    def encode(self, text):
        # Преобразуем текст в последовательность токенов
        tokens = []
        for char in text:
            tokens.append(self.vocab.get(char, self.vocab['<UNK>']))
        return tokens
    
    def decode(self, tokens):
        # Преобразуем токены обратно в текст
        text = []
        for token in tokens:
            if token in self.reverse_vocab:
                text.append(self.reverse_vocab[token])
            else:
                text.append('<UNK>')
        return ''.join(text)
    
    @property
    def vocab_size(self):
        return len(self.vocab)

# Инициализируем токенизатор
tokenizer = CodeTokenizer()
config.vocab_size = tokenizer.vocab_size

# Слой эмбеддингов с записью в регистр
class RegistryEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, registry):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.registry = registry

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        token_embeddings = self.token_emb(x)
        pos_embeddings = self.pos_emb(positions)
        embeddings = token_embeddings + pos_embeddings
        
        self.registry.embeddings.append(embeddings)
        return embeddings

# Многоголовое внимание с записью в регистр
class RegistryMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, registry, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.registry = registry

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)
        
        self.registry.self_attentions.append(output)
        return output

# Полносвязный слой с записью в регистр
class RegistryFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, registry, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.registry = registry

    def forward(self, x):
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        self.registry.ffns.append(output)
        return output

# Слой декодера
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, registry, dropout=0.1):
        super().__init__()
        self.self_attn = RegistryMultiHeadAttention(d_model, n_heads, registry, dropout)
        self.ffn = RegistryFeedForward(d_model, d_ff, registry, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        return x

# Механизм исторического внимания
class HistoryIntegrationAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(attn_output)

        return output

# Основная модель HATGPT
class HATGPT(nn.Module):
    def __init__(self, config, registry):
        super().__init__()
        self.config = config
        self.registry = registry

        self.embeddings = RegistryEmbeddings(config.vocab_size, config.d_model, config.max_len, registry)
        
        self.layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, registry, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.hia = HistoryIntegrationAttention(config.d_model, config.n_heads, config.dropout)
        self.norm_hia = nn.LayerNorm(config.d_model)
        
        self.norm_final = nn.LayerNorm(config.d_model)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, mask=None):
        self.registry.clear()
        
        x = self.embeddings(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        if self.config.use_hia:
            historical_kv = self.registry.get_historical_kv(max_history_tokens=256)
            if historical_kv is not None:
                query_len = x.size(1)
                key_len = historical_kv.size(1)
                cross_mask = torch.ones(query_len, key_len, device=x.device).bool()
                
                hia_output = self.hia(x, historical_kv, historical_kv, cross_mask)
                x = self.norm_hia(x + hia_output)
        
        x = self.norm_final(x)
        logits = self.output_layer(x)
        
        return logits

    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.9, 
                 repetition_penalty=1.0, do_sample=True):
        self.eval()
        self.registry.clear()
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Создаем маску для текущей последовательности
                seq_len = generated.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len, device=generated.device), diagonal=1).bool()
                mask = ~mask
                
                logits = self(generated, mask)
                next_token_logits = logits[:, -1, :] / temperature
                
                if repetition_penalty != 1.0:
                    for token_id in set(generated[0].tolist()):
                        next_token_logits[0, token_id] /= repetition_penalty
                
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                if generated.size(1) >= self.config.max_len:
                    break
        
        return generated

# Датасет для кода
class CodeDataset(Dataset):
    def __init__(self, file_path, seq_length, tokenizer):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Токенизируем весь текст
        tokens = tokenizer.encode(text)
        
        self.data = []
        
        # Создаем последовательности для обучения
        step_size = seq_length // 2
        for i in range(0, len(tokens) - seq_length, step_size):
            input_seq = tokens[i:i+seq_length]
            target_seq = tokens[i+1:i+seq_length+1]
            self.data.append((input_seq, target_seq))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)

# Функция для создания маски
def create_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return ~mask

# Функция для обработки батчей
def collate_fn(batch):
    input_batch, target_batch = zip(*batch)
    max_len = max(len(seq) for seq in input_batch)
    
    padded_inputs = []
    padded_targets = []
    
    for input_seq, target_seq in zip(input_batch, target_batch):
        pad_len = max_len - len(input_seq)
        padded_inputs.append(torch.cat([input_seq, torch.full((pad_len,), tokenizer.vocab['<PAD>'], dtype=torch.long)]))
        padded_targets.append(torch.cat([target_seq, torch.full((pad_len,), tokenizer.vocab['<PAD>'], dtype=torch.long)]))
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)

# Функция для обучения модели
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        seq_len = inputs.size(1)
        mask = create_mask(seq_len, device)
        
        optimizer.zero_grad()
        outputs = model(inputs, mask)
        
        # Игнорируем паддинг в вычислении потерь
        loss_mask = targets != tokenizer.vocab['<PAD>']
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        torch.cuda.empty_cache()
    
    return total_loss / len(dataloader)

# Функция для оценки модели
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            seq_len = inputs.size(1)
            mask = create_mask(seq_len, device)
            
            outputs = model(inputs, mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            total_loss += loss.item()
            
            torch.cuda.empty_cache()
    
    return total_loss / len(dataloader)

# Основная функция
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Создаем датасет
    dataset = CodeDataset("text.txt", seq_length=config.max_len, tokenizer=tokenizer)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    registry = HistoryRegistry()
    model = HATGPT(config, registry).to(device)
    
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Vocabulary size: {tokenizer.vocab_size} characters")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<PAD>'])
    
    train_losses = []
    val_losses = []
    
    try:
        for epoch in range(30):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            if (epoch + 1) % 5 == 0:
                prompts = ["def ", "class ", "import ", "Кто такой Румата"]
                for prompt in prompts:
                    # Токенизируем промпт
                    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
                    
                    generated = model.generate(input_ids, max_length=50, temperature=0.8)
                    generated_text = tokenizer.decode(generated[0].tolist())
                    
                    print(f"Prompt: '{prompt}' -> Generated: '{generated_text}'")
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("Out of memory error! Trying to recover...")
            torch.cuda.empty_cache()
            config.batch_size = 1
            config.max_len = 256
            print("Reduced batch size and sequence length. Please try again.")
        else:
            raise e
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': config.__dict__
    }, 'hatgpt_model.pth')
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()

if __name__ == "__main__":
    main()