# thtgpt.py
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
    d_ff = 2048
    n_layers = 4
    max_len = 512
    dropout = 0.1
    batch_size = 8
    use_hia = True
    use_persistent_history = True

config = GPTConfig()

# Новый регистр с Pre-store и Post-store
class PersistentHistoryRegistry:
    def __init__(self):
        # Pre-store: сырые данные с текущего прохода
        self.pre_store_embeddings = []
        self.pre_store_self_attentions = []
        self.pre_store_ffns = []
        
        # Post-store: очищенные данные для HIA
        self.post_store_embeddings = []
        self.post_store_self_attentions = []
        self.post_store_ffns = []

    def clear_pre_store(self):
        self.pre_store_embeddings.clear()
        self.pre_store_self_attentions.clear()
        self.pre_store_ffns.clear()

    def clear_post_store(self):
        self.post_store_embeddings.clear()
        self.post_store_self_attentions.clear()
        self.post_store_ffns.clear()

    def clear_all(self):
        self.clear_pre_store()
        self.clear_post_store()

    def prepare_for_hia(self, current_seq_len):
        """
        Переносит и 'очищает' данные из pre-store в post-store.
        current_seq_len - длина текущей последовательности.
        """
        self.clear_post_store()
        
        # Обрабатываем эмбеддинги
        for emb in self.pre_store_embeddings:
            safe_emb = emb.detach().clone()
            self.post_store_embeddings.append(safe_emb)

        # Обрабатываем self-attention и FFN с маскировкой
        for tensor in self.pre_store_self_attentions:
            batch, seq_len, d_model = tensor.shape
            mask = torch.ones_like(tensor)
            for i in range(seq_len):
                if i >= current_seq_len:
                    mask[:, i, :] = 0  # Обнуляем будущие токены
            safe_tensor = tensor.detach().clone() * mask
            self.post_store_self_attentions.append(safe_tensor)

        for tensor in self.pre_store_ffns:
            batch, seq_len, d_model = tensor.shape
            mask = torch.ones_like(tensor)
            for i in range(seq_len):
                if i >= current_seq_len:
                    mask[:, i, :] = 0  # Обнуляем будущие токены
            safe_tensor = tensor.detach().clone() * mask
            self.post_store_ffns.append(safe_tensor)

    def get_historical_kv(self, max_history_tokens=1024):
        """Возвращает исторические ключи и значения только из post-store"""
        all_tensors = (self.post_store_embeddings + 
                      self.post_store_self_attentions + 
                      self.post_store_ffns)
        
        if not all_tensors:
            return None
            
        historical_context = torch.cat(all_tensors, dim=1)
        
        if historical_context.size(1) > max_history_tokens:
            historical_context = historical_context[:, -max_history_tokens:, :]
            
        return historical_context

# Улучшенный токенизатор для кода
class CodeTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<code>': 4,
            '<literature>': 5,
            '<chat>': 6,
            '<logic>': 7,
            '<end>': 8
        }
        self._build_vocab()
        
    def _build_vocab(self):
        # Добавляем специальные токены
        self.vocab = self.special_tokens.copy()
        self.reverse_vocab = {v: k for k, v in self.special_tokens.items()}
        
        idx = len(self.vocab)
        # Добавляем ASCII символы
        for i in range(32, 127):
            char = chr(i)
            self.vocab[char] = idx
            self.reverse_vocab[idx] = char
            idx += 1
            
        # Добавляем кириллические символы
        for i in range(1040, 1104):
            char = chr(i)
            self.vocab[char] = idx
            self.reverse_vocab[idx] = char
            idx += 1
            
        # Дополнительные специальные символы
        extra_chars = ['\t', '\n', '\r', ' ', ' ', '→', '←', '↑', '↓', '§', '©', '®', '™', '°', '±', '×', '÷', 'π', '∞', '√', '∆', '∑', '∫']
        for char in extra_chars:
            if char not in self.vocab:
                self.vocab[char] = idx
                self.reverse_vocab[idx] = char
                idx += 1
    
    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            # Проверяем специальные токены
            found = False
            for special in self.special_tokens:
                if text.startswith(special, i):
                    tokens.append(self.vocab[special])
                    i += len(special)
                    found = True
                    break
            
            if not found:
                # Обычный символ
                char = text[i]
                tokens.append(self.vocab.get(char, self.vocab['<UNK>']))
                i += 1
        return tokens
    
    def decode(self, tokens):
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

tokenizer = CodeTokenizer()
config.vocab_size = tokenizer.vocab_size

# Слой эмбеддингов с записью в pre-store
class PersistentRegistryEmbeddings(nn.Module):
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
        
        if config.use_persistent_history:
            self.registry.pre_store_embeddings.append(embeddings.detach().clone())
        return embeddings

# Многоголовое внимание с записью в pre-store
class PersistentRegistryMultiHeadAttention(nn.Module):
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
        
        if config.use_persistent_history:
            self.registry.pre_store_self_attentions.append(output.detach().clone())
        return output

# Полносвязный слой с записью в pre-store
class PersistentRegistryFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, registry, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.registry = registry

    def forward(self, x):
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        if config.use_persistent_history:
            self.registry.pre_store_ffns.append(output.detach().clone())
        return output

# Слой декодера
class PersistentDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, registry, dropout=0.1):
        super().__init__()
        self.self_attn = PersistentRegistryMultiHeadAttention(d_model, n_heads, registry, dropout)
        self.ffn = PersistentRegistryFeedForward(d_model, d_ff, registry, dropout)
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
class PersistentHistoryIntegrationAttention(nn.Module):
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

# Основная модель THTGPT
class THTGPT(nn.Module):
    def __init__(self, config, registry):
        super().__init__()
        self.config = config
        self.registry = registry

        self.embeddings = PersistentRegistryEmbeddings(config.vocab_size, config.d_model, config.max_len, registry)
        
        self.layers = nn.ModuleList([
            PersistentDecoderLayer(config.d_model, config.n_heads, config.d_ff, registry, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.hia = PersistentHistoryIntegrationAttention(config.d_model, config.n_heads, config.dropout)
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

    def forward(self, x, mask=None, is_generation=False):
        self.registry.clear_pre_store()  # Очищаем pre-store в начале forward pass
        
        x = self.embeddings(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        if self.config.use_hia and self.config.use_persistent_history:
            # Подготавливаем данные для HIA
            self.registry.prepare_for_hia(x.size(1))
            historical_kv = self.registry.get_historical_kv(max_history_tokens=1024)
            
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
        self.registry.clear_all()  # Очищаем всю историю перед генерацией
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for i in range(max_length):
                seq_len = generated.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len, device=generated.device), diagonal=1).bool()
                mask = ~mask
                
                logits = self(generated, mask, is_generation=True)
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
    def __init__(self, file_paths, seq_length, tokenizer, is_train=True):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        file_path = file_paths['train'] if is_train else file_paths['val']
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Разбиваем текст на блоки
        blocks = self._split_into_blocks(text)
        
        # Токенизируем блоки
        self.tokens = []
        for block_type, content in blocks:
            # Добавляем токен типа блока
            self.tokens.append(self.tokenizer.vocab[f'<{block_type}>'])
            # Добавляем содержимое блока
            self.tokens.extend(self.tokenizer.encode(content))
            # Добавляем конечный токен
            self.tokens.append(self.tokenizer.vocab['<end>'])
        
        # Создаем последовательности для обучения
        self.data = []
        step_size = seq_length if not is_train else seq_length // 2
        for i in range(0, len(self.tokens) - seq_length, step_size):
            input_seq = self.tokens[i:i+seq_length]
            target_seq = self.tokens[i+1:i+seq_length+1]
            self.data.append((input_seq, target_seq))
    
    def _split_into_blocks(self, text):
        blocks = []
        pattern = r'<(code|literature|chat|logic)>(.*?)<end>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            block_type, content = match
            # Убираем лишние пробелы и переносы
            content = content.strip()
            blocks.append((block_type, content))
        
        return blocks
    
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
    
    file_paths = {
        'train': 'textt.txt',
        'val': 'textv.txt'
    }
    
    train_dataset = CodeDataset(file_paths, seq_length=config.max_len, 
                               tokenizer=tokenizer, is_train=True)
    val_dataset = CodeDataset(file_paths, seq_length=config.max_len, 
                             tokenizer=tokenizer, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, collate_fn=collate_fn)
    
    registry = PersistentHistoryRegistry()
    model = THTGPT(config, registry).to(device)
    
    print(f"Train dataset size: {len(train_dataset)} sequences")
    print(f"Validation dataset size: {len(val_dataset)} sequences")
    print(f"Vocabulary size: {tokenizer.vocab_size} characters")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("Using PERSISTENT history mechanism with Pre-store/Post-store")
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.85)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<PAD>'])
    
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    try:
        for epoch in range(50):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Сохраняем лучшую модель
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'best_model.pth')
                print(f"Новая лучшая модель сохранена с Val Loss = {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Ранняя остановка: Val Loss не улучшается")
                    break


            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f'checkpoint_epoch_{epoch+1}.pth')
            
            if (epoch + 1) % 5 == 0:
                prompts = ["<literature> Рама стоял ", "<code> Class MyClass ", "<chat> Отлично! Вот ", "<literature> В НИИЧАВО"]
                for prompt in prompts:
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
    }, 'thtgpt_model.pth')
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()

if __name__ == "__main__":
    main()