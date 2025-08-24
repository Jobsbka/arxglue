# demo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Callable, Optional, Dict, List, Tuple, Union, Set, Type
import numpy as np
import argparse
import sys
import os

# Добавляем путь к текущей директории для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем наши сети
from mnist import create_simple_network
from transformer import create_transformer

def generate_dummy_data(vocab_size, seq_length, batch_size, num_classes):
    """Генерация случайных данных для трансформера"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Создаем маску (все токены действительны)
    mask = torch.ones(batch_size, seq_length, seq_length).bool()
    
    return input_ids, mask, labels

def train_mnist(model, epochs=5, batch_size=32, lr=0.001):
    """Обучение простой сети на MNIST"""
    print("Подготовка данных MNIST...")
    
    # Для демо используем случайные данные вместо реального MNIST
    train_data = torch.randn(1000, 1, 28, 28)
    train_labels = torch.randint(0, 10, (1000,))
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print("Начало обучения...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Прямой проход
            output = model(image=data)
            loss = criterion(output['logits'], target)
            
            # Обратный проход
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} завершена | Средний loss: {avg_loss:.4f}')
    
    print("Обучение завершено!")
    return model

def train_transformer(model, epochs=5, batch_size=16, lr=0.0001):
    """Обучение трансформера на случайных данных"""
    print("Подготовка данных для трансформера...")
    
    vocab_size = 10000
    seq_length = 32
    num_classes = 10
    
    # Генерируем данные
    input_ids, mask, labels = generate_dummy_data(vocab_size, seq_length, 200, num_classes)
    train_dataset = TensorDataset(input_ids, mask, labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print("Начало обучения трансформера...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (input_batch, mask_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Прямой проход
            output = model(input_ids=input_batch, mask=mask_batch)
            loss = criterion(output['logits'], target_batch)
            
            # Обратный проход
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} завершена | Средний loss: {avg_loss:.4f}')
    
    print("Обучение трансформера завершено!")
    return model

def inference_mnist(model, num_samples=5):
    """Инференс на простой сети"""
    print("Запуск инференса на MNIST...")
    
    # Генерируем тестовые данные
    test_data = torch.randn(num_samples, 1, 28, 28)
    
    # Переводим модель в режим оценки
    model.eval()
    
    with torch.no_grad():
        output = model(image=test_data)
        predictions = torch.softmax(output['logits'], dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)
    
    print("Результаты инференса:")
    for i in range(num_samples):
        print(f"Пример {i+1}: предсказанный класс = {predicted_classes[i].item()}")
    
    return predicted_classes

def inference_transformer(model, num_samples=3):
    """Инференс на трансформере"""
    print("Запуск инференса на трансформере...")
    
    # Генерируем тестовые данные
    vocab_size = 10000
    seq_length = 32
    input_ids, mask, _ = generate_dummy_data(vocab_size, seq_length, num_samples, 10)
    
    # Переводим модель в режим оценки
    model.eval()
    
    with torch.no_grad():
        output = model(input_ids=input_ids, mask=mask)
        predictions = torch.softmax(output['logits'], dim=1)
        predicted_classes = torch.argmax(predictions, dim=1)
    
    print("Результаты инференса трансформера:")
    for i in range(num_samples):
        print(f"Пример {i+1}: предсказанный класс = {predicted_classes[i].item()}")
    
    return predicted_classes

def main():
    parser = argparse.ArgumentParser(description='Демонстрация нейросетей на GlueTorch')
    parser.add_argument('--model', type=str, choices=['mnist', 'transformer'], 
                        required=True, help='Выбор модели: mnist или transformer')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], 
                        required=True, help='Режим работы: обучение или инференс')
    parser.add_argument('--epochs', type=int, default=5, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')
    parser.add_argument('--lr', type=float, default=0.001, help='Скорость обучения')
    
    args = parser.parse_args()
    
    # Создаем выбранную модель
    if args.model == 'mnist':
        print("Создание простой сети для MNIST...")
        model = create_simple_network()
    else:
        print("Создание трансформера...")
        model = create_transformer()
    
    # Выполняем выбранный режим
    if args.mode == 'train':
        if args.model == 'mnist':
            model = train_mnist(model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        else:
            model = train_transformer(model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        
        # Сохраняем обученную модель
        model_path = f"{args.model}_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Модель сохранена в {model_path}")
    else:
        # Загружаем веса для инференса (если есть)
        model_path = f"{args.model}_model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Загружены веса из {model_path}")
        else:
            print("Предупреждение: веса модели не найдены, используется случайная инициализация")
        
        if args.model == 'mnist':
            inference_mnist(model)
        else:
            inference_transformer(model)

if __name__ == "__main__":
    main()