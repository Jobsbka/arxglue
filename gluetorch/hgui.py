import sys
import os
import torch
import pickle
import random
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QTextEdit, 
                             QListWidget, QLabel, QFileDialog, QComboBox,
                             QSplitter, QTableWidget, QTableWidgetItem,
                             QHeaderView, QProgressBar, QMessageBox, QGroupBox,
                             QLineEdit, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from collections import defaultdict

# Импортируем классы из вашего кода
try:
    from hgpt import BaselineGenerator, EnhancedHistoryAwareGenerator, Config
except ImportError:
    print("Ошибка: Не удалось импортировать классы из hgpt.py")
    sys.exit(1)

class ModelLoader:
    """Класс для загрузки и управления моделями с определением их типа"""
    def __init__(self):
        self.models = {}
        self.vocabs = {}
        self.config = Config()
        
    def load_model(self, model_path, vocab_path, name):
        """Загрузка модели и словаря с автоматическим определением типа модели"""
        try:
            # Загрузка словаря
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
            
            # Определяем тип модели по имени файла
            if 'baseline' in model_path.lower():
                model = BaselineGenerator(self.config, len(vocab)).to(self.config.device)
                model_type = 'baseline'
            else:
                model = EnhancedHistoryAwareGenerator(self.config, len(vocab)).to(self.config.device)
                model_type = 'history'
            
            # Загрузка весов модели
            if self.config.device.type == 'cpu':
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(model_path)
            
            # Загрузка state_dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            
            # Сохранение в кеше
            self.models[name] = model
            self.vocabs[name] = vocab
            
            return True, f"Модель {name} ({model_type}) успешно загружена"
            
        except Exception as e:
            return False, f"Ошибка загрузки: {str(e)}"
    
    def get_model(self, name):
        """Получение модели по имени"""
        return self.models.get(name)
    
    def get_vocab(self, name):
        """Получение словаря по имени"""
        return self.vocabs.get(name)
    
    def preprocess_input(self, model_name, text):
        """Преобразование текста в последовательность индексов"""
        vocab = self.get_vocab(model_name)
        tokens = text.lower().split()
        sequence = [vocab.get('<sos>', 2)] + [vocab.get(token, 1) for token in tokens]
        return torch.tensor(sequence).unsqueeze(0).to(self.config.device)
    
    def generate_response(self, model_name, input_tensor, max_length=50, temperature=0.8):
        """Генерация ответа для заданного промпта"""
        model = self.get_model(model_name)
        vocab = self.get_vocab(model_name)
        
        # Генерация ответа
        with torch.no_grad():
            generated = input_tensor.clone()
            
            for _ in range(max_length):
                logits = model(generated)
                next_token_logits = logits[:, -1, :] / temperature
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == vocab.get('<eos>', 3):
                    break
            
            # Преобразование обратно в текст
            tokens = generated.squeeze(0).cpu().numpy()
            words = []
            for idx in tokens:
                if idx == vocab.get('<eos>', 3):
                    break
                if idx not in [vocab.get('<sos>', 2), vocab.get('<pad>', 0)]:
                    word = [k for k, v in vocab.items() if v == idx]
                    if word:
                        words.append(word[0])
                    else:
                        words.append('<unk>')
            
            return ' '.join(words)
    
    def tokens_to_text(self, model_name, tokens):
        """Преобразование индексов в текст"""
        vocab = self.get_vocab(model_name)
        inv_vocab = {v: k for k, v in vocab.items()}
        
        tokens = tokens.cpu().numpy()
        words = []
        for idx in tokens:
            if idx == vocab.get('<eos>', 3):
                break
            if idx not in [vocab.get('<sos>', 2), vocab.get('<pad>', 0)]:
                words.append(inv_vocab.get(idx, '<unk>'))
        return ' '.join(words)


class ABTestWorker(QThread):
    """Рабочий поток для выполнения A/B тестирования"""
    progress = pyqtSignal(int)
    result = pyqtSignal(object)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, model_loader, model1_name, model2_name, prompts, num_responses=3):
        super().__init__()
        self.model_loader = model_loader
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.prompts = prompts
        self.num_responses = num_responses
    
    def run(self):
        try:
            results = []
            total = len(self.prompts) * self.num_responses
            
            for i, prompt in enumerate(self.prompts):
                for j in range(self.num_responses):
                    # Генерация ответа первой моделью
                    input_tensor = self.model_loader.preprocess_input(self.model1_name, prompt)
                    response1 = self.model_loader.generate_response(self.model1_name, input_tensor)
                    
                    # Генерация ответа второй моделью
                    input_tensor = self.model_loader.preprocess_input(self.model2_name, prompt)
                    response2 = self.model_loader.generate_response(self.model2_name, input_tensor)
                    
                    results.append({
                        'prompt': prompt,
                        'model1': self.model1_name,
                        'model2': self.model2_name,
                        'response1': response1,
                        'response2': response2,
                        'iteration': j
                    })
                    
                    # Отправка прогресса
                    progress = int((i * self.num_responses + j + 1) / total * 100)
                    self.progress.emit(progress)
            
            self.result.emit(results)
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_loader = ModelLoader()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('HistoryGPT A/B Testing Tool')
        self.setGeometry(100, 100, 1400, 900)
        
        # Создаем вкладки
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Вкладка A/B тестирования
        self.ab_tab = QWidget()
        self.setup_ab_tab()
        self.tabs.addTab(self.ab_tab, "A/B Testing")
        
        # Вкладка чата
        self.chat_tab = QWidget()
        self.setup_chat_tab()
        self.tabs.addTab(self.chat_tab, "Chat Interface")
        
        # Вкладка управления моделями
        self.models_tab = QWidget()
        self.setup_models_tab()
        self.tabs.addTab(self.models_tab, "Model Management")
    
    def setup_ab_tab(self):
        layout = QVBoxLayout()
        
        # Верхняя панель с выбором моделей и промптов
        top_panel = QHBoxLayout()
        
        # Выбор первой модели
        model1_group = QGroupBox("Model 1")
        model1_layout = QVBoxLayout()
        self.model1_combo = QComboBox()
        model1_layout.addWidget(QLabel("Model:"))
        model1_layout.addWidget(self.model1_combo)
        self.model1_btn = QPushButton("Load Model 1")
        self.model1_btn.clicked.connect(lambda: self.load_model(1))
        model1_layout.addWidget(self.model1_btn)
        model1_group.setLayout(model1_layout)
        top_panel.addWidget(model1_group)
        
        # Выбор второй модели
        model2_group = QGroupBox("Model 2")
        model2_layout = QVBoxLayout()
        self.model2_combo = QComboBox()
        model2_layout.addWidget(QLabel("Model:"))
        model2_layout.addWidget(self.model2_combo)
        self.model2_btn = QPushButton("Load Model 2")
        self.model2_btn.clicked.connect(lambda: self.load_model(2))
        model2_layout.addWidget(self.model2_btn)
        model2_group.setLayout(model2_layout)
        top_panel.addWidget(model2_group)
        
        # Выбор промптов
        prompts_group = QGroupBox("Prompts")
        prompts_layout = QVBoxLayout()
        self.prompts_list = QListWidget()
        prompts_layout.addWidget(self.prompts_list)
        
        prompts_btns = QHBoxLayout()
        self.load_prompts_btn = QPushButton("Load Prompts")
        self.load_prompts_btn.clicked.connect(self.load_prompts)
        prompts_btns.addWidget(self.load_prompts_btn)
        
        self.clear_prompts_btn = QPushButton("Clear")
        self.clear_prompts_btn.clicked.connect(self.prompts_list.clear)
        prompts_btns.addWidget(self.clear_prompts_btn)
        
        prompts_layout.addLayout(prompts_btns)
        prompts_group.setLayout(prompts_layout)
        top_panel.addWidget(prompts_group)
        
        layout.addLayout(top_panel)
        
        # Параметры тестирования
        params_group = QGroupBox("Test Parameters")
        params_layout = QHBoxLayout()
        
        params_layout.addWidget(QLabel("Responses per prompt:"))
        self.num_responses = QSpinBox()
        self.num_responses.setRange(1, 10)
        self.num_responses.setValue(3)
        params_layout.addWidget(self.num_responses)
        
        params_layout.addWidget(QLabel("Max length:"))
        self.max_length = QSpinBox()
        self.max_length.setRange(10, 200)
        self.max_length.setValue(50)
        params_layout.addWidget(self.max_length)
        
        params_layout.addWidget(QLabel("Temperature:"))
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.1, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.8)
        params_layout.addWidget(self.temperature)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Кнопка запуска теста
        self.run_test_btn = QPushButton("Run A/B Test")
        self.run_test_btn.clicked.connect(self.run_ab_test)
        layout.addWidget(self.run_test_btn)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Таблица результатов
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["Prompt", "Iteration", "Model 1 Response", "Model 2 Response", "Preferred", "Notes"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.results_table)
        
        # Кнопки экспорта
        export_btns = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export to CSV")
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        self.export_csv_btn.setEnabled(False)
        export_btns.addWidget(self.export_csv_btn)
        
        self.export_blind_btn = QPushButton("Export Blind Test")
        self.export_blind_btn.clicked.connect(self.export_blind_test)
        self.export_blind_btn.setEnabled(False)
        export_btns.addWidget(self.export_blind_btn)
        
        layout.addLayout(export_btns)
        
        self.ab_tab.setLayout(layout)
        
        # Переменные для хранения результатов
        self.test_results = []
    
    def setup_chat_tab(self):
        layout = QVBoxLayout()
        
        # Выбор модели для чата
        chat_top = QHBoxLayout()
        chat_top.addWidget(QLabel("Chat Model:"))
        self.chat_model_combo = QComboBox()
        chat_top.addWidget(self.chat_model_combo)
        self.load_chat_model_btn = QPushButton("Load Chat Model")
        self.load_chat_model_btn.clicked.connect(self.load_chat_model)
        chat_top.addWidget(self.load_chat_model_btn)
        layout.addLayout(chat_top)
        
        # Область чата
        chat_splitter = QSplitter(Qt.Vertical)
        
        # История чата
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        chat_splitter.addWidget(self.chat_history)
        
        # Ввод сообщения
        chat_bottom = QWidget()
        chat_bottom_layout = QVBoxLayout()
        
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(100)
        chat_bottom_layout.addWidget(self.chat_input)
        
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_chat_message)
        chat_bottom_layout.addWidget(send_btn)
        
        chat_bottom.setLayout(chat_bottom_layout)
        chat_splitter.addWidget(chat_bottom)
        
        chat_splitter.setSizes([600, 200])
        layout.addWidget(chat_splitter)
        
        self.chat_tab.setLayout(layout)
    
    def setup_models_tab(self):
        layout = QVBoxLayout()
        
        # Список загруженных моделей
        layout.addWidget(QLabel("Loaded Models:"))
        self.models_list = QListWidget()
        layout.addWidget(self.models_list)
        
        # Кнопка обновления списка
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.update_models_list)
        layout.addWidget(refresh_btn)
        
        self.models_tab.setLayout(layout)
    
    def load_model(self, model_num):
        """Загрузка модели"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.pth)")
        
        if not model_path:
            return
        
        vocab_path, _ = QFileDialog.getOpenFileName(
            self, "Select Vocab File", "", "Vocab Files (*.pkl)")
        
        if not vocab_path:
            return
        
        # Извлекаем имя модели из пути
        model_name = os.path.basename(model_path)
        
        # Загружаем модель
        success, message = self.model_loader.load_model(model_path, vocab_path, model_name)
        
        if success:
            # Обновляем комбобоксы
            if model_num == 1:
                self.model1_combo.addItem(model_name)
                self.model1_combo.setCurrentText(model_name)
            else:
                self.model2_combo.addItem(model_name)
                self.model2_combo.setCurrentText(model_name)
            
            self.chat_model_combo.addItem(model_name)
            self.update_models_list()
            
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Error", message)
    
    def load_chat_model(self):
        """Загрузка модели для чата"""
        model_name = self.chat_model_combo.currentText()
        if not model_name:
            QMessageBox.warning(self, "Error", "Please select a model first")
            return
        
        self.chat_history.append(f"Chat model loaded: {model_name}")
    
    def load_prompts(self):
        """Загрузка промптов из файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Prompts File", "", "Text Files (*.txt)")
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            self.prompts_list.clear()
            self.prompts_list.addItems(prompts)
            
            QMessageBox.information(self, "Success", f"Loaded {len(prompts)} prompts")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load prompts: {str(e)}")
    
    def run_ab_test(self):
        """Запуск A/B теста"""
        model1_name = self.model1_combo.currentText()
        model2_name = self.model2_combo.currentText()
        
        if not model1_name or not model2_name:
            QMessageBox.warning(self, "Error", "Please load both models first")
            return
        
        prompts = [self.prompts_list.item(i).text() for i in range(self.prompts_list.count())]
        if not prompts:
            QMessageBox.warning(self, "Error", "Please load prompts first")
            return
        
        # Получаем параметры теста
        num_responses = self.num_responses.value()
        max_length = self.max_length.value()
        temperature = self.temperature.value()
        
        # Обновляем конфиг
        self.model_loader.config.gen_length = max_length
        self.model_loader.config.temperature = temperature
        
        # Создаем и запускаем рабочий поток
        self.worker = ABTestWorker(
            self.model_loader, 
            model1_name, 
            model2_name, 
            prompts,
            num_responses
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.result.connect(self.display_results)
        self.worker.finished.connect(self.test_finished)
        self.worker.error.connect(self.test_error)
        self.worker.start()
        
        self.run_test_btn.setEnabled(False)
        self.export_csv_btn.setEnabled(False)
        self.export_blind_btn.setEnabled(False)
    
    def display_results(self, results):
        """Отображение результатов теста"""
        self.test_results = results
        self.results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(result['prompt']))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(result['iteration'])))
            self.results_table.setItem(i, 2, QTableWidgetItem(result['response1']))
            self.results_table.setItem(i, 3, QTableWidgetItem(result['response2']))
            
            # Добавляем комбобокс для выбора предпочтительного ответа
            combo = QComboBox()
            combo.addItems(["No preference", "Model 1", "Model 2", "Tie"])
            self.results_table.setCellWidget(i, 4, combo)
            
            # Добавляем поле для заметок
            notes_item = QTableWidgetItem("")
            self.results_table.setItem(i, 5, notes_item)
    
    def test_finished(self):
        """Завершение теста"""
        self.run_test_btn.setEnabled(True)
        self.export_csv_btn.setEnabled(True)
        self.export_blind_btn.setEnabled(True)
        QMessageBox.information(self, "Success", "A/B test completed")
    
    def test_error(self, error_msg):
        """Ошибка при выполнении теста"""
        self.run_test_btn.setEnabled(True)
        QMessageBox.warning(self, "Error", f"Test failed: {error_msg}")
    
    def export_to_csv(self):
        """Экспорт результатов в CSV файл"""
        if not self.test_results:
            QMessageBox.warning(self, "Error", "No test results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "", "CSV Files (*.csv)")
        
        if not file_path:
            return
        
        try:
            # Собираем данные из таблицы
            data = []
            for i in range(self.results_table.rowCount()):
                prompt = self.results_table.item(i, 0).text()
                iteration = self.results_table.item(i, 1).text()
                response1 = self.results_table.item(i, 2).text()
                response2 = self.results_table.item(i, 3).text()
                
                # Получаем выбранное предпочтение
                combo = self.results_table.cellWidget(i, 4)
                preferred = combo.currentText() if combo else "No preference"
                
                # Получаем заметки
                notes_item = self.results_table.item(i, 5)
                notes = notes_item.text() if notes_item else ""
                
                data.append({
                    'prompt': prompt,
                    'iteration': iteration,
                    'model1_response': response1,
                    'model2_response': response2,
                    'preferred': preferred,
                    'notes': notes,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Создаем DataFrame и сохраняем в CSV
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            QMessageBox.information(self, "Success", f"Results exported to {file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export results: {str(e)}")
    
    def export_blind_test(self):
        """Экспорт слепого теста для оценки"""
        if not self.test_results:
            QMessageBox.warning(self, "Error", "No test results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Blind Test File", "", "CSV Files (*.csv)")
        
        if not file_path:
            return
        
        try:
            blind_test_data = []
            
            for i, result in enumerate(self.test_results):
                # Случайным образом меняем порядок ответов
                if random.random() > 0.5:
                    response_a = result['response1']
                    response_b = result['response2']
                    model_a = 'A'
                    model_b = 'B'
                else:
                    response_a = result['response2']
                    response_b = result['response1']
                    model_a = 'B'
                    model_b = 'A'
                
                blind_test_data.append({
                    'id': i,
                    'prompt': result['prompt'],
                    'iteration': result['iteration'],
                    'response_a': response_a,
                    'response_b': response_b,
                    'model_a': model_a,
                    'model_b': model_b,
                    'preferred': '',  # Для заполнения оценщиком
                    'notes': ''       # Для комментариев оценщика
                })
            
            # Сохраняем в CSV
            df = pd.DataFrame(blind_test_data)
            df.to_csv(file_path, index=False, encoding='utf-8')
            
            QMessageBox.information(self, "Success", f"Blind test exported to {file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export blind test: {str(e)}")
    
    def send_chat_message(self):
        """Отправка сообщения в чате"""
        message = self.chat_input.toPlainText().strip()
        if not message:
            return
        
        model_name = self.chat_model_combo.currentText()
        if not model_name:
            QMessageBox.warning(self, "Error", "Please load a chat model first")
            return
        
        # Добавляем сообщение пользователя в историю
        self.chat_history.append(f"You: {message}")
        self.chat_input.clear()
        
        # Генерируем ответ
        try:
            input_tensor = self.model_loader.preprocess_input(model_name, message)
            response = self.model_loader.generate_response(model_name, input_tensor)
            self.chat_history.append(f"Bot: {response}")
        except Exception as e:
            self.chat_history.append(f"Error: {str(e)}")
    
    def update_models_list(self):
        """Обновление списка загруженных моделей"""
        self.models_list.clear()
        for model_name in self.model_loader.models.keys():
            self.models_list.addItem(model_name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())