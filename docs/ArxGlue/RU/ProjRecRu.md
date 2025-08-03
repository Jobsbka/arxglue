### 📁 Структура проекта (рекомендуемая)
```
my_project/
├── config.py       # Конфигурация системы
├── connects.py     # Декларация связей между компонентами
├── components.py   # Чистая бизнес-логика
└── executor.py     # Ваш кастомный исполнитель (опционально)
```

---

### 1. `config.py` - Стандартные конфиги
```python
# config.py
DATABASE_URL = "postgresql://user:pass@localhost/db"
API_KEY = "xxxxxx"
LOG_LEVEL = "DEBUG"

# Конфиг для конкретного пайплайна
DATA_PIPELINE = {
    "input_path": "/data/raw",
    "output_path": "/data/processed"
}
```

---

### 2. `components.py` - Чистая логика компонентов
```python
# components.py
from .config import DATABASE_URL, DATA_PIPELINE  # Только чтение конфигов!

def load_data(source: str) -> list:
    """Загрузка данных из источника"""
    # Логика загрузки (не зависит от других компонентов)
    print(f"Loading from {source}...")
    return [1, 2, 3]

def clean_data(data: list) -> list:
    """Очистка данных"""
    return [x for x in data if x > 1]

def analyze_data(data: list) -> dict:
    """Анализ данных"""
    return {"mean": sum(data)/len(data)}

def save_report(report: dict) -> None:
    """Сохранение отчета"""
    output_path = DATA_PIPELINE["output_path"]
    print(f"Saving report to {output_path}: {report}")

# Компонент с состоянием (если нужно)
class DatabaseWriter:
    def __init__(self, connection_str):
        self.conn = create_connection(connection_str)
    
    def __call__(self, data: dict):
        """Запись в БД"""
        self.conn.execute("INSERT ...", data)
```

---

### 3. `connects.py` - Декларация связей
```python
# connects.py
from arxglue import connect
from .components import load_data, clean_data, analyze_data, save_report
from .config import DATA_PIPELINE

# Основной пайплайн обработки
main_pipeline = [
    connect(
        source=load_data, 
        target=clean_data,
        transformer=lambda source_out: source_out(DATA_PIPELINE["input_path"])
    ),
    connect(clean_data, analyze_data),
    connect(analyze_data, save_report)
]

# Ветвление данных
monitoring = [
    connect(clean_data, log_stats, transformer=format_for_logging),
    connect(analyze_data, alert_system)
]
```

---

### 4. `executor.py` - Ваш исполнитель (пример)
```python
# executor.py
from .connects import main_pipeline
from .components import DatabaseWriter
from arxglue import ContextProtocol

class CustomContext(ContextProtocol):
    def __init__(self, input_data):
        super().__init__(input_data)
        self.timestamps = []
        
    def add_timestamp(self):
        self.timestamps.append(time.time())

def execute_pipeline(pipeline, input_data):
    """Кастомный исполнитель с контекстом"""
    ctx = CustomContext(input_data)
    
    for connection in pipeline:
        source_out = connection.source(ctx.input)
        
        # Применяем трансформер если есть
        if connection.transformer:
            target_in = connection.transformer(source_out)
        else:
            target_in = source_out
            
        # Передаем данные в target
        if isinstance(connection.target, tuple):
            for target in connection.target:
                target(target_in)
        else:
            connection.target(target_in)
        
        ctx.add_timestamp()
    
    return ctx.output

# Запуск
if __name__ == "__main__":
    results = execute_pipeline(main_pipeline, "initial_input")
    print(f"Pipeline completed: {results}")
```

---

### 🔄 Как это работает вместе
1. **Чистая сепарация ответственности**:
   - `components.py`: Что делать? (бизнес-логика)
   - `connects.py`: В каком порядке? (оркестровка)
   - `config.py`: С какими параметрами? (конфигурация)

2. **Преимущества подхода**:
   - **Тестируемость**: Компоненты тестируются изолированно
   ```python
   # test_components.py
   def test_clean_data():
       assert clean_data([1, 0, 3]) == [1, 3]
   ```
   
   - **Гибкость**: Переконфигурация без изменения логики
   ```python
   # connects.py (новая версия)
   connect(load_data, [clean_data, parallel_processor])
   ```
   
   - **Безопасность**: Компоненты не знают друг о друге
   ```python
   # components.py
   # НЕТ импортов из connects.py!
   ```

3. **Масштабирование**:
   - Добавление нового компонента:
     1. Реализовать в `components.py`
     2. Подключить в `connects.py`
     3. Ничего не менять в существующем коде

---

### 💡 Идеальный вариант использования
```bash
# Запуск всей системы
python -m executor --pipeline main --input sales_data.csv

# Результат:
# Loading from /data/raw/sales_data.csv...
# Cleaning data...
# Analyzing...
# Saving report to /data/processed: {'mean': 42.5}
```

**Ключевое преимущество**: При изменении бизнес-требований вы:
1. Меняете связи в `connects.py` (а не логику)
2. Генерируете новые компоненты через ИИ
3. Не переписываете рабочую систему!

Такой подход особенно мощный в сочетании с:
- Инструментами визуализации (автогенерация графов из `connects.py`)
- Системами развертывания (контейнеризация компонентов)
- Мониторингом (трассировка выполнения)