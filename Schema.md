### Визуальная схема системы с использованием ArxGlue

```plaintext
  +----------------------------------------------+
  |             Контекст выполнения              |
  |             (ContextProtocol)                |
  | +-----------------------------------------+  |
  | | input:  <исходные данные>               |  |
  | | output: <результат обработки>           |  |
  | | state: {                                |  |
  | |   "шаг1": результат компонента A,       |  |
  | |   "meta": метаданные трансформации      |  |
  | | }                                       |  |
  | +-----------------------------------------+  |
  +-----------------------------------------------+
           ↑                           ↑
           | (чтение/запись)           | (запись результата)
  +--------+--------+        +---------+---------+
  |   Компонент A   |        |   Компонент B     |
  |  (Callable)     |        |  (Callable)       |
  |                 |        |                   |
  | Любой вызываемый|        | Любой вызываемый  |
  | объект:         |        | объект:           |
  |  - функция      |        |  - класс          |
  |  - класс с __call__ |    |  - лямбда         |
  |  - внешний сервис |      |  - ИИ-модель      |
  +--------+---------+        +---------+--------+
           |                           ↑
           | Вывод данных A            | Ввод данных B
           | (произвольный формат)     | (формат после трансформации)
           ↓                           |
  +--------+---------+        +--------+---------+
  |   Трансформер T  |        |   Трансформер U  |  [опционально]
  |   (Callable)     +-------->                  |
  |                  |        |                  |
  | Любой вызываемый |        | Любой вызываемый |
  | объект:          |        | объект:          |
  |  - функция       |        |  - класс         |
  |  - класс-адаптер |        |  - цепочка       |
  |  - шифратор      |        |  преобразований  |
  +------------------+        +------------------+
           |
           | Преобразованныe данные
           | (в формате для Компонента B)
           ↓
```

### Ключевые элементы системы:

1. **Компонент A** (Источник)
   - Любой вызываемый объект Python
   - Примеры:
     ```python
     # Функция
     def extract_data(ctx): return ctx.input.upper()
     
     # Класс с состоянием
     class DataValidator:
         def __init__(self, rules):
             self.rules = rules
         def __call__(self, data):
             return validate(data, self.rules)
     
     # Внешний сервис
     bank_api = lambda data: requests.post(API_URL, json=data)
     ```

2. **Трансформер T** (Преобразователь)
   - Преобразует вывод A в формат для B
   - Может быть цепочкой преобразований
   - Примеры:
     ```python
     # Простое преобразование
     json_to_xml = lambda json_data: xmltodict.unparse(json_data)
     
     # Состоятельный трансформер
     class DataMasker:
         def __init__(self, fields):
             self.fields = fields
         def __call__(self, data):
             return mask_sensitive_fields(data, self.fields)
     ```

3. **Компонент B** (Приёмник)
   - Обрабатывает преобразованные данные
   - Примеры:
     ```python
     # Монолитный банковский сервис
     class CoreBankingSystem:
         def __call__(self, transaction):
             return process_transaction(transaction)
     
     # Асинхронный обработчик
     async def save_to_db(data):
         await database.insert(data)
     ```

4. **Контекст выполнения**
   - Сквозной объект для всего конвейера
   - Содержит:
     - Исходные данные (input)
     - Финальный результат (output)
     - Произвольное состояние (state)
   - Пример использования:
     ```python
     class ProcessingContext(ContextProtocol):
         def __init__(self, input_data):
             super().__init__(input_data)
             self.state["start_time"] = time.time()
     
     ctx = ProcessingContext({"transaction": "..."})
     ```

### Поток данных в системе:

```plaintext
(Начало)
  ↓
[Контекст] → input передается в → [Компонент A]
  ↓
[Компонент A] → сырые данные → [Трансформер T]
  ↓
[Трансформер T] → преобразованные данные → [Компонент B]
  ↓
[Компонент B] → записывает результат → output в [Контекст]
  ↓
(Конец)
```

### Особенности взаимодействия:

1. **Двунаправленная связь с контекстом**:
   - Все компоненты и трансформеры могут:
     - Читать `input`
     - Записывать в `state`
     - Обновлять `output`
   - Пример:
     ```python
     class FraudDetector:
         def __call__(self, data, ctx: ContextProtocol):
             score = calculate_fraud_score(data)
             ctx.state["fraud_score"] = score
             return score > 0.8
     ```

2. **Каскадные трансформеры**:
   ```python
   # Цепочка преобразований
   transforms = connect(
       source=extractor,
       target=loader,
       transformer=(
           DataCleaner(),
           FeatureEncoder(),
           Anonymizer(fields=["phone"])
       )
   )
   ```

3. **Динамическая композиция**:
   - Компоненты могут создаваться во время выполнения
   - Пример:
     ```python
     def create_pipeline(config):
         return connect(
             source=create_component(config['source']),
             target=create_component(config['target']),
             transformer=create_transformer_chain(config)
         )
     ```

### Пример банковской транзакции:

```plaintext
[Контекст]
input: {"card": "411111******1111", "amount": 10000, "currency": "RUB"}

[Компонент A] → Валидация карты (CardValidator)
  ↓ {"status": "valid", "bin": "411111"}
  
[Трансформер T] → Преобразование в формат ISO-8583
  ↓ {"MTI": "0200", "PAN": "4111111111", "AMOUNT": "00000010000"}

[Компонент B] → Система обработки платежей (CoreBanking)
  ↓ {"status": "approved", "code": "00"}

[Контекст]:
output: {"result": "approved"}
state: {
  "validation_result": {...},
  "iso_message": "...",
  "processing_time_ms": 45
}
```

Такая архитектура позволяет создавать гибкие, декомпозированные системы, где:
1. Компоненты реализуют бизнес-логику
2. Трансформеры решают проблемы интеграции
3. Контекст обеспечивает сквозную видимость данных
4. Связи явно декларируют потоки данных
