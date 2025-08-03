## ArxGLUE: Минималистичный интерфейс композиции (v1.0)

**Философия библиотеки:**  
ArxGLUE — это не фреймворк, а концептуальный *клей* для объединения любых Python-совместимых компонентов. Библиотека не навязывает способы решения задач, а предоставляет минималистичные примитивы для декларации связей между произвольными элементами вашей системы. Ваша логика, ваши исполнители, ваши данные — мы лишь даем инструмент для их "склеивания".

---

### Ключевые концепции

1. **`Component` (Компонент)**  
   Любой вызываемый объект Python:  
   ```python
   Component = Callable[[Any], Any]
   ```
   - Функции, лямбды, классы с `__call__`, методы объектов.
   - Не накладывает ограничений на вход/выход.
   - Пример: `lambda x: x*2`, `str.upper`, `pandas.DataFrame`.

2. **`connect()` — Декларация связей**  
   Объявляет отношение между компонентами без их непосредственного вызова:
   ```python
   connect(source, target, transformer=None) -> tuple
   ```
   - **`source`**: Источник данных (один компонент или кортеж компонентов).
   - **`target`**: Приемник данных (один компонент или кортеж компонентов).
   - **`transformer`**: Опциональная функция преобразования выходов `source` перед передачей в `target`.
   - **Возвращает** кортеж-дескриптор `(source, target, transformer)`.

3. **`ContextProtocol` (Опциональный протокол)**  
   Базовый класс для передачи состояния между компонентами:
   ```python
   class ContextProtocol:
       input: Any          # Входные данные
       output: Optional[Any]  # Выходные данные
       state: dict         # Произвольное состояние
   ```
   - Не обязателен к использованию.
   - Может быть расширен пользователем.

4. **Исполнители (Пример: `execute_linear`)**  
   ArxGLUE **не включает** готовые системы исполнения. Библиотека предлагает лишь *интерфейсы* для интеграции. Пример минималистичного исполнителя:
   ```python
   def execute_linear(components: list[Component], input_data: Any) -> Any:
       result = input_data
       for comp in components:
           result = comp(result)
       return result
   ```

---

### Принципы работы

#### 1. Декларативность связей
Связи объявляются *логически*, а не физически. Функция `connect()` лишь создает дескриптор, который может быть интерпретирован вашим исполнителем:
```python
loader = connect(data_source, preprocessor, transformer=json.loads)
analyzer = connect(preprocessor, [report_generator, saver])
```

#### 2. Трансформеры данных
Преобразуют выход `source` перед передачей в `target`:
```python
# Преобразуем число в строку перед передачей
connect(get_random_number, logger, transformer=str)
```

#### 3. Множественные связи
Поддержка ветвления и слияния потоков:
```python
# Один источник -> два приемника
connect(fetcher, [parser, validator])

# Два источника -> один приемник (требует трансформер!)
connect((sensor1, sensor2), aggregator, transformer=lambda a,b: (a+b)/2)
```

#### 4. Контекст (по желанию)
Пример кастомного контекста:
```python
class PipelineContext(ContextProtocol):
    def __init__(self, input_data):
        super().__init__(input_data)
        self.timestamps = []
    
    def add_timestamp(self):
        self.timestamps.append(time.time())

# Использование в компоненте:
def processor(ctx: PipelineContext):
    ctx.add_timestamp()
    ctx.output = process(ctx.input)
```

---

### Примеры использования

#### Простая цепочка
```python
def load(data): return f"Loaded: {data}"
def clean(data): return f"Cleaned: {data}"
def save(data): print(f"Saving: {data}")

chain = [
    connect(load, clean),
    connect(clean, save)
]

# Ваш кастомный исполнитель:
def run_chain(chain, input_data):
    # Здесь ваша логика обработки дескрипторов
    data = load(input_data)
    data = clean(data)
    save(data)

run_chain(chain, "test_file.csv")
```

#### Ветвление с трансформером
```python
def get_data(): return 42
def log(data): print(f"LOG: {data}")
def process_num(n): return n * 100

connections = connect(
    source=get_data,
    target=[log, process_num],
    transformer=lambda x: f"Value={x}"  # Для log
)

# Интерпретация в исполнителе:
data = get_data()
log_data = transformer(data)
process_num_data = data  # Без трансформации

log(log_data)           # LOG: Value=42
result = process_num(process_num_data)  # 4200
```

---

### FAQ

**Q: Как обрабатывать сложные зависимости?**  
A: ArxGLUE не управляет зависимостями. Реализуйте собственный исполнитель, который интерпретирует дескрипторы связей по вашим правилам.

**Q: Можно ли интегрировать с asyncio/threading?**  
A: Да! Компоненты могут быть асинхронными функциями, генераторами, или даже внешними сервисами. Ваш исполнитель управляет многопоточностью.

**Q: Зачем нужен `ContextProtocol`?**  
A: Это опциональный базовый класс для сквозной передачи данных. Если не нужен — не используйте.

**Q: Как интегрировать с Django/FastAPI?**  
A: Оберните компоненты в middleware/роутеры. Пример для FastAPI:
```python
@app.get("/")
def endpoint(request):
    return execute_linear([auth, process_request], request)
```

---

### Сильные стороны
- 🧩 **Агностичность**: Работает с любым Python-кодом.
- 🧠 **Простота**: 50 строк кода, 0 зависимостей.
- 🚀 **Гибкость**: Поддерживает любые парадигмы (ООП, ФП, реактивность).
- 🔌 **Интеграция**: Совместим с любыми библиотеками (Pandas, TensorFlow, Celery).

---

**ArxGLUE не решает ваши задачи — он дает вам свободу решать их так, как вы хотите, объединяя компоненты в любой конфигурации.**  
Ваша логика + наш клей = ❤️