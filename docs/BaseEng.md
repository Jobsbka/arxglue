## ArxGLUE: Minimalistic Composition Interface (v1.0)

**Library Philosophy:**  
ArxGLUE is not a framework, but a conceptual *ArxGLUE* for combining any Python-compatible components. The library doesn't impose solutions—it provides minimalistic primitives for declaratively connecting arbitrary elements of your system. Your logic, your executors, your data—we just give you the tools to "ArxGLUE" them together.

---

### Core Concepts

1. **`Component`**  
   Any Python callable object:  
   ```python
   Component = Callable[[Any], Any]
   ```
   - Functions, lambdas, classes with `__call__`, object methods.
   - No constraints on inputs/outputs.
   - Examples: `lambda x: x*2`, `str.upper`, `pandas.DataFrame`.

2. **`connect()` — Connection Declaration**  
   Declares relationships between components without immediate execution:
   ```python
   connect(source, target, transformer=None) -> tuple
   ```
   - **`source`**: Data source (single component or tuple).
   - **`target`**: Data receiver (single component or tuple).
   - **`transformer`**: Optional function to transform `source` outputs before passing to `target`.
   - **Returns** a connection descriptor tuple `(source, target, transformer)`.

3. **`ContextProtocol` (Optional)**  
   Base class for state management between components:
   ```python
   class ContextProtocol:
       input: Any          # Input data
       output: Optional[Any]  # Output data
       state: dict         # Arbitrary state
   ```
   - Not mandatory.
   - Can be extended by users.

4. **Executors (Example: `execute_linear`)**  
   ArxGLUE **does not include** ready-made executors. It only offers *interfaces* for integration. Minimal executor example:
   ```python
   def execute_linear(components: list[Component], input_data: Any) -> Any:
       result = input_data
       for comp in components:
           result = comp(result)
       return result
   ```

---

### Principles of Operation

#### 1. Declarative Connections
Connections are declared *logically*, not physically. `connect()` creates descriptors for your executor to interpret:
```python
loader = connect(data_source, preprocessor, transformer=json.loads)
analyzer = connect(preprocessor, [report_generator, saver])
```

#### 2. Data Transformers
Transform `source` outputs before passing to `target`:
```python
# Convert number to string before logging
connect(get_random_number, logger, transformer=str)
```

#### 3. Multiple Connections
Supports branching and merging data flows:
```python
# One source → two targets
connect(fetcher, [parser, validator])

# Two sources → one target (requires transformer!)
connect((sensor1, sensor2), aggregator, transformer=lambda a,b: (a+b)/2)
```

#### 4. Context (Optional)
Custom context example:
```python
class PipelineContext(ContextProtocol):
    def __init__(self, input_data):
        super().__init__(input_data)
        self.timestamps = []
    
    def add_timestamp(self):
        self.timestamps.append(time.time())

# Usage in component:
def processor(ctx: PipelineContext):
    ctx.add_timestamp()
    ctx.output = process(ctx.input)
```

---

### Usage Examples

#### Simple Chain
```python
def load(data): return f"Loaded: {data}"
def clean(data): return f"Cleaned: {data}"
def save(data): print(f"Saving: {data}")

chain = [
    connect(load, clean),
    connect(clean, save)
]

# Custom executor:
def run_chain(chain, input_data):
    data = load(input_data)
    data = clean(data)
    save(data)

run_chain(chain, "test_file.csv")
```

#### Branching with Transformer
```python
def get_data(): return 42
def log(data): print(f"LOG: {data}")
def process_num(n): return n * 100

connections = connect(
    source=get_data,
    target=[log, process_num],
    transformer=lambda x: f"Value={x}"  # For logger
)

# Executor interpretation:
data = get_data()
log(transformer(data))        # LOG: Value=42
result = process_num(data)    # 4200
```

---

### FAQ

**Q: How to handle complex dependencies?**  
A: ArxGLUE doesn't manage dependencies. Implement your own executor to interpret connection descriptors per your rules.

**Q: Can I integrate with asyncio/threading?**  
A: Yes! Components can be async functions, generators, or external services. Your executor handles concurrency.

**Q: Why use `ContextProtocol`?**  
A: Optional base class for cross-component data flow. Omit if unneeded.

**Q: How to integrate with Django/FastAPI?**  
A: Wrap components in middleware/routes. FastAPI example:
```python
@app.get("/")
def endpoint(request):
    return execute_linear([auth, process_request], request)
```

---

### Key Strengths
- 🧩 **Agnostic**: Works with any Python code.
- 🧠 **Simplicity**: 50 LOC, zero dependencies.
- 🚀 **Flexibility**: Supports all paradigms (OOP, FP, reactivity).
- 🔌 **Integration**: Compatible with any libraries (Pandas, TensorFlow, Celery).

---

**ArxGLUE doesn't solve your problems—it gives you the freedom to solve them by combining components in any configuration.**  
Your logic + our ArxGLUE = ❤️