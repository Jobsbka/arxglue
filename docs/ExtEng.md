## GLUE: Minimalistic Composition Interface (v1.0) - Extended Documentation

**Library Philosophy:**  
GLUE is a conceptual *glue* for the Python ecosystem. No restrictions, no ready-made solutions—only primitives for combining ANY components into a SINGLE system. Your code, your paradigms, your infrastructure decisions—we simply connect them.

---

### Ecosystem Integration Examples

#### 1. AI/ML Pipeline (TensorFlow + Scikit-learn)
```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Components
def load_data(_): 
    return np.random.rand(1000, 10)

def scale_data(data):
    return StandardScaler().fit_transform(data)

def train_model(data):
    model = tf.keras.Sequential([...])
    model.fit(data, epochs=10)
    return model

# GLUE composition
pipeline = [
    connect(load_data, scale_data),
    connect(scale_data, train_model)
]

# Custom executor
def run_ai_pipeline(connections):
    data = load_data(None)
    scaled = scale_data(data)
    model = train_model(scaled)
    return model
```

#### 2. Highload API Gateway (FastAPI + Redis)
```python
from fastapi import Request
import redis

r = redis.Redis()

# Components
async def auth_middleware(request: Request):
    if not request.headers.get('Auth'):
        raise HTTPException(401)
    return request

async def cache_check(request: Request):
    if cached := r.get(request.url.path):
        return JSONResponse(cached)
    return request

async def process_request(request: Request):
    # Heavy processing
    return {"result": "data"}

# GLUE router
@app.get("/data")
async def endpoint(request: Request):
    pipeline = [
        connect(auth_middleware, cache_check),
        connect(cache_check, process_request, transformer=lambda x: x if x is None else x)
    ]
    return await execute_async(pipeline, request)
```

#### 3. IoT Device Processing (Pandas + MQTT)
```python
import paho.mqtt.client as mqtt
import pandas as pd

# Components
def read_mqtt(topic: str) -> callable:
    def _read(_):
        client = mqtt.Client()
        client.connect("iot-broker")
        return client.subscribe(topic)
    return _read

def to_dataframe(payload):
    return pd.DataFrame([payload])

def detect_anomalies(df: pd.DataFrame):
    df['anomaly'] = df['value'] > df['value'].mean() + 3*df['value'].std()
    return df

# GLUE composition
device_processing = [
    connect(read_mqtt("sensors/temp"), to_dataframe),
    connect(to_dataframe, detect_anomalies)
]
```

#### 4. ETL Pipeline (Pandas + DuckDB)
```python
import duckdb
import pandas as pd

# Components
def extract(_):
    return pd.read_csv("data.csv")

def transform(df):
    df = df[df['quality'] > 3]
    df['category'] = pd.cut(df['price'], bins=5)
    return df

def load(df):
    with duckdb.connect("data.duckdb") as con:
        con.execute("CREATE TABLE IF NOT EXISTS data AS SELECT * FROM df")

# GLUE connections
etl = connect(extract, transform)
load = connect(transform, load)

# Execution
transform(extract())(load)
```

---

### Advanced Patterns

#### Parallel Execution (concurrent.futures)
```python
from concurrent.futures import ThreadPoolExecutor

def parallel_executor(connections, input_data):
    with ThreadPoolExecutor() as executor:
        futures = {}
        for src, tgt, transformer in connections:
            if src not in futures:
                futures[src] = executor.submit(src, input_data)
            
            src_result = futures[src].result()
            tgt_input = transformer(src_result) if transformer else src_result
            
            if isinstance(tgt, tuple):
                for t in tgt:
                    futures[t] = executor.submit(t, tgt_input)
            else:
                futures[tgt] = executor.submit(tgt, tgt_input)
                
        return list(futures.values())[-1].result()
```

#### Reactive Streams (RxPY integration)
```python
from rx import operators as op

def to_rx(connections):
    stream = rx.empty()
    for src, tgt, transformer in connections:
        if transformer:
            stream = stream.pipe(
                op.flat_map(lambda x: src(x)),
                op.map(transformer),
                op.flat_map(lambda y: tgt(y))
            )
        else:
            stream = stream.pipe(
                op.flat_map(lambda x: src(x)),
                op.flat_map(lambda y: tgt(y))
            )
    return stream
```

#### Stateful Service (ContextProtocol)
```python
class TradingContext(ContextProtocol):
    def __init__(self, order):
        super().__init__(order)
        self.market_data = None
        self.risk_level = 0

def fetch_market_data(ctx: TradingContext):
    ctx.market_data = stock_api.get(ctx.input['symbol'])

def risk_check(ctx: TradingContext):
    ctx.risk_level = calculate_risk(ctx.market_data)

def execute_order(ctx: TradingContext):
    if ctx.risk_level < 5:
        ctx.output = broker.execute(ctx.input)
    else:
        ctx.output = {"error": "Risk too high"}

# Composition
trading_flow = [
    connect(fetch_market_data, risk_check),
    connect(risk_check, execute_order)
]

def run_trading(order):
    ctx = TradingContext(order)
    for comp in [fetch_market_data, risk_check, execute_order]:
        comp(ctx)
    return ctx.output
```

---

### Infrastructure Integration

#### Distributed Tasks (Celery)
```python
from celery import Celery
app = Celery()

@app.task
def process_data_task(data):
    pipeline = [
        connect(clean_data, validate_data),
        connect(validate_data, enrich_data)
    ]
    return execute_distributed(pipeline, data)

# GLUE CLI trigger
connect(
    source=receive_kafka_message, 
    target=process_data_task.delay
)
```

#### Monitoring (Prometheus + Grafana)
```python
from prometheus_client import Counter

REQUESTS = Counter('glue_requests', 'Total requests')

def with_monitoring(component):
    def wrapper(*args):
        REQUESTS.inc()
        start = time.perf_counter()
        result = component(*args)
        latency = time.perf_counter() - start
        return result
    return wrapper

# Instrumentation
monitored_load = with_monitoring(load_data)
connect(monitored_load, process_data)
```

---

### Optimization Techniques

#### JIT Compilation (Numba)
```python
from numba import njit

@njit
def numba_optimized(data):
    # Compute-intensive part
    return result

connect(preprocess, numba_optimized)
```

#### GPU Acceleration (CuPy)
```python
import cupy as cp

def to_gpu(data):
    return cp.asarray(data)

def gpu_processing(gpu_data):
    with cp.cuda.Device(0):
        return cp.fft.fft(gpu_data)

connect(load_data, to_gpu)
connect(to_gpu, gpu_processing)
```

---

### Enterprise FAQ

**Q: How to integrate with legacy C++ code?**  
A: Via Python C-API:
```python
import ctypes

lib = ctypes.CDLL('legacy.so')

def legacy_adapter(data):
    # Data conversion
    result = lib.process_data(data)
    return python_friendly_result
```

**Q: GraphQL/Apollo Federation support?**  
A: Via custom resolver:
```python
def graphql_resolver(obj, info, **args):
    pipeline = [
        connect(fetch_user, fetch_metadata),
        connect(fetch_metadata, format_response)
    ]
    return execute_linear(pipeline, args)
```

**Q: Apache Kafka integration?**  
A: Consumer as source:
```python
def kafka_source(topic):
    consumer = KafkaConsumer(topic)
    for msg in consumer:
        yield msg.value

connect(kafka_source('events'), process_event)
```

---

### System Requirements
- Python 3.8+
- Zero external dependencies (integrations optional)
- Compatible with any infrastructure:
  * Kubernetes Operators
  * AWS Lambda / GCP Functions
  * Airflow / Prefect DAGs
  * Django / Flask / FastAPI
  * PySpark / Dask / Ray

**GLUE doesn't replace your stack—it connects its components into a single organism.**  
Your complexity + our simplicity = industrial solutions.