# arxglue

Minimalistic Component Composition Interface

[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/arxglue.svg)](https://pypi.org/project/arxglue/)
[![GitHub]](https://github.com/Jobsbka/arxglue)

```bash
pip install arxglue

Why arxglue?
Minimal Core: Only essential primitives (Component, Connection)

Zero Dependencies: Pure Python, no external packages

Framework Agnostic: Works with any Python code

Extremely Flexible: From simple scripts to complex systems

Apache 2.0 Licensed: Permissive for commercial use

Core Concepts
python
from arxglue import connect, execute_linear

# Any callable is a component
def uppercase(text: str) -> str:
    return text.upper()

# Create connections
connection = connect(uppercase, print)

# Execute sequentially
execute_linear([uppercase, print], "hello")
Advanced Patterns
python
# Group connections
def sensor1(): return 10
def sensor2(): return 20
def processor(data): return sum(data)

connect(
    source=(sensor1, sensor2),
    target=processor,
    transformer=lambda x, y: [x, y]
)

# Stateful processing
class ProcessingContext:
    def __init__(self, data):
        self.input = data
        self.output = None
        self.state = {}
        
    def __call__(self):
        self.output = process(self.input)
        self.state["processed"] = True



License
Apache License 2.0 - See LICENSE for details.