# arxglue

Minimalistic Component Composition Interface

[![Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/arxglue.svg)](https://pypi.org/project/arxglue/)
[![GitHub]](https://github.com/Jobsbka/arxglue)

```bash
pip install arxglue
```

### ArxGlue — Философия Макрокомпозиции

#### 🌟 **Введение**  
ArxGlue — это минималистичный подход к построению программных систем через **макрокомпозицию**. В отличие от традиционных методов, где логика и оркестрация переплетены, ArxGlue разделяет систему на три независимых слоя:  
1. **Что делать?** (чистая бизнес-логика)  
2. **Как связать?** (декларативная оркестровка)  
3. **Чем управлять?** (конфигурация и исполнение).  

Этот подход вдохновлен принципами Unix ("делай одну вещь и делай её хорошо") и микросервисной архитектурой, а применим как к отдельным компонентам в сложных приложениях, так и при общении композиции из сложных приложений.

---

####   **Проблемы, которые мы решаем**  
1. **Жёсткая связанность**  
   Компоненты знают друг о друге → изменение одного ломает другие.  
   *Решение в ArxGlue:* Компоненты изолированы, связи объявляются внешне.  

2. **Сложность тестирования**  
   Бизнес-логика смешана с инфраструктурным кодом → тесты хрупкие.  
   *Решение:* Компоненты — чистые функции/классы. Тестируются в изоляции.  

3. **Низкая гибкость**  
   Изменение потока данных требует переписывания кода.  
   *Решение:* Оркестровка описывается декларативно. Поток меняется без правки логики.  

---

#### ⚙️ **Три Столпа Философии**  

##### 1. **Компоненты (components.py)**  
- **Чистая бизнес-логика** без побочных эффектов.  
- Зависимости только через входные параметры.  
- Пример:  
  ```python
  def clean_data(data: list) -> list:
      return [x for x in data if x > 0]  # Только логика фильтрации!
  ```

##### 2. **Оркестровка (connects.py)**  
- **Декларативное описание связей** через `connect()`.  
- Компоненты комбинируются в пайплайны/графы.  
- Пример:  
  ```python
  main_pipeline = [
      connect(load_data, clean_data),
      connect(clean_data, analyze_data)
  ]
  ```

##### 3. **Конфигурация (config.py)**  
- **Параметры системы вынесены в конфиги**.  
- Компоненты читают конфиги, но не изменяют их.  
- Пример:  
  ```python
  DATABASE_URL = "postgresql://user:pass@localhost/db"  # Всё в одном месте!
  ```

---

#### ✨ **Ключевые Преимущества**  
- **🚀 Гибкость**  
  Измените порядок обработки, переписав только `connects.py`:  
  ```python
  connect(load_data, [clean_data, validate_data])  # Добавили валидацию!
  ```
  
- **🔒 Безопасность**  
  Компоненты не импортируют друг друга → нет скрытых зависимостей.  

- **📊 Масштабируемость**  
  Новый компонент = 2 шага:  
  1. Реализовать логику в `components.py`.  
  2. Подключить в `connects.py`.  

- **🤖 Тестируемость**  
  Компоненты тестируются изолированно:  
  ```python
  def test_clean_data():
      assert clean_data([0, 1, -5]) == [1]  # Никаких моков БД/API!
  ```

---

#### 🚀 **Идеальный Сценарий Использования**  
```bash
# Запуск системы = выбор пайплайна + входные данные
python -m executor --pipeline main --input sales.csv
```
**Результат:**  
```
Загрузка данных из /data/raw/sales.csv...
Очистка данных...
Анализ...
Сохранение отчёта в /data/processed: {'mean': 42.5}
```

---

#### 🌐 **Экосистема и Будущее**  
ArxGlue идеально сочетается с:  
- **Визуальными редакторами** (автогенерация графов из `connects.py`).  
- **FaaS/Serverless** (каждый компонент → отдельная функция).  
- **Трассировкой** (мониторинг выполнения через `ContextProtocol`).  
- **ИИ-генерацией** (автоматическое создание компонентов под задачи).  

---

#### 💎 **Заключение**  
ArxGlue — это **философия**, а не фреймворк. Её суть:  
> «Собирайте сложные системы из простых, изолированных компонентов,  
> управляя связями декларативно, а не императивно».  

Это снижает порог входа, ускоряет разработку и превращает поддержку кода из боли в удовольствие.  

**Сделайте шаг к макрокомпозиции — и ваши системы обретут элегантность и мощь!**


< | full content | >

============================================================
.\arxglue\arxsnake.py
============================================================
import pygame
import numpy as np
import random
import arxviz
import arxerr
import sys
from arxerr import ErrorContext, traced, logged
from arxviz import export_architecture_dot, render_dot_to_png, export_trace_plantuml
import matplotlib.pyplot as plt
import os
import tempfile
from arxglue import connect, Component, ContextProtocol
from typing import List, Tuple, Optional, Dict, Any

# Конфигурация
GRID_SIZE = 20
CELL_SIZE = 30
WIDTH, HEIGHT = 800, 600
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
DEBUG_SNAKE = 0  # Индекс змейки для отладки

@traced
@logged
class Neuron(Component):
    """Нейрон с адаптивными весами и трассировкой"""
    def __init__(self, weights: List[float], activation: str = 'relu'):
        self.weights = weights
        self.activation = activation
        self.output = 0.0
        
    def __call__(self, inputs: List[float], ctx: ErrorContext) -> float:
        potential = sum(i * w for i, w in zip(inputs, self.weights))
        
        if self.activation == 'relu':
            self.output = max(0, potential)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-potential))
        elif self.activation == 'tanh':
            self.output = np.tanh(potential)
            
        ctx.log(f"Neuron {self.activation}: input={inputs}, weights={self.weights}, output={self.output}")
        return self.output

class SnakeBrain:
    """Мозг змейки с трассировкой и визуализацией"""
    def __init__(self, debug=False):
        self.debug = debug
        
        # Сенсоры: расстояние до стены/еды/тела (8 направлений)
        self.sensors = [Neuron([1.0]) for _ in range(8)]
        
        # Скрытый слой
        self.hidden_layer = [
            Neuron([random.uniform(-1, 1) for _ in range(8)], 'relu')
            for _ in range(8)
        ]
        
        # Выход: вверх/вниз/влево/вправо
        self.motors = [
            Neuron([random.uniform(-1, 1) for _ in range(8)], 'sigmoid')
            for _ in range(4)
        ]
        
        # Связи через ArxGlue
        self.connections = []
        for sensor in self.sensors:
            self.connections += [connect(sensor, neuron) for neuron in self.hidden_layer]
            
        for neuron in self.hidden_layer:
            self.connections += [connect(neuron, motor) for motor in self.motors]
        
        # Визуализация архитектуры
        if debug:
            try:
                export_architecture_dot(self.connections, "snake_brain.dot")
                render_dot_to_png("snake_brain.dot", "snake_brain.png")
                print("Архитектура мозга сохранена в snake_brain.png")
            except Exception as e:
                print(f"Ошибка визуализации: {str(e)}")

    def decide(self, sensor_data: List[float], snake_id: int) -> int:
        """Принять решение с трассировкой"""
        ctx = ErrorContext({
            "snake_id": snake_id,
            "sensor_data": sensor_data
        })
        
        # Активируем сенсоры
        sensor_outputs = []
        for i, sensor in enumerate(self.sensors):
            output = sensor([sensor_data[i]], ctx)
            sensor_outputs.append(output)
            ctx.log(f"Sensor {i}: value={sensor_data[i]:.4f} => output={output:.4f}")
        
        # Активируем скрытый слой
        hidden_outputs = []
        for i, neuron in enumerate(self.hidden_layer):
            output = neuron(sensor_outputs, ctx)
            hidden_outputs.append(output)
            ctx.log(f"Hidden neuron {i}: output={output:.4f}")
        
        # Активируем выходной слой
        motor_outputs = []
        for i, motor in enumerate(self.motors):
            output = motor(hidden_outputs, ctx)
            motor_outputs.append(output)
            ctx.log(f"Motor {i}: output={output:.4f}")
        
        # Принимаем решение
        decision = motor_outputs.index(max(motor_outputs))
        ctx.log(f"Decision: direction={decision}")
        ctx.state["motor_outputs"] = motor_outputs
        ctx.state["decision"] = decision
        
        # Экспорт трассировки
        if self.debug:
            try:
                export_trace_plantuml(ctx.trace, f"snake_trace_{snake_id}.puml")
                print(f"Трассировка сохранена в snake_trace_{snake_id}.puml")
                
                # Визуализация активаций
                plt.figure(figsize=(12, 8))
                plt.subplot(311)
                plt.bar(range(len(sensor_outputs)), sensor_outputs)
                plt.title("Sensor Outputs")
                
                plt.subplot(312)
                plt.bar(range(len(hidden_outputs)), hidden_outputs)
                plt.title("Hidden Layer Outputs")
                
                plt.subplot(313)
                plt.bar(range(len(motor_outputs)), motor_outputs)
                plt.title("Motor Outputs")
                plt.savefig(f"neuron_activations_{snake_id}.png")
                plt.close()
            except Exception as e:
                print(f"Ошибка экспорта трассировки: {str(e)}")
        
        return decision

class Snake:
    """Змейка с улучшенной сенсорной системой"""
    def __init__(self, brain: Optional[SnakeBrain] = None, debug=False):
        self.debug = debug
        self.brain = brain or SnakeBrain(debug=debug)
        self.reset()
        
    def reset(self):
        self.id = random.randint(1000, 9999)
        self.length = 3
        self.positions = [(GRID_SIZE//2, GRID_SIZE//2)]
        self.direction = random.randint(0, 3)
        self.food = self.spawn_food()
        self.fitness = 0
        self.steps = 0
        self.alive = True
        
    def spawn_food(self) -> Tuple[int, int]:
        """Создать еду в случайной позиции"""
        while True:
            food = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if food not in self.positions:
                return food
    
    def get_sensor_data(self) -> List[float]:
        """Улучшенная сенсорная система с плавными значениями"""
        head_x, head_y = self.positions[0]
        data = []
        
        # Направления: 0°-315° с шагом 45°
        for angle in range(0, 360, 45):
            dx, dy = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            wall_distance = 0.0
            food_distance = 0.0
            body_distance = 0.0
            
            # Сканируем лучом
            for dist in range(1, GRID_SIZE+1):
                x = int(head_x + dx * dist)
                y = int(head_y + dy * dist)
                
                # Выход за границы
                if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
                    wall_distance = 1.0 / dist
                    break
                
                # Тело змейки
                if (x, y) in self.positions:
                    body_distance = 1.0 / dist
                    break
                
                # Еда
                if (x, y) == self.food:
                    food_distance = -1.0 / dist  # Отрицательное значение как "желаемый объект"
            
            # Комбинируем показания
            combined = wall_distance + body_distance + food_distance
            data.append(combined)
        
        return data
    
    def move(self):
        """Сделать шаг с улучшенной логикой"""
        if not self.alive:
            return
            
        self.steps += 1
        
        # Принять решение
        sensor_data = self.get_sensor_data()
        decision = self.brain.decide(sensor_data, self.id)
        
        # Изменить направление (0=вверх, 1=вправо, 2=вниз, 3=влево)
        if decision != (self.direction + 2) % 4:  # Не разворачиваемся на 180°
            self.direction = decision
            
        # Двигаемся
        x, y = self.positions[0]
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][self.direction]
        new_head = (x + dx, y + dy)
        
        # Проверка столкновений
        if (new_head in self.positions or 
            not (0 <= new_head[0] < GRID_SIZE) or 
            not (0 <= new_head[1] < GRID_SIZE)):
            self.alive = False
            return
            
        # Добавляем новую голову
        self.positions.insert(0, new_head)
        
        # Проверка еды
        if new_head == self.food:
            self.length += 1
            self.fitness += 100
            self.food = self.spawn_food()
        else:
            # Убираем хвост
            self.positions = self.positions[:self.length]
            
        # Начисляем фитнес за выживание
        self.fitness += 1
        
        # Штраф за бездействие
        if self.steps > 200 and self.length < 5:
            self.alive = False

class Evolution:
    """Эволюционный менеджер с улучшенной селекцией"""
    def __init__(self):
        self.population = [Snake(debug=(i == DEBUG_SNAKE)) for i in range(POPULATION_SIZE)]
        self.generation = 0
        self.best_fitness = 0
        
    def evolve(self):
        """Создать новое поколение с улучшенным скрещиванием"""
        # Оценка приспособленности
        fitnesses = [snake.fitness for snake in self.population]
        max_fitness = max(fitnesses)
        
        if max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
            
        # Селекция (турнирный отбор)
        new_population = []
        
        # Элитизм: сохраняем лучших
        elites = sorted(self.population, key=lambda s: s.fitness, reverse=True)[:5]
        new_population.extend(elites)
        
        # Скрещивание и мутация
        while len(new_population) < POPULATION_SIZE:
            # Турнирный отбор
            candidates = random.sample(self.population, 5)
            parent1 = max(candidates, key=lambda s: s.fitness)
            candidates = random.sample(self.population, 5)
            parent2 = max(candidates, key=lambda s: s.fitness)
            
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            new_population.append(child)
            
        self.population = new_population
        self.generation += 1
        
    def crossover(self, parent1: Snake, parent2: Snake) -> Snake:
        """Улучшенное скрещивание с сохранением структуры"""
        child_brain = SnakeBrain()
        
        # Скрещивание весов скрытого слоя
        for i in range(len(child_brain.hidden_layer)):
            for j in range(len(child_brain.hidden_layer[i].weights)):
                if random.random() > 0.5:
                    child_brain.hidden_layer[i].weights[j] = parent1.brain.hidden_layer[i].weights[j]
                else:
                    child_brain.hidden_layer[i].weights[j] = parent2.brain.hidden_layer[i].weights[j]
        
        # Скрещивание весов моторов
        for i in range(len(child_brain.motors)):
            for j in range(len(child_brain.motors[i].weights)):
                if random.random() > 0.5:
                    child_brain.motors[i].weights[j] = parent1.brain.motors[i].weights[j]
                else:
                    child_brain.motors[i].weights[j] = parent2.brain.motors[i].weights[j]
                    
        return Snake(child_brain)
    
    def mutate(self, snake: Snake):
        """Улучшенная мутация с нормальным распределением"""
        for neuron in snake.brain.hidden_layer + snake.brain.motors:
            for i in range(len(neuron.weights)):
                if random.random() < MUTATION_RATE:
                    neuron.weights[i] += random.gauss(0, 0.3)
                    # Гарантируем границы весов
                    neuron.weights[i] = max(-1.0, min(1.0, neuron.weights[i]))

class SnakeVisualizer:
    """Визуализация игры с улучшенным интерфейсом"""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Neuro-Evolution Snake with ArxGlue")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        
        self.evolution = Evolution()
        self.current_snake = 0
        self.fast_mode = False
        self.paused = False
        
    def draw_grid(self):
        """Отрисовать сетку"""
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (WIDTH, y))
    
    def draw_snake(self, snake: Snake):
        """Отрисовать змейку с градиентом"""
        # Голова
        head_x, head_y = snake.positions[0]
        pygame.draw.rect(self.screen, (0, 200, 0), 
                         (head_x * CELL_SIZE, head_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Тело с градиентом
        for i, (x, y) in enumerate(snake.positions[1:]):
            intensity = max(50, 255 - i * 5)
            pygame.draw.rect(self.screen, (0, intensity, 0), 
                            (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    def draw_food(self, food_pos: Tuple[int, int]):
        """Отрисовать еду с анимацией"""
        x, y = food_pos
        pygame.draw.circle(self.screen, (200, 0, 0), 
                          (x * CELL_SIZE + CELL_SIZE//2, y * CELL_SIZE + CELL_SIZE//2),
                          CELL_SIZE//2 - 2)
    
    def draw_hud(self, snake: Snake):
        """Отрисовать улучшенный интерфейс"""
        gen_text = self.font.render(f"Generation: {self.evolution.generation}", True, (255, 255, 255))
        snake_text = self.font.render(f"Snake: {self.current_snake+1}/{POPULATION_SIZE} (ID:{snake.id})", True, (255, 255, 255))
        fitness_text = self.font.render(f"Fitness: {snake.fitness}", True, (255, 255, 255))
        best_text = self.font.render(f"Best Fitness: {self.evolution.best_fitness}", True, (255, 255, 255))
        length_text = self.font.render(f"Length: {snake.length}", True, (255, 255, 255))
        steps_text = self.font.render(f"Steps: {snake.steps}", True, (255, 255, 255))
        
        self.screen.blit(gen_text, (10, 10))
        self.screen.blit(snake_text, (10, 40))
        self.screen.blit(fitness_text, (10, 70))
        self.screen.blit(best_text, (10, 100))
        self.screen.blit(length_text, (10, 130))
        self.screen.blit(steps_text, (10, 160))
        
        # Кнопки режимов
        mode_color = (100, 200, 100) if self.fast_mode else (200, 100, 100)
        mode_text = "FAST" if self.fast_mode else "NORMAL"
        pygame.draw.rect(self.screen, mode_color, (WIDTH-120, 10, 110, 30))
        text = self.font.render(f"Mode: {mode_text}", True, (0, 0, 0))
        self.screen.blit(text, (WIDTH-110, 15))
        
        pause_color = (200, 100, 100) if self.paused else (100, 200, 100)
        pause_text = "PAUSED" if self.paused else "RUNNING"
        pygame.draw.rect(self.screen, pause_color, (WIDTH-120, 50, 110, 30))
        text = self.font.render(f"State: {pause_text}", True, (0, 0, 0))
        self.screen.blit(text, (WIDTH-110, 55))
    
    def run(self):
        """Главный игровой цикл"""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.fast_mode = not self.fast_mode
                    if event.key == pygame.K_r:
                        self.evolution = Evolution()
                        self.current_snake = 0
                    if event.key == pygame.K_p:
                        self.paused = not self.paused
                    if event.key == pygame.K_d:
                        global DEBUG_SNAKE
                        DEBUG_SNAKE = self.current_snake
                        print(f"Debug enabled for snake {DEBUG_SNAKE}")
            
            if self.paused:
                self.clock.tick(1)
                continue
            
            # Получить текущую змейку
            snake = self.evolution.population[self.current_snake]
            
            # Обновить змейку
            if snake.alive:
                snake.move()
            
            # Переключить змейку, если текущая умерла
            if not snake.alive:
                self.current_snake += 1
                
                # Новое поколение
                if self.current_snake >= POPULATION_SIZE:
                    self.evolution.evolve()
                    self.current_snake = 0
            
            # Отрисовка
            self.screen.fill((0, 0, 0))
            self.draw_grid()
            
            if snake.alive:
                self.draw_snake(snake)
                self.draw_food(snake.food)
            
            self.draw_hud(snake)
            
            pygame.display.flip()
            
            # Скорость обновления
            if self.fast_mode:
                self.clock.tick(60)
            else:
                self.clock.tick(10)

if __name__ == "__main__":
    visualizer = SnakeVisualizer()
    visualizer.run()

============================================================
.\arxglue\setup.py
============================================================
from setuptools import setup, find_packages

setup(
    name="arxglue",
    version="1.0.1",
    packages=find_packages(),
    description="Minimalistic component composition interface",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="VKB Arcghitector",
    url="https://github.com/jobsbka/gluecore",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",

        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.7",
    keywords="composition, components, glue, minimal, architecture",
    project_urls={
        "Source": "https://github.com/jobsbka/gluecore",
    },
)

============================================================
.\arxglue\arxerr\adapters.py
============================================================
"""
Adapters for external logging systems
"""

import logging
from typing import Optional, Any

def setup_std_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure integration with standard logging module"""
    logger = logging.getLogger('arxerr')
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def log_to_std(ctx: Any, logger: Optional[logging.Logger] = None):
    """Forward context logs to standard logging system"""
    logger = logger or setup_std_logging()
    
    if not hasattr(ctx, 'logs'):
        return
    
    for entry in ctx.logs:
        if entry.startswith("[ERROR]"):
            logger.error(entry[8:])
        elif entry.startswith("[WARNING]"):
            logger.warning(entry[10:])
        elif entry.startswith("[INFO]"):
            logger.info(entry[7:])
        elif entry.startswith("[DEBUG]"):
            logger.debug(entry[8:])
        else:
            logger.info(entry)

============================================================
.\arxglue\arxerr\context.py
============================================================
"""
Enhanced context implementations
"""
from typing import Any
from .mixins import TracingMixin, LoggingMixin

class ErrorContext(TracingMixin, LoggingMixin):
    """
    Full-featured execution context with tracing and logging
    Compatible with ArxGLUE systems
    """
    def __init__(self, input_data: Any):
        super().__init__()
        self.input = input_data
        self.output = None
        self.state = {}

============================================================
.\arxglue\arxerr\decorators.py
============================================================
import functools
import time
from typing import Callable, Any

def traced(component: Callable) -> Callable:
    """Decorator for automatic execution tracing"""
    @functools.wraps(component)
    def wrapper(*args, **kwargs):
        # Ищем контекст в аргументах
        ctx = None
        for arg in args:
            if hasattr(arg, 'trace') and hasattr(arg, 'trace_call'):
                ctx = arg
                break
        
        if not ctx:
            return component(*args, **kwargs)
        
        # Всегда используем имя функции/компонента
        component_name = component.__name__
        
        ctx.trace_call(component_name, ctx.input)
        
        try:
            result = component(*args, **kwargs)
            output = result if result is not None else getattr(ctx, 'output', None)
            
            ctx.trace_return(output)
            return result
        except Exception as e:
            ctx.trace_error(e)
            raise
    return wrapper

def logged(component: Callable) -> Callable:
    """Decorator for automatic execution logging"""
    @functools.wraps(component)
    def wrapper(*args, **kwargs):
        # Ищем контекст в аргументах
        ctx = None
        for arg in args:
            if hasattr(arg, 'logs') and hasattr(arg, 'log'):
                ctx = arg
                break
        
        if not ctx:
            return component(*args, **kwargs)
        
        # Всегда используем имя функции/компонента
        component_name = component.__name__
        
        ctx.log(f"Entering {component_name} with input: {ctx.input}")
        
        start_time = time.time()
        try:
            result = component(*args, **kwargs)
            duration = time.time() - start_time
            output = result if result is not None else getattr(ctx, 'output', None)
            
            ctx.log(f"Completed {component_name} in {duration:.4f}s. Output: {output}")
            return result
        except Exception as e:
            ctx.log(f"Error in {component_name}: {str(e)}", "ERROR")
            raise
    return wrapper

============================================================
.\arxglue\arxerr\mixins.py
============================================================
import time
from typing import Any, Dict, List

class TracingMixin:
    """Mixin for execution tracing capabilities"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace: List[Dict[str, Any]] = []
        self._current_depth = 0

    def trace_call(self, component_name: str, input_data: Any):
        """Record component execution start"""
        self.trace.append({
            "component": component_name,
            "depth": self._current_depth,
            "input": input_data,
            "start_time": time.time(),
            "end_time": None,
            "success": None,
            "error": None
        })
        self._current_depth += 1

    def trace_return(self, output_data: Any = None):
        """Record successful component completion"""
        if self.trace:
            self._current_depth -= 1
            self.trace[-1].update({
                "output": output_data,
                "end_time": time.time(),
                "success": True
            })

    def trace_error(self, error: Exception):
        """Record component execution error"""
        if self.trace:
            self._current_depth -= 1
            self.trace[-1].update({
                "end_time": time.time(),
                "success": False,
                "error": str(error)
            })

class LoggingMixin:
    """Mixin for execution logging capabilities"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logs: List[str] = []
    
    def log(self, message: str, level: str = "INFO"):
        """Add log entry to context"""
        self.logs.append(f"[{level}] {message}")

============================================================
.\arxglue\arxerr\__init__.py
============================================================
"""
ArxErr - Logging and Tracing Tools for ArxGLUE Systems
Version: 1.0
"""

from .context import ErrorContext
from .mixins import TracingMixin, LoggingMixin
from .decorators import traced, logged
from .adapters import setup_std_logging, log_to_std

__all__ = [
    'ErrorContext',
    'TracingMixin',
    'LoggingMixin',
    'traced',
    'logged',
    'setup_std_logging',
    'log_to_std'
]

============================================================
.\arxglue\arxglue\core.py
============================================================
"""
ArxGLUE Core - Minimalistic Component Composition Implementation
"""

from typing import Any, Callable, Optional, Union

# 1. Core primitives
Component = Callable[[Any], Any]  # Any callable is a component

def connect(
    source: Union[Component, tuple[Component, ...]], 
    target: Union[Component, tuple[Component, ...]],
    transformer: Optional[Callable[[Any], Any]] = None
) -> tuple:
    """
    Declares a connection between components
    
    :param source: Source component(s)
    :param target: Target component(s)
    :param transformer: Optional data transformation function
    :return: Connection descriptor tuple
    """
    return (source, target, transformer)

# 2. Optional Context Protocol
class ContextProtocol:
    """
    Optional execution context protocol
    Usage:
        class MyContext(ContextProtocol):
            ...
    """
    input: Any
    output: Optional[Any]
    state: dict
    
    def __init__(self, input_data: Any):
        self.input = input_data
        self.output = None
        self.state = {}

============================================================
.\arxglue\arxglue\utils.py
============================================================
#NOT USE THIS MODULE IN YOUR PROJECTS THIS MODULE WILL DELETED IN NEXT REV JUST FOR TESTS!!!

from typing import Any, Callable, List, Tuple

def flatten_connections(connections: list) -> List[Tuple]:
    """
    Flattens group connections into 1:1 connections
    
    :param connections: List of connection descriptors
    :return: Flat list of (source, target, transformer) tuples
    """
    flattened = []
    for conn in connections:
        sources = conn[0] if isinstance(conn[0], tuple) else (conn[0],)
        targets = conn[1] if isinstance(conn[1], tuple) else (conn[1],)
        
        for src in sources:
            for tgt in targets:
                flattened.append((src, tgt, conn[2]))
    return flattened

def component(func: Callable) -> Callable:
    """
    Component decorator (optional)
    
    :param func: Component function
    :return: Marked component function
    """
    func._is_arxglue_component = True
    return func

# Перенесён из core.py
def execute_linear(
    components: list[Callable[[Any], Any]], 
    input_data: Any
) -> Any:
    """
    Sequential component execution (example)
    
    :param components: List of components to execute
    :param input_data: Input data
    :return: Processing result
    """
    result = input_data
    for comp in components:
        result = comp(result)
    return result

============================================================
.\arxglue\arxglue\__init__.py
============================================================
"""
ArxGLUE - Minimalistic Component Composition Interface
Version: 1.0
"""


from .core import Component, connect, ContextProtocol
from .utils import execute_linear, flatten_connections, component


__all__ = [
    'Component',
    'connect',
    'ContextProtocol',
    'execute_linear',
    'flatten_connections',
    'component'
]

============================================================
.\arxglue\arxviz\exporters.py
============================================================
import csv
from typing import Any, Dict, List, Tuple

def export_architecture_dot(
    components: List[Dict[str, Any]],
    connections: List[Dict[str, Any]]
) -> str:
    """
    Generate Graphviz DOT diagram from components and connections
    
    :param components: List of component dicts with 'id' and 'name'
    :param connections: List of connection dicts with 'source', 'target', 'transformer'
    :return: DOT diagram code
    """
    dot = ["digraph ArxArchitecture {"]
    dot.append("    rankdir=LR;")
    dot.append("    node [shape=box, style=rounded];")
    
    # Add components
    for comp in components:
        dot.append(f'    {comp["id"]} [label="{comp["name"]}"];')
    
    # Add connections
    for conn in connections:
        label = f' [label="{conn["transformer"]}"]' if conn.get("transformer") else ""
        dot.append(f'    {conn["source"]} -> {conn["target"]}{label};')
    
    dot.append("}")
    return "\n".join(dot)

def export_architecture_plantuml(
    components: List[Dict[str, Any]],
    connections: List[Dict[str, Any]]
) -> str:
    """
    Generate PlantUML diagram from components and connections
    
    :param components: List of component dicts
    :param connections: List of connection dicts
    :return: PlantUML diagram code
    """
    plantuml = ["@startuml"]
    plantuml.append("left to right direction")
    
    # Add components
    for comp in components:
        plantuml.append(f'component "{comp["name"]}" as {comp["id"]}')
    
    # Add connections
    for conn in connections:
        label = f' : {conn["transformer"]}' if conn.get("transformer") else ""
        plantuml.append(f'{conn["source"]} --> {conn["target"]}{label}')
    
    plantuml.append("@enduml")
    return "\n".join(plantuml)

def export_trace_plantuml(trace: List[Dict[str, Any]]) -> str:
    """
    Generate PlantUML sequence diagram from execution trace
    
    :param trace: Trace data from ErrorContext
    :return: PlantUML sequence diagram code
    """
    if not trace:
        return "@startuml\nnote: Empty trace\n@enduml"
    
    plantuml = ["@startuml"]
    plantuml.append("skinparam responseMessageBelowArrow true")
    
    # Collect participants
    participants = {entry["component"] for entry in trace}
    for p in participants:
        plantuml.append(f'participant "{p}" as {p}')
    
    # Add interactions
    for i, entry in enumerate(trace):
        comp = entry["component"]
        input_data = str(entry["input"])[:30] + "..." if len(str(entry["input"])) > 30 else entry["input"]
        
        if i > 0 and trace[i-1]["component"] != comp:
            plantuml.append(f'{trace[i-1]["component"]} -> {comp}: {input_data}')
        
        if entry.get("error"):
            plantuml.append(f'group Error')
            plantuml.append(f'note right of {comp}: {entry["error"]}')
            plantuml.append('end group')
    
    plantuml.append("@enduml")
    return "\n".join(plantuml)

def export_logs_text(logs: List[str]) -> str:
    """Convert logs to plain text format"""
    return "\n".join(logs)

def export_architecture_csv(
    components: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    base_name: str
):
    """
    Export architecture to CSV files
    
    :param components: List of component dicts
    :param connections: List of connection dicts
    :param base_name: Base filename
    """
    # Components CSV
    with open(f"{base_name}_components.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "type", "description"])
        writer.writeheader()
        writer.writerows(components)
    
    # Connections CSV
    with open(f"{base_name}_connections.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "transformer"])
        writer.writeheader()
        writer.writerows(connections)

def export_trace_mermaid(trace: List[Dict[str, Any]]) -> str:
    """
    Generate sequence diagram from execution trace
    
    :param trace: Trace data from ErrorContext
    :return: Mermaid sequence diagram code
    """
    diagram = [
        "sequenceDiagram",
        "    autonumber"
    ]
    
    for entry in trace:
        comp = entry["component"]
        input_data = str(entry["input"])[:30] + "..." if len(str(entry["input"])) > 30 else entry["input"]
        output_data = str(entry.get("output", ""))[:30] + "..." if entry.get("output") else ""
        
        diagram.append(f"    participant {comp}")
        
        if "calls" in entry:
            for call in entry["calls"]:
                diagram.append(f"    {comp}->>{call['component']}: {call['input']}")
        
        if output_data:
            diagram.append(f"    activate {comp}")
            diagram.append(f"    {comp}-->>Output: {output_data}")
            diagram.append(f"    deactivate {comp}")
        
        if entry.get("error"):
            diagram.append(f"    Note right of {comp}: ERROR: {entry['error']}")
    
    return "\n".join(diagram)

def export_logs_csv(logs: List[str], filename: str):
    """Export logs to CSV with improved parsing"""
    parsed_logs = []
    for entry in logs:
        # Улучшенный парсинг логов
        if entry.startswith("[") and "]" in entry:
            # Находим конец уровня
            end_index = entry.index("]")
            level = entry[1:end_index]
            message = entry[end_index+1:].strip()
            parsed_logs.append({
                "level": level,
                "message": message
            })
        else:
            # Для логов без стандартного формата
            parsed_logs.append({
                "level": "INFO",
                "message": entry
            })
    
    # Запись в файл
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "message"])
        writer.writeheader()
        writer.writerows(parsed_logs)


============================================================
.\arxglue\arxviz\importers.py
============================================================
import json
import re
from typing import Any, Dict, List, Tuple, Optional

def parse_dot_architecture(
    dot_code: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse DOT architecture diagram
    
    :param dot_code: Graphviz DOT code
    :param metadata: Component metadata
    :return: (components, connections)
    """
    components = []
    connections = []
    metadata = metadata or {}
    
    # Extract components
    node_pattern = r'(\w+)\s*\[label="([^"]+)"'
    for match in re.finditer(node_pattern, dot_code):
        comp_id = match.group(1)
        comp_name = match.group(2)
        components.append({
            "id": comp_id,
            "name": comp_name,
            **metadata.get(comp_id, {})
        })
    
    # Extract connections
    edge_pattern = r'(\w+)\s*->\s*(\w+)(\s*\[label="([^"]+)"\])?'
    for match in re.finditer(edge_pattern, dot_code):
        source = match.group(1)
        target = match.group(2)
        transformer = match.group(4) if match.group(4) else None
        
        connections.append({
            "source": source,
            "target": target,
            "transformer": transformer
        })
    
    return components, connections

def generate_code_skeleton(
    components: List[Dict[str, Any]],
    connections: List[Dict[str, Any]]
) -> str:
    """
    Generate Python code skeleton from architecture
    
    :param components: List of component dicts
    :param connections: List of connection dicts
    :return: Python code string
    """
    code = ["# Auto-generated by ArxViz", "from arxglue import connect, Component"]
    
    # Component definitions
    code.append("\n# Component definitions")
    for comp in components:
        comp_name = comp["name"].replace(" ", "_")
        if comp.get("type") == "class":
            code.append(f"\nclass {comp_name}(Component):")
            code.append("    def __call__(self, data):")
            code.append("        # Implement your logic here")
            code.append("        return data")
        else:
            code.append(f"\ndef {comp_name}(data):")
            code.append("    # Implement your logic here")
            code.append("    return data")
    
    # Connections
    code.append("\n# Architecture connections")
    for conn in connections:
        source = next((c["name"].replace(" ", "_") for c in components if c["id"] == conn["source"]), conn["source"])
        target = next((c["name"].replace(" ", "_") for c in components if c["id"] == conn["target"]), conn["target"])
        
        # Обработка трансформера
        transformer = ""
        if conn.get("transformer"):
            # Если трансформер уже определен как компонент
            if any(c["name"].replace(" ", "_") == conn["transformer"] for c in components):
                transformer = f", transformer={conn['transformer']}"
            else:
                # Добавляем новый трансформер как функцию
                code.append(f"\ndef {conn['transformer']}(data):")
                code.append("    # Implement transformation logic here")
                code.append("    return data")
                transformer = f", transformer={conn['transformer']}"
        
        code.append(f"connect({source}, {target}{transformer})")
    
    return "\n".join(code)


============================================================
.\arxglue\arxviz\renderers.py
============================================================
import os
import subprocess
from typing import Optional

def render_dot_to_png(dot_code: str, output_file: str, engine: str = "dot"):
    """
    Render DOT code to PNG using pure Python implementation
    
    :param dot_code: DOT diagram code
    :param output_file: Output PNG filename (without .png extension)
    :param engine: Layout engine (dot, neato, fdp, etc.)
    :return: Full path to the generated PNG file
    """
    try:
        from graphviz import Source
        # Убираем .png расширение, если оно есть
        if output_file.endswith('.png'):
            output_file = output_file[:-4]
            
        src = Source(dot_code, engine=engine)
        src.format = 'png'
        src.render(output_file, cleanup=True)
        
        # Возвращаем полный путь к сгенерированному файлу
        return output_file + '.png'
    except ImportError:
        # Fallback to PlantUML if Graphviz not available
        plantuml_code = f"@startdot\n{dot_code}\n@enddot"
        txt_file = output_file + '.txt'
        render_plantuml_to_ascii(plantuml_code, txt_file)
        raise RuntimeError("Graphviz not installed. Rendered PlantUML fallback.")
    except Exception as e:
        plantuml_code = f"@startdot\n{dot_code}\n@enddot"
        txt_file = output_file + '.txt'
        render_plantuml_to_ascii(plantuml_code, txt_file)
        raise RuntimeError(f"Graphviz error: {str(e)}. Rendered PlantUML fallback.")

def render_plantuml_to_ascii(
    plantuml_code: str, 
    output_file: str,
    server_url: str = "http://www.plantuml.com/plantuml"  # Параметр с умолчанием
):
    """
    Render PlantUML to ASCII art using pure Python
    
    :param plantuml_code: PlantUML diagram code
    :param output_file: Output text filename
    :param server_url: URL of PlantUML server (default: public server)
    """
    try:
        import plantuml
        plantuml.PlantUML(server_url).processes(plantuml_code, output_file)
    except ImportError:
        # Pure Python fallback
        with open(output_file, "w") as f:
            f.write("PlantUML rendering unavailable\n")
            f.write("Diagram source:\n\n")
            f.write(plantuml_code)
    except Exception as e:
        # Handle PlantUML server errors
        with open(output_file, "w") as f:
            f.write(f"PlantUML error: {str(e)}\n")
            f.write("Diagram source:\n\n")
            f.write(plantuml_code)

============================================================
.\arxglue\arxviz\utils.py
============================================================
import json
import re
from typing import Any, Dict

def extract_metadata(diagram_code: str) -> Dict[str, Any]:
    """
    Extract JSON metadata from diagram comments
    
    :param diagram_code: Diagram source code
    :return: Metadata dictionary
    """
    metadata = {}
    pattern = r"//\s*METADATA\s*:\s*(\{.*?\})"
    
    for match in re.finditer(pattern, diagram_code, re.DOTALL):
        try:
            metadata.update(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    
    return metadata

def generate_component_id(name: str) -> str:
    """Generate consistent component ID from name"""
    return name.lower().replace(" ", "_").replace("-", "_")

============================================================
.\arxglue\arxviz\__init__.py
============================================================
"""
ArxViz - Self-contained Visualization Tools for Arx Systems
Version: 1.0
"""

from .exporters import (
    export_architecture_dot,
    export_architecture_plantuml,
    export_trace_plantuml,
    export_logs_text,
    export_architecture_csv,
    export_logs_csv
)
from .importers import (
    parse_dot_architecture,
    generate_code_skeleton
)
from .renderers import (
    render_dot_to_png,
    render_plantuml_to_ascii
)

__all__ = [
    'export_architecture_dot',
    'export_architecture_plantuml',
    'export_trace_plantuml',
    'export_logs_text',
    'export_architecture_csv',
    'export_logs_csv',
    'parse_dot_architecture',
    'generate_code_skeleton',
    'render_dot_to_png',
    'render_plantuml_to_ascii'
]

============================================================
.\arxglue\gluetorch\hgpt.py
============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import requests
import zipfile
import io
from tqdm import tqdm
import warnings
import argparse
import psutil
import GPUtil
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    # Параметры данных
    dataset_name = 'imdb'
    max_seq_length = 64
    vocab_size = 100000  # Уменьшим для более быстрого тестирования
    batch_size = 32
    
    # Параметры модели
    d_model = 256
    nhead = 4
    num_layers = 7
    dim_feedforward = 512
    dropout = 0.1
    
    # Параметры истории (новые настраиваемые параметры)
    history_heads = 2
    history_dropout = 0.1
    
    # Стратегия выбора слоев для истории
    history_layer_strategy = "all"  # "all", "first_last", "custom" "first_mid_prelast"
    custom_history_layers = [0, 3, 5]  # если strategy = "custom"
    
    # Тип агрегации исторических состояний
    history_aggregation = "attention"  # "concat", "sum", "max_pool", "gated"
    
    # Уровень применения исторического внимания
    history_application_level = "output"  # "per_layer"
    
    # Метод объединения с основным потоком
    history_fusion = "residual"  # "gate", "concat"
    
    # Включение различных улучшений
    use_memory_bank = False  # Экспериментальная функция
    use_layerwise_gating = False  # Разные ворота для каждого исторического слоя
    use_learnable_weights = False  # Обучаемые веса для разных исторических слоев
    
    # Параметры обучения
    learning_rate = 5e-4
    num_epochs = 100  # Уменьшим для более быстрого тестирования
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Параметры генерации
    gen_length = 30  # Уменьшим для более быстрого тестирования
    temperature = 0.8
    
    # Визуализация и отладка
    print_interval = 50
    save_models = True
    visualize_attention = False  # Визуализация весов внимания

# ==================== УТИЛИТЫ ДАННЫХ ====================
def download_imdb_data():
    """Загрузка датасета IMDB если он отсутствует"""
    data_dir = "imdb_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    file_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    
    if not os.path.exists(file_path):
        print("Загрузка IMDB датасета...")
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc="Загрузка",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))
            
            print("Распаковка датасета...")
            import tarfile
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=data_dir)
            
            print("Датасет успешно загружен и распакован.")
        except Exception as e:
            print(f"Ошибка при загрузке датасета: {e}")
            print("Создаем демо-данные для тестирования...")
            create_demo_data(data_dir)
    else:
        print("Датасет уже существует.")
    
    return os.path.join(data_dir, "aclImdb")

def create_demo_data(data_dir):
    """Создание демо-данных если загрузка не удалась"""
    imdb_path = os.path.join(data_dir, "aclImdb", "train")
    pos_path = os.path.join(imdb_path, "pos")
    neg_path = os.path.join(imdb_path, "neg")
    
    os.makedirs(pos_path, exist_ok=True)
    os.makedirs(neg_path, exist_ok=True)
    
    # Создаем несколько демо-отзывов
    demo_reviews = [
        ("pos", "This movie was absolutely fantastic! Great acting and storyline."),
        ("pos", "One of the best films I've seen this year. Highly recommend."),
        ("pos", "Brilliant performance by the lead actor. The plot was engaging."),
        ("neg", "Terrible movie. Poor acting and boring storyline."),
        ("neg", "Waste of time. The plot made no sense and the acting was awful."),
        ("neg", "Disappointing film. Expected much more from this director.")
    ]
    
    for i, (label, review) in enumerate(demo_reviews):
        path = pos_path if label == "pos" else neg_path
        with open(os.path.join(path, f"review_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(review)
    
    print("Созданы демо-данные для тестирования.")

def load_imdb_texts(data_path):
    """Загрузка текстов из IMDB"""
    texts = []
    
    for label in ['pos', 'neg']:
        labeled_path = os.path.join(data_path, 'train', label)
        if os.path.exists(labeled_path):
            for file_name in os.listdir(labeled_path):
                if file_name.endswith('.txt'):
                    with open(os.path.join(labeled_path, file_name), 'r', encoding='utf-8') as f:
                        texts.append(f.read())
        else:
            print(f"Предупреждение: путь {labeled_path} не существует.")
    
    # Если текстов нет, создаем несколько демо-текстов
    if len(texts) == 0:
        texts = [
            "This movie was great! I really enjoyed it.",
            "Terrible film. Waste of time.",
            "Amazing acting and plot. Highly recommend.",
            "Poor storyline and bad acting.",
            "One of the best movies I've seen this year.",
            "Disappointing. Expected more from this director."
        ]
    
    return texts

def build_vocab(texts, vocab_size):
    """Построение словаря"""
    word_freq = defaultdict(int)
    for text in texts:
        for word in text.lower().replace('<br />', ' ').split():
            word_freq[word] += 1

    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:vocab_size-4]
    
    for idx, (word, freq) in enumerate(most_common):
        vocab[word] = idx + 4
    
    return vocab

def text_to_sequence(text, vocab, max_length):
    """Преобразование текста в последовательность индексов"""
    tokens = text.lower().replace('<br />', ' ').split()[:max_length-2]  # -2 для sos и eos
    sequence = [vocab['<sos>']] + [vocab.get(token, 1) for token in tokens] + [vocab['<eos>']]
    return torch.tensor(sequence)

def preprocess_data(texts, vocab, max_length):
    """Препроцессинг данных для языкового моделирования"""
    processed_texts = []
    
    for text in texts:
        sequence = text_to_sequence(text, vocab, max_length)
        if len(sequence) > 3:  # Убедимся, что последовательность достаточно длинная
            processed_texts.append(sequence)
    
    return processed_texts

def collate_fn(batch):
    """Функция для объединения примеров в батчи"""
    # Добавляем padding к последовательностям
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    
    # Для языкового моделирования цель - это та же последовательность, но сдвинутая на 1
    # Вход: все токены кроме последнего, цель: все токены кроме первого
    data = padded[:, :-1]
    target = padded[:, 1:]

    if data.size(1) != target.size(1):
        min_len = min(data.size(1), target.size(1))
        data = data[:, :min_len]
        target = target[:, :min_len]

    return data, target

# ==================== ПОЗИЦИОННОЕ КОДИРОВАНИЕ ====================
class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==================== БАЗОВАЯ МОДЕЛЬ ====================
class BaselineGenerator(nn.Module):
    """Базовая генеративная модель на трансформере (только декодер)"""
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Эмбеддинги
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)
        
        # Декодер трансформера
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.num_layers)
        
        # Выходной слой
        self.output_layer = nn.Linear(config.d_model, vocab_size)
        
    def forward(self, tgt, memory=None):
        # Эмбеддинги и позиционное кодирование
        tgt_emb = self.dropout(self.pos_encoding(self.token_embedding(tgt)))
        
        # Создаем маску для декодера
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.config.device)
        
        # Прямой проход через трансформер
        if memory is None:
            # Если memory не предоставлена, используем zeros like tgt_emb
            memory = torch.zeros_like(tgt_emb)
        
        output = self.transformer(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True
        )
        
        # Выходные вероятности
        logits = self.output_layer(output)
        return logits
    
    def generate(self, prompt, max_length, temperature=1.0):
        """Генерация текста по промпту"""
        self.eval()
        with torch.no_grad():
            # Начинаем с промпта
            generated = prompt.clone().unsqueeze(0)
            
            for _ in range(max_length):
                # Прямой проход для получения следующего токена
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Применяем softmax и выбираем следующий токен
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                # Добавляем к сгенерированной последовательности
                generated = torch.cat([generated, next_token], dim=1)
                
                # Если сгенерировали конец последовательности, останавливаемся
                if next_token.item() == 3:  # <eos>
                    break
            
            return generated.squeeze(0)

# ==================== УЛУЧШЕННАЯ МОДЕЛЬ ====================
class EnhancedHistoryAwareGenerator(nn.Module):
    """Улучшенная генеративная модель с историческим вниманием"""
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Эмбеддинги
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)
        
        # Декодерные слои
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # Инициализация механизмов истории
        self._init_history_mechanisms()
        
        # Внешняя память (экспериментальная функция)
        if config.use_memory_bank:
            self.memory_bank = nn.ParameterList()
            self.memory_attention = MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.history_heads,
                dropout=config.history_dropout,
                batch_first=True
            )
        
        # Выходной слой
        self.output_layer = nn.Linear(config.d_model, vocab_size)
        
        # Визуализация внимания
        self.attention_weights = None
    
    def _init_history_mechanisms(self):
        """Инициализация механизмов работы с истории"""
        config = self.config
        
        # Механизм исторического внимания
        if config.history_aggregation == "attention":
            self.history_attention = MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.history_heads,
                dropout=config.history_dropout,
                batch_first=True
            )
        
        # Нормализация и dropout для истории
        self.history_norm = nn.LayerNorm(config.d_model)
        self.history_dropout = nn.Dropout(config.history_dropout)
        
        # Механизмы слияния
        if config.history_fusion == "gate":
            if config.use_layerwise_gating:
                # Отдельные ворота для каждого исторического слоя
                self.history_gates = nn.ModuleList([
                    nn.Linear(config.d_model * 2, config.d_model)
                    for _ in range(config.num_layers)
                ])
            else:
                # Общие ворота
                self.history_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.sigmoid = nn.Sigmoid()
        
        elif config.history_fusion == "concat":
            self.history_projection = nn.Linear(config.d_model * 2, config.d_model)
        
        # Обучаемые веса для исторических слоев
        if config.use_learnable_weights:
            self.layer_weights = nn.Parameter(torch.ones(config.num_layers))
        
        # Проекционные слои для разных типов агрегации
        if config.history_aggregation in ["concat", "sum", "max_pool"]:
            self.history_proj = nn.Linear(config.d_model, config.d_model)
    
    def _select_history_layers(self, layer_idx):
        """Выбор слоев для истории на основе стратегии"""
        config = self.config
        
        if config.history_layer_strategy == "all":
            return True
        elif config.history_layer_strategy == "first_mid_prelast":
            mid_layer = config.num_layers // 2
            prelast_layer = config.num_layers - 2
            return layer_idx == 0 or layer_idx == mid_layer or layer_idx == prelast_layer
        elif config.history_layer_strategy == "first_last":
            return layer_idx == 0 or layer_idx == config.num_layers - 1
        elif config.history_layer_strategy == "custom":
            return layer_idx in config.custom_history_layers
        elif config.history_layer_strategy == "skip_even":
            return layer_idx % 2 == 0
        elif config.history_layer_strategy == "skip_odd":
            return layer_idx % 2 == 1
        
        return False
    
    def _apply_history_aggregation(self, history_states, current_state):
        """Применение выбранного метода агрегации исторических состояний"""
        config = self.config
        
        if config.history_aggregation == "attention":
            # Объединяем исторические состояния
            history = torch.stack(history_states, dim=2)
            batch_size, seq_len, history_len, d_model = history.shape
            
            # Изменяем форму для внимания
            history_flat = history.reshape(batch_size * seq_len, history_len, d_model)
            current_flat = current_state.reshape(batch_size * seq_len, 1, d_model)
            
            # Применяем внимание к истории
            attn_output, attn_weights = self.history_attention(
                query=current_flat,
                key=history_flat,
                value=history_flat
            )
            
            # Сохраняем веса для визуализации
            if config.visualize_attention:
                self.attention_weights = attn_weights.detach().cpu().numpy()
            
            attn_output = attn_output.reshape(batch_size, seq_len, d_model)
            return attn_output
        
        elif config.history_aggregation == "concat":
            # Конкатенация всех исторических состояний
            concat_states = torch.cat(history_states, dim=-1)
            return self.history_proj(concat_states)
        
        elif config.history_aggregation == "sum":
            # Суммирование исторических состояний
            sum_states = torch.stack(history_states, dim=0).sum(dim=0)
            return self.history_proj(sum_states)
        
        elif config.history_aggregation == "max_pool":
            # Max-pooling по историческим состояниям
            stacked_states = torch.stack(history_states, dim=0)
            max_states, _ = stacked_states.max(dim=0)
            return self.history_proj(max_states)
        
        elif config.history_aggregation == "gated":
            # Гейтированная агрегация
            weighted_states = []
            for i, state in enumerate(history_states):
                if self.config.use_learnable_weights:
                    weight = torch.sigmoid(self.layer_weights[i])
                else:
                    weight = 1.0 / len(history_states)
                weighted_states.append(state * weight)
            
            return torch.stack(weighted_states, dim=0).sum(dim=0)
        
        return None
    
    def _apply_history_fusion(self, current_state, history_output):
        """Применение выбранного метода слияния с историей"""
        config = self.config
        
        if config.history_fusion == "residual":
            return current_state + self.history_dropout(history_output)
        
        elif config.history_fusion == "gate":
            combined = torch.cat([current_state, history_output], dim=-1)
            if config.use_layerwise_gating:
                # Используем разные ворота для каждого слоя
                gate = self.sigmoid(self.history_gates[self.current_layer](combined))
            else:
                gate = self.sigmoid(self.history_gate(combined))
            return gate * current_state + (1 - gate) * history_output
        
        elif config.history_fusion == "concat":
            combined = torch.cat([current_state, history_output], dim=-1)
            return self.history_projection(combined)
        
        return current_state
    
    def forward(self, tgt, memory=None, use_history=True):
        # Эмбеддинги и позиционное кодирование
        tgt_emb = self.dropout(self.pos_encoding(self.token_embedding(tgt)))
        
        # Создаем маску для декодера
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.config.device)
        
        if memory is None:
            memory = torch.zeros_like(tgt_emb)
        
        # Собираем историю состояний
        history_states = []
        x = tgt_emb
        
        # Прямой проход через слои декодера
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, tgt_mask=tgt_mask, tgt_is_causal=True)
            
            # Сохраняем текущий слой для layerwise_gating
            self.current_layer = i
            
            # Сохраняем состояния выбранных слоев
            if use_history and self._select_history_layers(i):
                history_states.append(x)
                
                # Применяем историческое внимание на уровне каждого слоя
                if self.config.history_application_level == "per_layer" and len(history_states) > 1:
                    history_output = self._apply_history_aggregation(history_states[:-1], x)
                    if history_output is not None:
                        x = self._apply_history_fusion(x, history_output)
        
        # Применяем внимание к истории на выходном уровне
        if use_history and len(history_states) > 0 and self.config.history_application_level == "output":
            history_output = self._apply_history_aggregation(history_states, x)
            if history_output is not None:
                x = self._apply_history_fusion(x, history_output)
                x = self.history_norm(x)
        
        # Работа с внешней памятью (экспериментальная функция)
        if use_history and self.config.use_memory_bank and len(self.memory_bank) > 0:
            memory_output, _ = self.memory_attention(x, self.memory_bank[-1], self.memory_bank[-1])
            x = x + self.history_dropout(memory_output)
        
        # Выходные вероятности
        logits = self.output_layer(x)
        return logits
    
    def generate(self, prompt, max_length, temperature=1.0):
        """Генерация текста по промпту с использованием истории"""
        self.eval()
        with torch.no_grad():
            # Начинаем с промпта
            generated = prompt.clone().unsqueeze(0)
            
            # Инициализируем память для хранения состояний
            if self.config.use_memory_bank:
                self.memory_bank = nn.ParameterList()
            
            for i in range(max_length):
                # Прямой проход для получения следующего токена
                logits = self.forward(generated, use_history=True)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Применяем softmax и выбираем следующий токен
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                # Добавляем к сгенерированной последовательности
                generated = torch.cat([generated, next_token], dim=1)
                
                # Сохраняем состояния во внешнюю память
                if self.config.use_memory_bank and i % 5 == 0:  # Сохраняем каждые 5 шагов
                    with torch.no_grad():
                        # Для simplicity, просто сохраняем последнее состояние
                        self.memory_bank.append(nn.Parameter(generated.detach()))
                
                # Если сгенерировали конец последовательности, останавливаемся
                if next_token.item() == 3:  # <eos>
                    break
            
            return generated.squeeze(0)

# ==================== УТИЛИТЫ ОБУЧЕНИЯ ====================
def count_parameters(model):
    """Подсчет количества параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_usage():
    """Получение информации об использовании памяти"""
    memory_info = {}
    
    # CPU память
    memory_info['cpu_memory'] = psutil.virtual_memory().percent
    
    # GPU память (если доступно)
    if torch.cuda.is_available():
        memory_info['gpu_memory'] = torch.cuda.memory_allocated() / 1024**3  # в GB
        memory_info['gpu_memory_max'] = torch.cuda.max_memory_allocated() / 1024**3  # в GB
    else:
        memory_info['gpu_memory'] = 0
        memory_info['gpu_memory_max'] = 0
    
    return memory_info

def train_epoch(model, dataloader, criterion, optimizer, config, epoch, model_name):
    """Обучение на одной эпохе"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(config.device), target.to(config.device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Вычисляем потерю
        loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()
        
        if batch_idx % config.print_interval == 0:
            perplexity = np.exp(loss.item())
            print(f'{model_name} - Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Perplexity: {perplexity:.2f}')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def generate_sample(model, vocab, inv_vocab, config, prompt_text):
    """Генерация примера текста"""
    # Преобразуем промпт в последовательность индексов
    prompt_tokens = prompt_text.lower().split()
    prompt_indices = [vocab.get(token, 1) for token in prompt_tokens]
    prompt_indices = [vocab['<sos>']] + prompt_indices
    
    prompt_tensor = torch.tensor(prompt_indices).to(config.device)
    
    # Генерируем продолжение
    generated = model.generate(prompt_tensor, config.gen_length, config.temperature)
    
    # Преобразуем обратно в текст
    generated_tokens = []
    for idx in generated.cpu().numpy():
        if idx == vocab['<eos>']:
            break
        if idx not in [vocab['<sos>'], vocab['<pad>'], vocab['<unk>']]:
            generated_tokens.append(inv_vocab.get(idx, '<unk>'))
    
    return ' '.join(generated_tokens)

# ==================== УТИЛИТЫ ДЛЯ ВИЗУАЛИЗАЦИИ ====================
def visualize_attention_weights(attention_weights, layer_names, save_path):
    """Визуализация весов внимания"""
    if attention_weights is None:
        return
    
    plt.figure(figsize=(12, 8))
    
    # attention_weights shape: [batch*seq, history_len, 1]
    attn = attention_weights.reshape(-1, attention_weights.shape[1])
    avg_attn = attn.mean(axis=0)
    
    plt.bar(range(len(avg_attn)), avg_attn)
    plt.xticks(range(len(avg_attn)), layer_names)
    plt.xlabel("Исторические слои")
    plt.ylabel("Средний вес внимания")
    plt.title("Распределение внимания по историческим слоям")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_training_results(baseline_history, history_model_history, save_path):
    """Визуализация результатов обучения"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if baseline_history:
        plt.plot(baseline_history['train_perplexity'], label='Baseline')
    if history_model_history:
        plt.plot(history_model_history['train_perplexity'], label='History Model')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.title('Training Perplexity')
    
    plt.subplot(1, 2, 2)
    if baseline_history:
        plt.plot(baseline_history['train_loss'], label='Baseline')
    if history_model_history:
        plt.plot(history_model_history['train_loss'], label='History Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==================== РЕЖИМЫ РАБОТЫ ====================
def competitive_mode(config):
    """Соревновательный режим: обучение обеих моделей и сравнение"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== РЕЖИМ: СОРЕВНОВАТЕЛЬНЫЙ (запуск: {timestamp}) ===")
    
    # Загрузка данных
    data_path = download_imdb_data()
    texts = load_imdb_texts(data_path)
    
    # Построение словаря
    print("Построение словаря...")
    vocab = build_vocab(texts, config.vocab_size)
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_path = f"vocab_с{timestamp}.pkl"
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Словарь сохранен как: {vocab_path}")
    
    # Предобработка данных
    print("Предобработка данных...")
    processed_texts = preprocess_data(texts[:1000], vocab, config.max_seq_length)  # Используем только часть данных
    
    # Создание DataLoader
    train_loader = DataLoader(
        processed_texts, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Инициализация моделей
    baseline_model = BaselineGenerator(config, len(vocab)).to(config.device)
    history_model = EnhancedHistoryAwareGenerator(config, len(vocab)).to(config.device)
    
    # Подсчет параметров
    baseline_params = count_parameters(baseline_model)
    history_params = count_parameters(history_model)
    
    print(f"Параметры Baseline модели: {baseline_params:,}")
    print(f"Параметры History модели: {history_params:,}")
    print(f"Разница: {history_params - baseline_params:,} параметров")
    
    # Функция потерь и оптимизаторы
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Игнорируем padding
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=config.learning_rate)
    history_optimizer = torch.optim.Adam(history_model.parameters(), lr=config.learning_rate)
    
    # Обучение моделей
    print("Начало обучения...")
    
    baseline_history = {
        'train_loss': [], 'train_perplexity': [],
        'memory_usage': [], 'training_time': 0
    }
    
    history_model_history = {
        'train_loss': [], 'train_perplexity': [],
        'memory_usage': [], 'training_time': 0
    }
    
    # Обучение Baseline модели
    print("\n=== ОБУЧЕНИЕ BASELINE МОДЕЛИ ===")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        train_loss, train_ppl = train_epoch(
            baseline_model, train_loader, criterion, baseline_optimizer, config, epoch, "Baseline"
        )
        
        baseline_history['train_loss'].append(train_loss)
        baseline_history['train_perplexity'].append(train_ppl)
        baseline_history['memory_usage'].append(get_memory_usage())
        
        print(f"Baseline - Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
    
    baseline_history['training_time'] = time.time() - start_time
    print(f"Baseline обучение заняло: {baseline_history['training_time']:.2f} секунд")
    
    # Обучение History модели
    print("\n=== ОБУЧЕНИЕ HISTORY МОДЕЛИ ===")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        train_loss, train_ppl = train_epoch(
            history_model, train_loader, criterion, history_optimizer, config, epoch, "History"
        )
        
        history_model_history['train_loss'].append(train_loss)
        history_model_history['train_perplexity'].append(train_ppl)
        history_model_history['memory_usage'].append(get_memory_usage())
        
        print(f"History - Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
    
    history_model_history['training_time'] = time.time() - start_time
    print(f"History обучение заняло: {history_model_history['training_time']:.2f} секунд")
    
    # Генерация примеров
    print("\n=== ГЕНЕРАЦИЯ ПРИМЕРОВ ТЕКСТА ===")
    prompts = [
        "How are you today?",
        "i really liked",
        "the acting was",
        "the story is",
        "this movie is"
    ]
    
    baseline_results = []
    history_results = []
    
    print("\n=== BASELINE MODEL ===")
    for prompt in prompts:
        generated = generate_sample(baseline_model, vocab, inv_vocab, config, prompt)
        baseline_results.append((prompt, generated))
        print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
    
    print("\n=== HISTORY MODEL ===")
    for prompt in prompts:
        generated = generate_sample(history_model, vocab, inv_vocab, config, prompt)
        history_results.append((prompt, generated))
        print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
    
    # Визуализация весов внимания
    if config.visualize_attention and hasattr(history_model, 'attention_weights'):
        layer_names = []
        for i in range(config.num_layers):
            if history_model._select_history_layers(i):
                layer_names.append(f"Layer {i}")
        
        attention_path = f"attention_weights_{timestamp}.png"
        visualize_attention_weights(
            history_model.attention_weights, 
            layer_names,
            attention_path
        )
        print(f"Визуализация внимания сохранена как: {attention_path}")
    
    # Визуализация результатов
    training_plot_path = f"training_history_{timestamp}.png"
    visualize_training_results(baseline_history, history_model_history, training_plot_path)
    print(f"График обучения сохранен как: {training_plot_path}")
    
    # Сохранение моделей
    if config.save_models:
        baseline_path = f'baseline_generator_{timestamp}.pth'
        history_path = f'history_generator_{timestamp}.pth'
        
        torch.save(baseline_model.state_dict(), baseline_path)
        torch.save(history_model.state_dict(), history_path)
        print(f"Модели сохранены как '{baseline_path}' и '{history_path}'")
    
    # Сравнение финальных результатов
    print("\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ===")
    if baseline_history['train_perplexity']:
        print(f"Baseline Model - Final Train Perplexity: {baseline_history['train_perplexity'][-1]:.2f}")
    if history_model_history['train_perplexity']:
        print(f"History Model - Final Train Perplexity: {history_model_history['train_perplexity'][-1]:.2f}")
    
    if baseline_history['train_perplexity'] and history_model_history['train_perplexity']:
        improvement = baseline_history['train_perplexity'][-1] - history_model_history['train_perplexity'][-1]
        print(f"Improvement: {improvement:.2f}")
    
    # Аналитика использования ресурсов
    print("\n=== АНАЛИТИКА ИСПОЛЬЗОВАНИЯ РЕСУРСОВ ===")
    if baseline_history['memory_usage']:
        avg_cpu_memory = np.mean([m['cpu_memory'] for m in baseline_history['memory_usage']])
        avg_gpu_memory = np.mean([m['gpu_memory'] for m in baseline_history['memory_usage']])
        max_gpu_memory = np.max([m['gpu_memory_max'] for m in baseline_history['memory_usage']])
        
        print(f"Baseline - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%")
        print(f"Baseline - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB")
        print(f"Baseline - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB")
        print(f"Baseline - Время обучения: {baseline_history['training_time']:.2f} секунд")
    
    if history_model_history['memory_usage']:
        avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
        avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
        max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
        
        print(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%")
        print(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB")
        print(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB")
        print(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд")
    
    # Сохранение результатов в файл
    results_path = f"results_{timestamp}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ===\n")
        f.write(f"Дата и время: {timestamp}\n")
        f.write(f"Режим: Соревновательный\n\n")
        
        f.write("=== МОДЕЛИ ===\n")
        f.write(f"Baseline параметров: {baseline_params:,}\n")
        f.write(f"History параметров: {history_params:,}\n")
        f.write(f"Разница: {history_params - baseline_params:,} параметров\n\n")
        
        f.write("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===\n")
        if baseline_history['train_perplexity']:
            f.write(f"Baseline - Финальная перплексия: {baseline_history['train_perplexity'][-1]:.2f}\n")
        if history_model_history['train_perplexity']:
            f.write(f"History - Финальная перплексия: {history_model_history['train_perplexity'][-1]:.2f}\n")
        
        if baseline_history['train_perplexity'] and history_model_history['train_perplexity']:
            improvement = baseline_history['train_perplexity'][-1] - history_model_history['train_perplexity'][-1]
            f.write(f"Улучшение: {improvement:.2f}\n")
        
        f.write("\n=== РЕСУРСЫ ===\n")
        if baseline_history['memory_usage']:
            avg_cpu_memory = np.mean([m['cpu_memory'] for m in baseline_history['memory_usage']])
            avg_gpu_memory = np.mean([m['gpu_memory'] for m in baseline_history['memory_usage']])
            max_gpu_memory = np.max([m['gpu_memory_max'] for m in baseline_history['memory_usage']])
            
            f.write(f"Baseline - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%\n")
            f.write(f"Baseline - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB\n")
            f.write(f"Baseline - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB\n")
            f.write(f"Baseline - Время обучения: {baseline_history['training_time']:.2f} секунд\n")
        
        if history_model_history['memory_usage']:
            avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
            avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
            max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
            
            f.write(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%\n")
            f.write(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB\n")
            f.write(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB\n")
            f.write(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд\n")
        
        f.write("\n=== ГЕНЕРАЦИЯ ===\n")
        f.write("Baseline Model:\n")
        for prompt, generated in baseline_results:
            f.write(f"Prompt: '{prompt}' -> Generated: '{generated}'\n")
        
        f.write("\nHistory Model:\n")
        for prompt, generated in history_results:
            f.write(f"Prompt: '{prompt}' -> Generated: '{generated}'\n")
    
    print(f"Полные результаты сохранены в: {results_path}")

def history_only_mode(config):
    """Режим только для History модели"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== РЕЖИМ: ТОЛЬКО HISTORY (запуск: {timestamp}) ===")
    
    # Загрузка данных
    data_path = download_imdb_data()
    texts = load_imdb_texts(data_path)
    
    # Построение словаря
    print("Построение словаря...")
    vocab = build_vocab(texts, config.vocab_size)
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_path = f"vocab_h{timestamp}.pkl"
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Словарь сохранен как: {vocab_path}")
    
    # Предобработка данных
    print("Предобработка данных...")
    processed_texts = preprocess_data(texts[:1000], vocab, config.max_seq_length)  # Используем только часть данных
    
    # Создание DataLoader
    train_loader = DataLoader(
        processed_texts, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Инициализация History модели
    history_model = EnhancedHistoryAwareGenerator(config, len(vocab)).to(config.device)
    
    # Подсчет параметров
    history_params = count_parameters(history_model)
    print(f"Параметры History модели: {history_params:,}")
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Игнорируем padding
    history_optimizer = torch.optim.Adam(history_model.parameters(), lr=config.learning_rate)
    
    # Обучение History модели
    print("\n=== ОБУЧЕНИЕ HISTORY МОДЕЛИ ===")
    
    history_model_history = {
        'train_loss': [], 'train_perplexity': [],
        'memory_usage': [], 'training_time': 0
    }
    
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        train_loss, train_ppl = train_epoch(
            history_model, train_loader, criterion, history_optimizer, config, epoch, "History"
        )
        
        history_model_history['train_loss'].append(train_loss)
        history_model_history['train_perplexity'].append(train_ppl)
        history_model_history['memory_usage'].append(get_memory_usage())
        
        print(f"History - Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
    
    history_model_history['training_time'] = time.time() - start_time
    print(f"History обучение заняло: {history_model_history['training_time']:.2f} секунд")
    
    # Генерация примеров
    print("\n=== ГЕНЕРАЦИЯ ПРИМЕРОВ ТЕКСТА ===")
    prompts = [
        "How are you?",
        "i really liked",
        "the acting was",
        "the story is"
    ]
    
    history_results = []
    
    print("\n=== HISTORY MODEL ===")
    for prompt in prompts:
        generated = generate_sample(history_model, vocab, inv_vocab, config, prompt)
        history_results.append((prompt, generated))
        print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
    
    # Визуализация весов внимания
    if config.visualize_attention and hasattr(history_model, 'attention_weights'):
        layer_names = []
        for i in range(config.num_layers):
            if history_model._select_history_layers(i):
                layer_names.append(f"Layer {i}")
        
        attention_path = f"history_attention_weights_{timestamp}.png"
        visualize_attention_weights(
            history_model.attention_weights, 
            layer_names,
            attention_path
        )
        print(f"Визуализация внимания сохранена как: {attention_path}")
    
    # Визуализация результатов
    training_plot_path = f"history_training_{timestamp}.png"
    visualize_training_results(None, history_model_history, training_plot_path)
    print(f"График обучения сохранен как: {training_plot_path}")
    
    # Сохранение модели
    if config.save_models:
        history_path = f'history_generator_{timestamp}.pth'
        torch.save(history_model.state_dict(), history_path)
        print(f"Модель сохранена как '{history_path}'")
    
    # Аналитика использования ресурсов
    print("\n=== АНАЛИТИКА ИСПОЛЬЗОВАНИЯ РЕСУРСОВ ===")
    if history_model_history['memory_usage']:
        avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
        avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
        max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
        
        print(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%")
        print(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB")
        print(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB")
        print(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд")
    
    # Сохранение результатов в файл
    results_path = f"history_results_{timestamp}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ===\n")
        f.write(f"Дата и время: {timestamp}\n")
        f.write(f"Режим: Только History\n\n")
        
        f.write("=== МОДЕЛЬ ===\n")
        f.write(f"History параметров: {history_params:,}\n\n")
        
        f.write("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===\n")
        if history_model_history['train_perplexity']:
            f.write(f"History - Финальная перплексия: {history_model_history['train_perplexity'][-1]:.2f}\n")
        
        f.write("\n=== РЕСУРСЫ ===\n")
        if history_model_history['memory_usage']:
            avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
            avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
            max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
            
            f.write(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%\n")
            f.write(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB\n")
            f.write(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB\n")
            f.write(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд\n")
        
        f.write("\n=== ГЕНЕРАЦИЯ ===\n")
        f.write("History Model:\n")
        for prompt, generated in history_results:
            f.write(f"Prompt: '{prompt}' -> Generated: '{generated}'\n")
    
    print(f"Полные результаты сохранены в: {results_path}")

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    parser = argparse.ArgumentParser(description='HistoryGPT - Генеративная модель с историческим вниманием')
    parser.add_argument('--mode', type=str, default='competitive', 
                       choices=['competitive', 'history'],
                       help='Режим работы: competitive (обе модели) или history (только history модель)')
    
    args = parser.parse_args()
    
    config = Config()
    print(f"Используется устройство: {config.device}")
    
    if args.mode == 'competitive':
        competitive_mode(config)
    elif args.mode == 'history':
        history_only_mode(config)

if __name__ == "__main__":
    main()

============================================================
.\arxglue\gluetorch\hgpt2.py
============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import requests
import zipfile
import io
from tqdm import tqdm
import warnings
import argparse
import psutil
import GPUtil
import pickle
from datetime import datetime
import re

warnings.filterwarnings('ignore')

# ==================== КОНФИГУРАЦИЯ ====================
class Config:
    # Параметры данных
    dataset_type = "imdb"  # "imdb" или "literature"
    data_path = "literature"  # путь к папке с данными
    max_seq_length = 64
    vocab_size = 10000000  # Уменьшим для более быстрого тестирования
    batch_size = 32
    
    # Параметры модели
    d_model = 256
    nhead = 4
    num_layers = 7
    dim_feedforward = 512
    dropout = 0.1
    
    # Параметры истории (новые настраиваемые параметры)
    history_heads = 2
    history_dropout = 0.1
    
    # Стратегия выбора слоев для истории
    history_layer_strategy = "all"  # "all", "first_last", "custom" "first_mid_prelast"
    custom_history_layers = [0, 3, 5]  # если strategy = "custom"
    
    # Тип агрегации исторических состояний
    history_aggregation = "attention"  # "concat", "sum", "max_pool", "gated"
    
    # Уровень применения исторического внимания
    history_application_level = "output"  # "per_layer"
    
    # Метод объединения с основным потоком
    history_fusion = "residual"  # "gate", "concat"
    
    # Включение различных улучшений
    use_memory_bank = False  # Экспериментальная функция
    use_layerwise_gating = False  # Разные ворота для каждого исторического слоя
    use_learnable_weights = False  # Обучаемые веса для разных исторических слоев
    
    # Параметры обучения
    learning_rate = 5e-4
    num_epochs = 100  # Уменьшим для более быстрого тестирования
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Параметры генерации
    gen_length = 30  # Уменьшим для более быстрого тестирования
    temperature = 0.8
    
    # Визуализация и отладка
    print_interval = 50
    save_models = True
    visualize_attention = False  # Визуализация весов внимания

# ==================== УТИЛИТЫ ДАННЫХ ====================
def download_imdb_data():
    """Загрузка датасета IMDB если он отсутствует"""
    data_dir = "imdb_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    file_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    
    if not os.path.exists(file_path):
        print("Загрузка IMDB датасета...")
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc="Загрузка",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    pbar.update(len(data))
            
            print("Распаковка датасета...")
            import tarfile
            with tarfile.open(file_path, 'r:gz') as tar:
                tar.extractall(path=data_dir)
            
            print("Датасет успешно загружен и распакован.")
        except Exception as e:
            print(f"Ошибка при загрузке датасета: {e}")
            print("Создаем демо-данные для тестирования...")
            create_demo_data(data_dir)
    else:
        print("Датасет уже существует.")
    
    return os.path.join(data_dir, "aclImdb")

def create_demo_data(data_dir):
    """Создание демо-данных если загрузка не удалась"""
    imdb_path = os.path.join(data_dir, "aclImdb", "train")
    pos_path = os.path.join(imdb_path, "pos")
    neg_path = os.path.join(imdb_path, "neg")
    
    os.makedirs(pos_path, exist_ok=True)
    os.makedirs(neg_path, exist_ok=True)
    
    # Создаем несколько демо-отзывов
    demo_reviews = [
        ("pos", "This movie was absolutely fantastic! Great acting and storyline."),
        ("pos", "One of the best films I've seen this year. Highly recommend."),
        ("pos", "Brilliant performance by the lead actor. The plot was engaging."),
        ("neg", "Terrible movie. Poor acting and boring storyline."),
        ("neg", "Waste of time. The plot made no sense and the acting was awful."),
        ("neg", "Disappointing film. Expected much more from this director.")
    ]
    
    for i, (label, review) in enumerate(demo_reviews):
        path = pos_path if label == "pos" else neg_path
        with open(os.path.join(path, f"review_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(review)
    
    print("Созданы демо-данные для тестирования.")

def load_imdb_texts(data_path):
    """Загрузка текстов из IMDB"""
    texts = []
    
    for label in ['pos', 'neg']:
        labeled_path = os.path.join(data_path, 'train', label)
        if os.path.exists(labeled_path):
            for file_name in os.listdir(labeled_path):
                if file_name.endswith('.txt'):
                    with open(os.path.join(labeled_path, file_name), 'r', encoding='utf-8') as f:
                        texts.append(f.read())
        else:
            print(f"Предупреждение: путь {labeled_path} не существует.")
    
    # Если текстов нет, создаем несколько демо-текстов
    if len(texts) == 0:
        texts = [
            "This movie was great! I really enjoyed it.",
            "Terrible film. Waste of time.",
            "Amazing acting and plot. Highly recommend.",
            "Poor storyline and bad acting.",
            "One of the best movies I've seen this year.",
            "Disappointing. Expected more from this director."
        ]
    
    return texts

def load_literature_texts(folder_path):
    """Загрузка текстов из папки с литературой с обработкой различных кодировок"""
    texts = []
    encodings_to_try = ['utf-8', 'cp1251', 'iso-8859-1', 'cp866', 'koi8-r']
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            content = None
            
            # Пытаемся прочитать файл с разными кодировками
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break  # Если удалось прочитать, выходим из цикла
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Ошибка при чтении файла {filename} с кодировкой {encoding}: {e}")
                    continue
            
            # Если не удалось прочитать ни одной кодировкой, пробуем бинарный режим
            if content is None:
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read().decode('utf-8', errors='ignore')
                    print(f"Файл {filename} прочитан в бинарном режиме с игнорированием ошибок")
                except Exception as e:
                    print(f"Не удалось прочитать файл {filename}: {e}")
                    continue
            
            if content:
                texts.append(content)
    
    return texts

def build_vocab(texts, vocab_size, is_russian=False):
    """Построение словаря"""
    word_freq = defaultdict(int)
    
    for text in texts:
        if is_russian:
            # Токенизация для русского языка
            words = re.findall(r'\b[а-яё]+\b', text.lower())
        else:
            # Токенизация для английского языка
            words = text.lower().replace('<br />', ' ').split()
            
        for word in words:
            word_freq[word] += 1

    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:vocab_size-4]
    
    for idx, (word, freq) in enumerate(most_common):
        vocab[word] = idx + 4
    
    return vocab

def text_to_sequence(text, vocab, max_length, is_russian=False):
    """Преобразование текста в последовательность индексов"""
    if is_russian:
        # Токенизация для русского языка
        words = re.findall(r'\b[а-яё]+\b', text.lower())
        words = words[:max_length-2]
    else:
        # Токенизация для английского языка
        words = text.lower().replace('<br />', ' ').split()[:max_length-2]
    
    sequence = [vocab['<sos>']] + [vocab.get(word, 1) for word in words] + [vocab['<eos>']]
    return torch.tensor(sequence)

def preprocess_data(texts, vocab, max_length, is_russian=False):
    """Препроцессинг данных для языкового моделирования"""
    processed_texts = []
    
    for text in texts:
        sequence = text_to_sequence(text, vocab, max_length, is_russian)
        if len(sequence) > 3:  # Убедимся, что последовательность достаточно длинная
            processed_texts.append(sequence)
    
    return processed_texts

def collate_fn(batch):
    """Функция для объединения примеров в батчи"""
    # Добавляем padding к последовательностям
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    
    # Для языкового моделирования цель - это та же последовательность, но сдвинутая на 1
    # Вход: все токены кроме последнего, цель: все токены кроме первого
    data = padded[:, :-1]
    target = padded[:, 1:]

    if data.size(1) != target.size(1):
        min_len = min(data.size(1), target.size(1))
        data = data[:, :min_len]
        target = target[:, :min_len]

    return data, target

# ==================== ПОЗИЦИОННОЕ КОДИРОВАНИЕ ====================
class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==================== БАЗОВАЯ МОДЕЛЬ ====================
class BaselineGenerator(nn.Module):
    """Базовая генеративная модель на трансформере (только декодер)"""
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Эмбеддинги
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)
        
        # Декодер трансформера
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.num_layers)
        
        # Выходной слой
        self.output_layer = nn.Linear(config.d_model, vocab_size)
        
    def forward(self, tgt, memory=None):
        # Эмбеддинги и позиционное кодирование
        tgt_emb = self.dropout(self.pos_encoding(self.token_embedding(tgt)))
        
        # Создаем маску для декодера
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.config.device)
        
        # Прямой проход через трансформер
        if memory is None:
            # Если memory не предоставлена, используем zeros like tgt_emb
            memory = torch.zeros_like(tgt_emb)
        
        output = self.transformer(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True
        )
        
        # Выходные вероятности
        logits = self.output_layer(output)
        return logits
    
    def generate(self, prompt, max_length, temperature=1.0):
        """Генерация текста по промпту"""
        self.eval()
        with torch.no_grad():
            # Начинаем с промпта
            generated = prompt.clone().unsqueeze(0)
            
            for _ in range(max_length):
                # Прямой проход для получения следующего токена
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Применяем softmax и выбираем следующий токен
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                # Добавляем к сгенерированной последовательности
                generated = torch.cat([generated, next_token], dim=1)
                
                # Если сгенерировали конец последовательности, останавливаемся
                if next_token.item() == 3:  # <eos>
                    break
            
            return generated.squeeze(0)

# ==================== УЛУЧШЕННАЯ МОДЕЛЬ ====================
class EnhancedHistoryAwareGenerator(nn.Module):
    """Улучшенная генеративная модель с историческим вниманием"""
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Эмбеддинги
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)
        
        # Декодерные слои
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # Инициализация механизмов истории
        self._init_history_mechanisms()
        
        # Внешняя память (экспериментальная функция)
        if config.use_memory_bank:
            self.memory_bank = nn.ParameterList()
            self.memory_attention = MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.history_heads,
                dropout=config.history_dropout,
                batch_first=True
            )
        
        # Выходной слой
        self.output_layer = nn.Linear(config.d_model, vocab_size)
        
        # Визуализация внимания
        self.attention_weights = None
    
    def _init_history_mechanisms(self):
        """Инициализация механизмов работы с истории"""
        config = self.config
        
        # Механизм исторического внимания
        if config.history_aggregation == "attention":
            self.history_attention = MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.history_heads,
                dropout=config.history_dropout,
                batch_first=True
            )
        
        # Нормализация и dropout для истории
        self.history_norm = nn.LayerNorm(config.d_model)
        self.history_dropout = nn.Dropout(config.history_dropout)
        
        # Механизмы слияния
        if config.history_fusion == "gate":
            if config.use_layerwise_gating:
                # Отдельные ворота для каждого исторического слоя
                self.history_gates = nn.ModuleList([
                    nn.Linear(config.d_model * 2, config.d_model)
                    for _ in range(config.num_layers)
                ])
            else:
                # Общие ворота
                self.history_gate = nn.Linear(config.d_model * 2, config.d_model)
            self.sigmoid = nn.Sigmoid()
        
        elif config.history_fusion == "concat":
            self.history_projection = nn.Linear(config.d_model * 2, config.d_model)
        
        # Обучаемые веса для исторических слоев
        if config.use_learnable_weights:
            self.layer_weights = nn.Parameter(torch.ones(config.num_layers))
        
        # Проекционные слои для разных типов агрегации
        if config.history_aggregation in ["concat", "sum", "max_pool"]:
            self.history_proj = nn.Linear(config.d_model, config.d_model)
    
    def _select_history_layers(self, layer_idx):
        """Выбор слоев для истории на основе стратегии"""
        config = self.config
        
        if config.history_layer_strategy == "all":
            return True
        elif config.history_layer_strategy == "first_mid_prelast":
            mid_layer = config.num_layers // 2
            prelast_layer = config.num_layers - 2
            return layer_idx == 0 or layer_idx == mid_layer or layer_idx == prelast_layer
        elif config.history_layer_strategy == "first_last":
            return layer_idx == 0 or layer_idx == config.num_layers - 1
        elif config.history_layer_strategy == "custom":
            return layer_idx in config.custom_history_layers
        elif config.history_layer_strategy == "skip_even":
            return layer_idx % 2 == 0
        elif config.history_layer_strategy == "skip_odd":
            return layer_idx % 2 == 1
        
        return False
    
    def _apply_history_aggregation(self, history_states, current_state):
        """Применение выбранного метода агрегации исторических состояний"""
        config = self.config
        
        if config.history_aggregation == "attention":
            # Объединяем исторические состояния
            history = torch.stack(history_states, dim=2)
            batch_size, seq_len, history_len, d_model = history.shape
            
            # Изменяем форму для внимания
            history_flat = history.reshape(batch_size * seq_len, history_len, d_model)
            current_flat = current_state.reshape(batch_size * seq_len, 1, d_model)
            
            # Применяем внимание к истории
            attn_output, attn_weights = self.history_attention(
                query=current_flat,
                key=history_flat,
                value=history_flat
            )
            
            # Сохраняем веса для визуализации
            if config.visualize_attention:
                self.attention_weights = attn_weights.detach().cpu().numpy()
            
            attn_output = attn_output.reshape(batch_size, seq_len, d_model)
            return attn_output
        
        elif config.history_aggregation == "concat":
            # Конкатенация всех исторических состояний
            concat_states = torch.cat(history_states, dim=-1)
            return self.history_proj(concat_states)
        
        elif config.history_aggregation == "sum":
            # Суммирование исторических состояний
            sum_states = torch.stack(history_states, dim=0).sum(dim=0)
            return self.history_proj(sum_states)
        
        elif config.history_aggregation == "max_pool":
            # Max-pooling по историческим состояниям
            stacked_states = torch.stack(history_states, dim=0)
            max_states, _ = stacked_states.max(dim=0)
            return self.history_proj(max_states)
        
        elif config.history_aggregation == "gated":
            # Гейтированная агрегация
            weighted_states = []
            for i, state in enumerate(history_states):
                if self.config.use_learnable_weights:
                    weight = torch.sigmoid(self.layer_weights[i])
                else:
                    weight = 1.0 / len(history_states)
                weighted_states.append(state * weight)
            
            return torch.stack(weighted_states, dim=0).sum(dim=0)
        
        return None
    
    def _apply_history_fusion(self, current_state, history_output):
        """Применение выбранного метода слияния с историей"""
        config = self.config
        
        if config.history_fusion == "residual":
            return current_state + self.history_dropout(history_output)
        
        elif config.history_fusion == "gate":
            combined = torch.cat([current_state, history_output], dim=-1)
            if config.use_layerwise_gating:
                # Используем разные ворота для каждого слоя
                gate = self.sigmoid(self.history_gates[self.current_layer](combined))
            else:
                gate = self.sigmoid(self.history_gate(combined))
            return gate * current_state + (1 - gate) * history_output
        
        elif config.history_fusion == "concat":
            combined = torch.cat([current_state, history_output], dim=-1)
            return self.history_projection(combined)
        
        return current_state
    
    def forward(self, tgt, memory=None, use_history=True):
        # Эмбеддинги и позиционное кодирование
        tgt_emb = self.dropout(self.pos_encoding(self.token_embedding(tgt)))
        
        # Создаем маску для декодера
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.config.device)
        
        if memory is None:
            memory = torch.zeros_like(tgt_emb)
        
        # Собираем историю состояний
        history_states = []
        x = tgt_emb
        
        # Прямой проход через слои декодера
        for i, layer in enumerate(self.layers):
            x = layer(x, memory, tgt_mask=tgt_mask, tgt_is_causal=True)
            
            # Сохраняем текущий слой для layerwise_gating
            self.current_layer = i
            
            # Сохраняем состояния выбранных слоев
            if use_history and self._select_history_layers(i):
                history_states.append(x)
                
                # Применяем историческое внимание на уровне каждого слоя
                if self.config.history_application_level == "per_layer" and len(history_states) > 1:
                    history_output = self._apply_history_aggregation(history_states[:-1], x)
                    if history_output is not None:
                        x = self._apply_history_fusion(x, history_output)
        
        # Применяем внимание к истории на выходном уровне
        if use_history and len(history_states) > 0 and self.config.history_application_level == "output":
            history_output = self._apply_history_aggregation(history_states, x)
            if history_output is not None:
                x = self._apply_history_fusion(x, history_output)
                x = self.history_norm(x)
        
        # Работа с внешней памятью (экспериментальная функция)
        if use_history and self.config.use_memory_bank and len(self.memory_bank) > 0:
            memory_output, _ = self.memory_attention(x, self.memory_bank[-1], self.memory_bank[-1])
            x = x + self.history_dropout(memory_output)
        
        # Выходные вероятности
        logits = self.output_layer(x)
        return logits
    
    def generate(self, prompt, max_length, temperature=1.0):
        """Генерация текста по промпту с использованием истории"""
        self.eval()
        with torch.no_grad():
            # Начинаем с промпта
            generated = prompt.clone().unsqueeze(0)
            
            # Инициализируем память для хранения состояний
            if self.config.use_memory_bank:
                self.memory_bank = nn.ParameterList()
            
            for i in range(max_length):
                # Прямой проход для получения следующего токена
                logits = self.forward(generated, use_history=True)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Применяем softmax и выбираем следующий токен
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_probs, num_samples=1)
                
                # Добавляем к сгенерированной последовательности
                generated = torch.cat([generated, next_token], dim=1)
                
                # Сохраняем состояния во внешнюю память
                if self.config.use_memory_bank and i % 5 == 0:  # Сохраняем каждые 5 шагов
                    with torch.no_grad():
                        # Для simplicity, просто сохраняем последнее состояние
                        self.memory_bank.append(nn.Parameter(generated.detach()))
                
                # Если сгенерировали конец последовательности, останавливаемся
                if next_token.item() == 3:  # <eos>
                    break
            
            return generated.squeeze(0)

# ==================== УТИЛИТЫ ОБУЧЕНИЯ ====================
def count_parameters(model):
    """Подсчет количества параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_usage():
    """Получение информации об использовании памяти"""
    memory_info = {}
    
    # CPU память
    memory_info['cpu_memory'] = psutil.virtual_memory().percent
    
    # GPU память (если доступно)
    if torch.cuda.is_available():
        memory_info['gpu_memory'] = torch.cuda.memory_allocated() / 1024**3  # в GB
        memory_info['gpu_memory_max'] = torch.cuda.max_memory_allocated() / 1024**3  # в GB
    else:
        memory_info['gpu_memory'] = 0
        memory_info['gpu_memory_max'] = 0
    
    return memory_info

def train_epoch(model, dataloader, criterion, optimizer, config, epoch, model_name):
    """Обучение на одной эпохе"""
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(config.device), target.to(config.device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Вычисляем потерю
        loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()
        
        if batch_idx % config.print_interval == 0:
            perplexity = np.exp(loss.item())
            print(f'{model_name} - Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Perplexity: {perplexity:.2f}')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

def generate_sample(model, vocab, inv_vocab, config, prompt_text, is_russian=False):
    """Генерация примера текста"""
    if is_russian:
        # Токенизация для русского языка
        words = re.findall(r'\b[а-яё]+\b', prompt_text.lower())
        prompt_indices = [vocab.get(word, 1) for word in words]
    else:
        # Токенизация для английского языка
        prompt_tokens = prompt_text.lower().split()
        prompt_indices = [vocab.get(token, 1) for token in prompt_tokens]
    
    prompt_indices = [vocab['<sos>']] + prompt_indices
    prompt_tensor = torch.tensor(prompt_indices).to(config.device)
    
    # Генерируем продолжение
    generated = model.generate(prompt_tensor, config.gen_length, config.temperature)
    
    # Преобразуем обратно в текст
    generated_tokens = []
    for idx in generated.cpu().numpy():
        if idx == vocab['<eos>']:
            break
        if idx not in [vocab['<sos>'], vocab['<pad>'], vocab['<unk>']]:
            generated_tokens.append(inv_vocab.get(idx, '<unk>'))
    
    return ' '.join(generated_tokens)

# ==================== УТИЛИТЫ ДЛЯ ВИЗУАЛИЗАЦИИ ====================
def visualize_attention_weights(attention_weights, layer_names, save_path):
    """Визуализация весов внимания"""
    if attention_weights is None:
        return
    
    plt.figure(figsize=(12, 8))
    
    # attention_weights shape: [batch*seq, history_len, 1]
    attn = attention_weights.reshape(-1, attention_weights.shape[1])
    avg_attn = attn.mean(axis=0)
    
    plt.bar(range(len(avg_attn)), avg_attn)
    plt.xticks(range(len(avg_attn)), layer_names)
    plt.xlabel("Исторические слои")
    plt.ylabel("Средний вес внимания")
    plt.title("Распределение внимания по историческим слоям")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_training_results(baseline_history, history_model_history, save_path):
    """Визуализация результатов обучения"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if baseline_history:
        plt.plot(baseline_history['train_perplexity'], label='Baseline')
    if history_model_history:
        plt.plot(history_model_history['train_perplexity'], label='History Model')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.title('Training Perplexity')
    
    plt.subplot(1, 2, 2)
    if baseline_history:
        plt.plot(baseline_history['train_loss'], label='Baseline')
    if history_model_history:
        plt.plot(history_model_history['train_loss'], label='History Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==================== РЕЖИМЫ РАБОТЫ ====================
def competitive_mode(config):
    """Соревновательный режим: обучение обеих моделей и сравнение"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== РЕЖИМ: СОРЕВНОВАТЕЛЬНЫЙ (запуск: {timestamp}) ===")
    
    # Определяем тип датасета
    is_russian = config.dataset_type == "literature"
    
    # Загрузка данных
    if config.dataset_type == "imdb":
        data_path = download_imdb_data()
        texts = load_imdb_texts(data_path)
    else:
        texts = load_literature_texts(config.data_path)
    
    # Построение словаря
    print("Построение словаря...")
    vocab = build_vocab(texts, config.vocab_size, is_russian=is_russian)
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_path = f"vocab_{timestamp}.pkl"
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Словарь сохранен как: {vocab_path}")
    
    # Предобработка данных
    print("Предобработка данных...")
    processed_texts = preprocess_data(texts, vocab, config.max_seq_length, is_russian=is_russian)
    
    # Создание DataLoader
    train_loader = DataLoader(
        processed_texts, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Инициализация моделей
    baseline_model = BaselineGenerator(config, len(vocab)).to(config.device)
    history_model = EnhancedHistoryAwareGenerator(config, len(vocab)).to(config.device)
    
    # Подсчет параметров
    baseline_params = count_parameters(baseline_model)
    history_params = count_parameters(history_model)
    
    print(f"Параметры Baseline модели: {baseline_params:,}")
    print(f"Параметры History модели: {history_params:,}")
    print(f"Разница: {history_params - baseline_params:,} параметров")
    
    # Функция потерь и оптимизаторы
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Игнорируем padding
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=config.learning_rate)
    history_optimizer = torch.optim.Adam(history_model.parameters(), lr=config.learning_rate)
    
    # Обучение моделей
    print("Начало обучения...")
    
    baseline_history = {
        'train_loss': [], 'train_perplexity': [],
        'memory_usage': [], 'training_time': 0
    }
    
    history_model_history = {
        'train_loss': [], 'train_perplexity': [],
        'memory_usage': [], 'training_time': 0
    }
    
    # Обучение Baseline модели
    print("\n=== ОБУЧЕНИЕ BASELINE МОДЕЛИ ===")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        train_loss, train_ppl = train_epoch(
            baseline_model, train_loader, criterion, baseline_optimizer, config, epoch, "Baseline"
        )
        
        baseline_history['train_loss'].append(train_loss)
        baseline_history['train_perplexity'].append(train_ppl)
        baseline_history['memory_usage'].append(get_memory_usage())
        
        print(f"Baseline - Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
    
    baseline_history['training_time'] = time.time() - start_time
    print(f"Baseline обучение заняло: {baseline_history['training_time']:.2f} секунд")
    
    # Обучение History модели
    print("\n=== ОБУЧЕНИЕ HISTORY МОДЕЛИ ===")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        train_loss, train_ppl = train_epoch(
            history_model, train_loader, criterion, history_optimizer, config, epoch, "History"
        )
        
        history_model_history['train_loss'].append(train_loss)
        history_model_history['train_perplexity'].append(train_ppl)
        history_model_history['memory_usage'].append(get_memory_usage())
        
        print(f"History - Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
    
    history_model_history['training_time'] = time.time() - start_time
    print(f"History обучение заняло: {history_model_history['training_time']:.2f} секунд")
    
    # Генерация примеров
    print("\n=== ГЕНЕРАЦИЯ ПРИМЕРОВ ТЕКСТА ===")
    if is_russian:
        prompts = [
            "в белой ночи",
            "он размышлял о",
            "любовь это",
            "природа вокруг"
        ]
    else:
        prompts = [
            "How are you today?",
            "i really liked",
            "the acting was",
            "the story is",
            "this movie is"
        ]
    
    baseline_results = []
    history_results = []
    
    print("\n=== BASELINE MODEL ===")
    for prompt in prompts:
        generated = generate_sample(baseline_model, vocab, inv_vocab, config, prompt, is_russian)
        baseline_results.append((prompt, generated))
        print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
    
    print("\n=== HISTORY MODEL ===")
    for prompt in prompts:
        generated = generate_sample(history_model, vocab, inv_vocab, config, prompt, is_russian)
        history_results.append((prompt, generated))
        print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
    
    # Визуализация весов внимания
    if config.visualize_attention and hasattr(history_model, 'attention_weights'):
        layer_names = []
        for i in range(config.num_layers):
            if history_model._select_history_layers(i):
                layer_names.append(f"Layer {i}")
        
        attention_path = f"attention_weights_{timestamp}.png"
        visualize_attention_weights(
            history_model.attention_weights, 
            layer_names,
            attention_path
        )
        print(f"Визуализация внимания сохранена как: {attention_path}")
    
    # Визуализация результатов
    training_plot_path = f"training_history_{timestamp}.png"
    visualize_training_results(baseline_history, history_model_history, training_plot_path)
    print(f"График обучения сохранен как: {training_plot_path}")
    
    # Сохранение моделей
    if config.save_models:
        baseline_path = f'baseline_generator_{timestamp}.pth'
        history_path = f'history_generator_{timestamp}.pth'
        
        torch.save(baseline_model.state_dict(), baseline_path)
        torch.save(history_model.state_dict(), history_path)
        print(f"Модели сохранены как '{baseline_path}' и '{history_path}'")
    
    # Сравнение финальных результатов
    print("\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ===")
    if baseline_history['train_perplexity']:
        print(f"Baseline Model - Final Train Perplexity: {baseline_history['train_perplexity'][-1]:.2f}")
    if history_model_history['train_perplexity']:
        print(f"History Model - Final Train Perplexity: {history_model_history['train_perplexity'][-1]:.2f}")
    
    if baseline_history['train_perplexity'] and history_model_history['train_perplexity']:
        improvement = baseline_history['train_perplexity'][-1] - history_model_history['train_perplexity'][-1]
        print(f"Improvement: {improvement:.2f}")
    
    # Аналитика использования ресурсов
    print("\n=== АНАЛИТИКА ИСПОЛЬЗОВАНИЯ РЕСУРСОВ ===")
    if baseline_history['memory_usage']:
        avg_cpu_memory = np.mean([m['cpu_memory'] for m in baseline_history['memory_usage']])
        avg_gpu_memory = np.mean([m['gpu_memory'] for m in baseline_history['memory_usage']])
        max_gpu_memory = np.max([m['gpu_memory_max'] for m in baseline_history['memory_usage']])
        
        print(f"Baseline - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%")
        print(f"Baseline - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB")
        print(f"Baseline - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB")
        print(f"Baseline - Время обучения: {baseline_history['training_time']:.2f} секунд")
    
    if history_model_history['memory_usage']:
        avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
        avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
        max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
        
        print(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%")
        print(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB")
        print(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB")
        print(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд")
    
    # Сохранение результатов в файл
    results_path = f"results_{timestamp}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ===\n")
        f.write(f"Дата и время: {timestamp}\n")
        f.write(f"Режим: Соревновательный\n")
        f.write(f"Тип датасета: {config.dataset_type}\n")
        f.write(f"Путь к данным: {config.data_path}\n\n")
        
        f.write("=== МОДЕЛИ ===\n")
        f.write(f"Baseline параметров: {baseline_params:,}\n")
        f.write(f"History параметров: {history_params:,}\n")
        f.write(f"Разница: {history_params - baseline_params:,} параметров\n\n")
        
        f.write("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===\n")
        if baseline_history['train_perplexity']:
            f.write(f"Baseline - Финальная перплексия: {baseline_history['train_perplexity'][-1]:.2f}\n")
        if history_model_history['train_perplexity']:
            f.write(f"History - Финальная перплексия: {history_model_history['train_perplexity'][-1]:.2f}\n")
        
        if baseline_history['train_perplexity'] and history_model_history['train_perplexity']:
            improvement = baseline_history['train_perplexity'][-1] - history_model_history['train_perplexity'][-1]
            f.write(f"Улучшение: {improvement:.2f}\n")
        
        f.write("\n=== РЕСУРСЫ ===\n")
        if baseline_history['memory_usage']:
            avg_cpu_memory = np.mean([m['cpu_memory'] for m in baseline_history['memory_usage']])
            avg_gpu_memory = np.mean([m['gpu_memory'] for m in baseline_history['memory_usage']])
            max_gpu_memory = np.max([m['gpu_memory_max'] for m in baseline_history['memory_usage']])
            
            f.write(f"Baseline - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%\n")
            f.write(f"Baseline - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB\n")
            f.write(f"Baseline - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB\n")
            f.write(f"Baseline - Время обучения: {baseline_history['training_time']:.2f} секунд\n")
        
        if history_model_history['memory_usage']:
            avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
            avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
            max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
            
            f.write(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%\n")
            f.write(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB\n")
            f.write(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB\n")
            f.write(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд\n")
        
        f.write("\n=== ГЕНЕРАЦИЯ ===\n")
        f.write("Baseline Model:\n")
        for prompt, generated in baseline_results:
            f.write(f"Prompt: '{prompt}' -> Generated: '{generated}'\n")
        
        f.write("\nHistory Model:\n")
        for prompt, generated in history_results:
            f.write(f"Prompt: '{prompt}' -> Generated: '{generated}'\n")
    
    print(f"Полные результаты сохранены в: {results_path}")

def history_only_mode(config):
    """Режим только для History модели"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== РЕЖИМ: ТОЛЬКО HISTORY (запуск: {timestamp}) ===")
    
    # Определяем тип датасета
    is_russian = config.dataset_type == "literature"
    
    # Загрузка данных
    if config.dataset_type == "imdb":
        data_path = download_imdb_data()
        texts = load_imdb_texts(data_path)
    else:
        texts = load_literature_texts(config.data_path)
    
    # Построение словаря
    print("Построение словаря...")
    vocab = build_vocab(texts, config.vocab_size, is_russian=is_russian)
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_path = f"vocab_{timestamp}.pkl"
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Словарь сохранен как: {vocab_path}")
    
    # Предобработка данных
    print("Предобработка данных...")
    processed_texts = preprocess_data(texts[:1000], vocab, config.max_seq_length, is_russian=is_russian)
    
    # Создание DataLoader
    train_loader = DataLoader(
        processed_texts, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Инициализация History модели
    history_model = EnhancedHistoryAwareGenerator(config, len(vocab)).to(config.device)
    
    # Подсчет параметров
    history_params = count_parameters(history_model)
    print(f"Параметры History модели: {history_params:,}")
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Игнорируем padding
    history_optimizer = torch.optim.Adam(history_model.parameters(), lr=config.learning_rate)
    
    # Обучение History модели
    print("\n=== ОБУЧЕНИЕ HISTORY МОДЕЛИ ===")
    
    history_model_history = {
        'train_loss': [], 'train_perplexity': [],
        'memory_usage': [], 'training_time': 0
    }
    
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        train_loss, train_ppl = train_epoch(
            history_model, train_loader, criterion, history_optimizer, config, epoch, "History"
        )
        
        history_model_history['train_loss'].append(train_loss)
        history_model_history['train_perplexity'].append(train_ppl)
        history_model_history['memory_usage'].append(get_memory_usage())
        
        print(f"History - Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Perplexity: {train_ppl:.2f}")
    
    history_model_history['training_time'] = time.time() - start_time
    print(f"History обучение заняло: {history_model_history['training_time']:.2f} секунд")
    
    # Генерация примеров
    print("\n=== ГЕНЕРАЦИЯ ПРИМЕРОВ ТЕКСТА ===")
    if is_russian:
        prompts = [
            "в белой ночи",
            "он размышлял о",
            "любовь это",
            "природа вокруг"
        ]
    else:
        prompts = [
            "How are you?",
            "i really liked",
            "the acting was",
            "the story is"
        ]
    
    history_results = []
    
    print("\n=== HISTORY MODEL ===")
    for prompt in prompts:
        generated = generate_sample(history_model, vocab, inv_vocab, config, prompt, is_russian)
        history_results.append((prompt, generated))
        print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
    
    # Визуализация весов внимания
    if config.visualize_attention and hasattr(history_model, 'attention_weights'):
        layer_names = []
        for i in range(config.num_layers):
            if history_model._select_history_layers(i):
                layer_names.append(f"Layer {i}")
        
        attention_path = f"history_attention_weights_{timestamp}.png"
        visualize_attention_weights(
            history_model.attention_weights, 
            layer_names,
            attention_path
        )
        print(f"Визуализация внимания сохранена как: {attention_path}")
    
    # Визуализация результатов
    training_plot_path = f"history_training_{timestamp}.png"
    visualize_training_results(None, history_model_history, training_plot_path)
    print(f"График обучения сохранен как: {training_plot_path}")
    
    # Сохранение модели
    if config.save_models:
        history_path = f'history_generator_{timestamp}.pth'
        torch.save(history_model.state_dict(), history_path)
        print(f"Модель сохранена как '{history_path}'")
    
    # Аналитика использования ресурсов
    print("\n=== АНАЛИТИКА ИСПОЛЬЗОВАНИЯ РЕСУРСОВ ===")
    if history_model_history['memory_usage']:
        avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
        avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
        max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
        
        print(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%")
        print(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB")
        print(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB")
        print(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд")
    
    # Сохранение результатов в файл
    results_path = f"history_results_{timestamp}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ===\n")
        f.write(f"Дата и время: {timestamp}\n")
        f.write(f"Режим: Только History\n")
        f.write(f"Тип датасета: {config.dataset_type}\n")
        f.write(f"Путь к данным: {config.data_path}\n\n")
        
        f.write("=== МОДЕЛЬ ===\n")
        f.write(f"History параметров: {history_params:,}\n\n")
        
        f.write("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===\n")
        if history_model_history['train_perplexity']:
            f.write(f"History - Финальная перплексия: {history_model_history['train_perplexity'][-1]:.2f}\n")
        
        f.write("\n=== РЕСУРСЫ ===\n")
        if history_model_history['memory_usage']:
            avg_cpu_memory = np.mean([m['cpu_memory'] for m in history_model_history['memory_usage']])
            avg_gpu_memory = np.mean([m['gpu_memory'] for m in history_model_history['memory_usage']])
            max_gpu_memory = np.max([m['gpu_memory_max'] for m in history_model_history['memory_usage']])
            
            f.write(f"History - Среднее использование CPU памяти: {avg_cpu_memory:.2f}%\n")
            f.write(f"History - Среднее использование GPU памяти: {avg_gpu_memory:.2f}GB\n")
            f.write(f"History - Максимальное использование GPU памяти: {max_gpu_memory:.2f}GB\n")
            f.write(f"History - Время обучения: {history_model_history['training_time']:.2f} секунд\n")
        
        f.write("\n=== ГЕНЕРАЦИЯ ===\n")
        f.write("History Model:\n")
        for prompt, generated in history_results:
            f.write(f"Prompt: '{prompt}' -> Generated: '{generated}'\n")
    
    print(f"Полные результаты сохранены в: {results_path}")

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    parser = argparse.ArgumentParser(description='HistoryGPT - Генеративная модель с историческим вниманием')
    parser.add_argument('--mode', type=str, default='competitive', 
                       choices=['competitive', 'history'],
                       help='Режим работы: competitive (обе модели) или history (только history модель)')
    parser.add_argument('--dataset', type=str, default='imdb', 
                       choices=['imdb', 'literature'],
                       help='Тип датасета: imdb или literature')
    parser.add_argument('--data_path', type=str, default='literature',
                       help='Путь к папке с данными (для literature)')
    
    args = parser.parse_args()
    
    config = Config()
    config.dataset_type = args.dataset
    config.data_path = args.data_path
    
    print(f"Используется устройство: {config.device}")
    print(f"Тип датасета: {config.dataset_type}")
    if config.dataset_type == "literature":
        print(f"Путь к данным: {config.data_path}")
    
    if args.mode == 'competitive':
        competitive_mode(config)
    elif args.mode == 'history':
        history_only_mode(config)

if __name__ == "__main__":
    main()

============================================================
.\arxglue\gluetorch\hgui.py
============================================================
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

============================================================
.\arxglue\gluetorch\GluePaddle\gluepaddle.py
============================================================
"""
GluePaddle - Модульная система для PaddlePaddle
с поддержкой портов, валидацией и компиляцией графов
"""

import paddle
from typing import Any, Callable, Optional, Dict, List, Tuple, Union, Set, Type
import hashlib
import inspect
from collections import defaultdict
from dataclasses import dataclass
import math
from enum import Enum
import warnings
import json
import time
from contextlib import contextmanager

class PaddleAggregationStrategy(Enum):
    """Стратегии агрегации входных данных для портов"""
    CONCAT = "concat"  # Конкатенация по последнему измерению
    SUM = "sum"        # Поэлементное суммирование
    MEAN = "mean"      # Поэлементное усреднение
    MAX = "max"        # Поэлементный максимум
    STACK = "stack"    # Создание нового измерения
    CUSTOM = "custom"  # Пользовательская агрегация

class PaddlePortType(Enum):
    """Типы данных для портов"""
    TENSOR = "tensor"
    SCALAR = "scalar"
    SEQUENCE = "sequence"
    DICT = "dict"
    ANY = "any"

@dataclass
class PaddlePortSpec:
    """Спецификация порта с расширенной валидацией"""
    name: str
    type: PaddlePortType = PaddlePortType.TENSOR
    shape: Optional[Tuple[Optional[int]]] = None  # None для любых размерностей
    dtype: Optional[paddle.dtype] = None
    required: bool = True
    aggregation: PaddleAggregationStrategy = PaddleAggregationStrategy.CONCAT
    custom_aggregator: Optional[Callable[[List[Any]], Any]] = None
    
    def is_compatible_with(self, other: 'PaddlePortSpec') -> bool:
        """Проверяет совместимость с другим портом"""
        # Проверка типов
        if self.type != other.type and self.type != PaddlePortType.ANY and other.type != PaddlePortType.ANY:
            return False
        
        # Проверка dtype
        if self.dtype is not None and other.dtype is not None and self.dtype != other.dtype:
            return False
        
        # Проверка формы (если обе спецификации имеют форму)
        if self.shape is not None and other.shape is not None:
            if len(self.shape) != len(other.shape):
                return False
            
            for dim_self, dim_other in zip(self.shape, other.shape):
                if dim_self is not None and dim_other is not None and dim_self != dim_other:
                    return False
        
        return True
    
    def validate_value(self, value: Any) -> bool:
        """Валидирует значение по спецификации порта"""
        try:
            # Проверка типа
            if self.type == PaddlePortType.TENSOR and not isinstance(value, paddle.Tensor):
                return False
            elif self.type == PaddlePortType.SCALAR and not isinstance(value, (int, float, paddle.Tensor)):
                return False
            elif self.type == PaddlePortType.SEQUENCE and not isinstance(value, (list, tuple)):
                return False
            elif self.type == PaddlePortType.DICT and not isinstance(value, dict):
                return False
            
            # Проверка dtype для тензоров
            if isinstance(value, paddle.Tensor) and self.dtype is not None and value.dtype != self.dtype:
                return False
            
            # Проверка формы для тензоров
            if isinstance(value, paddle.Tensor) and self.shape is not None:
                if len(value.shape) != len(self.shape):
                    return False
                
                for dim_value, dim_spec in zip(value.shape, self.shape):
                    if dim_spec is not None and dim_value != dim_spec:
                        return False
            
            return True
        except:
            return False

class PaddleComponent(paddle.nn.Layer):
    """Базовый компонент системы с поддержкой портов"""
    
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name or self.__class__.__name__
        self.input_ports: Dict[str, PaddlePortSpec] = self._define_input_ports()
        self.output_ports: Dict[str, PaddlePortSpec] = self._define_output_ports()
        self._compiled = False
        
    def _define_input_ports(self) -> Dict[str, PaddlePortSpec]:
        """Определяет входные порты компонента (переопределяется в подклассах)"""
        return {'default': PaddlePortSpec('default')}
    
    def _define_output_ports(self) -> Dict[str, PaddlePortSpec]:
        """Определяет выходные порты компонента (переопределяется в подклассах)"""
        return {'default': PaddlePortSpec('default')}
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Прямой проход с поддержкой множественных входов/выходов"""
        raise NotImplementedError("Components must implement forward method")
    
    def validate_ports(self) -> List[str]:
        """Проверка корректности определения портов"""
        errors = []
    
        # Проверяем, что все обязательные порты имеют спецификации
        for port_name, port_spec in self.input_ports.items():
            if port_spec.required and port_spec.type == PaddlePortType.ANY:
                warnings.warn(f"Input port {port_name} is required but has type ANY")
    
        # Проверяем сигнатуру forward метода только если не используется **inputs
        try:
            sig = inspect.signature(self.forward)
            forward_params = list(sig.parameters.keys())
        
            # Если метод использует **kwargs или **inputs, пропускаем проверку параметров
            has_var_keyword = any(
                param.kind == param.VAR_KEYWORD 
                for param in sig.parameters.values()
            )
        
            if not has_var_keyword:
                # Проверяем, что все входные порты есть в параметрах forward
                for port_name in self.input_ports:
                    if port_name not in forward_params and port_name != 'default':
                        errors.append(f"Input port {port_name} not found in forward method parameters")
        except:
            pass  # Пропускаем проверку, если невозможно получить сигнатуру
    
        return errors
    
    def compile(self):
        """Компиляция компонента (если нужна)"""
        self._compiled = True
        
    def reset(self):
        """Сброс состояния компонента (переопределяется в подклассах)"""
        pass
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, inputs={list(self.input_ports.keys())}, outputs={list(self.output_ports.keys())})"

@dataclass
class PaddleConnection:
    """Соединение между портами компонентов"""
    source: str  # format: "component_name.port_name"
    target: str  # format: "component_name.port_name"
    transformer: Optional[Callable] = None
    delay: int = 0  # Для рекуррентных соединений
    
    def __post_init__(self):
        if '.' not in self.source:
            self.source = f"{self.source}.default"
        if '.' not in self.target:
            self.target = f"{self.target}.default"
    
    @property
    def source_component(self) -> str:
        return self.source.split('.')[0]
    
    @property
    def source_port(self) -> str:
        return self.source.split('.')[1]
    
    @property
    def target_component(self) -> str:
        return self.target.split('.')[0]
    
    @property
    def target_port(self) -> str:
        return self.target.split('.')[1]
    
    def is_recurrent(self) -> bool:
        """Проверяет, является ли соединение рекуррентным"""
        return self.delay > 0
    
    def __repr__(self):
        delay_str = f", delay={self.delay}" if self.delay > 0 else ""
        return f"PaddleConnection({self.source} -> {self.target}{delay_str})"

class PaddleSharedMemory:
    """Общая память для быстрого обмена данными между компонентами"""
    
    def __init__(self):
        self._storage = {}
        self._access_times = {}
        self._max_size = 1000  # Максимальное количество элементов в памяти
        
    def set(self, key: str, value: Any):
        """Устанавливает значение в общей памяти"""
        self._storage[key] = value
        self._access_times[key] = time.time()
        self._cleanup()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Получает значение из общей памяти"""
        value = self._storage.get(key, default)
        if key in self._storage:
            self._access_times[key] = time.time()
        return value
    
    def delete(self, key: str):
        """Удаляет значение из общей памяти"""
        if key in self._storage:
            del self._storage[key]
            del self._access_times[key]
    
    def clear(self):
        """Очищает всю общую память"""
        self._storage.clear()
        self._access_times.clear()
    
    def _cleanup(self):
        """Очищает старые элементы при превышении лимита"""
        if len(self._storage) > self._max_size:
            # Удаляем самые старые элементы
            sorted_keys = sorted(self._access_times.keys(), key=lambda k: self._access_times[k])
            for key in sorted_keys[:len(self._storage) - self._max_size]:
                self.delete(key)

class GluePaddle(paddle.nn.Layer):
    """Модульная система для построения нейросетевых архитектур на PaddlePaddle"""
    
    def __init__(self):
        super().__init__()
        self.components: Dict[str, PaddleComponent] = {}
        self.connections: List[PaddleConnection] = []
        self.execution_order: List[str] = []
        self._compiled = False
        self._connection_map: Dict[str, List[PaddleConnection]] = defaultdict(list)
        self._recurrent_connections: List[PaddleConnection] = []
        self._state_buffers: Dict[str, Any] = {}
        self._input_map: Dict[str, List[PaddleConnection]] = defaultdict(list)
        self.shared_memory = PaddleSharedMemory()
        self._performance_stats: Dict[str, List[float]] = defaultdict(list)
        
    def register_component(self, component: PaddleComponent):
        """Регистрирует компонент в системе с валидацией"""
        if component.name in self.components:
            raise ValueError(f"Component {component.name} already registered")
        
        # Валидируем порты компонента
        errors = component.validate_ports()
        if errors:
            raise ValueError(f"Component {component.name} has port errors: {errors}")
        
        self.components[component.name] = component
        self.add_sublayer(component.name, component)
        
    def add_connection(self, source: Union[str, Tuple[str, str]], 
                      target: Union[str, Tuple[str, str]],
                      transformer: Optional[Callable] = None,
                      delay: int = 0):
        """Добавляет соединение между портами компонентов с валидации"""
        if isinstance(source, tuple):
            source_str = f"{source[0]}.{source[1]}"
        else:
            source_str = source
            
        if isinstance(target, tuple):
            target_str = f"{target[0]}.{target[1]}"
        else:
            target_str = target
            
        connection = PaddleConnection(source_str, target_str, transformer, delay)
        
        # Проверяем существование компонентов и портов
        self._validate_connection(connection)
        
        # Проверяем совместимость портов
        self._validate_port_compatibility(connection)
        
        self.connections.append(connection)
        
        # Добавляем в карту соединений для быстрого доступа
        if connection.is_recurrent():
            self._recurrent_connections.append(connection)
        else:
            self._connection_map[connection.target].append(connection)
            self._input_map[connection.target].append(connection)
    
    def _validate_connection(self, connection: PaddleConnection):
        """Валидирует соединение на существование компонентов и портов"""
        # Проверяем существование компонентов
        if (connection.source_component != 'input' and 
            connection.source_component not in self.components):
            raise ValueError(f"Source component {connection.source_component} not found")
            
        if (connection.target_component != 'output' and 
            connection.target_component not in self.components):
            raise ValueError(f"Target component {connection.target_component} not found")
            
        # Проверяем существование портов
        if (connection.source_component != 'input' and 
            connection.source_port not in self.components[connection.source_component].output_ports):
            raise ValueError(f"Source port {connection.source_port} not found in component {connection.source_component}")
            
        if (connection.target_component != 'output' and 
            connection.target_port not in self.components[connection.target_component].input_ports):
            raise ValueError(f"Target port {connection.target_port} not found in component {connection.target_component}")
    
    def _validate_port_compatibility(self, connection: PaddleConnection):
        """Проверяет совместимость портов для соединения с учетом агрегации"""
        if connection.source_component == 'input' or connection.target_component == 'output':
            return  # Пропускаем проверку для входов/выходов системы

        source_comp = self.components[connection.source_component]
        target_comp = self.components[connection.target_component]

        source_port = source_comp.output_ports[connection.source_port]
        target_port = target_comp.input_ports[connection.target_port]

        # Для агрегируемых портов откладываем проверку до компиляции
        if target_port.aggregation != PaddleAggregationStrategy.CUSTOM:
            # Проверяем только базовую совместимость типов
            if source_port.type != target_port.type and source_port.type != PaddlePortType.ANY and target_port.type != PaddlePortType.ANY:
                raise ValueError(
                    f"Port type mismatch: {connection.source} ({source_port.type}) -> "
                    f"{connection.target} ({target_port.type})"
                )
            return

        # Для неагрегируемых портов выполняем полную проверку
        if not source_port.is_compatible_with(target_port):
            raise ValueError(
                f"Port compatibility error: {connection.source} ({source_port.type}, "
                f"{source_port.shape}, {source_port.dtype}) -> {connection.target} "
                f"({target_port.type}, {target_port.shape}, {target_port.dtype})"
            )

    def _validate_aggregation_compatibility(self, target_component: str, target_port: str):
        """Проверяет совместимость всех входов для агрегируемого порта"""
        component = self.components[target_component]
        port_spec = component.input_ports[target_port]
    
        # Получаем все соединения к целевому порту
        target_connections = [
            conn for conn in self.connections 
            if conn.target_component == target_component and conn.target_port == target_port
        ]
    
        if len(target_connections) <= 1:
            return  # Для одиночных соединений нет необходимости проверять агрегацию
    
        # Собираем спецификации всех исходных портов
        source_specs = []
        for conn in target_connections:
            if conn.source_component == 'input':
                # Для входных портов создаем временную спецификацию
                source_specs.append(PaddlePortSpec('input', PaddlePortType.ANY))
            else:
                source_comp = self.components[conn.source_component]
                source_specs.append(source_comp.output_ports[conn.source_port])
    
        # Проверяем совместимость в зависимости от стратегии агрегации
        if port_spec.aggregation == PaddleAggregationStrategy.CONCAT:
            # Для конкатенации проверяем, что все размерности кроме последней совпадают
            first_shape = source_specs[0].shape
            if first_shape is None:
                return  # Не можем проверить
            
            for spec in source_specs[1:]:
                if spec.shape is None:
                    continue
                
                if len(spec.shape) != len(first_shape):
                    raise ValueError(f"Shape rank mismatch for aggregation: "
                                   f"{len(spec.shape)} vs {len(first_shape)}")
            
                for i in range(len(first_shape) - 1):
                    if (spec.shape[i] != first_shape[i] and 
                        spec.shape[i] is not None and 
                        first_shape[i] is not None):
                        raise ValueError(f"Shape mismatch for aggregation at dimension {i}: "
                                       f"{spec.shape[i]} vs {first_shape[i]}")
    
        elif port_spec.aggregation in [PaddleAggregationStrategy.SUM, PaddleAggregationStrategy.MEAN, 
                                      PaddleAggregationStrategy.MAX, PaddleAggregationStrategy.STACK]:
            # Для этих стратегий все формы должны совпадать
            first_shape = source_specs[0].shape
            for spec in source_specs[1:]:
                if not spec.is_compatible_with(source_specs[0]):
                    raise ValueError(f"Shape mismatch for {port_spec.aggregation.value} aggregation: "
                                   f"{spec.shape} vs {first_shape}")

    def _precompile_connections(self):
        """Предварительная компиляция карты соединений для оптимизации производительности"""
        self._input_map.clear()
        for conn in self.connections:
            self._input_map[conn.target].append(conn)
    
    def compile(self):
        """Компилирует граф выполнения с расширенной валидацией"""
        # Валидируем все компоненты
        for comp_name, component in self.components.items():
            errors = component.validate_ports()
            if errors:
                raise ValueError(f"Component {comp_name} has validation errors: {errors}")
    
        # Предварительная компиляция соединений
        self._precompile_connections()
    
        # Строим граф зависимостей на уровне компонентов
        dependency_graph = defaultdict(list)
        in_degree = defaultdict(int)
    
        # Добавляем все компоненты в граф
        for comp_name in self.components:
            dependency_graph[comp_name] = []
            in_degree[comp_name] = 0
    
        # Строим зависимости на основе соединений
        for conn in self.connections:
            if conn.is_recurrent():
                continue  # Пропускаем рекуррентные соединения для топологической сортировки
            
            if (conn.source_component != 'input' and 
                conn.target_component != 'output' and
                conn.source_component != conn.target_component):
            
                dependency_graph[conn.source_component].append(conn.target_component)
                in_degree[conn.target_component] += 1
    
        # Топологическая сортировка (Kahn's algorithm)
        queue = [comp for comp in in_degree if in_degree[comp] == 0]
        execution_order = []
    
        while queue:
            node = queue.pop(0)
            execution_order.append(node)
        
            for neighbor in dependency_graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                
        if len(execution_order) != len(self.components):
            # Проверяем наличие циклов (кроме рекуррентных)
            cyclic_components = set(self.components.keys()) - set(execution_order)
            if cyclic_components:
                raise ValueError(f"Graph contains cycles involving components: {cyclic_components}")
        
        self.execution_order = execution_order
    
        # Проверяем совместимость агрегации для всех портов
        for comp_name, component in self.components.items():
            for port_name, port_spec in component.input_ports.items():
                self._validate_aggregation_compatibility(comp_name, port_name)
    
        # Компилируем все компоненты
        for component in self.components.values():
            component.compile()
    
        self._compiled = True
    
        return self
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Прямой проход через всю систему с поддержкой рекуррентных соединений"""
        if not self._compiled:
            self.compile()
            
        # Словарь для хранения промежуточных результатов по портам
        results = {'input': inputs}
        
        # Инициализируем буферы состояний для рекуррентных соединений
        self._init_state_buffers()
        
        # Выполняем компоненты в порядке топологической сортировки
        for comp_name in self.execution_order:
            component = self.components[comp_name]
            
            # Собираем входные данные для каждого порта компонента
            component_inputs = self._gather_inputs(comp_name, results)
            
            # Выполняем компонент
            try:
                start_time = time.time()
                component_outputs = component(component_inputs)
                end_time = time.time()
                
                # Сохраняем метрики производительности
                self._performance_stats[comp_name].append(end_time - start_time)
                
                results[comp_name] = component_outputs
            except Exception as e:
                raise RuntimeError(f"Error executing component {comp_name}: {str(e)}")
            
        # Обрабатываем рекуррентные соединения (обновляем состояния)
        self._update_recurrent_states(results)
            
        # Собираем выходные данные
        output_data = self._gather_outputs(results)
        
        return output_data
    
    def _gather_inputs(self, comp_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Собирает входные данные для компонента"""
        component = self.components[comp_name]
        component_inputs = {}
    
        for port_name, port_spec in component.input_ports.items():
            port_inputs = []
            target_key = f"{comp_name}.{port_name}"
        
            # Ищем все соединения, ведущие к этому порту
            for conn in self._input_map.get(target_key, []):
                # Получаем данные из источника
                source_data = self._get_source_data(conn, results)
            
                # Пропускаем None значения (например, для рекуррентных соединений в начале)
                if source_data is None:
                    continue
                
                # Применяем трансформер если есть
                if conn.transformer:
                    try:
                        source_data = self._apply_transformer(conn, source_data)
                    except Exception as e:
                        self._log_transformer_error(conn, e)
                        raise
            
                # Для агрегируемых портов откладываем валидацию до после агрегации
                if port_spec.aggregation == PaddleAggregationStrategy.CUSTOM or len(self._input_map.get(target_key, [])) <= 1:
                    # Валидируем данные по спецификации порта
                    if not port_spec.validate_value(source_data):
                        raise ValueError(
                            f"Data validation failed for {conn.source} -> {conn.target}: "
                            f"value {type(source_data)} doesn't match port spec"
                        )
            
                port_inputs.append(source_data)
        
            # Обрабатываем собранные входы для порта
            if not port_inputs:
                if port_spec.required:
                    raise ValueError(f"No connections to required port {comp_name}.{port_name}")
                continue
        
            # Применяем стратегию агрегации
            if len(port_inputs) == 1:
                component_inputs[port_name] = port_inputs[0]
            else:
                component_inputs[port_name] = self._aggregate_inputs(port_inputs, port_spec)
                # После агрегации валидируем результат для агрегируемых портов
                if port_spec.aggregation != PaddleAggregationStrategy.CUSTOM:
                    if not port_spec.validate_value(component_inputs[port_name]):
                        raise ValueError(
                            f"Data validation failed after aggregation for port {comp_name}.{port_name}: "
                            f"value {type(component_inputs[port_name])} doesn't match port spec"
                        )
    
        return component_inputs   

    def _apply_transformer(self, connection: PaddleConnection, data: Any) -> Any:
        """Применяет трансформер к данным с обработкой ошибок"""
        try:
            return connection.transformer(data)
        except Exception as e:
            error_msg = f"Transformer error in connection {connection}: {str(e)}"
            if hasattr(self, '_log_transformer_error'):
                self._log_transformer_error(connection, e)
            raise RuntimeError(error_msg)
    
    def _log_transformer_error(self, connection: PaddleConnection, error: Exception):
        """Логирует ошибку трансформера"""
        warnings.warn(f"Transformer error in connection {connection}: {str(error)}")
    
    def _aggregate_inputs(self, inputs: List[Any], port_spec: PaddlePortSpec) -> Any:
        """Агрегирует входные данные согласно стратегии агрегации"""
        if port_spec.aggregation == PaddleAggregationStrategy.CONCAT:
            return paddle.concat(inputs, axis=-1)
        elif port_spec.aggregation == PaddleAggregationStrategy.SUM:
            return sum(inputs)
        elif port_spec.aggregation == PaddleAggregationStrategy.MEAN:
            return sum(inputs) / len(inputs)
        elif port_spec.aggregation == PaddleAggregationStrategy.MAX:
            return paddle.stack(inputs).max(axis=0)
        elif port_spec.aggregation == PaddleAggregationStrategy.STACK:
            return paddle.stack(inputs)
        elif port_spec.aggregation == PaddleAggregationStrategy.CUSTOM and port_spec.custom_aggregator:
            return port_spec.custom_aggregator(inputs)
        else:
            # По умолчанию конкатенируем
            return paddle.concat(inputs, axis=-1)
    
    def _get_source_data(self, connection: PaddleConnection, results: Dict[str, Any]) -> Any:
        """Получает данные из источника соединения"""
        if connection.source_component == 'input':
            source_data = results['input'].get(connection.source_port, None)
            if source_data is None:
                raise ValueError(f"Input port {connection.source_port} not provided")
        else:
            # Для рекуррентных соединений, если компонент еще не выполнен, используем буфер состояний
            if connection.is_recurrent() and connection.source_component not in results:
                buffer_key = f"{connection.source}->{connection.target}"
                if buffer_key in self._state_buffers and self._state_buffers[buffer_key]:
                    # Берем последнее состояние из буферов
                    return self._state_buffers[buffer_key][-1]
                else:
                    # Если буфер пуст, возвращаем None (должно обрабатываться вызывающим кодом)
                    return None
        
            source_data = results.get(connection.source_component, {}).get(connection.source_port, None)
            if source_data is None:
                raise ValueError(f"Source port {connection.source_component}.{connection.source_port} not found")
    
        return source_data
    
    def _init_state_buffers(self):
        """Инициализирует буферы состояний для рекуррентных соединений"""
        for conn in self._recurrent_connections:
            buffer_key = f"{conn.source}->{conn.target}"
            if buffer_key not in self._state_buffers:
                self._state_buffers[buffer_key] = []
    
    def _update_recurrent_states(self, results: Dict[str, Any]):
        """Обновляет состояния для рекуррентных соединений"""
        for conn in self._recurrent_connections:
            buffer_key = f"{conn.source}->{conn.target}"
            source_data = self._get_source_data(conn, results)
            
            # Добавляем текущее состояние в буфер
            self._state_buffers[buffer_key].append(source_data)
            
            # Ограничиваем размер буфера для предотвращения утечек памяти
            if len(self._state_buffers[buffer_key]) > conn.delay * 2:
                self._state_buffers[buffer_key] = self._state_buffers[buffer_key][-conn.delay:]
    
    def _gather_outputs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Собирает выходные данные системы"""
        output_data = {}
        
        for conn in self.connections:
            if conn.target_component == 'output':
                source_data = self._get_source_data(conn, results)
                
                if conn.transformer:
                    try:
                        source_data = self._apply_transformer(conn, source_data)
                    except Exception as e:
                        self._log_transformer_error(conn, e)
                        raise
                
                output_data[conn.target_port] = source_data
        
        return output_data
    
    def reset(self):
        """Сброс состояния системы и всех компонентов"""
        for component in self.components.values():
            if hasattr(component, 'reset'):
                component.reset()
        self._state_buffers.clear()
        self.shared_memory.clear()
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Возвращает статистику производительности компонентов"""
        stats = {}
        for comp_name, times in self._performance_stats.items():
            if times:
                stats[comp_name] = {
                    'calls': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times)
                }
        return stats
    
    def get_connection_graph(self) -> Dict:
        """Возвращает граф соединений для визуализации"""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Добавляем специальные узлы input/output
        graph['nodes'].append({'id': 'input', 'type': 'input'})
        graph['nodes'].append({'id': 'output', 'type': 'output'})
        
        # Добавляем компоненты как узлы
        for comp_name, component in self.components.items():
            graph['nodes'].append({
                'id': comp_name,
                'type': 'component',
                'inputs': list(component.input_ports.keys()),
                'outputs': list(component.output_ports.keys())
            })
        
        # Добавляем соединения как ребра
        for conn in self.connections:
            graph['edges'].append({
                'source': conn.source,
                'target': conn.target,
                'recurrent': conn.is_recurrent(),
                'delay': conn.delay
            })
        
        return graph
    
    def visualize(self, output_file: str = None):
        """Визуализирует граф соединений"""
        try:
            import graphviz
            dot = graphviz.Digraph()
            
            # Добавляем узлы
            dot.node('input', 'Input', shape='ellipse', color='green')
            dot.node('output', 'Output', shape='ellipse', color='red')
            
            for comp_name, component in self.components.items():
                label = f"{comp_name}|{{{'|'.join(component.input_ports.keys())}}}|{{{'|'.join(component.output_ports.keys())}}}"
                dot.node(comp_name, f"{{{label}}}", shape='record')
            
            # Добавляем ребра
            for conn in self.connections:
                if conn.is_recurrent():
                    dot.edge(conn.source, conn.target, label=f"delay={conn.delay}", color='blue', style='dashed')
                else:
                    dot.edge(conn.source, conn.target)
            
            if output_file:
                dot.render(output_file, format='png', cleanup=True)
            
            return dot
        except ImportError:
            warnings.warn("Install graphviz for visualization: pip install graphviz")
            # Fallback to text visualization
            graph = self.get_connection_graph()
            print("Graph visualization:")
            print(f"Nodes: {[node['id'] for node in graph['nodes']]}")
            for edge in graph['edges']:
                rec_str = " (recurrent)" if edge['recurrent'] else ""
                print(f"  {edge['source']} -> {edge['target']}{rec_str}")
            return None

    def save(self, filepath: str):
        """Сохраняет конфигурацию системы в файл"""
        config = {
            'components': {},
            'connections': []
        }
        
        # Сохраняем информацию о компонентах
        for comp_name, component in self.components.items():
            config['components'][comp_name] = {
                'type': component.__class__.__name__,
                'input_ports': {name: {'type': port.type.value, 
                                      'shape': port.shape,
                                      'dtype': str(port.dtype) if port.dtype else None,
                                      'required': port.required,
                                      'aggregation': port.aggregation.value}
                               for name, port in component.input_ports.items()},
                'output_ports': {name: {'type': port.type.value, 
                                       'shape': port.shape,
                                       'dtype': str(port.dtype) if port.dtype else None}
                                for name, port in component.output_ports.items()}
            }
        
        # Сохраняем соединения
        for conn in self.connections:
            config['connections'].append({
                'source': conn.source,
                'target': conn.target,
                'delay': conn.delay
            })
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, component_registry: Dict[str, Type[PaddleComponent]]):
        """Загружает конфигурацию системы из файла"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        system = cls()
        
        # Создаем и регистрируем компоненты
        for comp_name, comp_config in config['components'].items():
            comp_type = comp_config['type']
            if comp_type not in component_registry:
                raise ValueError(f"Component type {comp_type} not found in registry")
            
            component = component_registry[comp_type]()
            component.name = comp_name
            
            # Восстанавливаем порты (упрощенная версия)
            system.register_component(component)
        
        # Восстанавливаем соединения
        for conn_config in config['connections']:
            system.add_connection(
                conn_config['source'],
                conn_config['target'],
                delay=conn_config.get('delay', 0)
            )
        
        return system

    def to_distributed(self, device_map: Dict[str, str]):
        """Распределяет компоненты по устройствам"""
        for comp_name, device in device_map.items():
            if comp_name in self.components:
                # PaddlePaddle автоматически размещает операции на устройстве
                # через paddle.set_device() перед созданием модели
                pass
    
    def to_inference_model(self, filepath: str, sample_input: Dict[str, Any]):
        """Экспортирует систему для инференса"""
        # Сохраняем модель для инференса
        paddle.jit.save(self, filepath, input_spec=[sample_input])

# Базовые компоненты для PaddlePaddle
class PaddleLinearComponent(PaddleComponent):
    """Линейный компонент с расширенной спецификацией портов"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 activation: str = 'relu', name: str = None):
        super().__init__(name)
        self.linear = paddle.nn.Linear(input_dim, output_dim)
        self.activation = self._get_activation(activation)
        
        # Переопределяем спецификацию портов
        self.input_ports = {
            'default': PaddlePortSpec(
                'default', 
                type=PaddlePortType.TENSOR,
                shape=(None, input_dim),
                dtype=paddle.float32
            )
        }
        self.output_ports = {
            'default': PaddlePortSpec(
                'default',
                type=PaddlePortType.TENSOR,
                shape=(None, output_dim),
                dtype=paddle.float32
            )
        }
        
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return paddle.nn.ReLU()
        elif activation == 'sigmoid':
            return paddle.nn.Sigmoid()
        elif activation == 'tanh':
            return paddle.nn.Tanh()
        else:
            return paddle.nn.Identity()
            
    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        return {'default': self.activation(self.linear(inputs['default']))}

class PaddleMultiHeadAttentionComponent(PaddleComponent):
    """Компонент multi-head attention с расширенной спецификацией портов"""
    
    def __init__(self, d_model: int, num_heads: int, name: str = None):
        super().__init__(name)
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Определяем порты с точными спецификациями
        self.input_ports = {
            'query': PaddlePortSpec(
                'query', 
                type=PaddlePortType.TENSOR,
                shape=(None, None, d_model),
                dtype=paddle.float32,
                required=True
            ),
            'key': PaddlePortSpec(
                'key',
                type=PaddlePortType.TENSOR, 
                shape=(None, None, d_model),
                dtype=paddle.float32,
                required=True
            ),
            'value': PaddlePortSpec(
                'value',
                type=PaddlePortType.TENSOR,
                shape=(None, None, d_model), 
                dtype=paddle.float32,
                required=True
            ),
            'mask': PaddlePortSpec(
                'mask',
                type=PaddlePortType.TENSOR,
                shape=(None, None, None),
                dtype=paddle.bool,
                required=False
            )
        }
        self.output_ports = {
            'output': PaddlePortSpec(
                'output',
                type=PaddlePortType.TENSOR,
                shape=(None, None, d_model),
                dtype=paddle.float32
            )
        }
        
        # Инициализируем слои
        self.q_linear = paddle.nn.Linear(d_model, d_model)
        self.k_linear = paddle.nn.Linear(d_model, d_model)
        self.v_linear = paddle.nn.Linear(d_model, d_model)
        self.out_linear = paddle.nn.Linear(d_model, d_model)
        
    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        query = inputs['query']
        key = inputs['key']
        value = inputs['value']
        mask = inputs.get('mask', None)
        
        batch_size = query.shape[0]
        d_k = self.d_model // self.num_heads
        
        # Linear transformations and split into heads
        q = self.q_linear(query).reshape([batch_size, -1, self.num_heads, d_k]).transpose([0, 2, 1, 3])
        k = self.k_linear(key).reshape([batch_size, -1, self.num_heads, d_k]).transpose([0, 2, 1, 3])
        v = self.v_linear(value).reshape([batch_size, -1, self.num_heads, d_k]).transpose([0, 2, 1, 3])
        
        # Calculate attention scores
        scores = paddle.matmul(q, k.transpose([0, 1, 3, 2])) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + (mask.astype(scores.dtype) - 1) * 1e9
        
        # Apply softmax to get attention weights
        attn_weights = paddle.nn.functional.softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_output = paddle.matmul(attn_weights, v)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([batch_size, -1, self.d_model])
        return {'output': self.out_linear(attn_output)}

# Пример использования
def paddle_example_usage():
    """Пример использования библиотеки GluePaddle"""
    
    # Создаем систему
    system = GluePaddle()
    
    # Создаем и регистрируем компоненты
    attention = PaddleMultiHeadAttentionComponent(d_model=512, num_heads=8, name="attention")
    linear = PaddleLinearComponent(input_dim=512, output_dim=256, name="linear")
    
    system.register_component(attention)
    system.register_component(linear)
    
    # Добавляем соединения с указанием портов
    system.add_connection(('input', 'query'), ('attention', 'query'))
    system.add_connection(('input', 'key'), ('attention', 'key')) 
    system.add_connection(('input', 'value'), ('attention', 'value'))
    system.add_connection(('attention', 'output'), ('linear', 'default'))
    system.add_connection(('linear', 'default'), ('output', 'result'))
    
    # Визуализируем граф
    system.visualize()
    
    # Компилируем и выполняем
    system.compile()
    
    # Входные данные для каждого порта
    inputs = {
        'query': paddle.randn([1, 10, 512]),
        'key': paddle.randn([1, 10, 512]),
        'value': paddle.randn([1, 10, 512])
    }
    
    # Прямой проход
    outputs = system(inputs)
    print(f"Output shape: {outputs['result'].shape}")

if __name__ == "__main__":
    paddle_example_usage()

============================================================
.\arxglue\gluetorch\GlueTensorFlow\gluetensorflow.py
============================================================
"""
GlueTensorFlow - Модульная система для TensorFlow/Keras
с поддержкой портов, валидацией и компиляцией графов
"""

import tensorflow as tf
from typing import Any, Callable, Optional, Dict, List, Tuple, Union, Set, Type
import hashlib
import inspect
from collections import defaultdict
from dataclasses import dataclass
import math
from enum import Enum
import warnings
import json
import time
from contextlib import contextmanager

class TFAggregationStrategy(Enum):
    """Стратегии агрегации входных данных для портов"""
    CONCAT = "concat"  # Конкатенация по последнему измерению
    SUM = "sum"        # Поэлементное суммирование
    MEAN = "mean"      # Поэлементное усреднение
    MAX = "max"        # Поэлементный максимум
    STACK = "stack"    # Создание нового измерения
    CUSTOM = "custom"  # Пользовательская агрегация

class TFPortType(Enum):
    """Типы данных для портов"""
    TENSOR = "tensor"
    SCALAR = "scalar"
    SEQUENCE = "sequence"
    DICT = "dict"
    ANY = "any"

@dataclass
class TFPortSpec:
    """Спецификация порта с расширенной валидацией"""
    name: str
    type: TFPortType = TFPortType.TENSOR
    shape: Optional[Tuple[Optional[int]]] = None  # None для любых размерностей
    dtype: Optional[tf.dtypes.DType] = None
    required: bool = True
    aggregation: TFAggregationStrategy = TFAggregationStrategy.CONCAT
    custom_aggregator: Optional[Callable[[List[Any]], Any]] = None
    
    def is_compatible_with(self, other: 'TFPortSpec') -> bool:
        """Проверяет совместимость с другим портом"""
        # Проверка типов
        if self.type != other.type and self.type != TFPortType.ANY and other.type != TFPortType.ANY:
            return False
        
        # Проверка dtype
        if self.dtype is not None and other.dtype is not None and self.dtype != other.dtype:
            return False
        
        # Проверка формы (если обе спецификации имеют форму)
        if self.shape is not None and other.shape is not None:
            if len(self.shape) != len(other.shape):
                return False
            
            for dim_self, dim_other in zip(self.shape, other.shape):
                if dim_self is not None and dim_other is not None and dim_self != dim_other:
                    return False
        
        return True
    
    def validate_value(self, value: Any) -> bool:
        """Валидирует значение по спецификации порта"""
        try:
            # Проверка типа
            if self.type == TFPortType.TENSOR and not isinstance(value, tf.Tensor):
                return False
            elif self.type == TFPortType.SCALAR and not isinstance(value, (int, float, tf.Tensor)):
                return False
            elif self.type == TFPortType.SEQUENCE and not isinstance(value, (list, tuple)):
                return False
            elif self.type == TFPortType.DICT and not isinstance(value, dict):
                return False
            
            # Проверка dtype для тензоров
            if isinstance(value, tf.Tensor) and self.dtype is not None and value.dtype != self.dtype:
                return False
            
            # Проверка формы для тензоров
            if isinstance(value, tf.Tensor) and self.shape is not None:
                if len(value.shape) != len(self.shape):
                    return False
                
                for dim_value, dim_spec in zip(value.shape, self.shape):
                    if dim_spec is not None and dim_value != dim_spec:
                        return False
            
            return True
        except:
            return False

class TFComponent(tf.keras.Model):
    """Базовый компонент системы с поддержкой портов"""
    
    def __init__(self, name: str = None):
        super().__init__(name=name)
        self.name = name or self.__class__.__name__
        self.input_ports: Dict[str, TFPortSpec] = self._define_input_ports()
        self.output_ports: Dict[str, TFPortSpec] = self._define_output_ports()
        self._compiled = False
        
    def _define_input_ports(self) -> Dict[str, TFPortSpec]:
        """Определяет входные порты компонента (переопределяется в подклассах)"""
        return {'default': TFPortSpec('default')}
    
    def _define_output_ports(self) -> Dict[str, TFPortSpec]:
        """Определяет выходные порты компонента (переопределяется в подклассах)"""
        return {'default': TFPortSpec('default')}
    
    def call(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Прямой проход с поддержкой множественных входов/выходов"""
        raise NotImplementedError("Components must implement call method")
    
    def validate_ports(self) -> List[str]:
        """Проверка корректности определения портов"""
        errors = []
    
        # Проверяем, что все обязательные порты имеют спецификации
        for port_name, port_spec in self.input_ports.items():
            if port_spec.required and port_spec.type == TFPortType.ANY:
                warnings.warn(f"Input port {port_name} is required but has type ANY")
    
        # Проверяем сигнатуру call метода только если не используется **inputs
        try:
            sig = inspect.signature(self.call)
            call_params = list(sig.parameters.keys())
        
            # Если метод использует **kwargs или **inputs, пропускаем проверку параметров
            has_var_keyword = any(
                param.kind == param.VAR_KEYWORD 
                for param in sig.parameters.values()
            )
        
            if not has_var_keyword:
                # Проверяем, что все входные порты есть в параметрах call
                for port_name in self.input_ports:
                    if port_name not in call_params and port_name != 'default':
                        errors.append(f"Input port {port_name} not found in call method parameters")
        except:
            pass  # Пропускаем проверку, если невозможно получить сигнатуру
    
        return errors
    
    def compile(self):
        """Компиляция компонента (если нужна)"""
        self._compiled = True
        
    def reset(self):
        """Сброс состояния компонента (переопределяется в подклассах)"""
        pass
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, inputs={list(self.input_ports.keys())}, outputs={list(self.output_ports.keys())})"

@dataclass
class TFConnection:
    """Соединение между портами компонентов"""
    source: str  # format: "component_name.port_name"
    target: str  # format: "component_name.port_name"
    transformer: Optional[Callable] = None
    delay: int = 0  # Для рекуррентных соединений
    
    def __post_init__(self):
        if '.' not in self.source:
            self.source = f"{self.source}.default"
        if '.' not in self.target:
            self.target = f"{self.target}.default"
    
    @property
    def source_component(self) -> str:
        return self.source.split('.')[0]
    
    @property
    def source_port(self) -> str:
        return self.source.split('.')[1]
    
    @property
    def target_component(self) -> str:
        return self.target.split('.')[0]
    
    @property
    def target_port(self) -> str:
        return self.target.split('.')[1]
    
    def is_recurrent(self) -> bool:
        """Проверяет, является ли соединение рекуррентным"""
        return self.delay > 0
    
    def __repr__(self):
        delay_str = f", delay={self.delay}" if self.delay > 0 else ""
        return f"TFConnection({self.source} -> {self.target}{delay_str})"

class TFSharedMemory:
    """Общая память для быстрого обмена данными между компонентами"""
    
    def __init__(self):
        self._storage = {}
        self._access_times = {}
        self._max_size = 1000  # Максимальное количество элементов в памяти
        
    def set(self, key: str, value: Any):
        """Устанавливает значение в общей памяти"""
        self._storage[key] = value
        self._access_times[key] = time.time()
        self._cleanup()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Получает значение из общей памяти"""
        value = self._storage.get(key, default)
        if key in self._storage:
            self._access_times[key] = time.time()
        return value
    
    def delete(self, key: str):
        """Удаляет значение из общей памяти"""
        if key in self._storage:
            del self._storage[key]
            del self._access_times[key]
    
    def clear(self):
        """Очищает всю общую память"""
        self._storage.clear()
        self._access_times.clear()
    
    def _cleanup(self):
        """Очищает старые элементы при превышении лимита"""
        if len(self._storage) > self._max_size:
            # Удаляем самые старые элементы
            sorted_keys = sorted(self._access_times.keys(), key=lambda k: self._access_times[k])
            for key in sorted_keys[:len(self._storage) - self._max_size]:
                self.delete(key)

class GlueTensorFlow(tf.keras.Model):
    """Модульная система для построения нейросетевых архитектур на TensorFlow"""
    
    def __init__(self):
        super().__init__()
        self.components: Dict[str, TFComponent] = {}
        self.connections: List[TFConnection] = []
        self.execution_order: List[str] = []
        self._compiled = False
        self._connection_map: Dict[str, List[TFConnection]] = defaultdict(list)
        self._recurrent_connections: List[TFConnection] = []
        self._state_buffers: Dict[str, Any] = {}
        self._input_map: Dict[str, List[TFConnection]] = defaultdict(list)
        self.shared_memory = TFSharedMemory()
        self._performance_stats: Dict[str, List[float]] = defaultdict(list)
        
    def register_component(self, component: TFComponent):
        """Регистрирует компонент в системе с валидацией"""
        if component.name in self.components:
            raise ValueError(f"Component {component.name} already registered")
        
        # Валидируем порты компонента
        errors = component.validate_ports()
        if errors:
            raise ValueError(f"Component {component.name} has port errors: {errors}")
        
        self.components[component.name] = component
        self._track_trackable(component, name=component.name)
        
    def add_connection(self, source: Union[str, Tuple[str, str]], 
                      target: Union[str, Tuple[str, str]],
                      transformer: Optional[Callable] = None,
                      delay: int = 0):
        """Добавляет соединение между портами компонентов с валидацией"""
        if isinstance(source, tuple):
            source_str = f"{source[0]}.{source[1]}"
        else:
            source_str = source
            
        if isinstance(target, tuple):
            target_str = f"{target[0]}.{target[1]}"
        else:
            target_str = target
            
        connection = TFConnection(source_str, target_str, transformer, delay)
        
        # Проверяем существование компонентов и портов
        self._validate_connection(connection)
        
        # Проверяем совместимость портов
        self._validate_port_compatibility(connection)
        
        self.connections.append(connection)
        
        # Добавляем в карту соединений для быстрого доступа
        if connection.is_recurrent():
            self._recurrent_connections.append(connection)
        else:
            self._connection_map[connection.target].append(connection)
            self._input_map[connection.target].append(connection)
    
    def _validate_connection(self, connection: TFConnection):
        """Валидирует соединение на существование компонентов и портов"""
        # Проверяем существование компонентов
        if (connection.source_component != 'input' and 
            connection.source_component not in self.components):
            raise ValueError(f"Source component {connection.source_component} not found")
            
        if (connection.target_component != 'output' and 
            connection.target_component not in self.components):
            raise ValueError(f"Target component {connection.target_component} not found")
            
        # Проверяем существование портов
        if (connection.source_component != 'input' and 
            connection.source_port not in self.components[connection.source_component].output_ports):
            raise ValueError(f"Source port {connection.source_port} not found in component {connection.source_component}")
            
        if (connection.target_component != 'output' and 
            connection.target_port not in self.components[connection.target_component].input_ports):
            raise ValueError(f"Target port {connection.target_port} not found in component {connection.target_component}")
    
    def _validate_port_compatibility(self, connection: TFConnection):
        """Проверяет совместимость портов для соединения с учетом агрегации"""
        if connection.source_component == 'input' or connection.target_component == 'output':
            return  # Пропускаем проверку для входов/выходов системы

        source_comp = self.components[connection.source_component]
        target_comp = self.components[connection.target_component]

        source_port = source_comp.output_ports[connection.source_port]
        target_port = target_comp.input_ports[connection.target_port]

        # Для агрегируемых портов откладываем проверку до компиляции
        if target_port.aggregation != TFAggregationStrategy.CUSTOM:
            # Проверяем только базовую совместимость типов
            if source_port.type != target_port.type and source_port.type != TFPortType.ANY and target_port.type != TFPortType.ANY:
                raise ValueError(
                    f"Port type mismatch: {connection.source} ({source_port.type}) -> "
                    f"{connection.target} ({target_port.type})"
                )
            return

        # Для неагрегируемых портов выполняем полную проверку
        if not source_port.is_compatible_with(target_port):
            raise ValueError(
                f"Port compatibility error: {connection.source} ({source_port.type}, "
                f"{source_port.shape}, {source_port.dtype}) -> {connection.target} "
                f"({target_port.type}, {target_port.shape}, {target_port.dtype})"
            )

    def _validate_aggregation_compatibility(self, target_component: str, target_port: str):
        """Проверяет совместимость всех входов для агрегируемого порта"""
        component = self.components[target_component]
        port_spec = component.input_ports[target_port]
    
        # Получаем все соединения к целевому порту
        target_connections = [
            conn for conn in self.connections 
            if conn.target_component == target_component and conn.target_port == target_port
        ]
    
        if len(target_connections) <= 1:
            return  # Для одиночных соединений нет необходимости проверять агрегацию
    
        # Собираем спецификации всех исходных портов
        source_specs = []
        for conn in target_connections:
            if conn.source_component == 'input':
                # Для входных портов создаем временную спецификацию
                source_specs.append(TFPortSpec('input', TFPortType.ANY))
            else:
                source_comp = self.components[conn.source_component]
                source_specs.append(source_comp.output_ports[conn.source_port])
    
        # Проверяем совместимость в зависимости от стратегии агрегации
        if port_spec.aggregation == TFAggregationStrategy.CONCAT:
            # Для конкатенации проверяем, что все размерности кроме последней совпадают
            first_shape = source_specs[0].shape
            if first_shape is None:
                return  # Не можем проверить
            
            for spec in source_specs[1:]:
                if spec.shape is None:
                    continue
                
                if len(spec.shape) != len(first_shape):
                    raise ValueError(f"Shape rank mismatch for aggregation: "
                                   f"{len(spec.shape)} vs {len(first_shape)}")
            
                for i in range(len(first_shape) - 1):
                    if (spec.shape[i] != first_shape[i] and 
                        spec.shape[i] is not None and 
                        first_shape[i] is not None):
                        raise ValueError(f"Shape mismatch for aggregation at dimension {i}: "
                                       f"{spec.shape[i]} vs {first_shape[i]}")
    
        elif port_spec.aggregation in [TFAggregationStrategy.SUM, TFAggregationStrategy.MEAN, 
                                      TFAggregationStrategy.MAX, TFAggregationStrategy.STACK]:
            # Для этих стратегий все формы должны совпадать
            first_shape = source_specs[0].shape
            for spec in source_specs[1:]:
                if not spec.is_compatible_with(source_specs[0]):
                    raise ValueError(f"Shape mismatch for {port_spec.aggregation.value} aggregation: "
                                   f"{spec.shape} vs {first_shape}")

    def _precompile_connections(self):
        """Предварительная компиляция карты соединений для оптимизации производительности"""
        self._input_map.clear()
        for conn in self.connections:
            self._input_map[conn.target].append(conn)
    
    def compile(self):
        """Компилирует граф выполнения с расширенной валидацией"""
        # Валидируем все компоненты
        for comp_name, component in self.components.items():
            errors = component.validate_ports()
            if errors:
                raise ValueError(f"Component {comp_name} has validation errors: {errors}")
    
        # Предварительная компиляция соединений
        self._precompile_connections()
    
        # Строим граф зависимостей на уровне компонентов
        dependency_graph = defaultdict(list)
        in_degree = defaultdict(int)
    
        # Добавляем все компоненты в граф
        for comp_name in self.components:
            dependency_graph[comp_name] = []
            in_degree[comp_name] = 0
    
        # Строим зависимости на основе соединений
        for conn in self.connections:
            if conn.is_recurrent():
                continue  # Пропускаем рекуррентные соединения для топологической сортировки
            
            if (conn.source_component != 'input' and 
                conn.target_component != 'output' and
                conn.source_component != conn.target_component):
            
                dependency_graph[conn.source_component].append(conn.target_component)
                in_degree[conn.target_component] += 1
    
        # Топологическая сортировка (Kahn's algorithm)
        queue = [comp for comp in in_degree if in_degree[comp] == 0]
        execution_order = []
    
        while queue:
            node = queue.pop(0)
            execution_order.append(node)
        
            for neighbor in dependency_graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                
        if len(execution_order) != len(self.components):
            # Проверяем наличие циклов (кроме рекуррентных)
            cyclic_components = set(self.components.keys()) - set(execution_order)
            if cyclic_components:
                raise ValueError(f"Graph contains cycles involving components: {cyclic_components}")
        
        self.execution_order = execution_order
    
        # Проверяем совместимость агрегации для всех портов
        for comp_name, component in self.components.items():
            for port_name, port_spec in component.input_ports.items():
                self._validate_aggregation_compatibility(comp_name, port_name)
    
        # Компилируем все компоненты
        for component in self.components.values():
            component.compile()
    
        self._compiled = True
    
        return self
    
    def call(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Прямой проход через всю систему с поддержкой рекуррентных соединений"""
        if not self._compiled:
            self.compile()
            
        # Словарь для хранения промежуточных результатов по портам
        results = {'input': inputs}
        
        # Инициализируем буферы состояний для рекуррентных соединений
        self._init_state_buffers()
        
        # Выполняем компоненты в порядке топологической сортировки
        for comp_name in self.execution_order:
            component = self.components[comp_name]
            
            # Собираем входные данные для каждого порта компонента
            component_inputs = self._gather_inputs(comp_name, results)
            
            # Выполняем компонент
            try:
                start_time = time.time()
                component_outputs = component(component_inputs)
                end_time = time.time()
                
                # Сохраняем метрики производительности
                self._performance_stats[comp_name].append(end_time - start_time)
                
                results[comp_name] = component_outputs
            except Exception as e:
                raise RuntimeError(f"Error executing component {comp_name}: {str(e)}")
            
        # Обрабатываем рекуррентные соединения (обновляем состояния)
        self._update_recurrent_states(results)
            
        # Собираем выходные данные
        output_data = self._gather_outputs(results)
        
        return output_data
    
    def _gather_inputs(self, comp_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Собирает входные данные для компонента"""
        component = self.components[comp_name]
        component_inputs = {}
    
        for port_name, port_spec in component.input_ports.items():
            port_inputs = []
            target_key = f"{comp_name}.{port_name}"
        
            # Ищем все соединения, ведущие к этому порту
            for conn in self._input_map.get(target_key, []):
                # Получаем данные из источника
                source_data = self._get_source_data(conn, results)
            
                # Пропускаем None значения (например, для рекуррентных соединений в начале)
                if source_data is None:
                    continue
                
                # Применяем трансформер если есть
                if conn.transformer:
                    try:
                        source_data = self._apply_transformer(conn, source_data)
                    except Exception as e:
                        self._log_transformer_error(conn, e)
                        raise
            
                # Для агрегируемых портов откладываем валидацию до после агрегации
                if port_spec.aggregation == TFAggregationStrategy.CUSTOM or len(self._input_map.get(target_key, [])) <= 1:
                    # Валидируем данные по спецификации порта
                    if not port_spec.validate_value(source_data):
                        raise ValueError(
                            f"Data validation failed for {conn.source} -> {conn.target}: "
                            f"value {type(source_data)} doesn't match port spec"
                        )
            
                port_inputs.append(source_data)
        
            # Обрабатываем собранные входы для порта
            if not port_inputs:
                if port_spec.required:
                    raise ValueError(f"No connections to required port {comp_name}.{port_name}")
                continue
        
            # Применяем стратегию агрегации
            if len(port_inputs) == 1:
                component_inputs[port_name] = port_inputs[0]
            else:
                component_inputs[port_name] = self._aggregate_inputs(port_inputs, port_spec)
                # После агрегации валидируем результат для агрегируемых портов
                if port_spec.aggregation != TFAggregationStrategy.CUSTOM:
                    if not port_spec.validate_value(component_inputs[port_name]):
                        raise ValueError(
                            f"Data validation failed after aggregation for port {comp_name}.{port_name}: "
                            f"value {type(component_inputs[port_name])} doesn't match port spec"
                        )
    
        return component_inputs   

    def _apply_transformer(self, connection: TFConnection, data: Any) -> Any:
        """Применяет трансформер к данным с обработкой ошибок"""
        try:
            return connection.transformer(data)
        except Exception as e:
            error_msg = f"Transformer error in connection {connection}: {str(e)}"
            if hasattr(self, '_log_transformer_error'):
                self._log_transformer_error(connection, e)
            raise RuntimeError(error_msg)
    
    def _log_transformer_error(self, connection: TFConnection, error: Exception):
        """Логирует ошибку трансформера"""
        warnings.warn(f"Transformer error in connection {connection}: {str(error)}")
    
    def _aggregate_inputs(self, inputs: List[Any], port_spec: TFPortSpec) -> Any:
        """Агрегирует входные данные согласно стратегии агрегации"""
        if port_spec.aggregation == TFAggregationStrategy.CONCAT:
            return tf.concat(inputs, axis=-1)
        elif port_spec.aggregation == TFAggregationStrategy.SUM:
            return tf.add_n(inputs)
        elif port_spec.aggregation == TFAggregationStrategy.MEAN:
            return tf.reduce_mean(tf.stack(inputs), axis=0)
        elif port_spec.aggregation == TFAggregationStrategy.MAX:
            return tf.reduce_max(tf.stack(inputs), axis=0)
        elif port_spec.aggregation == TFAggregationStrategy.STACK:
            return tf.stack(inputs)
        elif port_spec.aggregation == TFAggregationStrategy.CUSTOM and port_spec.custom_aggregator:
            return port_spec.custom_aggregator(inputs)
        else:
            # По умолчанию конкатенируем
            return tf.concat(inputs, axis=-1)
    
    def _get_source_data(self, connection: TFConnection, results: Dict[str, Any]) -> Any:
        """Получает данные из источника соединения"""
        if connection.source_component == 'input':
            source_data = results['input'].get(connection.source_port, None)
            if source_data is None:
                raise ValueError(f"Input port {connection.source_port} not provided")
        else:
            # Для рекуррентных соединений, если компонент еще не выполнен, используем буфер состояний
            if connection.is_recurrent() and connection.source_component not in results:
                buffer_key = f"{connection.source}->{connection.target}"
                if buffer_key in self._state_buffers and self._state_buffers[buffer_key]:
                    # Берем последнее состояние из буфера
                    return self._state_buffers[buffer_key][-1]
                else:
                    # Если буфер пуст, возвращаем None (должно обрабатываться вызывающим кодом)
                    return None
        
            source_data = results.get(connection.source_component, {}).get(connection.source_port, None)
            if source_data is None:
                raise ValueError(f"Source port {connection.source_component}.{connection.source_port} not found")
    
        return source_data
    
    def _init_state_buffers(self):
        """Инициализирует буферы состояний для рекуррентных соединений"""
        for conn in self._recurrent_connections:
            buffer_key = f"{conn.source}->{conn.target}"
            if buffer_key not in self._state_buffers:
                self._state_buffers[buffer_key] = []
    
    def _update_recurrent_states(self, results: Dict[str, Any]):
        """Обновляет состояния для рекуррентных соединений"""
        for conn in self._recurrent_connections:
            buffer_key = f"{conn.source}->{conn.target}"
            source_data = self._get_source_data(conn, results)
            
            # Добавляем текущее состояние в буфер
            self._state_buffers[buffer_key].append(source_data)
            
            # Ограничиваем размер буфера для предотвращения утечек памяти
            if len(self._state_buffers[buffer_key]) > conn.delay * 2:
                self._state_buffers[buffer_key] = self._state_buffers[buffer_key][-conn.delay:]
    
    def _gather_outputs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Собирает выходные данные системы"""
        output_data = {}
        
        for conn in self.connections:
            if conn.target_component == 'output':
                source_data = self._get_source_data(conn, results)
                
                if conn.transformer:
                    try:
                        source_data = self._apply_transformer(conn, source_data)
                    except Exception as e:
                        self._log_transformer_error(conn, e)
                        raise
                
                output_data[conn.target_port] = source_data
        
        return output_data
    
    def reset(self):
        """Сброс состояния системы и всех компонентов"""
        for component in self.components.values():
            if hasattr(component, 'reset'):
                component.reset()
        self._state_buffers.clear()
        self.shared_memory.clear()
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Возвращает статистику производительности компонентов"""
        stats = {}
        for comp_name, times in self._performance_stats.items():
            if times:
                stats[comp_name] = {
                    'calls': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times)
                }
        return stats
    
    def get_connection_graph(self) -> Dict:
        """Возвращает граф соединений для визуализации"""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Добавляем специальные узлы input/output
        graph['nodes'].append({'id': 'input', 'type': 'input'})
        graph['nodes'].append({'id': 'output', 'type': 'output'})
        
        # Добавляем компоненты как узлы
        for comp_name, component in self.components.items():
            graph['nodes'].append({
                'id': comp_name,
                'type': 'component',
                'inputs': list(component.input_ports.keys()),
                'outputs': list(component.output_ports.keys())
            })
        
        # Добавляем соединения как ребра
        for conn in self.connections:
            graph['edges'].append({
                'source': conn.source,
                'target': conn.target,
                'recurrent': conn.is_recurrent(),
                'delay': conn.delay
            })
        
        return graph
    
    def visualize(self, output_file: str = None):
        """Визуализирует граф соединений"""
        try:
            import graphviz
            dot = graphviz.Digraph()
            
            # Добавляем узлы
            dot.node('input', 'Input', shape='ellipse', color='green')
            dot.node('output', 'Output', shape='ellipse', color='red')
            
            for comp_name, component in self.components.items():
                label = f"{comp_name}|{{{'|'.join(component.input_ports.keys())}}}|{{{'|'.join(component.output_ports.keys())}}}"
                dot.node(comp_name, f"{{{label}}}", shape='record')
            
            # Добавляем ребра
            for conn in self.connections:
                if conn.is_recurrent():
                    dot.edge(conn.source, conn.target, label=f"delay={conn.delay}", color='blue', style='dashed')
                else:
                    dot.edge(conn.source, conn.target)
            
            if output_file:
                dot.render(output_file, format='png', cleanup=True)
            
            return dot
        except ImportError:
            warnings.warn("Install graphviz for visualization: pip install graphviz")
            # Fallback to text visualization
            graph = self.get_connection_graph()
            print("Graph visualization:")
            print(f"Nodes: {[node['id'] for node in graph['nodes']]}")
            for edge in graph['edges']:
                rec_str = " (recurrent)" if edge['recurrent'] else ""
                print(f"  {edge['source']} -> {edge['target']}{rec_str}")
            return None

    def save(self, filepath: str):
        """Сохраняет конфигурацию системы в файл"""
        config = {
            'components': {},
            'connections': []
        }
        
        # Сохраняем информацию о компонентами
        for comp_name, component in self.components.items():
            config['components'][comp_name] = {
                'type': component.__class__.__name__,
                'input_ports': {name: {'type': port.type.value, 
                                      'shape': port.shape,
                                      'dtype': str(port.dtype) if port.dtype else None,
                                      'required': port.required,
                                      'aggregation': port.aggregation.value}
                               for name, port in component.input_ports.items()},
                'output_ports': {name: {'type': port.type.value, 
                                       'shape': port.shape,
                                       'dtype': str(port.dtype) if port.dtype else None}
                                for name, port in component.output_ports.items()}
            }
        
        # Сохраняем соединения
        for conn in self.connections:
            config['connections'].append({
                'source': conn.source,
                'target': conn.target,
                'delay': conn.delay
            })
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, component_registry: Dict[str, Type[TFComponent]]):
        """Загружает конфигурацию системы из файла"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        system = cls()
        
        # Создаем и регистрируем компоненты
        for comp_name, comp_config in config['components'].items():
            comp_type = comp_config['type']
            if comp_type not in component_registry:
                raise ValueError(f"Component type {comp_type} not found in registry")
            
            component = component_registry[comp_type]()
            component.name = comp_name
            
            # Восстанавливаем порты (упрощенная версия)
            system.register_component(component)
        
        # Восстанавливаем соединения
        for conn_config in config['connections']:
            system.add_connection(
                conn_config['source'],
                conn_config['target'],
                delay=conn_config.get('delay', 0)
            )
        
        return system

    def to_distributed(self, device_map: Dict[str, str]):
        """Распределяет компоненты по устройствам"""
        for comp_name, device in device_map.items():
            if comp_name in self.components:
                with tf.device(device):
                    # TensorFlow автоматически размещает операции на устройстве
                    pass
    
    def to_saved_model(self, filepath: str, sample_input: Dict[str, Any]):
        """Экспортирует систему в SavedModel формат"""
        # Создаем сигнатуру для экспорта
        input_signature = {}
        for port_name, value in sample_input.items():
            input_signature[port_name] = tf.TensorSpec.from_tensor(value)
        
        # Экспортируем модель
        tf.saved_model.save(self, filepath, signatures={
            'serving_default': self.call.get_concrete_function(input_signature)
        })
    
    def _get_output_ports(self) -> List[str]:
        """Получает список выходных портов системы"""
        output_ports = set()
        for conn in self.connections:
            if conn.target_component == 'output':
                output_ports.add(conn.target_port)
        return list(output_ports)
    
    @contextmanager
    def profile(self):
        """Контекстный менеджер для профилирования выполнения"""
        # TensorFlow имеет встроенные инструменты профилирования
        try:
            from tensorflow.python.profiler import profiler_v2 as profiler
            profiler.start()
            yield
            profiler.stop()
            profiler.save('logs', profiler.ProfileOption('time', 'ops'))
        except ImportError:
            warnings.warn("TensorFlow profiler not available")
            yield

# Базовые компоненты для TensorFlow
class TFLinearComponent(TFComponent):
    """Линейный компонент с расширенной спецификацией портов"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 activation: str = 'relu', name: str = None):
        super().__init__(name)
        self.linear = tf.keras.layers.Dense(output_dim)
        self.activation = self._get_activation(activation)
        
        # Переопределяем спецификацию портов
        self.input_ports = {
            'default': TFPortSpec(
                'default', 
                type=TFPortType.TENSOR,
                shape=(None, input_dim),
                dtype=tf.float32
            )
        }
        self.output_ports = {
            'default': TFPortSpec(
                'default',
                type=TFPortType.TENSOR,
                shape=(None, output_dim),
                dtype=tf.float32
            )
        }
        
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return tf.keras.layers.ReLU()
        elif activation == 'sigmoid':
            return tf.keras.layers.Activation('sigmoid')
        elif activation == 'tanh':
            return tf.keras.layers.Activation('tanh')
        else:
            return tf.keras.layers.Lambda(lambda x: x)
            
    def call(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        return {'default': self.activation(self.linear(inputs['default']))}

class TFMultiHeadAttentionComponent(TFComponent):
    """Компонент multi-head attention с расширенной спецификацией портов"""
    
    def __init__(self, d_model: int, num_heads: int, name: str = None):
        super().__init__(name)
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Определяем порты с точными спецификациями
        self.input_ports = {
            'query': TFPortSpec(
                'query', 
                type=TFPortType.TENSOR,
                shape=(None, None, d_model),
                dtype=tf.float32,
                required=True
            ),
            'key': TFPortSpec(
                'key',
                type=TFPortType.TENSOR, 
                shape=(None, None, d_model),
                dtype=tf.float32,
                required=True
            ),
            'value': TFPortSpec(
                'value',
                type=TFPortType.TENSOR,
                shape=(None, None, d_model), 
                dtype=tf.float32,
                required=True
            ),
            'mask': TFPortSpec(
                'mask',
                type=TFPortType.TENSOR,
                shape=(None, None, None),
                dtype=tf.bool,
                required=False
            )
        }
        self.output_ports = {
            'output': TFPortSpec(
                'output',
                type=TFPortType.TENSOR,
                shape=(None, None, d_model),
                dtype=tf.float32
            )
        }
        
        # Инициализируем слои
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
        
    def call(self, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        query = inputs['query']
        key = inputs['key']
        value = inputs['value']
        mask = inputs.get('mask', None)
        
        output = self.mha(query=query, value=value, key=key, attention_mask=mask)
        return {'output': output}

# Пример использования
def tf_example_usage():
    """Пример использования библиотеки GlueTensorFlow"""
    
    # Создаем систему
    system = GlueTensorFlow()
    
    # Создаем и регистрируем компоненты
    attention = TFMultiHeadAttentionComponent(d_model=512, num_heads=8, name="attention")
    linear = TFLinearComponent(input_dim=512, output_dim=256, name="linear")
    
    system.register_component(attention)
    system.register_component(linear)
    
    # Добавляем соединения с указанием портов
    system.add_connection(('input', 'query'), ('attention', 'query'))
    system.add_connection(('input', 'key'), ('attention', 'key')) 
    system.add_connection(('input', 'value'), ('attention', 'value'))
    system.add_connection(('attention', 'output'), ('linear', 'default'))
    system.add_connection(('linear', 'default'), ('output', 'result'))
    
    # Визуализируем граф
    system.visualize()
    
    # Компилируем и выполняем
    system.compile()
    
    # Входные данные для каждого порта
    inputs = {
        'query': tf.random.normal((1, 10, 512)),
        'key': tf.random.normal((1, 10, 512)),
        'value': tf.random.normal((1, 10, 512))
    }
    
    # Прямой проход
    outputs = system(inputs)
    print(f"Output shape: {outputs['result'].shape}")

if __name__ == "__main__":
    tf_example_usage()

============================================================
.\arxglue\gluetorch\GlueTorch\demo.py
============================================================
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

============================================================
.\arxglue\gluetorch\GlueTorch\gluetorch.py
============================================================
"""
GlueTorch - Модульная система для построения нейросетевых архитектур
с поддержкой портов, валидацией, компиляцией графов и общей памятью
"""

import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Dict, List, Tuple, Union, Set, Type
import hashlib
import inspect
from collections import defaultdict
from dataclasses import dataclass
import math
from enum import Enum
import warnings
import json
import time
from contextlib import contextmanager

class AggregationStrategy(Enum):
    """Стратегии агрегации входных данных для портов"""
    CONCAT = "concat"  # Конкатенация по последнему измерению
    SUM = "sum"        # Поэлементное суммирование
    MEAN = "mean"      # Поэлементное усреднение
    MAX = "max"        # Поэлементный максимум
    STACK = "stack"    # Создание нового измерения
    CUSTOM = "custom"  # Пользовательская агрегация

class PortType(Enum):
    """Типы данных для портов"""
    TENSOR = "tensor"
    SCALAR = "scalar"
    SEQUENCE = "sequence"
    DICT = "dict"
    ANY = "any"

@dataclass
class PortSpec:
    """Спецификация порта с расширенной валидацией"""
    name: str
    type: PortType = PortType.TENSOR
    shape: Optional[Tuple[Optional[int]]] = None  # None для любых размерностей
    dtype: Optional[torch.dtype] = None
    required: bool = True
    aggregation: AggregationStrategy = AggregationStrategy.CONCAT
    custom_aggregator: Optional[Callable[[List[Any]], Any]] = None
    
    def is_compatible_with(self, other: 'PortSpec') -> bool:
        """Проверяет совместимость с другим портом"""
        # Проверка типов
        if self.type != other.type and self.type != PortType.ANY and other.type != PortType.ANY:
            return False
        
        # Проверка dtype
        if self.dtype is not None and other.dtype is not None and self.dtype != other.dtype:
            return False
        
        # Проверка формы (если обе спецификации имеют форму)
        if self.shape is not None and other.shape is not None:
            if len(self.shape) != len(other.shape):
                return False
            
            for dim_self, dim_other in zip(self.shape, other.shape):
                if dim_self is not None and dim_other is not None and dim_self != dim_other:
                    return False
        
        return True
    
    def validate_value(self, value: Any) -> bool:
        """Валидирует значение по спецификации порта"""
        try:
            # Проверка типа
            if self.type == PortType.TENSOR and not isinstance(value, torch.Tensor):
                return False
            elif self.type == PortType.SCALAR and not isinstance(value, (int, float, torch.Tensor)):
                return False
            elif self.type == PortType.SEQUENCE and not isinstance(value, (list, tuple)):
                return False
            elif self.type == PortType.DICT and not isinstance(value, dict):
                return False
            
            # Проверка dtype для тензоров
            if isinstance(value, torch.Tensor) and self.dtype is not None and value.dtype != self.dtype:
                return False
            
            # Проверка формы для тензоров
            if isinstance(value, torch.Tensor) and self.shape is not None:
                if len(value.shape) != len(self.shape):
                    return False
                
                for dim_value, dim_spec in zip(value.shape, self.shape):
                    if dim_spec is not None and dim_value != dim_spec:
                        return False
            
            return True
        except:
            return False

class Component(nn.Module):
    """Базовый компонент системы с поддержкой портов"""
    
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name or self.__class__.__name__
        self.input_ports: Dict[str, PortSpec] = self._define_input_ports()
        self.output_ports: Dict[str, PortSpec] = self._define_output_ports()
        self._compiled = False
        
    def _define_input_ports(self) -> Dict[str, PortSpec]:
        """Определяет входные порты компонента (переопределяется в подклассах)"""
        return {'default': PortSpec('default')}
    
    def _define_output_ports(self) -> Dict[str, PortSpec]:
        """Определяет выходные порты компонента (переопределяется в подклассах)"""
        return {'default': PortSpec('default')}
    
    def forward(self, **inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Прямой проход с поддержкой множественных входов/выходов"""
        raise NotImplementedError("Components must implement forward method")
    
    def validate_ports(self) -> List[str]:
        """Проверка корректности определения портов"""
        errors = []
    
        # Проверяем, что все обязательные порты имеют спецификации
        for port_name, port_spec in self.input_ports.items():
            if port_spec.required and port_spec.type == PortType.ANY:
                warnings.warn(f"Input port {port_name} is required but has type ANY")
    
        # Проверяем сигнатуру forward метода только если не используется **inputs
        try:
            sig = inspect.signature(self.forward)
            forward_params = list(sig.parameters.keys())
        
            # Если метод использует **kwargs или **inputs, пропускаем проверку параметров
            has_var_keyword = any(
                param.kind == param.VAR_KEYWORD 
                for param in sig.parameters.values()
            )
        
            if not has_var_keyword:
                # Проверяем, что все входные порты есть в параметрах forward
                for port_name in self.input_ports:
                    if port_name not in forward_params and port_name != 'default':
                        errors.append(f"Input port {port_name} not found in forward method parameters")
        except:
            pass  # Пропускаем проверку, если невозможно получить сигнатуру
    
        return errors
    
    def compile(self):
        """Компиляция компонента (если нужна)"""
        self._compiled = True
        
    def reset(self):
        """Сброс состояния компонента (переопределяется в подклассах)"""
        pass
        
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, inputs={list(self.input_ports.keys())}, outputs={list(self.output_ports.keys())})"

@dataclass
class Connection:
    """Соединение между портами компонентов"""
    source: str  # format: "component_name.port_name"
    target: str  # format: "component_name.port_name"
    transformer: Optional[Callable] = None
    delay: int = 0  # Для рекуррентных соединений
    
    def __post_init__(self):
        if '.' not in self.source:
            self.source = f"{self.source}.default"
        if '.' not in self.target:
            self.target = f"{self.target}.default"
    
    @property
    def source_component(self) -> str:
        return self.source.split('.')[0]
    
    @property
    def source_port(self) -> str:
        return self.source.split('.')[1]
    
    @property
    def target_component(self) -> str:
        return self.target.split('.')[0]
    
    @property
    def target_port(self) -> str:
        return self.target.split('.')[1]
    
    def is_recurrent(self) -> bool:
        """Проверяет, является ли соединение рекуррентным"""
        return self.delay > 0
    
    def __repr__(self):
        delay_str = f", delay={self.delay}" if self.delay > 0 else ""
        return f"Connection({self.source} -> {self.target}{delay_str})"

class SharedMemory:
    """Общая память для быстрого обмена данными между компонентами"""
    
    def __init__(self):
        self._storage = {}
        self._access_times = {}
        self._max_size = 1000  # Максимальное количество элементов в памяти
        
    def set(self, key: str, value: Any):
        """Устанавливает значение в общей памяти"""
        self._storage[key] = value
        self._access_times[key] = time.time()
        self._cleanup()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Получает значение из общей памяти"""
        value = self._storage.get(key, default)
        if key in self._storage:
            self._access_times[key] = time.time()
        return value
    
    def delete(self, key: str):
        """Удаляет значение из общей памяти"""
        if key in self._storage:
            del self._storage[key]
            del self._access_times[key]
    
    def clear(self):
        """Очищает всю общую память"""
        self._storage.clear()
        self._access_times.clear()
    
    def _cleanup(self):
        """Очищает старые элементы при превышении лимита"""
        if len(self._storage) > self._max_size:
            # Удаляем самые старые элементы
            sorted_keys = sorted(self._access_times.keys(), key=lambda k: self._access_times[k])
            for key in sorted_keys[:len(self._storage) - self._max_size]:
                self.delete(key)

class GlueTorch(nn.Module):
    """Модульная система для построения нейросетевых архитектур"""
    
    def __init__(self):
        super().__init__()
        self.components: Dict[str, Component] = {}
        self.connections: List[Connection] = []
        self.execution_order: List[str] = []
        self._compiled = False
        self._connection_map: Dict[str, List[Connection]] = defaultdict(list)
        self._recurrent_connections: List[Connection] = []
        self._state_buffers: Dict[str, Any] = {}
        self._input_map: Dict[str, List[Connection]] = defaultdict(list)
        self.shared_memory = SharedMemory()
        self._performance_stats: Dict[str, List[float]] = defaultdict(list)
        
    def register_component(self, component: Component):
        """Регистрирует компонент в системе с валидацией"""
        if component.name in self.components:
            raise ValueError(f"Component {component.name} already registered")
        
        # Валидируем порты компонента
        errors = component.validate_ports()
        if errors:
            raise ValueError(f"Component {component.name} has port errors: {errors}")
        
        self.components[component.name] = component
        self.add_module(component.name, component)
        
    def add_connection(self, source: Union[str, Tuple[str, str]], 
                      target: Union[str, Tuple[str, str]],
                      transformer: Optional[Callable] = None,
                      delay: int = 0):
        """Добавляет соединение между портами компонентов с валидацией"""
        if isinstance(source, tuple):
            source_str = f"{source[0]}.{source[1]}"
        else:
            source_str = source
            
        if isinstance(target, tuple):
            target_str = f"{target[0]}.{target[1]}"
        else:
            target_str = target
            
        connection = Connection(source_str, target_str, transformer, delay)
        
        # Проверяем существование компонентов и портов
        self._validate_connection(connection)
        
        # Проверяем совместимость портов
        self._validate_port_compatibility(connection)
        
        self.connections.append(connection)
        
        # Добавляем в карту соединений для быстрого доступа
        if connection.is_recurrent():
            self._recurrent_connections.append(connection)
        else:
            self._connection_map[connection.target].append(connection)
            self._input_map[connection.target].append(connection)
    
    def _validate_connection(self, connection: Connection):
        """Валидирует соединение на существование компонентов и портов"""
        # Проверяем существование компонентов
        if (connection.source_component != 'input' and 
            connection.source_component not in self.components):
            raise ValueError(f"Source component {connection.source_component} not found")
            
        if (connection.target_component != 'output' and 
            connection.target_component not in self.components):
            raise ValueError(f"Target component {connection.target_component} not found")
            
        # Проверяем существование портов
        if (connection.source_component != 'input' and 
            connection.source_port not in self.components[connection.source_component].output_ports):
            raise ValueError(f"Source port {connection.source_port} not found in component {connection.source_component}")
            
        if (connection.target_component != 'output' and 
            connection.target_port not in self.components[connection.target_component].input_ports):
            raise ValueError(f"Target port {connection.target_port} not found in component {connection.target_component}")
    
    def _validate_port_compatibility(self, connection: Connection):
        """Проверяет совместимость портов для соединения с учетом агрегации"""
        if connection.source_component == 'input' or connection.target_component == 'output':
            return  # Пропускаем проверку для входов/выходов системы

        source_comp = self.components[connection.source_component]
        target_comp = self.components[connection.target_component]

        source_port = source_comp.output_ports[connection.source_port]
        target_port = target_comp.input_ports[connection.target_port]

        # Для агрегируемых портов откладываем проверку до компиляции
        if target_port.aggregation != AggregationStrategy.CUSTOM:
            # Проверяем только базовую совместимость типов
            if source_port.type != target_port.type and source_port.type != PortType.ANY and target_port.type != PortType.ANY:
                raise ValueError(
                    f"Port type mismatch: {connection.source} ({source_port.type}) -> "
                    f"{connection.target} ({target_port.type})"
                )
            return

        # Для неагрегируемых портов выполняем полную проверку
        if not source_port.is_compatible_with(target_port):
            raise ValueError(
                f"Port compatibility error: {connection.source} ({source_port.type}, "
                f"{source_port.shape}, {source_port.dtype}) -> {connection.target} "
                f"({target_port.type}, {target_port.shape}, {target_port.dtype})"
            )

    def _validate_aggregation_compatibility(self, target_component: str, target_port: str):
        """Проверяет совместимость всех входов для агрегируемого порта"""
        component = self.components[target_component]
        port_spec = component.input_ports[target_port]
    
        # Получаем все соединения к целевому порту
        target_connections = [
            conn for conn in self.connections 
            if conn.target_component == target_component and conn.target_port == target_port
        ]
    
        if len(target_connections) <= 1:
            return  # Для одиночных соединений нет необходимости проверять агрегацию
    
        # Собираем спецификации всех исходных портов
        source_specs = []
        for conn in target_connections:
            if conn.source_component == 'input':
                # Для входных портов создаем временную спецификацию
                source_specs.append(PortSpec('input', PortType.ANY))
            else:
                source_comp = self.components[conn.source_component]
                source_specs.append(source_comp.output_ports[conn.source_port])
    
        # Проверяем совместимость в зависимости от стратегии агрегации
        if port_spec.aggregation == AggregationStrategy.CONCAT:
            # Для конкатенации проверяем, что все размерности кроме последней совпадают
            first_shape = source_specs[0].shape
            if first_shape is None:
                return  # Не можем проверить
            
            for spec in source_specs[1:]:
                if spec.shape is None:
                    continue
                
                if len(spec.shape) != len(first_shape):
                    raise ValueError(f"Shape rank mismatch for aggregation: "
                                   f"{len(spec.shape)} vs {len(first_shape)}")
            
                for i in range(len(first_shape) - 1):
                    if (spec.shape[i] != first_shape[i] and 
                        spec.shape[i] is not None and 
                        first_shape[i] is not None):
                        raise ValueError(f"Shape mismatch for aggregation at dimension {i}: "
                                       f"{spec.shape[i]} vs {first_shape[i]}")
    
        elif port_spec.aggregation in [AggregationStrategy.SUM, AggregationStrategy.MEAN, 
                                      AggregationStrategy.MAX, AggregationStrategy.STACK]:
            # Для этих стратегий все формы должны совпадать
            first_shape = source_specs[0].shape
            for spec in source_specs[1:]:
                if not spec.is_compatible_with(source_specs[0]):
                    raise ValueError(f"Shape mismatch for {port_spec.aggregation.value} aggregation: "
                                   f"{spec.shape} vs {first_shape}")

    def _precompile_connections(self):
        """Предварительная компиляция карты соединений для оптимизации производительности"""
        self._input_map.clear()
        for conn in self.connections:
            self._input_map[conn.target].append(conn)
    
    def compile(self):
        """Компилирует граф выполнения с расширенной валидацией"""
        # Валидируем все компоненты
        for comp_name, component in self.components.items():
            errors = component.validate_ports()
            if errors:
                raise ValueError(f"Component {comp_name} has validation errors: {errors}")
    
        # Предварительная компиляция соединений
        self._precompile_connections()
    
        # Строим граф зависимостей на уровне компонентов
        dependency_graph = defaultdict(list)
        in_degree = defaultdict(int)
    
        # Добавляем все компоненты в граф
        for comp_name in self.components:
            dependency_graph[comp_name] = []
            in_degree[comp_name] = 0
    
        # Строим зависимости на основе соединений
        for conn in self.connections:
            if conn.is_recurrent():
                continue  # Пропускаем рекуррентные соединения для топологической сортировки
            
            if (conn.source_component != 'input' and 
                conn.target_component != 'output' and
                conn.source_component != conn.target_component):
            
                dependency_graph[conn.source_component].append(conn.target_component)
                in_degree[conn.target_component] += 1
    
        # Топологическая сортировка (Kahn's algorithm)
        queue = [comp for comp in in_degree if in_degree[comp] == 0]
        execution_order = []
    
        while queue:
            node = queue.pop(0)
            execution_order.append(node)
        
            for neighbor in dependency_graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                
        if len(execution_order) != len(self.components):
            # Проверяем наличие циклов (кроме рекуррентных)
            cyclic_components = set(self.components.keys()) - set(execution_order)
            if cyclic_components:
                raise ValueError(f"Graph contains cycles involving components: {cyclic_components}")
        
        self.execution_order = execution_order
    
        # Проверяем совместимость агрегации для всех портов
        for comp_name, component in self.components.items():
            for port_name, port_spec in component.input_ports.items():
                self._validate_aggregation_compatibility(comp_name, port_name)
    
        # Компилируем все компоненты
        for component in self.components.values():
            component.compile()
    
        self._compiled = True
    
        return self
    
    def forward(self, **inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Прямой проход через всю систему с поддержкой рекуррентных соединений"""
        if not self._compiled:
            self.compile()
            
        # Словарь для хранения промежуточных результатов по портам
        results = {'input': inputs}
        
        # Инициализируем буферы состояний для рекуррентных соединений
        self._init_state_buffers()
        
        # Выполняем компоненты в порядке топологической сортировки
        for comp_name in self.execution_order:
            component = self.components[comp_name]
            
            # Собираем входные данные для каждого порта компонента
            component_inputs = self._gather_inputs(comp_name, results)
            
            # Выполняем компонент
            try:
                start_time = time.time()
                component_outputs = component(**component_inputs)
                end_time = time.time()
                
                # Сохраняем метрики производительности
                self._performance_stats[comp_name].append(end_time - start_time)
                
                results[comp_name] = component_outputs
            except Exception as e:
                raise RuntimeError(f"Error executing component {comp_name}: {str(e)}")
            
        # Обрабатываем рекуррентные соединения (обновляем состояния)
        self._update_recurrent_states(results)
            
        # Собираем выходные данные
        output_data = self._gather_outputs(results)
        
        return output_data
    
    def _gather_inputs(self, comp_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Собирает входные данные для компонента"""
        component = self.components[comp_name]
        component_inputs = {}
    
        for port_name, port_spec in component.input_ports.items():
            port_inputs = []
            target_key = f"{comp_name}.{port_name}"
        
            # Ищем все соединения, ведущие к этому порту
            for conn in self._input_map.get(target_key, []):
                # Получаем данные из источника
                source_data = self._get_source_data(conn, results)
            
                # Пропускаем None значения (например, для рекуррентных соединений в начале)
                if source_data is None:
                    continue
                
                # Применяем трансформер если есть
                if conn.transformer:
                    try:
                        source_data = self._apply_transformer(conn, source_data)
                    except Exception as e:
                        self._log_transformer_error(conn, e)
                        raise
            
                # Для агрегируемых портов откладываем валидацию до после агрегации
                if port_spec.aggregation == AggregationStrategy.CUSTOM or len(self._input_map.get(target_key, [])) <= 1:
                    # Валидируем данные по спецификации порта
                    if not port_spec.validate_value(source_data):
                        raise ValueError(
                            f"Data validation failed for {conn.source} -> {conn.target}: "
                            f"value {type(source_data)} doesn't match port spec"
                        )
            
                port_inputs.append(source_data)
        
            # Обрабатываем собранные входы для порта
            if not port_inputs:
                if port_spec.required:
                    raise ValueError(f"No connections to required port {comp_name}.{port_name}")
                continue
        
            # Применяем стратегию агрегации
            if len(port_inputs) == 1:
                component_inputs[port_name] = port_inputs[0]
            else:
                component_inputs[port_name] = self._aggregate_inputs(port_inputs, port_spec)
                # После агрегации валидируем результат для агрегируемых портов
                if port_spec.aggregation != AggregationStrategy.CUSTOM:
                    if not port_spec.validate_value(component_inputs[port_name]):
                        raise ValueError(
                            f"Data validation failed after aggregation for port {comp_name}.{port_name}: "
                            f"value {type(component_inputs[port_name])} doesn't match port spec"
                        )
    
        return component_inputs   

    def _apply_transformer(self, connection: Connection, data: Any) -> Any:
        """Применяет трансформер к данным с обработкой ошибок"""
        try:
            return connection.transformer(data)
        except Exception as e:
            error_msg = f"Transformer error in connection {connection}: {str(e)}"
            if hasattr(self, '_log_transformer_error'):
                self._log_transformer_error(connection, e)
            raise RuntimeError(error_msg)
    
    def _log_transformer_error(self, connection: Connection, error: Exception):
        """Логирует ошибку трансформера"""
        warnings.warn(f"Transformer error in connection {connection}: {str(error)}")
    
    def _aggregate_inputs(self, inputs: List[Any], port_spec: PortSpec) -> Any:
        """Агрегирует входные данные согласно стратегии агрегации"""
        if port_spec.aggregation == AggregationStrategy.CONCAT:
            return torch.cat(inputs, dim=-1)
        elif port_spec.aggregation == AggregationStrategy.SUM:
            return sum(inputs)
        elif port_spec.aggregation == AggregationStrategy.MEAN:
            return sum(inputs) / len(inputs)
        elif port_spec.aggregation == AggregationStrategy.MAX:
            return torch.stack(inputs).max(dim=0).values
        elif port_spec.aggregation == AggregationStrategy.STACK:
            return torch.stack(inputs)
        elif port_spec.aggregation == AggregationStrategy.CUSTOM and port_spec.custom_aggregator:
            return port_spec.custom_aggregator(inputs)
        else:
            # По умолчанию конкатенируем
            return torch.cat(inputs, dim=-1)
    
    def _get_source_data(self, connection: Connection, results: Dict[str, Any]) -> Any:
        """Получает данные из источника соединения"""
        if connection.source_component == 'input':
            source_data = results['input'].get(connection.source_port, None)
            if source_data is None:
                raise ValueError(f"Input port {connection.source_port} not provided")
        else:
            # Для рекуррентных соединений, если компонент еще не выполнен, используем буфер состояний
            if connection.is_recurrent() and connection.source_component not in results:
                buffer_key = f"{connection.source}->{connection.target}"
                if buffer_key in self._state_buffers and self._state_buffers[buffer_key]:
                    # Берем последнее состояние из буфера
                    return self._state_buffers[buffer_key][-1]
                else:
                    # Если буфер пуст, возвращаем None (должно обрабатываться вызывающим кодом)
                    return None
        
            source_data = results.get(connection.source_component, {}).get(connection.source_port, None)
            if source_data is None:
                raise ValueError(f"Source port {connection.source_component}.{connection.source_port} not found")
    
        return source_data
    
    def _init_state_buffers(self):
        """Инициализирует буферы состояний для рекуррентных соединений"""
        for conn in self._recurrent_connections:
            buffer_key = f"{conn.source}->{conn.target}"
            if buffer_key not in self._state_buffers:
                self._state_buffers[buffer_key] = []
    
    def _update_recurrent_states(self, results: Dict[str, Any]):
        """Обновляет состояния для рекуррентных соединений"""
        for conn in self._recurrent_connections:
            buffer_key = f"{conn.source}->{conn.target}"
            source_data = self._get_source_data(conn, results)
            
            # Добавляем текущее состояние в буфер
            self._state_buffers[buffer_key].append(source_data)
            
            # Ограничиваем размер буфера для предотвращения утечек памяти
            if len(self._state_buffers[buffer_key]) > conn.delay * 2:
                self._state_buffers[buffer_key] = self._state_buffers[buffer_key][-conn.delay:]
    
    def _gather_outputs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Собирает выходные данные системы"""
        output_data = {}
        
        for conn in self.connections:
            if conn.target_component == 'output':
                source_data = self._get_source_data(conn, results)
                
                if conn.transformer:
                    try:
                        source_data = self._apply_transformer(conn, source_data)
                    except Exception as e:
                        self._log_transformer_error(conn, e)
                        raise
                
                output_data[conn.target_port] = source_data
        
        return output_data
    
    def reset(self):
        """Сброс состояния системы и всех компонентов"""
        for component in self.components.values():
            if hasattr(component, 'reset'):
                component.reset()
        self._state_buffers.clear()
        self.shared_memory.clear()
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Возвращает статистику производительности компонентов"""
        stats = {}
        for comp_name, times in self._performance_stats.items():
            if times:
                stats[comp_name] = {
                    'calls': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'max_time': max(times),
                    'min_time': min(times)
                }
        return stats
    
    def get_connection_graph(self) -> Dict:
        """Возвращает граф соединений для визуализации"""
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Добавляем специальные узлы input/output
        graph['nodes'].append({'id': 'input', 'type': 'input'})
        graph['nodes'].append({'id': 'output', 'type': 'output'})
        
        # Добавляем компоненты как узлы
        for comp_name, component in self.components.items():
            graph['nodes'].append({
                'id': comp_name,
                'type': 'component',
                'inputs': list(component.input_ports.keys()),
                'outputs': list(component.output_ports.keys())
            })
        
        # Добавляем соединения как ребра
        for conn in self.connections:
            graph['edges'].append({
                'source': conn.source,
                'target': conn.target,
                'recurrent': conn.is_recurrent(),
                'delay': conn.delay
            })
        
        return graph
    
    def visualize(self, output_file: str = None):
        """Визуализирует граф соединений"""
        try:
            import graphviz
            dot = graphviz.Digraph()
            
            # Добавляем узлы
            dot.node('input', 'Input', shape='ellipse', color='green')
            dot.node('output', 'Output', shape='ellipse', color='red')
            
            for comp_name, component in self.components.items():
                label = f"{comp_name}|{{{'|'.join(component.input_ports.keys())}}}|{{{'|'.join(component.output_ports.keys())}}}"
                dot.node(comp_name, f"{{{label}}}", shape='record')
            
            # Добавляем ребра
            for conn in self.connections:
                if conn.is_recurrent():
                    dot.edge(conn.source, conn.target, label=f"delay={conn.delay}", color='blue', style='dashed')
                else:
                    dot.edge(conn.source, conn.target)
            
            if output_file:
                dot.render(output_file, format='png', cleanup=True)
            
            return dot
        except ImportError:
            warnings.warn("Install graphviz for visualization: pip install graphviz")
            # Fallback to text visualization
            graph = self.get_connection_graph()
            print("Graph visualization:")
            print(f"Nodes: {[node['id'] for node in graph['nodes']]}")
            for edge in graph['edges']:
                rec_str = " (recurrent)" if edge['recurrent'] else ""
                print(f"  {edge['source']} -> {edge['target']}{rec_str}")
            return None

    def save(self, filepath: str):
        """Сохраняет конфигурацию системы в файл"""
        config = {
            'components': {},
            'connections': []
        }
        
        # Сохраняем информацию о компонентах
        for comp_name, component in self.components.items():
            config['components'][comp_name] = {
                'type': component.__class__.__name__,
                'input_ports': {name: {'type': port.type.value, 
                                      'shape': port.shape,
                                      'dtype': str(port.dtype) if port.dtype else None,
                                      'required': port.required,
                                      'aggregation': port.aggregation.value}
                               for name, port in component.input_ports.items()},
                'output_ports': {name: {'type': port.type.value, 
                                       'shape': port.shape,
                                       'dtype': str(port.dtype) if port.dtype else None}
                                for name, port in component.output_ports.items()}
            }
        
        # Сохраняем соединения
        for conn in self.connections:
            config['connections'].append({
                'source': conn.source,
                'target': conn.target,
                'delay': conn.delay
            })
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, component_registry: Dict[str, Type[Component]]):
        """Загружает конфигурацию системы из файла"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        system = cls()
        
        # Создаем и регистрируем компоненты
        for comp_name, comp_config in config['components'].items():
            comp_type = comp_config['type']
            if comp_type not in component_registry:
                raise ValueError(f"Component type {comp_type} not found in registry")
            
            component = component_registry[comp_type]()
            component.name = comp_name
            
            # Восстанавливаем порты (упрощенная версия)
            system.register_component(component)
        
        # Восстанавливаем соединения
        for conn_config in config['connections']:
            system.add_connection(
                conn_config['source'],
                conn_config['target'],
                delay=conn_config.get('delay', 0)
            )
        
        return system

    def to_distributed(self, device_map: Dict[str, torch.device]):
        """Распределяет компоненты по устройствам"""
        for comp_name, device in device_map.items():
            if comp_name in self.components:
                self.components[comp_name].to(device)
    
    def to_onnx(self, filepath: str, sample_input: Dict[str, Any]):
        """Экспортирует систему в ONNX формат"""
        # Временная реализация - может потребоваться доработка для сложных графов
        dynamic_axes = {}
        input_names = []
        output_names = []
        
        # Подготавливаем имена входов и выходов
        for port_name in sample_input.keys():
            input_names.append(f"input_{port_name}")
            dynamic_axes[f"input_{port_name}"] = {0: 'batch_size'}
        
        for port_name in self._get_output_ports():
            output_names.append(f"output_{port_name}")
            dynamic_axes[f"output_{port_name}"] = {0: 'batch_size'}
        
        # Экспортируем
        torch.onnx.export(
            self,
            sample_input,
            filepath,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=13
        )
    
    def _get_output_ports(self) -> List[str]:
        """Получает список выходных портов системы"""
        output_ports = set()
        for conn in self.connections:
            if conn.target_component == 'output':
                output_ports.add(conn.target_port)
        return list(output_ports)
    
    @contextmanager
    def profile(self):
        """Контекстный менеджер для профилирования выполнения"""
        try:
            import torch.autograd.profiler as profiler
        except ImportError:
            yield
            return
            
        with profiler.profile(record_shapes=True) as prof:
            yield
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

class NeuralCompiler(nn.Module):
    """
    Компилирует декларативные связи в исполняемый граф
    с возможностью адаптации параметров трансформеров
    """
    
    def __init__(self, glue_system: GlueTorch):
        super().__init__()
        self.glue_system = glue_system
        self.learnable_transformers: Dict[str, nn.Module] = {}
        self._compiled_pipeline: Optional[Callable] = None
        self._parameter_map: Dict[str, nn.Parameter] = {}
        
    def register_learnable_transformer(self, connection_id: str, transformer: nn.Module):
        """Регистрирует обучаемый трансформер для соединения"""
        self.learnable_transformers[connection_id] = transformer
        self.add_module(f"transformer_{connection_id}", transformer)
        
    def compile(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Компилирует граф с учетом обучаемых трансформеров"""
        # Строим оптимизированный граф выполнения
        execution_graph = self._build_optimized_graph()
        
        def compiled_pipeline(inputs: Dict[str, Any]) -> Dict[str, Any]:
            # Инициализируем контекст выполнения
            context = {
                'inputs': inputs,
                'intermediate': {},
                'outputs': {}
            }
            
            # Выполняем компоненты в порядке графа
            for component_name, dependencies in execution_graph:
                # Пропускаем специальные компоненты
                if component_name in ['input', 'output']:
                    continue
                    
                # Собираем входные данные для компонента
                component_inputs = self._gather_component_inputs(
                    component_name, dependencies, context
                )
                
                # Выполняем компонент
                component = self.glue_system.components[component_name]
                component_outputs = component(**component_inputs)
                
                # Сохраняем результаты
                context['intermediate'][component_name] = component_outputs
            
            # Собираем выходные данные
            return self._gather_outputs(context)
        
        self._compiled_pipeline = compiled_pipeline
        return compiled_pipeline
    
    def _build_optimized_graph(self) -> List[Tuple[str, List[Dict]]]:
        """Строит оптимизированный граф выполнения с кэшированием зависимостей"""
        # Используем существующий механизм GlueTorch для получения порядка выполнения
        if not self.glue_system._compiled:
            self.glue_system.compile()
        
        # Строим карту зависимостей для каждого компонента
        dependency_map = {}
        for comp_name in self.glue_system.execution_order:
            dependencies = []
            
            # Находим все соединения, ведущие к этому компоненту
            for conn in self.glue_system.connections:
                if conn.target_component == comp_name:
                    source_data = {
                        'source': conn.source_component,
                        'source_port': conn.source_port,
                        'target_port': conn.target_port,
                        'transformer': conn.transformer,
                        'connection_id': f"{conn.source}->{conn.target}"
                    }
                    dependencies.append(source_data)
            
            dependency_map[comp_name] = dependencies
        
        # Возвращаем упорядоченный список с зависимостями
        return [(comp_name, dependency_map[comp_name]) 
                for comp_name in self.glue_system.execution_order]
    
    def _gather_component_inputs(self, component_name: str, 
                               dependencies: List[Dict], 
                               context: Dict) -> Dict[str, Any]:
        """Собирает входные данные для компонента с учетом обучаемых трансформеров"""
        inputs = {}
        
        for dep in dependencies:
            # Получаем данные из источника
            if dep['source'] == 'input':
                source_data = context['inputs'].get(dep['source_port'], None)
            else:
                source_data = context['intermediate'][dep['source']].get(dep['source_port'], None)
            
            if source_data is None:
                raise ValueError(f"Source data not found for {dep['source']}.{dep['source_port']}")
            
            # Применяем трансформер
            connection_id = dep['connection_id']
            if connection_id in self.learnable_transformers:
                # Используем обучаемый трансформер
                transformed = self.learnable_transformers[connection_id](source_data)
            elif dep['transformer']:
                # Используем статический трансформер
                transformed = dep['transformer'](source_data)
            else:
                transformed = source_data
            
            # Агрегируем данные для целевого порта
            target_port = dep['target_port']
            if target_port in inputs:
                # Если на порт уже есть данные, применяем стратегию агрегации
                inputs[target_port] = self._aggregate_inputs(
                    inputs[target_port], transformed, 
                    self.glue_system.components[component_name].input_ports[target_port]
                )
            else:
                inputs[target_port] = transformed
        
        return inputs
    
    def _aggregate_inputs(self, existing_data: Any, new_data: Any, port_spec: PortSpec) -> Any:
        """Агрегирует входные данные согласно стратегии агрегации порта"""
        if port_spec.aggregation == AggregationStrategy.CONCAT:
            return torch.cat([existing_data, new_data], dim=-1)
        elif port_spec.aggregation == AggregationStrategy.SUM:
            return existing_data + new_data
        elif port_spec.aggregation == AggregationStrategy.MEAN:
            return (existing_data + new_data) / 2
        elif port_spec.aggregation == AggregationStrategy.MAX:
            return torch.max(existing_data, new_data)
        elif port_spec.aggregation == AggregationStrategy.STACK:
            return torch.stack([existing_data, new_data])
        else:
            # По умолчанию конкатенируем
            return torch.cat([existing_data, new_data], dim=-1)
    
    def _gather_outputs(self, context: Dict) -> Dict[str, Any]:
        """Собирает выходные данные системы"""
        outputs = {}
        
        for conn in self.glue_system.connections:
            if conn.target_component == 'output':
                # Получаем данные из источника
                if conn.source_component == 'input':
                    source_data = context['inputs'].get(conn.source_port, None)
                else:
                    source_data = context['intermediate'][conn.source_component].get(conn.source_port, None)
                
                if source_data is None:
                    raise ValueError(f"Source data not found for {conn.source}")
                
                # Применяем трансформер если есть
                connection_id = f"{conn.source}->{conn.target}"
                if connection_id in self.learnable_transformers:
                    transformed = self.learnable_transformers[connection_id](source_data)
                elif conn.transformer:
                    transformed = conn.transformer(source_data)
                else:
                    transformed = source_data
                
                outputs[conn.target_port] = transformed
        
        return outputs
    
    def forward(self, **inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Прямой проход через скомпилированный граф"""
        if self._compiled_pipeline is None:
            self.compile()
        
        return self._compiled_pipeline(inputs)
    
    def optimize_for_inference(self):
        """Оптимизирует граф для инференса"""
        # Переводим все обучаемые трансформеры в режим инференса
        for transformer in self.learnable_transformers.values():
            transformer.eval()
        
        # Замораживаем параметры
        for param in self.parameters():
            param.requires_grad = False
        
        # Конвертация в half precision если доступно CUDA
        if torch.cuda.is_available():
            self.half()
        
        # Компилируем TorScript если возможно
        try:
            self._compiled_pipeline = torch.jit.script(self._compiled_pipeline)
        except:
            warnings.warn("Could not compile pipeline with TorScript")
        
        # Используем torch.compile если доступно
        if hasattr(torch, 'compile'):
            try:
                self._compiled_pipeline = torch.compile(self._compiled_pipeline)
            except:
                warnings.warn("Could not compile pipeline with torch.compile")
    
    def get_learnable_parameters(self) -> Dict[str, nn.Parameter]:
        """Возвращает словарь обучаемых параметров"""
        params = {}
        for name, transformer in self.learnable_transformers.items():
            for param_name, param in transformer.named_parameters():
                params[f"{name}.{param_name}"] = param
        return params

# Базовые компоненты для быстрого старта
class LinearComponent(Component):
    """Линейный компонент с расширенной спецификацией портов"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 activation: str = 'relu', name: str = None):
        super().__init__(name)
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = self._get_activation(activation)
        
        # Переопределяем спецификацию портов
        self.input_ports = {
            'default': PortSpec(
                'default', 
                type=PortType.TENSOR,
                shape=(None, input_dim),
                dtype=torch.float32
            )
        }
        self.output_ports = {
            'default': PortSpec(
                'default',
                type=PortType.TENSOR,
                shape=(None, output_dim),
                dtype=torch.float32
            )
        }
        
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.Identity()
            
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'default': self.activation(self.linear(default))}

class MultiHeadAttentionComponent(Component):
    """Компонент multi-head attention с расширенной спецификацией портов"""
    
    def __init__(self, d_model: int, num_heads: int, name: str = None):
        super().__init__(name)
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Определяем порты с точными спецификациями
        self.input_ports = {
            'query': PortSpec(
                'query', 
                type=PortType.TENSOR,
                shape=(None, None, d_model),
                dtype=torch.float32,
                required=True
            ),
            'key': PortSpec(
                'key',
                type=PortType.TENSOR, 
                shape=(None, None, d_model),
                dtype=torch.float32,
                required=True
            ),
            'value': PortSpec(
                'value',
                type=PortType.TENSOR,
                shape=(None, None, d_model), 
                dtype=torch.float32,
                required=True
            ),
            'mask': PortSpec(
                'mask',
                type=PortType.TENSOR,
                shape=(None, None, None),
                dtype=torch.bool,
                required=False
            )
        }
        self.output_ports = {
            'output': PortSpec(
                'output',
                type=PortType.TENSOR,
                shape=(None, None, d_model),
                dtype=torch.float32
            )
        }
        
        # Инициализируем слои
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = query.size(0)
        d_k = self.d_model // self.num_heads
        
        # Linear transformations and split into heads
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return {'output': self.out_linear(attn_output)}

class VisionProcessor(Component):
    """Компонент для обработки визуальной информации"""
    def __init__(self, output_dim: int, name: str = None):
        super().__init__(name)
        self.output_dim = output_dim
        
        self.input_ports = {
            'default': PortSpec(
                'default',
                type=PortType.TENSOR,
                shape=(None, None, None, None),  # (batch, height, width, channels)
                dtype=torch.float32
            )
        }
        self.output_ports = {
            'default': PortSpec(
                'default',
                type=PortType.TENSOR,
                shape=(None, output_dim),
                dtype=torch.float32
            )
        }
        
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x shape: (batch, height, width, channels)
        x = default.flatten(start_dim=1)  # Flatten spatial dimensions
        if x.size(1) > self.output_dim:
            # Проецируем на меньшую размерность
            x = x[:, :self.output_dim]
        elif x.size(1) < self.output_dim:
            # Дополняем нулями
            padding = torch.zeros(x.size(0), self.output_dim - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        return {'default': x}

class MemoryComponent(Component):
    """Компонент для общей памяти с фиксированными ключами"""
    
    def __init__(self, keys: List[str], initial_values: Dict[str, Any] = None, name: str = None):
        super().__init__(name)
        self.keys = keys
        self.initial_values = initial_values or {}
        self.memory = {key: self.initial_values.get(key) for key in keys}
        
        # Создаем порты для записи и чтения для каждого ключа
        self.input_ports = {}
        self.output_ports = {}
        
        for key in keys:
            self.input_ports[f'write_{key}'] = PortSpec(
                f'write_{key}', 
                type=PortType.ANY,
                required=False
            )
            self.output_ports[f'read_{key}'] = PortSpec(
                f'read_{key}',
                type=PortType.ANY
            )
    
    def forward(self, **inputs) -> Dict[str, Any]:
        # Обрабатываем запись
        for key in self.keys:
            write_port = f'write_{key}'
            if write_port in inputs and inputs[write_port] is not None:
                self.memory[key] = inputs[write_port]
        
        # Формируем выходы для чтения
        outputs = {}
        for key in self.keys:
            outputs[f'read_{key}'] = self.memory[key]
        
        return outputs
    
    def reset(self):
        """Сброс памяти к начальным значениям"""
        self.memory = {key: self.initial_values.get(key) for key in self.keys}

# Утилиты для работы с системой
def connect(source: Union[str, Tuple[str, str]], 
           target: Union[str, Tuple[str, str]],
           transformer: Optional[Callable] = None,
           delay: int = 0) -> Connection:
    """Создает соединение между портами компонентов"""
    return Connection(source, target, transformer, delay)

# DSL для декларативного описания архитектур
class GlueGraph:
    """DSL для декларативного описания архитектур"""
    
    def __init__(self):
        self.system = GlueTorch()
        self._components = {}
    
    def add_input(self, name: str, shape: Tuple[Optional[int]] = None, dtype: torch.dtype = None) -> 'GlueGraph':
        """Добавляет входной порт"""
        # Входные порты обрабатываются автоматически в GlueTorch
        return self
    
    def add_component(self, name: str, component: Component) -> 'GlueGraph':
        """Добавляет компонент в граф"""
        self.system.register_component(component)
        self._components[name] = component
        return self
    
    def connect(self, source: str, target: str, transformer: Optional[Callable] = None) -> 'GlueGraph':
        """Добавляет соединение между компонентами"""
        self.system.add_connection(source, target, transformer)
        return self
    
    def build(self) -> GlueTorch:
        """Строит и возвращает систему"""
        self.system.compile()
        return self.system

# Пример использования
def example_usage():
    """Пример использования библиотеки GlueTorch"""
    
    # Создаем систему
    system = GlueTorch()
    
    # Создаем и регистрируем компоненты
    attention = MultiHeadAttentionComponent(d_model=512, num_heads=8, name="attention")
    linear = LinearComponent(input_dim=512, output_dim=256, name="linear")
    
    system.register_component(attention)
    system.register_component(linear)
    
    # Добавляем соединения с указанием портов
    system.add_connection(('input', 'query'), ('attention', 'query'))
    system.add_connection(('input', 'key'), ('attention', 'key')) 
    system.add_connection(('input', 'value'), ('attention', 'value'))
    system.add_connection(('attention', 'output'), ('linear', 'default'))
    system.add_connection(('linear', 'default'), ('output', 'result'))
    
    # Визуализируем граф
    system.visualize()
    
    # Компилируем и выполняем
    system.compile()
    
    # Входные данные для каждого порта
    inputs = {
        'query': torch.randn(1, 10, 512),
        'key': torch.randn(1, 10, 512),
        'value': torch.randn(1, 10, 512)
    }
    
    # Прямой проход
    outputs = system(**inputs)
    print(f"Output shape: {outputs['result'].shape}")
    
    # Использование NeuralCompiler
    compiler = NeuralCompiler(system)
    
    # Регистрируем обучаемый трансформер
    custom_transformer = nn.Linear(512, 512)
    compiler.register_learnable_transformer("attention.output->linear.default", custom_transformer)
    
    # Компилируем и выполняем
    compiled_outputs = compiler(**inputs)
    print(f"Compiled output shape: {compiled_outputs['result'].shape}")

if __name__ == "__main__":
    example_usage()

============================================================
.\arxglue\gluetorch\GlueTorch\mnist.py
============================================================
import torch
import torch.nn as nn
from typing import Any, Callable, Optional, Dict, List, Tuple, Union, Set, Type
from gluetorch import GlueTorch, Component, PortSpec, PortType, AggregationStrategy, LinearComponent

# Компонент для flattening
class FlattenComponent(Component):
    def __init__(self, name=None):
        super().__init__(name)
        self.input_ports = {
            'default': PortSpec('default', PortType.TENSOR, shape=(None, 1, 28, 28))
        }
        self.output_ports = {
            'default': PortSpec('default', PortType.TENSOR, shape=(None, 784))
        }
        
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'default': torch.flatten(default, start_dim=1)}

# Компонент с Dropout
class DropoutComponent(Component):
    def __init__(self, p=0.5, name=None):
        super().__init__(name)
        self.dropout = nn.Dropout(p)
        self.input_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        self.output_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'default': self.dropout(default)}

# Компонент с ReLU активацией
class ReLUComponent(Component):
    def __init__(self, name=None):
        super().__init__(name)
        self.relu = nn.ReLU()
        self.input_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        self.output_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'default': self.relu(default)}

# Создаем и настраиваем простую сеть
def create_simple_network():
    system = GlueTorch()
    
    # Создаем компоненты
    flatten = FlattenComponent(name='flatten')
    linear1 = LinearComponent(input_dim=784, output_dim=128, name='linear1')
    relu1 = ReLUComponent(name='relu1')
    dropout = DropoutComponent(p=0.5, name='dropout')
    linear2 = LinearComponent(input_dim=128, output_dim=10, name='linear2')
    
    # Регистрируем компоненты
    system.register_component(flatten)
    system.register_component(linear1)
    system.register_component(relu1)
    system.register_component(dropout)
    system.register_component(linear2)
    
    # Создаем соединения
    system.add_connection(('input', 'image'), ('flatten', 'default'))
    system.add_connection(('flatten', 'default'), ('linear1', 'default'))
    system.add_connection(('linear1', 'default'), ('relu1', 'default'))
    system.add_connection(('relu1', 'default'), ('dropout', 'default'))
    system.add_connection(('dropout', 'default'), ('linear2', 'default'))
    system.add_connection(('linear2', 'default'), ('output', 'logits'))
    
    # Компилируем систему
    system.compile()
    
    return system

# Тестируем простую сеть
if __name__ == "__main__":
    simple_net = create_simple_network()
    test_input = torch.randn(32, 1, 28, 28)  # Пакет из 32 изображений MNIST
    output = simple_net(image=test_input)
    print(f"Simple network output shape: {output['logits'].shape}")

============================================================
.\arxglue\gluetorch\GlueTorch\recruiter_components.py
============================================================
# recruiter_components.py
import torch
import torch.nn as nn
import re
from typing import Any, Callable, Optional, Dict, List, Tuple, Union, Set, Type
from gluetorch import Component, PortSpec, PortType, LinearComponent
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ResumeParserComponent(Component):
    """Компонент для парсинга резюме и извлечения структурированной информации"""
    def __init__(self, name: str = None):
        super().__init__(name)
        self.input_ports = {
            'resume_text': PortSpec('resume_text', PortType.ANY)  # Изменено на ANY
        }
        self.output_ports = {
            'skills': PortSpec('skills', PortType.SEQUENCE),
            'experience': PortSpec('experience', PortType.SEQUENCE),
            'education': PortSpec('education', PortType.SEQUENCE)
        }
        
    def forward(self, resume_text: Any) -> Dict[str, List[str]]:  # Изменен тип параметра
        # Преобразуем входные данные в строку
        text = str(resume_text)
        
        # Извлечение навыков (простая реализация)
        skills = self._extract_skills(text)
        
        # Извлечение опыта работы
        experience = self._extract_experience(text)
        
        # Извлечение образования
        education = self._extract_education(text)
        
        return {
            'skills': skills,
            'experience': experience,
            'education': education
        }
    
    def _extract_skills(self, text: str) -> List[str]:
        # Простое извлечение навыков по ключевым словам
        skill_keywords = ['python', 'java', 'sql', 'machine learning', 'docker', 'kubernetes']
        found_skills = []
        for skill in skill_keywords:
            if skill in text.lower():
                found_skills.append(skill)
        return found_skills
    
    def _extract_experience(self, text: str) -> List[str]:
        # Простое извлечение опыта работы
        experience_patterns = [
            r'(\d+)\s* years? of experience',
            r'experience.*?(\d+)\s* years?',
        ]
        experiences = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            experiences.extend(matches)
        return experiences if experiences else ['3']  # Значение по умолчанию
    
    def _extract_education(self, text: str) -> List[str]:
        # Простое извлечение образования
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma']
        education_levels = []
        for education in education_keywords:
            if education in text.lower():
                education_levels.append(education)
        return education_levels if education_levels else ['bachelor']  # Значение по умолчанию

class JDProcessorComponent(Component):
    """Компонент для обработки описания вакансии"""
    def __init__(self, name: str = None):
        super().__init__(name)
        self.input_ports = {
            'jd_text': PortSpec('jd_text', PortType.ANY)  # Изменено на ANY
        }
        self.output_ports = {
            'required_skills': PortSpec('required_skills', PortType.SEQUENCE),
            'required_experience': PortSpec('required_experience', PortType.SEQUENCE),
            'required_education': PortSpec('required_education', PortType.SEQUENCE)
        }
        
    def forward(self, jd_text: Any) -> Dict[str, List[str]]:  # Изменен тип параметра
        # Преобразуем входные данные в строку
        text = str(jd_text)
        
        required_skills = self._extract_required_skills(text)
        required_experience = self._extract_required_experience(text)
        required_education = self._extract_required_education(text)
        
        return {
            'required_skills': required_skills,
            'required_experience': required_experience,
            'required_education': required_education
        }
    
    def _extract_required_skills(self, text: str) -> List[str]:
        skill_keywords = ['python', 'java', 'sql', 'machine learning', 'docker', 'kubernetes']
        found_skills = []
        for skill in skill_keywords:
            if skill in text.lower():
                found_skills.append(skill)
        return found_skills
    
    def _extract_required_experience(self, text: str) -> List[str]:
        experience_patterns = [
            r'(\d+)\s* years? of experience',
            r'experience.*?(\d+)\s* years?',
        ]
        experiences = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            experiences.extend(matches)
        return experiences if experiences else ['3']  # Значение по умолчанию
    
    def _extract_required_education(self, text: str) -> List[str]:
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'diploma']
        education_levels = []
        for education in education_keywords:
            if education in text.lower():
                education_levels.append(education)
        return education_levels if education_levels else ['bachelor']  # Значение по умолчанию

class SkillMatcherComponent(Component):
    """Компонент для сравнения навыков кандидата и требований вакансии"""
    def __init__(self, name: str = None):
        super().__init__(name)
        self.input_ports = {
            'candidate_skills': PortSpec('candidate_skills', PortType.SEQUENCE),
            'required_skills': PortSpec('required_skills', PortType.SEQUENCE)
        }
        self.output_ports = {
            'skill_match_score': PortSpec('skill_match_score', PortType.TENSOR, shape=(1,))
        }
        
    def forward(self, candidate_skills: List[str], required_skills: List[str]) -> Dict[str, torch.Tensor]:
        # Вычисление соответствия навыков
        if not required_skills:
            return {'skill_match_score': torch.tensor([1.0])}
        
        matched_skills = set(candidate_skills) & set(required_skills)
        match_score = len(matched_skills) / len(required_skills)
        
        return {'skill_match_score': torch.tensor([match_score])}

class ExperienceMatcherComponent(Component):
    """Компонент для сравнения опыта кандидата и требований вакансии"""
    def __init__(self, name: str = None):
        super().__init__(name)
        self.input_ports = {
            'candidate_experience': PortSpec('candidate_experience', PortType.SEQUENCE),
            'required_experience': PortSpec('required_experience', PortType.SEQUENCE)
        }
        self.output_ports = {
            'experience_match_score': PortSpec('experience_match_score', PortType.TENSOR, shape=(1,))
        }
        
    def forward(self, candidate_experience: List[str], required_experience: List[str]) -> Dict[str, torch.Tensor]:
        # Вычисление соответствия опыта
        if not required_experience:
            return {'experience_match_score': torch.tensor([1.0])}
        
        candidate_exp = int(candidate_experience[0]) if candidate_experience else 0
        required_exp = int(required_experience[0]) if required_experience else 0
        
        if candidate_exp >= required_exp:
            return {'experience_match_score': torch.tensor([1.0])}
        else:
            return {'experience_match_score': torch.tensor([candidate_exp / required_exp])}

class EducationMatcherComponent(Component):
    """Компонент для сравнения образования кандидата и требований вакансии"""
    def __init__(self, name: str = None):
        super().__init__(name)
        self.input_ports = {
            'candidate_education': PortSpec('candidate_education', PortType.SEQUENCE),
            'required_education': PortSpec('required_education', PortType.SEQUENCE)
        }
        self.output_ports = {
            'education_match_score': PortSpec('education_match_score', PortType.TENSOR, shape=(1,))
        }
        
    def forward(self, candidate_education: List[str], required_education: List[str]) -> Dict[str, torch.Tensor]:
        # Вычисление соответствия образования
        if not required_education:
            return {'education_match_score': torch.tensor([1.0])}
        
        education_levels = {'bachelor': 1, 'master': 2, 'phd': 3, 'degree': 1, 'diploma': 1}
        
        candidate_edu_level = max([education_levels.get(edu.lower(), 0) for edu in candidate_education]) if candidate_education else 0
        required_edu_level = max([education_levels.get(edu.lower(), 0) for edu in required_education]) if required_education else 0
        
        if candidate_edu_level >= required_edu_level:
            return {'education_match_score': torch.tensor([1.0])}
        else:
            return {'education_match_score': torch.tensor([candidate_edu_level / required_edu_level])}

class OverallScorerComponent(Component):
    """Компонент для вычисления общего score кандидата"""
    def __init__(self, weights: Optional[Dict[str, float]] = None, name: str = None):
        super().__init__(name)
        self.weights = weights or {'skill': 0.5, 'experience': 0.3, 'education': 0.2}
        
        self.input_ports = {
            'skill_score': PortSpec('skill_score', PortType.TENSOR, shape=(1,)),
            'experience_score': PortSpec('experience_score', PortType.TENSOR, shape=(1,)),
            'education_score': PortSpec('education_score', PortType.TENSOR, shape=(1,))
        }
        self.output_ports = {
            'overall_score': PortSpec('overall_score', PortType.TENSOR, shape=(1,)),
            'detailed_scores': PortSpec('detailed_scores', PortType.DICT)
        }
        
    def forward(self, skill_score: torch.Tensor, experience_score: torch.Tensor, 
                education_score: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Вычисление общего score
        overall = (self.weights['skill'] * skill_score + 
                  self.weights['experience'] * experience_score + 
                  self.weights['education'] * education_score)
        
        return {
            'overall_score': overall,
            'detailed_scores': {
                'skill': skill_score,
                'experience': experience_score,
                'education': education_score
            }
        }

class SemanticSimilarityComponent(Component):
    """Компонент для вычисления семантического сходства на основе эмбеддингов"""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', name: str = None):
        super().__init__(name)
        self.model = SentenceTransformer(model_name)
        
        self.input_ports = {
            'text1': PortSpec('text1', PortType.ANY),  # Изменено на ANY
            'text2': PortSpec('text2', PortType.ANY)   # Изменено на ANY
        }
        self.output_ports = {
            'semantic_similarity': PortSpec('semantic_similarity', PortType.TENSOR, shape=(1,))
        }
        
    def forward(self, text1: Any, text2: Any) -> Dict[str, torch.Tensor]:  # Изменены типы параметров
        # Преобразуем входные данные в строки
        text1_str = str(text1)
        text2_str = str(text2)
            
        embedding1 = self.model.encode(text1_str, convert_to_tensor=True)
        embedding2 = self.model.encode(text2_str, convert_to_tensor=True)
        
        similarity = cosine_similarity(
            embedding1.cpu().numpy().reshape(1, -1), 
            embedding2.cpu().numpy().reshape(1, -1)
        )[0][0]
        
        return {'semantic_similarity': torch.tensor([similarity])}


============================================================
.\arxglue\gluetorch\GlueTorch\recruiter_pipeline.py
============================================================
# recruiter_pipeline.py
from gluetorch import GlueTorch
from recruiter_components import (
    ResumeParserComponent, 
    JDProcessorComponent,
    SkillMatcherComponent,
    ExperienceMatcherComponent,
    EducationMatcherComponent,
    OverallScorerComponent,
    SemanticSimilarityComponent
)

def create_recruiter_pipeline():
    """Создание пайплайна для анализа кандидатов"""
    system = GlueTorch()
    
    # Создаем компоненты
    resume_parser = ResumeParserComponent(name='resume_parser')
    jd_processor = JDProcessorComponent(name='jd_processor')
    skill_matcher = SkillMatcherComponent(name='skill_matcher')
    experience_matcher = ExperienceMatcherComponent(name='experience_matcher')
    education_matcher = EducationMatcherComponent(name='education_matcher')
    overall_scorer = OverallScorerComponent(name='overall_scorer')
    semantic_similarity = SemanticSimilarityComponent(name='semantic_similarity')
    
    # Регистрируем компоненты
    system.register_component(resume_parser)
    system.register_component(jd_processor)
    system.register_component(skill_matcher)
    system.register_component(experience_matcher)
    system.register_component(education_matcher)
    system.register_component(overall_scorer)
    system.register_component(semantic_similarity)
    
    # Создаем соединения
    # Парсинг резюме и вакансии
    system.add_connection(('input', 'resume_text'), ('resume_parser', 'resume_text'))
    system.add_connection(('input', 'jd_text'), ('jd_processor', 'jd_text'))
    
    # Соединяем парсеры с матчерами
    system.add_connection(('resume_parser', 'skills'), ('skill_matcher', 'candidate_skills'))
    system.add_connection(('jd_processor', 'required_skills'), ('skill_matcher', 'required_skills'))
    
    system.add_connection(('resume_parser', 'experience'), ('experience_matcher', 'candidate_experience'))
    system.add_connection(('jd_processor', 'required_experience'), ('experience_matcher', 'required_experience'))
    
    system.add_connection(('resume_parser', 'education'), ('education_matcher', 'candidate_education'))
    system.add_connection(('jd_processor', 'required_education'), ('education_matcher', 'required_education'))
    
    # Соединяем матчеры с общим scorer'ом
    system.add_connection(('skill_matcher', 'skill_match_score'), ('overall_scorer', 'skill_score'))
    system.add_connection(('experience_matcher', 'experience_match_score'), ('overall_scorer', 'experience_score'))
    system.add_connection(('education_matcher', 'education_match_score'), ('overall_scorer', 'education_score'))
    
    # Добавляем семантическое сходство
    system.add_connection(('input', 'resume_text'), ('semantic_similarity', 'text1'))
    system.add_connection(('input', 'jd_text'), ('semantic_similarity', 'text2'))
    
    # Выходы системы
    system.add_connection(('overall_scorer', 'overall_score'), ('output', 'overall_score'))
    system.add_connection(('overall_scorer', 'detailed_scores'), ('output', 'detailed_scores'))
    system.add_connection(('semantic_similarity', 'semantic_similarity'), ('output', 'semantic_similarity'))
    system.add_connection(('resume_parser', 'skills'), ('output', 'candidate_skills'))
    system.add_connection(('jd_processor', 'required_skills'), ('output', 'required_skills'))
    
    # Компилируем систему
    system.compile()
    
    return system

# Пример использования
if __name__ == "__main__":
    # Создаем пайплайн
    pipeline = create_recruiter_pipeline()
    
    # Тестовые данные
    resume_text = """
    John Doe, Python Developer
    Skills: Python, Machine Learning, SQL, Docker
    Experience: 5 years of software development
    Education: Master's degree in Computer Science
    """
    
    jd_text = """
    We are looking for a Senior Python Developer with:
    - Strong Python skills
    - Experience with Machine Learning
    - Knowledge of Docker and Kubernetes
    - At least 3 years of experience
    - Bachelor's degree or higher
    """
    
    # Запускаем пайплайн
    result = pipeline(resume_text=resume_text, jd_text=jd_text)
    
    print("Overall Score:", result['overall_score'].item())
    print("Detailed Scores:", result['detailed_scores'])
    print("Semantic Similarity:", result['semantic_similarity'].item())
    print("Candidate Skills:", result['candidate_skills'])
    print("Required Skills:", result['required_skills'])

============================================================
.\arxglue\gluetorch\GlueTorch\setup.py
============================================================
"""
Setup script for TransGlue libraries
"""

from setuptools import setup, find_packages

# Setup for TransGlueCore
setup(
    name="gluetorch",
    version="0.1.0",
    description="WIP",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/transglue",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)



============================================================
.\arxglue\gluetorch\GlueTorch\test_gluetorch.py
============================================================
import torch
import pytest
import sys
import os
from typing import Any, Callable, Optional, Dict, List, Tuple, Union, Set, Type

# Добавляем путь к модулю
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gluetorch import (
    GlueTorch, Component, LinearComponent, MultiHeadAttentionComponent,
    PortSpec, PortType, AggregationStrategy, Connection, SharedMemory,
    MemoryComponent
)

def test_port_spec_validation():
    """Тестирование валидации спецификаций портов"""
    # Создаем спецификацию для тензора определенной формы
    spec = PortSpec('test', PortType.TENSOR, shape=(None, 10), dtype=torch.float32)
    
    # Valid cases
    valid_tensor = torch.randn(5, 10)
    assert spec.validate_value(valid_tensor) == True
    
    # Invalid cases
    invalid_shape = torch.randn(5, 5)  # Wrong shape
    assert spec.validate_value(invalid_shape) == False
    
    invalid_type = [1, 2, 3]  # Not a tensor
    assert spec.validate_value(invalid_type) == False

def test_linear_component():
    """Тестирование линейного компонента"""
    comp = LinearComponent(input_dim=10, output_dim=5, name='linear')
    
    # Test forward pass
    input_tensor = torch.randn(2, 10)
    output = comp(default=input_tensor)
    
    assert 'default' in output
    assert output['default'].shape == (2, 5)

def test_attention_component():
    """Тестирование компонента внимания"""
    comp = MultiHeadAttentionComponent(d_model=12, num_heads=3, name='attention')
    
    # Test forward pass
    query = torch.randn(2, 5, 12)
    key = torch.randn(2, 5, 12)
    value = torch.randn(2, 5, 12)
    
    output = comp(query=query, key=key, value=value)
    
    assert 'output' in output
    assert output['output'].shape == (2, 5, 12)

def test_memory_component():
    """Тестирование компонента памяти"""
    comp = MemoryComponent(keys=['state'], name='memory')
    
    # Test write and read
    input_tensor = torch.randn(3, 5)
    output = comp(write_state=input_tensor)
    
    assert 'read_state' in output
    assert torch.allclose(output['read_state'], input_tensor)

def test_basic_connection():
    """Тестирование соединения двух компонентов"""
    system = GlueTorch()
    
    # Create components
    linear1 = LinearComponent(input_dim=10, output_dim=8, name='linear1')
    linear2 = LinearComponent(input_dim=8, output_dim=5, name='linear2')
    
    # Register components
    system.register_component(linear1)
    system.register_component(linear2)
    
    # Add connections
    system.add_connection(('input', 'data'), ('linear1', 'default'))
    system.add_connection(('linear1', 'default'), ('linear2', 'default'))
    system.add_connection(('linear2', 'default'), ('output', 'result'))
    
    # Compile system
    system.compile()
    
    # Test forward pass
    input_data = torch.randn(3, 10)
    result = system(data=input_data)
    
    assert 'result' in result
    assert result['result'].shape == (3, 5)

def test_shared_memory():
    """Тестирование общей памяти"""
    memory = SharedMemory()
    
    # Test set and get
    test_data = torch.randn(5, 3)
    memory.set('key1', test_data)
    
    retrieved = memory.get('key1')
    assert torch.allclose(retrieved, test_data)
    
    # Test nonexistent key
    assert memory.get('nonexistent') is None
    assert memory.get('nonexistent', 'default') == 'default'

def test_multiple_connections():
    """Тестирование множественных соединений к одному порту"""
    system = GlueTorch()
    
    # Create components
    linear1 = LinearComponent(input_dim=5, output_dim=3, name='linear1')
    linear2 = LinearComponent(input_dim=5, output_dim=3, name='linear2')
    
    # Создаем компонент с портом, поддерживающим агрегацию CONCAT
    class AggregatingComponent(Component):
        def __init__(self, name: str = None):
            super().__init__(name)
            # Порт с агрегацией CONCAT - ожидаем 6 features после конкатенации
            self.input_ports = {
                'default': PortSpec(
                    'default', 
                    PortType.TENSOR, 
                    shape=(None, 6),  # После конкатенации 3+3=6
                    aggregation=AggregationStrategy.CONCAT
                )
            }
            self.output_ports = {
                'default': PortSpec('default', PortType.TENSOR, shape=(None, 6))
            }
            
        def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
            return {'default': default}
    
    aggregator = AggregatingComponent(name='aggregator')
    linear3 = LinearComponent(input_dim=6, output_dim=4, name='linear3')
    
    # Register components
    system.register_component(linear1)
    system.register_component(linear2)
    system.register_component(aggregator)
    system.register_component(linear3)
    
    # Add connections (оба выхода идут к aggregator)
    system.add_connection(('input', 'data1'), ('linear1', 'default'))
    system.add_connection(('input', 'data2'), ('linear2', 'default'))
    system.add_connection(('linear1', 'default'), ('aggregator', 'default'))
    system.add_connection(('linear2', 'default'), ('aggregator', 'default'))
    system.add_connection(('aggregator', 'default'), ('linear3', 'default'))
    system.add_connection(('linear3', 'default'), ('output', 'result'))
    
    # Compile system - здесь произойдет проверка агрегации
    system.compile()
    
    # Test forward pass
    input1 = torch.randn(3, 5)
    input2 = torch.randn(3, 5)
    result = system(data1=input1, data2=input2)
    
    assert 'result' in result
    assert result['result'].shape == (3, 4)

def test_recurrent_connection():
    """Тестирование рекуррентных соединений"""
    system = GlueTorch()
    
    # Create components with correct dimensions
    rnn_cell = LinearComponent(input_dim=5, output_dim=5, name='rnn_cell')
    memory = MemoryComponent(keys=['hidden'], name='memory')
    
    # Изменяем стратегию агрегации на SUM для порта default
    rnn_cell.input_ports['default'].aggregation = AggregationStrategy.SUM
    
    # Register components
    system.register_component(rnn_cell)
    system.register_component(memory)
    
    # Add connections with delay for recurrence
    system.add_connection(('input', 'x'), ('rnn_cell', 'default'))
    system.add_connection(('memory', 'read_hidden'), ('rnn_cell', 'default'), delay=1)
    system.add_connection(('rnn_cell', 'default'), ('memory', 'write_hidden'))
    system.add_connection(('rnn_cell', 'default'), ('output', 'y'))
    
    # Compile system
    system.compile()
    
    # Test forward pass with sequence
    sequence = [torch.randn(2, 5) for _ in range(3)]
    outputs = []
    
    for step, x in enumerate(sequence):
        result = system(x=x)
        outputs.append(result['y'])
        
        # Reset memory between independent sequences
        if step == len(sequence) - 1:
            system.reset()
    
    assert len(outputs) == 3
    assert all(out.shape == (2, 5) for out in outputs)

def test_integration_simple_ai():
    """Интеграционный тест: простая нейросеть для классификации"""
    system = GlueTorch()
    
    # Create components for a simple classifier
    encoder = LinearComponent(input_dim=20, output_dim=10, name='encoder')
    classifier = LinearComponent(input_dim=10, output_dim=3, name='classifier')
    
    # Register components
    system.register_component(encoder)
    system.register_component(classifier)
    
    # Add connections
    system.add_connection(('input', 'features'), ('encoder', 'default'))
    system.add_connection(('encoder', 'default'), ('classifier', 'default'))
    system.add_connection(('classifier', 'default'), ('output', 'logits'))
    
    # Compile system
    system.compile()
    
    # Test with sample data (batch_size=4, features=20)
    features = torch.randn(4, 20)
    result = system(features=features)
    
    assert 'logits' in result
    assert result['logits'].shape == (4, 3)
    
    # Test prediction
    predictions = torch.softmax(result['logits'], dim=1)
    assert predictions.shape == (4, 3)
    assert torch.allclose(predictions.sum(dim=1), torch.ones(4))

def test_aggregation_compatibility():
    """Тестирование проверки совместимости при агрегации"""
    system = GlueTorch()
    
    # Create components with incompatible shapes
    linear1 = LinearComponent(input_dim=5, output_dim=3, name='linear1')  # output: (None, 3)
    linear2 = LinearComponent(input_dim=5, output_dim=4, name='linear2')  # output: (None, 4)
    
    # Компонент с агрегацией CONCAT
    class AggregatingComponent(Component):
        def __init__(self, name: str = None):
            super().__init__(name)
            self.input_ports = {
                'default': PortSpec(
                    'default', 
                    PortType.TENSOR, 
                    shape=(None, 7),  # Ожидаем 7 features (3+4=7)
                    aggregation=AggregationStrategy.CONCAT
                )
            }
            self.output_ports = {
                'default': PortSpec('default', PortType.TENSOR, shape=(None, 7))
            }
            
        def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
            return {'default': default}
    
    aggregator = AggregatingComponent(name='aggregator')
    
    system.register_component(linear1)
    system.register_component(linear2)
    system.register_component(aggregator)
    
    system.add_connection(('input', 'data1'), ('linear1', 'default'))
    system.add_connection(('input', 'data2'), ('linear2', 'default'))
    system.add_connection(('linear1', 'default'), ('aggregator', 'default'))
    system.add_connection(('linear2', 'default'), ('aggregator', 'default'))
    
    # This should work - 3 + 4 = 7
    system.compile()
    
    # Теперь попробуем с несовместимыми формами
    linear3 = LinearComponent(input_dim=5, output_dim=2, name='linear3')  # output: (None, 2)
    system.register_component(linear3)
    system.add_connection(('input', 'data3'), ('linear3', 'default'))
    system.add_connection(('linear3', 'default'), ('aggregator', 'default'))
    
    # This should fail - нельзя конкатенировать (None,3), (None,4) и (None,2)
    # потому что первые размерности должны совпадать
    with pytest.raises(ValueError):
        system.compile()

def test_aggregation_error():
    """Тестирование обработки ошибок при несовместимой агрегации"""
    system = GlueTorch()
    
    # Create components with incompatible shapes for aggregation
    linear1 = LinearComponent(input_dim=5, output_dim=3, name='linear1')  # output: (None, 3)
    linear2 = LinearComponent(input_dim=5, output_dim=4, name='linear2')  # output: (None, 4)
    
    # Компонент с агрегацией CONCAT, но ожидающий другую форму
    class AggregatingComponent(Component):
        def __init__(self, name: str = None):
            super().__init__(name)
            self.input_ports = {
                'default': PortSpec(
                    'default', 
                    PortType.TENSOR, 
                    shape=(None, 5),  # Ожидаем 5 features, но получим 3+4=7
                    aggregation=AggregationStrategy.CONCAT
                )
            }
            self.output_ports = {
                'default': PortSpec('default', PortType.TENSOR, shape=(None, 5))
            }
            
        def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
            return {'default': default}
    
    aggregator = AggregatingComponent(name='aggregator')
    
    system.register_component(linear1)
    system.register_component(linear2)
    system.register_component(aggregator)
    
    system.add_connection(('input', 'data1'), ('linear1', 'default'))
    system.add_connection(('input', 'data2'), ('linear2', 'default'))
    system.add_connection(('linear1', 'default'), ('aggregator', 'default'))
    system.add_connection(('linear2', 'default'), ('aggregator', 'default'))
    
    # This should raise an error due to incompatible aggregation
    with pytest.raises(ValueError):
        system.compile()

def test_error_handling():
    """Тестирование обработки ошибок"""
    system = GlueTorch()
    
    # Create incompatible components
    linear1 = LinearComponent(input_dim=10, output_dim=5, name='linear1')
    linear2 = LinearComponent(input_dim=8, output_dim=3, name='linear2')  # Expects 8 dims
    
    # Изменяем тип агрегации на CUSTOM, который требует строгой проверки
    linear2.input_ports['default'].aggregation = AggregationStrategy.CUSTOM
    
    system.register_component(linear1)
    system.register_component(linear2)
    
    # This should fail - incompatible dimensions
    # Теперь ошибка должна возникать при добавлении соединения, а не при компиляции
    with pytest.raises(ValueError):
        system.add_connection(('linear1', 'default'), ('linear2', 'default'))

if __name__ == "__main__":
    # Run tests
    test_port_spec_validation()
    test_linear_component()
    test_attention_component()
    test_memory_component()
    test_basic_connection()
    test_shared_memory()
    test_multiple_connections()
    test_recurrent_connection()
    test_integration_simple_ai()
    test_error_handling()
    
    print("All tests passed!")

============================================================
.\arxglue\gluetorch\GlueTorch\test_recruiter.py
============================================================
from recruiter_pipeline import create_recruiter_pipeline

def test_recruiter_pipeline():
    print("Testing Recruiter Pipeline...")
    
    # Создаем пайплайн
    pipeline = create_recruiter_pipeline()
    
    # Тестовые данные
    test_cases = [
        {
            'resume': "Python developer with 5 years of experience. Skills: Python, ML, Docker. Education: Master's degree.",
            'jd': "Looking for Python developer with ML experience and Docker knowledge. Minimum 3 years of experience. Bachelor's degree required."
        },
        {
            'resume': "Java developer with 2 years of experience. Skills: Java, Spring. Education: Bachelor's degree.",
            'jd': "Senior Python developer needed with 5+ years of experience. Master's degree preferred."
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        result = pipeline(
            resume_text=test_case['resume'],
            jd_text=test_case['jd']
        )
        
        print(f"Overall Score: {result['overall_score'].item():.3f}")
        print(f"Semantic Similarity: {result['semantic_similarity'].item():.3f}")
        print(f"Candidate Skills: {result['candidate_skills']}")
        print(f"Required Skills: {result['required_skills']}")

if __name__ == "__main__":
    test_recruiter_pipeline()

============================================================
.\arxglue\gluetorch\GlueTorch\transformer.py
============================================================
import torch
import torch.nn as nn
import math
from typing import Dict, Optional
from gluetorch import GlueTorch, Component, PortSpec, PortType, LinearComponent, MultiHeadAttentionComponent

# Позиционное кодирование
class PositionalEncodingComponent(Component):
    def __init__(self, d_model, max_len=5000, name=None):
        super().__init__(name)
        self.d_model = d_model
        
        # Создаем матрицу позиционного кодирования
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        self.input_ports = {
            'default': PortSpec('default', PortType.TENSOR, shape=(None, None, d_model))
        }
        self.output_ports = {
            'default': PortSpec('default', PortType.TENSOR, shape=(None, None, d_model))
        }
        
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq_len = default.size(1)
        return {'default': default + self.pe[:, :seq_len, :]}

# Нормализация слоя
class LayerNormComponent(Component):
    def __init__(self, normalized_shape, eps=1e-5, name=None):
        super().__init__(name)
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.input_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        self.output_ports = {
            'default': PortSpec('default', PortType.TENSOR)
        }
        
    def forward(self, default: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'default': self.norm(default)}

# Блок Трансформера
class TransformerBlockComponent(Component):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1, name=None):
        super().__init__(name)
        self.d_model = d_model
        
        # Создаем компоненты блока
        self.attention = MultiHeadAttentionComponent(d_model, num_heads, name=f'{name}_attention')
        self.norm1 = LayerNormComponent(d_model, name=f'{name}_norm1')
        self.linear1 = LinearComponent(d_model, dim_feedforward, name=f'{name}_linear1')
        self.dropout = nn.Dropout(dropout)
        self.linear2 = LinearComponent(dim_feedforward, d_model, name=f'{name}_linear2')
        self.norm2 = LayerNormComponent(d_model, name=f'{name}_norm2')
        self.relu = nn.ReLU()
        
        # Определяем порты
        self.input_ports = {
            'x': PortSpec('x', PortType.TENSOR, shape=(None, None, d_model)),
            'mask': PortSpec('mask', PortType.TENSOR, shape=(None, None, None), required=False)
        }
        self.output_ports = {
            'output': PortSpec('output', PortType.TENSOR, shape=(None, None, d_model))
        }
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Self-attention
        attn_output = self.attention(query=x, key=x, value=x, mask=mask)['output']
        x = x + attn_output
        x = self.norm1(x)['default']
        
        # Feedforward
        ff_output = self.linear1(x)['default']
        ff_output = self.relu(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)['default']
        
        x = x + ff_output
        x = self.norm2(x)['default']
        
        return {'output': x}

# Полный Трансформер
class TransformerComponent(Component):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes, max_len=5000, name=None):
        super().__init__(name)
        self.d_model = d_model
        
        # Embedding слой
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Позиционное кодирование
        self.pos_encoding = PositionalEncodingComponent(d_model, max_len, name=f'{name}_pos_encoding')
        
        # Блоки Трансформера
        self.blocks = nn.ModuleList([
            TransformerBlockComponent(d_model, num_heads, name=f'{name}_block_{i}')
            for i in range(num_layers)
        ])
        
        # Классификатор
        self.classifier = LinearComponent(d_model, num_classes, name=f'{name}_classifier')
        
        # Определяем порты
        self.input_ports = {
            'input_ids': PortSpec('input_ids', PortType.TENSOR, shape=(None, None)),
            'mask': PortSpec('mask', PortType.TENSOR, shape=(None, None, None), required=False)
        }
        self.output_ports = {
            'logits': PortSpec('logits', PortType.TENSOR, shape=(None, num_classes))
        }
        
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Embedding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        # Позиционное кодирование
        x = self.pos_encoding(x)['default']
        
        # Блоки Трансформера
        for block in self.blocks:
            x = block(x=x, mask=mask)['output']
        
        # Усреднение по последовательности (можно заменить на [CLS] токен)
        x = x.mean(dim=1)
        
        # Классификация
        logits = self.classifier(x)['default']
        
        return {'logits': logits}

# Создаем и тестируем Трансформер
def create_transformer():
    system = GlueTorch()
    
    # Создаем и регистрируем компонент Трансформера
    transformer = TransformerComponent(
        vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_layers=6,
        num_classes=10,
        name='transformer'
    )
    
    system.register_component(transformer)
    
    # Создаем соединения
    system.add_connection(('input', 'input_ids'), ('transformer', 'input_ids'))
    system.add_connection(('input', 'mask'), ('transformer', 'mask'))
    system.add_connection(('transformer', 'logits'), ('output', 'logits'))
    
    # Компилируем систему
    system.compile()
    
    return system

# Тестируем Трансформер
if __name__ == "__main__":
    transformer_net = create_transformer()
    test_input_ids = torch.randint(0, 10000, (16, 32))  # Пакет из 16 последовательностей по 32 токена
    test_mask = torch.ones(16, 32, 32).bool()  # Маска без маскирования
    output = transformer_net(input_ids=test_input_ids, mask=test_mask)
    print(f"Transformer output shape: {output['logits'].shape}")

============================================================
.\arxglue\gluetorch\han\chtgpt.py
============================================================
# chatgpt.py
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

# Увеличенная конфигурация модели (~4M параметров)
class GPTConfig:
    vocab_size = 1000
    d_model = 320      # Увеличили с 256 до 320
    n_heads = 8        # Оставили прежним
    d_ff = 1280        # Увеличили с 1024 до 1280 (4 * d_model)
    n_layers = 5       # Увеличили с 4 до 5
    max_len = 512
    dropout = 0.1
    batch_size = 2

config = GPTConfig()

# Токенизатор (такой же как в hatgpt.py)
class CodeTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        
        self._build_vocab()
        
    def _build_vocab(self):
        self.vocab = self.special_tokens.copy()
        self.reverse_vocab = {v: k for k, v in self.special_tokens.items()}
        
        idx = len(self.vocab)
        for i in range(32, 127):
            char = chr(i)
            self.vocab[char] = idx
            self.reverse_vocab[idx] = char
            idx += 1
            
        for i in range(1040, 1104):
            char = chr(i)
            self.vocab[char] = idx
            self.reverse_vocab[idx] = char
            idx += 1
            
        extra_chars = ['\t', '\n', '\r', ' ', ' ', '→', '←', '↑', '↓', '§', '©', '®', '™', '°', '±', '×', '÷', 'π', '∞', '√', '∆', '∑', '∫']
        for char in extra_chars:
            if char not in self.vocab:
                self.vocab[char] = idx
                self.reverse_vocab[idx] = char
                idx += 1
    
    def encode(self, text):
        tokens = []
        for char in text:
            tokens.append(self.vocab.get(char, self.vocab['<UNK>']))
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

# Инициализируем токенизатор
tokenizer = CodeTokenizer()
config.vocab_size = tokenizer.vocab_size

# Базовые компоненты трансформера (без History Registry)
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        token_embeddings = self.token_emb(x)
        pos_embeddings = self.pos_emb(positions)
        return token_embeddings + pos_embeddings

class MultiHeadAttention(nn.Module):
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
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        return x

# Базовая модель GPT (без History Integration)
class BaseGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = Embeddings(config.vocab_size, config.d_model, config.max_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
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
        x = self.embeddings(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm_final(x)
        return self.output_layer(x)

    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50, top_p=0.9, 
                 repetition_penalty=1.0, do_sample=True):
        self.eval()
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
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

# Датасет (такой же как в hatgpt.py)
class CodeDataset(Dataset):
    def __init__(self, file_path, seq_length, tokenizer):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokens = tokenizer.encode(text)
        
        self.data = []
        
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
    
    model = BaseGPT(config).to(device)
    
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
    }, 'basegpt_model.pth')
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('base_training_curve.png')
    plt.show()

if __name__ == "__main__":
    main()

============================================================
.\arxglue\gluetorch\han\htgpt.py
============================================================
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

============================================================
.\arxglue\gluetorch\han\TransformerXS.py
============================================================
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
< | end of content | >
