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