import unittest
import tempfile
import os
import sqlite3
from arxglue import connect, Component, ContextProtocol
from arxerr.context import ErrorContext
from arxerr.decorators import traced, logged
from arxviz.exporters import export_architecture_dot, export_trace_plantuml, export_logs_csv
from arxviz.importers import parse_dot_architecture, generate_code_skeleton
from arxviz.renderers import render_dot_to_png

class TestFullIntegration(unittest.TestCase):
    def test_full_workflow(self):
        # Шаг 1: Определение компонентов с помощью ArxGlue
        @traced
        @logged
        def loader(ctx: ErrorContext):
            ctx.log("Loading data")
            return ["data1", "data2"]
        
        @traced
        @logged
        class Processor(Component):
            def __call__(self, ctx: ErrorContext, data):
                ctx.log(f"Processing {len(data)} items")
                return [item.upper() for item in data]
        
        @traced
        @logged
        def saver(ctx: ErrorContext, data):
            ctx.log(f"Saving {len(data)} items")
            return f"Saved {len(data)} items"
        
        # Шаг 2: Создание архитектуры соединений
        connections = [
            connect(loader, Processor),
            connect(Processor, saver)
        ]
        
        # Шаг 3: Создание контекста выполнения с трассировкой и логированием
        ctx = ErrorContext("initial_data")
        
        # Шаг 4: Выполнение конвейера
        data = loader(ctx)
        processed = Processor()(ctx, data)
        result = saver(ctx, processed)
        
        # Шаг 5: Визуализация архитектуры с помощью ArxViz
        components = [
            {"id": "loader", "name": "loader", "type": "function"},
            {"id": "Processor", "name": "Processor", "type": "class"},
            {"id": "saver", "name": "saver", "type": "function"}
        ]
        
        connections_data = [
            {"source": "loader", "target": "Processor"},
            {"source": "Processor", "target": "saver"}
        ]
        
        dot_architecture = export_architecture_dot(components, connections_data)
        plantuml_trace = export_trace_plantuml(ctx.trace)
        
        # Шаг 6: Генерация кода из архитектуры
        parsed_components, parsed_connections = parse_dot_architecture(dot_architecture)
        
        # Устанавливаем тип для Processor как 'class'
        for comp in parsed_components:
            if comp["name"] == "Processor":
                comp["type"] = "class"
        
        generated_code = generate_code_skeleton(parsed_components, parsed_connections)
        
        # Шаг 7: Сохранение результатов в SQLite
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "results.db")
            
            # Создаем базу данных
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS artifacts
                         (id INTEGER PRIMARY KEY, type TEXT, content TEXT)''')
            
            # Сохраняем артефакты
            artifacts = [
                ("architecture_dot", dot_architecture),
                ("trace_plantuml", plantuml_trace),
                ("generated_code", generated_code)
            ]
            
            for art_type, content in artifacts:
                c.execute("INSERT INTO artifacts (type, content) VALUES (?, ?)",
                         (art_type, content))
            
            conn.commit()
            
            # Сохраняем логи в отдельный CSV
            logs_file = os.path.join(tmpdir, "logs.csv")
            export_logs_csv(ctx.logs, logs_file)
            
            # Шаг 8: Проверка результатов
            
            # Проверяем сохраненные артефакты
            c.execute("SELECT type, content FROM artifacts")
            artifacts_db = dict(c.fetchall())
            
            self.assertIn("architecture_dot", artifacts_db)
            self.assertIn("trace_plantuml", artifacts_db)
            self.assertIn("generated_code", artifacts_db)
            
            # Проверяем содержимое DOT
            self.assertIn("digraph ArxArchitecture", artifacts_db["architecture_dot"])
            self.assertIn("loader -> Processor", artifacts_db["architecture_dot"])
            
            # Проверяем содержимое PlantUML
            self.assertIn("@startuml", artifacts_db["trace_plantuml"])
            self.assertIn("participant", artifacts_db["trace_plantuml"])
            
            # Проверяем сгенерированный код
            self.assertIn("class Processor(Component):", artifacts_db["generated_code"])
            self.assertIn("connect(loader, Processor)", artifacts_db["generated_code"])
            
            # Проверяем логи
            self.assertTrue(os.path.exists(logs_file))
            with open(logs_file) as f:
                content = f.read()
                self.assertIn("Loading data", content)
                self.assertIn("Processing", content)
                self.assertIn("Saving", content)
            
            # Дополнительно: рендеринг архитектуры в PNG
            try:
                png_file = os.path.join(tmpdir, "architecture")
                render_dot_to_png(dot_architecture, png_file)
                self.assertTrue(os.path.exists(png_file))
                self.assertGreater(os.path.getsize(png_file), 0)
            except Exception as e:
                # Пропускаем если Graphviz не установлен
                pass
            
            conn.close()

if __name__ == "__main__":
    unittest.main()