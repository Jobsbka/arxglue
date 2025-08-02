import unittest
import tempfile
import os
from arxviz.exporters import (
    export_architecture_dot,
    export_architecture_plantuml,
    export_trace_plantuml,
    export_logs_text,
    export_architecture_csv,
    export_logs_csv
)

class TestExporters(unittest.TestCase):
    def test_export_architecture_dot(self):
        components = [{"id": "loader", "name": "Data Loader"}]
        connections = [{"source": "loader", "target": "processor"}]
        dot = export_architecture_dot(components, connections)
        self.assertIn('digraph ArxArchitecture', dot)
        self.assertIn('loader [label="Data Loader"]', dot)

    def test_export_architecture_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            components = [{"id": "loader", "type": "function", "description": "Load data"}]
            connections = [{"source": "loader", "target": "processor", "transformer": "parse"}]
            base_path = os.path.join(tmpdir, "test")
            export_architecture_csv(components, connections, base_path)
            
            comp_file = f"{base_path}_components.csv"
            with open(comp_file) as f:
                content = f.read()
                self.assertIn("loader,function,Load data", content)
            
            conn_file = f"{base_path}_connections.csv"
            with open(conn_file) as f:
                content = f.read()
                self.assertIn("loader,processor,parse", content)

    def test_export_trace_plantuml(self):
        trace = [{
            "component": "loader",
            "input": "test.csv",
             "error": "File not found"
        }]
        plantuml = export_trace_plantuml(trace)
        self.assertIn('@startuml', plantuml)
        self.assertIn('note right of loader: File not found', plantuml)

    def test_export_logs_text(self):
        logs = ["[INFO] Starting", "[ERROR] Failed"]
        text = export_logs_text(logs)
        self.assertEqual(text, "[INFO] Starting\n[ERROR] Failed")

    def test_export_architecture_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            components = [{"id": "loader", "type": "function", "description": "Load data"}]
            connections = [{"source": "loader", "target": "processor", "transformer": "parse"}]
            export_architecture_csv(components, connections, f"{tmpdir}/test")
            
            with open(f"{tmpdir}/test_components.csv") as f:
                content = f.read()
                self.assertIn("loader,function,Load data", content)
            
            with open(f"{tmpdir}/test_connections.csv") as f:
                content = f.read()
                self.assertIn("loader,processor,parse", content)

    def test_export_logs_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logs = ["[INFO] 2023-01-01 Starting", "[ERROR] 2023-01-01 Failed"]
            log_file = os.path.join(tmpdir, "logs.csv")
        
        # Вызываем функцию экспорта
            export_logs_csv(logs, log_file)
        
        # Проверяем существование файла перед открытием
            self.assertTrue(os.path.exists(log_file), f"File not found: {log_file}")
        
            with open(log_file) as f:
                content = f.read()
            # Обновленные проверки - новый формат заголовка
                self.assertIn("level,message", content)
            # Проверяем первую запись
                self.assertIn("INFO,2023-01-01 Starting", content)
            # Проверяем вторую запись
                self.assertIn("ERROR,2023-01-01 Failed", content)
    
if __name__ == "__main__":
    unittest.main()