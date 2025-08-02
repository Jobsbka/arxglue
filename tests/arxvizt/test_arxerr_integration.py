import unittest
import tempfile
import os
from arxerr.context import ErrorContext
from arxerr.decorators import traced, logged
from arxviz.exporters import export_trace_plantuml, export_logs_csv

@traced
@logged
def sample_component(ctx, data):
    ctx.log("Processing data")
    if data == "error":
        raise ValueError("Test error")
    return data.upper()

class TestArxErrIntegration(unittest.TestCase):
    def test_trace_visualization(self):
        ctx = ErrorContext("test")
        sample_component(ctx, "hello")
        
        plantuml = export_trace_plantuml(ctx.trace)
        # Обновленные проверки
        self.assertIn('@startuml', plantuml)
        self.assertIn('participant "sample_component"', plantuml)
    
    def test_error_trace_visualization(self):
        ctx = ErrorContext("test")
        with self.assertRaises(ValueError):
            sample_component(ctx, "error")
        
        plantuml = export_trace_plantuml(ctx.trace)
        # Обновленная проверка - убрали "ERROR:"
        self.assertIn('@startuml', plantuml)
        self.assertIn('participant "sample_component"', plantuml)
        self.assertIn('note right of sample_component: Test error', plantuml)
    
    def test_logs_export(self):
        ctx = ErrorContext("test")
        sample_component(ctx, "hello")
    
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "logs.csv")
            export_logs_csv(ctx.logs, log_file)
        
            self.assertTrue(os.path.exists(log_file))
            with open(log_file) as f:
                content = f.read()
                # Проверяем общее содержимое
                self.assertIn("Entering sample_component", content)
                self.assertIn("Completed sample_component", content)
                # Проверяем формат
                self.assertIn("INFO,Entering sample_component", content)
                self.assertIn("INFO,Completed sample_component", content)

if __name__ == "__main__":
    unittest.main()
