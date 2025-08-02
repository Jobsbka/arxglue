import unittest
import sqlite3
import tempfile
import os
import json
from arxviz.exporters import export_architecture_dot
from arxviz.importers import parse_dot_architecture
import pytest

class TestExternalIntegration(unittest.TestCase):
    def test_sqlite_storage(self):
        # Create test architecture
        components = [{"id": "loader", "name": "Data Loader"}]
        connections = [{"source": "loader", "target": "processor"}]
        dot_code = export_architecture_dot(components, connections)
        
        # Store in SQLite
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            db_path = tmp.name
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE architectures
                     (id INTEGER PRIMARY KEY, name TEXT, dot_code TEXT, metadata TEXT)''')
        c.execute("INSERT INTO architectures (name, dot_code, metadata) VALUES (?, ?, ?)",
                 ("Test Architecture", dot_code, json.dumps({"version": "1.0"})))
        conn.commit()
        
        # Retrieve from SQLite
        c.execute("SELECT dot_code, metadata FROM architectures WHERE name=?", ("Test Architecture",))
        row = c.fetchone()
        self.assertIsNotNone(row)
        
        loaded_dot, metadata = row
        self.assertEqual(loaded_dot, dot_code)
        self.assertEqual(json.loads(metadata)["version"], "1.0")
        
        # Parse and validate
        parsed_components, parsed_connections = parse_dot_architecture(loaded_dot)
        self.assertEqual(len(parsed_components), 1)
        self.assertEqual(parsed_components[0]["name"], "Data Loader")
        
        conn.close()
        os.unlink(db_path)
    
    def test_file_storage(self):
        # Create test architecture
        components = [{"id": "loader", "name": "Data Loader"}]
        connections = [{"source": "loader", "target": "processor"}]
        dot_code = export_architecture_dot(components, connections)
        
        # Save to file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(dot_code.encode())
            file_path = tmp.name
        
        # Load from file
        with open(file_path) as f:
            loaded_dot = f.read()
            self.assertEqual(loaded_dot, dot_code)
        
        # Parse and validate
        parsed_components, parsed_connections = parse_dot_architecture(loaded_dot)
        self.assertEqual(len(parsed_components), 1)
        self.assertEqual(parsed_components[0]["name"], "Data Loader")
        
        os.unlink(file_path)
   
     
    def test_graphviz_rendering(self):
        dot_code = "digraph { a -> b; }"
        try:
            from graphviz import Source
            src = Source(dot_code)
            png = src.pipe(format='png')
            self.assertIsInstance(png, bytes)
            self.assertTrue(len(png) > 0)
        except ImportError:
            # Skip if Graphviz not installed
            pass
    
    def test_plantuml_integration(self):
        plantuml_code = "@startuml\na -> b: test\n@enduml"
        try:
            import plantuml
        # Явно указываем URL сервера
            server = plantuml.PlantUML("http://www.plantuml.com/plantuml")
            url = server.get_url(plantuml_code)
            self.assertTrue(url.startswith("http"))
        except ImportError:
            self.skipTest("PlantUML not installed")
        except Exception as e:
            if "Failed to contact" in str(e):
                self.skipTest("PlantUML server not available")
            else:
                raise

if __name__ == "__main__":
    unittest.main()