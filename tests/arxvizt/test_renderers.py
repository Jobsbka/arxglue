import unittest
import tempfile
import os
from arxviz.exporters import export_architecture_dot
from arxviz.renderers import render_dot_to_png, render_plantuml_to_ascii

class TestRenderers(unittest.TestCase):
    def test_render_dot_to_png(self):
        dot_code = "digraph { a -> b; }"
        with tempfile.TemporaryDirectory() as tmpdir:
            # Указываем имя файла БЕЗ расширения .png
            output_base = os.path.join(tmpdir, "test")
            
            try:
                # Функция вернет полный путь к PNG файлу
                png_file = render_dot_to_png(dot_code, output_base)
                
                # Проверяем что файл существует
                self.assertTrue(os.path.exists(png_file))
                
                # Проверяем что это действительно PNG файл
                self.assertTrue(png_file.endswith('.png'))
                self.assertGreater(os.path.getsize(png_file), 0)
            except RuntimeError as e:
                if "Graphviz not installed" in str(e):
                    self.skipTest("Graphviz not installed")
                elif "Graphviz error" in str(e):
                    self.fail(f"Graphviz error: {e}")
                else:
                    raise

    def test_render_plantuml_to_ascii(self):
        plantuml_code = "@startuml\na -> b: test\n@enduml"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test.txt")
        # Явно указываем URL сервера
            render_plantuml_to_ascii(
                plantuml_code, 
                output_file,
                server_url="http://www.plantuml.com/plantuml"
            )
            self.assertTrue(os.path.exists(output_file))
            with open(output_file) as f:
                content = f.read()
                self.assertNotEqual(content, "")

if __name__ == "__main__":
    unittest.main()