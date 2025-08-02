import unittest
from arxglue import connect, Component
from arxviz.exporters import export_architecture_dot
from arxviz.importers import parse_dot_architecture, generate_code_skeleton

class TestArxGlueIntegration(unittest.TestCase):
    def test_component_generation(self):
        # Define components using ArxGlue
        def loader(data): return data
        class Processor(Component):
            def __call__(self, data): return data
        def saver(data): return data
    
        # Create connections
        connections = [
            connect(loader, Processor),
            connect(Processor, saver, transformer="format_output")
        ]
    
        # Export architecture
        components = [
            {"id": "loader", "name": "loader", "type": "function"},
            {"id": "Processor", "name": "Processor", "type": "class"},
            {"id": "saver", "name": "saver", "type": "function"}
        ]
        dot_code = export_architecture_dot(components, [
            {"source": "loader", "target": "Processor"},
            {"source": "Processor", "target": "saver", "transformer": "format_output"}
        ])
    
        # Generate code from architecture
        parsed_components, parsed_connections = parse_dot_architecture(dot_code)
    
        # Устанавливаем тип для Processor как 'class'
        for comp in parsed_components:
             if comp["name"] == "Processor":
                comp["type"] = "class"
    
        code = generate_code_skeleton(parsed_components, parsed_connections)
    
        # Validate generated code
        self.assertIn("def loader(data):", code)
        self.assertIn("class Processor(Component):", code)
        self.assertIn("def format_output(data):", code)
        self.assertIn("connect(loader, Processor)", code)
        self.assertIn('connect(Processor, saver, transformer=format_output)', code)

if __name__ == "__main__":
    unittest.main()