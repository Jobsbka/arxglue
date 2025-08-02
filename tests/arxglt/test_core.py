import unittest
from arxglue import connect, execute_linear, ContextProtocol

class Testarxglue(unittest.TestCase):
    def test_component_definition(self):
        def sample_component(x): return x * 2
        self.assertEqual(sample_component(3), 6)
    
    def test_connect_basic(self):
        def a(x): return x
        def b(x): return x
        
        conn = connect(a, b)
        self.assertEqual(conn[0], a)
        self.assertEqual(conn[1], b)
        self.assertIsNone(conn[2])
    
    def test_group_connection(self):
        def a(x): return x
        def b(x): return x
        def c(x): return x
        
        conn = connect((a, b), c)
        self.assertEqual(len(conn[0]), 2)
        self.assertEqual(conn[1], c)
    
    def test_transformer(self):
        def a(x): return x
        def b(x): return x
        transformer = lambda x: x.upper()
        
        conn = connect(a, b, transformer)
        self.assertEqual(conn[2]("test"), "TEST")
    
    def test_linear_execution(self):
        pipeline = [
            lambda x: x + 1,
            lambda x: x * 2,
            lambda x: f"Result: {x}"
        ]
        result = execute_linear(pipeline, 3)
        self.assertEqual(result, "Result: 8")
    
    def test_context_protocol(self):
        class CustomContext(ContextProtocol):
            def add_meta(self, value):
                self.state['meta'] = value
        
        ctx = CustomContext("input")
        ctx.output = "processed"
        ctx.add_meta("info")
        
        self.assertEqual(ctx.input, "input")
        self.assertEqual(ctx.output, "processed")
        self.assertEqual(ctx.state['meta'], "info")

if __name__ == "__main__":
    unittest.main()