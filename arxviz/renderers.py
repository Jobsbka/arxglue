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