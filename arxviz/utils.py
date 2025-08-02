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