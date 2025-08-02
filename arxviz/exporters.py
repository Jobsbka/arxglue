import csv
from typing import Any, Dict, List, Tuple

def export_architecture_dot(
    components: List[Dict[str, Any]],
    connections: List[Dict[str, Any]]
) -> str:
    """
    Generate Graphviz DOT diagram from components and connections
    
    :param components: List of component dicts with 'id' and 'name'
    :param connections: List of connection dicts with 'source', 'target', 'transformer'
    :return: DOT diagram code
    """
    dot = ["digraph ArxArchitecture {"]
    dot.append("    rankdir=LR;")
    dot.append("    node [shape=box, style=rounded];")
    
    # Add components
    for comp in components:
        dot.append(f'    {comp["id"]} [label="{comp["name"]}"];')
    
    # Add connections
    for conn in connections:
        label = f' [label="{conn["transformer"]}"]' if conn.get("transformer") else ""
        dot.append(f'    {conn["source"]} -> {conn["target"]}{label};')
    
    dot.append("}")
    return "\n".join(dot)

def export_architecture_plantuml(
    components: List[Dict[str, Any]],
    connections: List[Dict[str, Any]]
) -> str:
    """
    Generate PlantUML diagram from components and connections
    
    :param components: List of component dicts
    :param connections: List of connection dicts
    :return: PlantUML diagram code
    """
    plantuml = ["@startuml"]
    plantuml.append("left to right direction")
    
    # Add components
    for comp in components:
        plantuml.append(f'component "{comp["name"]}" as {comp["id"]}')
    
    # Add connections
    for conn in connections:
        label = f' : {conn["transformer"]}' if conn.get("transformer") else ""
        plantuml.append(f'{conn["source"]} --> {conn["target"]}{label}')
    
    plantuml.append("@enduml")
    return "\n".join(plantuml)

def export_trace_plantuml(trace: List[Dict[str, Any]]) -> str:
    """
    Generate PlantUML sequence diagram from execution trace
    
    :param trace: Trace data from ErrorContext
    :return: PlantUML sequence diagram code
    """
    if not trace:
        return "@startuml\nnote: Empty trace\n@enduml"
    
    plantuml = ["@startuml"]
    plantuml.append("skinparam responseMessageBelowArrow true")
    
    # Collect participants
    participants = {entry["component"] for entry in trace}
    for p in participants:
        plantuml.append(f'participant "{p}" as {p}')
    
    # Add interactions
    for i, entry in enumerate(trace):
        comp = entry["component"]
        input_data = str(entry["input"])[:30] + "..." if len(str(entry["input"])) > 30 else entry["input"]
        
        if i > 0 and trace[i-1]["component"] != comp:
            plantuml.append(f'{trace[i-1]["component"]} -> {comp}: {input_data}')
        
        if entry.get("error"):
            plantuml.append(f'group Error')
            plantuml.append(f'note right of {comp}: {entry["error"]}')
            plantuml.append('end group')
    
    plantuml.append("@enduml")
    return "\n".join(plantuml)

def export_logs_text(logs: List[str]) -> str:
    """Convert logs to plain text format"""
    return "\n".join(logs)

def export_architecture_csv(
    components: List[Dict[str, Any]],
    connections: List[Dict[str, Any]],
    base_name: str
):
    """
    Export architecture to CSV files
    
    :param components: List of component dicts
    :param connections: List of connection dicts
    :param base_name: Base filename
    """
    # Components CSV
    with open(f"{base_name}_components.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "type", "description"])
        writer.writeheader()
        writer.writerows(components)
    
    # Connections CSV
    with open(f"{base_name}_connections.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "transformer"])
        writer.writeheader()
        writer.writerows(connections)

def export_trace_mermaid(trace: List[Dict[str, Any]]) -> str:
    """
    Generate sequence diagram from execution trace
    
    :param trace: Trace data from ErrorContext
    :return: Mermaid sequence diagram code
    """
    diagram = [
        "sequenceDiagram",
        "    autonumber"
    ]
    
    for entry in trace:
        comp = entry["component"]
        input_data = str(entry["input"])[:30] + "..." if len(str(entry["input"])) > 30 else entry["input"]
        output_data = str(entry.get("output", ""))[:30] + "..." if entry.get("output") else ""
        
        diagram.append(f"    participant {comp}")
        
        if "calls" in entry:
            for call in entry["calls"]:
                diagram.append(f"    {comp}->>{call['component']}: {call['input']}")
        
        if output_data:
            diagram.append(f"    activate {comp}")
            diagram.append(f"    {comp}-->>Output: {output_data}")
            diagram.append(f"    deactivate {comp}")
        
        if entry.get("error"):
            diagram.append(f"    Note right of {comp}: ERROR: {entry['error']}")
    
    return "\n".join(diagram)

def export_logs_csv(logs: List[str], filename: str):
    """Export logs to CSV with improved parsing"""
    parsed_logs = []
    for entry in logs:
        # Улучшенный парсинг логов
        if entry.startswith("[") and "]" in entry:
            # Находим конец уровня
            end_index = entry.index("]")
            level = entry[1:end_index]
            message = entry[end_index+1:].strip()
            parsed_logs.append({
                "level": level,
                "message": message
            })
        else:
            # Для логов без стандартного формата
            parsed_logs.append({
                "level": "INFO",
                "message": entry
            })
    
    # Запись в файл
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "message"])
        writer.writeheader()
        writer.writerows(parsed_logs)
