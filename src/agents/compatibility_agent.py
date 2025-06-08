from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import numpy as np
from enum import Enum

class ComponentType(Enum):
    CPU = "CPU"
    GPU = "GPU"
    MOTHERBOARD = "Motherboard"
    RAM = "RAM"
    PSU = "Power Supply"
    SSD = "SSD"
    HDD = "HDD"
    COOLER = "Cooler"

@dataclass
class HardwareComponent:
    id: str
    type: ComponentType
    specs: Dict[str, any]
    metadata: Dict[str, any] = None

class CompatibilityRule:
    def __init__(
        self, 
        name: str,
        description: str,
        component_types: List[ComponentType],
        verification_fn: Callable,
        severity: str = "error"  # "error"|"warning"|"info"
    ):
        self.name = name
        self.description = description
        self.component_types = component_types
        self.verify = verification_fn
        self.severity = severity

class CompatibilityAgent:
    def __init__(self):
        self.rules = self._initialize_rules()
        self.compatibility_matrix = self._build_compatibility_matrix()

    def _initialize_rules(self) -> List[CompatibilityRule]:
        """Reglas técnicas fundamentales"""
        return [
            CompatibilityRule(
                name="cpu_motherboard_socket",
                description="El socket del CPU debe coincidir con el de la motherboard",
                component_types=[ComponentType.CPU, ComponentType.MOTHERBOARD],
                verification_fn=lambda cpu, mb: cpu.specs['socket'] == mb.specs['socket']
            ),
            CompatibilityRule(
                name="psu_wattage",
                description="La PSU debe tener suficiente wattage para los componentes",
                component_types=[ComponentType.PSU, ComponentType.CPU, ComponentType.GPU],
                verification_fn=self._check_power_supply,
                severity="warning"
            ),
            CompatibilityRule(
                name="ram_motherboard_compatibility",
                description="La RAM debe ser compatible con la motherboard",
                component_types=[ComponentType.RAM, ComponentType.MOTHERBOARD],
                verification_fn=lambda ram, mb: (
                    ram.specs['type'] in mb.specs['supported_memory'] and
                    ram.specs['speed'] <= mb.specs['max_memory_speed']
                )
            ),
            CompatibilityRule(
                name="cooler_tdp",
                description="El cooler debe soportar el TDP del CPU",
                component_types=[ComponentType.COOLER, ComponentType.CPU],
                verification_fn=lambda cooler, cpu: cooler.specs['max_tdp'] >= cpu.specs['tdp']
            )
        ]

    def _build_compatibility_matrix(self) -> Dict[ComponentType, List[ComponentType]]:
        """Matriz de qué componentes requieren verificación cruzada"""
        return {
            ComponentType.CPU: [ComponentType.MOTHERBOARD, ComponentType.COOLER],
            ComponentType.GPU: [ComponentType.PSU],
            ComponentType.RAM: [ComponentType.MOTHERBOARD],
            ComponentType.MOTHERBOARD: [ComponentType.CPU, ComponentType.RAM, ComponentType.SSD, ComponentType.HDD]
        }

    def verify_system(self, components: List[HardwareComponent]) -> Dict:
        """
        Verifica la compatibilidad de todo el sistema
        
        Returns:
            {
                "is_compatible": bool,
                "issues": List[Dict],
                "compatibility_score": float
            }
        """
        issues = []
        compatible_pairs = 0
        total_checks = 0

        # Verificación por pares según matriz de compatibilidad
        for component in components:
            related_types = self.compatibility_matrix.get(component.type, [])
            
            for related in [c for c in components if c.type in related_types]:
                for rule in [r for r in self.rules 
                           if {component.type, related.type} <= set(r.component_types)]:
                    
                    total_checks += 1
                    try:
                        if not rule.verify(component, related):
                            issues.append({
                                "rule": rule.name,
                                "components": [component.id, related.id],
                                "message": f"Incompatibilidad {rule.name}: {component.type} vs {related.type}",
                                "severity": rule.severity
                            })
                        else:
                            compatible_pairs += 1
                    except Exception as e:
                        issues.append({
                            "rule": rule.name,
                            "components": [component.id, related.id],
                            "message": f"Error verificando {rule.name}: {str(e)}",
                            "severity": "error"
                        })

        # Verificación de requisitos del sistema completo
        system_issues = self._check_system_requirements(components)
        issues.extend(system_issues)

        return {
            "is_compatible": len(issues) == 0,
            "issues": issues,
            "compatibility_score": compatible_pairs / total_checks if total_checks > 0 else 1.0
        }

    def _check_power_supply(self, psu: HardwareComponent, *other_components: HardwareComponent) -> bool:
        """Verifica si la PSU puede alimentar todos los componentes"""
        total_power = sum(
            comp.specs.get('tdp', 0) 
            for comp in other_components 
            if hasattr(comp, 'specs')
        )
        
        # Margen del 20% para estabilidad
        required_wattage = total_power * 1.2
        
        # Considerar conectores PCIe para GPU
        gpu_connectors_required = sum(
            1 for comp in other_components 
            if comp.type == ComponentType.GPU and comp.specs.get('pcie_connectors', 0) > 0
        )
        
        return (
            psu.specs['wattage'] >= required_wattage and
            psu.specs['pcie_6+2_pin'] >= gpu_connectors_required
        )

    def _check_system_requirements(self, components: List[HardwareComponent]) -> List[Dict]:
        """Verificaciones a nivel de todo el sistema"""
        issues = []
        
        # Verificar que existan componentes esenciales
        essential_types = {ComponentType.CPU, ComponentType.MOTHERBOARD, ComponentType.PSU}
        present_types = {c.type for c in components}
        missing = essential_types - present_types
        
        if missing:
            issues.append({
                "rule": "missing_essential_components",
                "components": [],
                "message": f"Faltan componentes esenciales: {', '.join([t.value for t in missing])}",
                "severity": "error"
            })
            
        # Verificar espacio físico en gabinete (simplificado)
        if any(c.type == ComponentType.GPU for c in components):
            gpu_length = max(c.specs.get('length_mm', 0) 
                           for c in components 
                           if c.type == ComponentType.GPU)
                           
            if gpu_length > 300:  # Asumiendo gabinete estándar
                issues.append({
                    "rule": "gpu_too_large",
                    "components": [c.id for c in components if c.type == ComponentType.GPU],
                    "message": f"GPU demasiado larga ({gpu_length}mm) para gabinete estándar",
                    "severity": "warning"
                })
                
        return issues

    def add_custom_rule(self, rule: CompatibilityRule):
        """Permite añadir reglas personalizadas dinámicamente"""
        self.rules.append(rule)
        # Actualizar matriz de compatibilidad
        for comp_type in rule.component_types:
            related_types = [t for t in rule.component_types if t != comp_type]
            self.compatibility_matrix.setdefault(comp_type, []).extend(related_types)
            self.compatibility_matrix[comp_type] = list(set(self.compatibility_matrix[comp_type]))