from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum
from blackboard import Blackboard, EventType

class ComponentType(Enum):
    CPU = "CPU"
    GPU = "GPU"
    MOTHERBOARD = "Motherboard"
    RAM = "RAM"
    STORAGE = "Storage"
    PSU = "Power Supply"
    CASE = "Case"
    COOLER = "Cooler"

@dataclass
class CompatibilityRule:
    description: str
    component_types: List[ComponentType]
    validator: Callable[[Dict, Dict], bool]
    severity: str  # "error" or "warning"

class CompatibilityAgent:
    def __init__(self, blackboard: Blackboard, rules_db: Dict[str, Any]):
        """
        :param blackboard: Instancia compartida del blackboard
        :param rules_db: Base de datos de reglas de compatibilidad
        """
        self.blackboard = blackboard
        self.rules_db = rules_db
        self.compatibility_rules = self._load_compatibility_rules()
        
        # Suscribirse a eventos de actualización de componentes
        self.blackboard.subscribe(
            EventType.COMPONENTS_PROPOSED,
            self.check_compatibility
        )

    def _load_compatibility_rules(self) -> List[CompatibilityRule]:
        """Carga las reglas de compatibilidad desde la base de datos"""
        rules = []
        
        # 1. Reglas de Socket CPU-Motherboard
        rules.append(CompatibilityRule(
            description="Socket del CPU debe coincidir con el de la motherboard",
            component_types=[ComponentType.CPU, ComponentType.MOTHERBOARD],
            validator=self._validate_socket_compatibility,
            severity="error"
        ))
        
        # 2. Reglas de RAM-Motherboard
        rules.append(CompatibilityRule(
            description="Tipo de RAM debe ser compatible con la motherboard",
            component_types=[ComponentType.RAM, ComponentType.MOTHERBOARD],
            validator=self._validate_ram_type_compatibility,
            severity="error"
        ))
        
        # 3. Reglas de GPU-Motherboard (PCIe)
        rules.append(CompatibilityRule(
            description="GPU debe ser compatible con slot PCIe de la motherboard",
            component_types=[ComponentType.GPU, ComponentType.MOTHERBOARD],
            validator=self._validate_pcie_compatibility,
            severity="error"
        ))
        
        # 4. Reglas de GPU-Case (tamaño físico)
        rules.append(CompatibilityRule(
            description="GPU debe caber en el gabinete",
            component_types=[ComponentType.GPU, ComponentType.CASE],
            validator=self._validate_gpu_size_compatibility,
            severity="error"
        ))
        
        # 5. Reglas de TDP Cooler-CPU
        rules.append(CompatibilityRule(
            description="Cooler debe soportar TDP del CPU",
            component_types=[ComponentType.COOLER, ComponentType.CPU],
            validator=self._validate_tdp_compatibility,
            severity="warning"
        ))
        
        # Pueden agregarse más reglas aquí...
        
        return rules

    def check_compatibility(self):
        """Verifica la compatibilidad entre todos los componentes propuestos"""
        component_proposals = self.blackboard.get('component_proposals', {})
        if not component_proposals:
            return
        
        issues = []
        
        # Verificar cada regla de compatibilidad
        for rule in self.compatibility_rules:
            # Obtener los componentes relevantes para esta regla
            components_to_check = {}
            for comp_type in rule.component_types:
                if comp_type.value in component_proposals:
                    components_to_check[comp_type] = component_proposals[comp_type.value][0]['metadata']
            
            # Si tenemos todos los componentes necesarios para esta regla
            if len(components_to_check) == len(rule.component_types):
                is_valid = rule.validator(components_to_check, self.rules_db)
                if not is_valid:
                    issues.append({
                        'rule': rule.description,
                        'components': [ct.value for ct in components_to_check.keys()],
                        'severity': rule.severity
                    })
        
        # Actualizar el blackboard con los problemas encontrados
        if issues:
            self.blackboard.update(
                section='compatibility_issues',
                data=issues,
                agent_id='compatibility_agent',
                notify=True
            )
        else:
            self.blackboard.update(
                section='compatibility_issues',
                data=[],
                agent_id='compatibility_agent',
                notify=True
            )

    # Implementaciones de validadores específicos
    def _validate_socket_compatibility(self, components: Dict[ComponentType, Dict], rules_db: Dict) -> bool:
        """Valida que el socket del CPU coincida con el de la motherboard"""
        cpu_socket = components[ComponentType.CPU].get('CPU Socket Type_CPU Socket Type', '').strip()
        mb_socket = components[ComponentType.MOTHERBOARD].get('CPU Socket Type_CPU Socket Type', '').strip()
        
        if not cpu_socket or not mb_socket:
            return False
        
        # Verificar compatibilidad exacta de socket
        if cpu_socket != mb_socket:
            return False
        
        # Verificar compatibilidad adicional de chipset (si está disponible)
        cpu_model = components[ComponentType.CPU].get('Model_Name', '').lower()
        mb_chipset = components[ComponentType.MOTHERBOARD].get('Chipset - Chipset Manufacturer', '').lower()
        
        # Aquí podríamos añadir lógica más compleja usando rules_db
        return True

    def _validate_ram_type_compatibility(self, components: Dict[ComponentType, Dict], rules_db: Dict) -> bool:
        """Valida que el tipo de RAM sea compatible con la motherboard"""
        ram_type = components[ComponentType.RAM].get('Memory - Memory Type', '').upper()
        mb_ram_types = components[ComponentType.MOTHERBOARD].get('Memory - Memory Type', '').upper()
        
        if not ram_type or not mb_ram_types:
            return True  # Asumir compatible si no hay información
        
        # Verificar si el tipo de RAM está en los soportados por la motherboard
        return ram_type in mb_ram_types

    def _validate_pcie_compatibility(self, components: Dict[ComponentType, Dict], rules_db: Dict) -> bool:
        """Valida compatibilidad de slot PCIe entre GPU y motherboard"""
        gpu_interface = components[ComponentType.GPU].get('Interface - InterfaceInterface', '').lower()
        mb_pcie_slots = components[ComponentType.MOTHERBOARD].get('Expansion Slots - PCI Express x16', '0')
        
        # Verificación básica - asumir compatible si la motherboard tiene al menos un slot PCIe x16
        return 'pcie' in gpu_interface and int(mb_pcie_slots) >= 1

    def _validate_gpu_size_compatibility(self, components: Dict[ComponentType, Dict], rules_db: Dict) -> bool:
        """Valida que la GPU quepa en el gabinete"""
        if ComponentType.CASE not in components:
            return True  # No hay gabinete seleccionado, no podemos verificar
        
        gpu_length = components[ComponentType.GPU].get('Form Factor & Dimensions - Max GPU Length', '0 mm')
        case_max_gpu = components[ComponentType.CASE].get('Max GPU Length', '0 mm')
        
        try:
            gpu_length_mm = float(gpu_length.split()[0])
            case_max_mm = float(case_max_gpu.split()[0])
            return gpu_length_mm <= case_max_mm
        except (ValueError, IndexError):
            return True  # Asumir compatible si no podemos parsear las dimensiones

    def _validate_tdp_compatibility(self, components: Dict[ComponentType, Dict], rules_db: Dict) -> bool:
        """Valida que el cooler pueda manejar el TDP del CPU"""
        cpu_tdp = components[ComponentType.CPU].get('Details_Thermal Design PowerThermal Design Power', '0W')
        cooler_tdp = components[ComponentType.COOLER].get('Max TDP Support', '0W')
        
        try:
            cpu_tdp_val = float(cpu_tdp[:-1]) if cpu_tdp.endswith('W') else float(cpu_tdp)
            cooler_tdp_val = float(cooler_tdp[:-1]) if cooler_tdp.endswith('W') else float(cooler_tdp)
            return cooler_tdp_val >= cpu_tdp_val
        except ValueError:
            return True  # Asumir compatible si no podemos parsear los valores

    def get_compatibility_report(self, issues: List[Dict]) -> str:
        """Genera un reporte de compatibilidad"""
        if not issues:
            return "✅ Todos los componentes son compatibles entre sí"
        
        report = ["## Reporte de Compatibilidad", ""]
        
        error_issues = [i for i in issues if i['severity'] == 'error']
        warning_issues = [i for i in issues if i['severity'] == 'warning']
        
        if error_issues:
            report.append("### ❌ Problemas críticos de compatibilidad:")
            for issue in error_issues:
                report.append(f"- {issue['rule']} (Componentes afectados: {', '.join(issue['components'])})")
            report.append("")
        
        if warning_issues:
            report.append("### ⚠ Advertencias de compatibilidad:")
            for issue in warning_issues:
                report.append(f"- {issue['rule']} (Componentes afectados: {', '.join(issue['components'])})")
            report.append("")
        
        report.append("### Recomendaciones:")
        report.append("- Los problemas marcados como ❌ deben resolverse antes de continuar")
        report.append("- Las advertencias ⚠ pueden no ser críticas pero deberían revisarse")
        
        return "\n".join(report)