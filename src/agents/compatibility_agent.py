from typing import Dict, List, Any, Tuple, Set
from dataclasses import dataclass
from blackboard import *
from enum import Enum
import re

class ComponentType(Enum):
    CPU = "CPU"
    GPU = "GPU"
    MOTHERBOARD = "Motherboard"
    RAM = "RAM"
    #STORAGE = "Storage"
    SSD = "SSD"
    HDD = "HDD"
    PSU = "Power Supply"
    CASE = "Case"
    COOLER = "Cooler"

@dataclass
class ComponentInfo:
    type: ComponentType
    model_name: str
    key_features: Dict[str, Any]
    full_metadata: Dict[str, Any]

@dataclass
class CompatibilityIssue:
    component_a: ComponentInfo
    component_b: ComponentInfo
    rule: str
    reason: str
    severity: str  # "critical", "warning", "info"

class CompatibilityAgent:
    def __init__(self, blackboard: Blackboard):
        self.blackboard = blackboard
        self.compatibility_rules = self._load_compatibility_rules()
        
        # Suscribirse a eventos de actualización de componentes
        self.blackboard.subscribe(
            EventType.TRIGGER_COMPATIBILITY,
            self.check_compatibility
        )

    def _load_compatibility_rules(self) -> Dict[Tuple[ComponentType, ComponentType], List[Callable]]:
        """Carga las reglas de compatibilidad entre pares de componentes"""
        rules = {}
        
        # Reglas CPU-Motherboard
        rules[(ComponentType.CPU, ComponentType.MOTHERBOARD)] = [
            self._validate_socket_compatibility,
            self._validate_chipset_compatibility
        ]
        
        # Reglas CPU-Cooler
        rules[(ComponentType.CPU, ComponentType.COOLER)] = [
            self._validate_tdp_compatibility,
            self._validate_socket_support
        ]
        
        # Reglas GPU-Motherboard
        rules[(ComponentType.GPU, ComponentType.MOTHERBOARD)] = [
            self._validate_pcie_compatibility
        ]
        
        # Reglas GPU-Case
        rules[(ComponentType.GPU, ComponentType.CASE)] = [
            self._validate_size_compatibility
        ]
        
        # Reglas RAM-Motherboard
        rules[(ComponentType.RAM, ComponentType.MOTHERBOARD)] = [
            self._validate_ram_type_compatibility,
            self._validate_ram_speed_compatibility
        ]
        
        # Reglas PSU-GPU
        rules[(ComponentType.PSU, ComponentType.GPU)] = [
            self._validate_power_compatibility
        ]
        
        return rules

    def check_compatibility(self):
        """Verifica la compatibilidad entre todos los componentes propuestos"""
        component_proposals = self.blackboard.get_consolidated_components() or {}
        
        if not component_proposals:
            return
        
        # Extraer información estructurada de los componentes
        components = self._extract_component_info(component_proposals)
        
        # Verificar todas las combinaciones posibles de componentes
        issues = []
        component_types = list(components.keys())
        
        for i, type_a in enumerate(component_types):
            for type_b in component_types[i+1:]:
                # Verificar si hay reglas para este par
                rule_key = (type_a, type_b)
                reverse_key = (type_b, type_a)
                
                rules = self.compatibility_rules.get(rule_key, [])
                if not rules:
                    rules = self.compatibility_rules.get(reverse_key, []) 
                
                for component_a in components[type_a]:
                    for component_b in components[type_b]:
                        for rule_func in rules:
                            is_compatible, reason = rule_func(component_a, component_b)
                            
                            if not is_compatible:
                                severity = "critical" if "socket" in reason.lower() else "warning"
                                issues.append(CompatibilityIssue(
                                    component_a=component_a,
                                    component_b=component_b,
                                    rule=rule_func.__name__,
                                    reason=reason,
                                    severity=severity
                                ))
        
        # Actualizar el blackboard con los problemas encontrados
        self.blackboard.update(
            section='compatibility_issues',
            data=issues,
            agent_id='compatibility_agent',
            notify=True
        )

    def _extract_component_info(self, proposals: Dict[str, List[Dict]]) -> Dict[ComponentType, List[ComponentInfo]]:
        """Convierte las propuestas en una estructura más manejable"""
        components = {}
        
        for comp_type, proposals_list in proposals.items():
            try:
                enum_type = ComponentType(comp_type)
            except ValueError:
                continue  # Saltar tipos no reconocidos
            
            components[enum_type] = []
            
            for proposal in proposals_list:
                metadata = proposal['metadata']
                model_name = metadata.get('Model_Name', 'Unknown')
                
                # Extraer características clave según el tipo de componente
                key_features = {}
                if enum_type == ComponentType.CPU:
                    key_features = {
                        'socket': metadata.get('Details_CPU Socket TypeCPU Socket Type', ''),
                        'generation': metadata.get('Model_Series',''),
                        'tdp': metadata.get('Details_Thermal Design PowerThermal Design Power', ''),
                        'ram_type': metadata.get('Details_Memory Types', '')
                    }
                elif enum_type == ComponentType.MOTHERBOARD:
                    key_features = {
                        'socket': metadata.get('Supported CPU_CPU Socket TypeCPU Socket Type', ''),
                        'max_ram': metadata.get('Memory_Maximum Memory Supported', ''),
                        'ram_type_spped': metadata.get('Memory_Memory Standard', ''),
                        'ram_slots': metadata.get('Memory_Number of Memory Slots', ''),
                        'pcie_slots': metadata.get('Expansion Slots_PCI Express 5.0 x16', ''),
                        'pcie_slots2': metadata.get('Memory_Buffer Supported', ''),
                        'supported_gpu': metadata.get('Supported CPU_CPU Type', '')
                    }
                elif enum_type == ComponentType.GPU:
                    key_features = {
                        'length': metadata.get('Form Factor & Dimensions - Max GPU Length', ''),
                        'power': metadata.get('Details - Recommended PSU Wattage', ''),
                        'interface': metadata.get('Interface - InterfaceInterface', '')
                    }
                elif enum_type == ComponentType.SSD:
                    key_features = {
                        'form_factor': metadata.get('Details_Form FactorForm Factor', ''),
                        'protocol': metadata.get('Details_Protocol', '')
                    }
                    
                elif enum_type == ComponentType.HDD:
                    key_features = {
                        
                    }
                    
                elif enum_type == ComponentType.RAM:
                    key_features = {
                        'ram_type_spped': metadata.get('Details_SpeedSpeed', '')
                    }
                
                elif enum_type == ComponentType.CASE:
                    key_features = {
                        'max_gpu_length': metadata.get('Dimensions & Weight_Max GPU Length', '')
                    }
                    
                components[enum_type].append(ComponentInfo(
                    type=enum_type,
                    model_name=model_name,
                    key_features=key_features,
                    full_metadata=metadata
                ))
        
        return components

    # Implementaciones de validadores específicos
    def _validate_socket_compatibility(self, mobo: ComponentInfo, cpu: ComponentInfo) -> Tuple[bool, str]:
        """Valida que el socket del CPU coincida con el de la motherboard"""
        cpu_socket = cpu.key_features.get('socket', '').strip()
        mobo_socket = mobo.key_features.get('socket', '').strip()
        
        cpu_socket = cpu_socket[len('Socket '):] if cpu_socket.startswith('Socket ') else cpu_socket
        
        if not cpu_socket or not mobo_socket:
            return False, "Información de socket no disponible"
        
        if cpu_socket != mobo_socket:
            return False, f"Socket incompatible: CPU ({cpu_socket}) vs Motherboard ({mobo_socket})"
        
        return True, "Sockets compatibles"

    def _validate_chipset_compatibility(self, mobo: ComponentInfo, cpu: ComponentInfo) -> Tuple[bool, str]:
        """Valida compatibilidad de chipset (ej: Z790 con Intel 13th/14th gen)"""
        # Implementación simplificada - en una implementación real usarías una DB de compatibilidad
        cpu_model = cpu.key_features.get('generation', '').lower()
        mobo_chipset = mobo.key_features.get('supported_gpu', '').lower()
        
        # print(cpu_model)
        # print(mobo_chipset)
        
        if '12th' in cpu_model:
            if '12th' in mobo_chipset:
                return True, "Chipset compatible"
            return False, f"Chipset no compatible con CPU {cpu.model_name}"
        
        if '13th' in cpu_model:
            if '13th' in mobo_chipset:
                return True, "Chipset compatible"
            return False, f"Chipset no compatible con CPU {cpu.model_name}"
        
        if '14th' in cpu_model:
            if '14th' in mobo_chipset:
                return True, "Chipset compatible"
            return False, f"Chipset no compatible con CPU {cpu.model_name}"
        
        if '7000' in cpu_model:
            if '7000' in mobo_chipset:
                return True, "Chipset compatible"
            return False, f"Chipset no compatible con CPU {cpu.model_name}"
        
        if '8000' in cpu_model:
            if '8000' in mobo_chipset:
                return True, "Chipset compatible"
            return False, f"Chipset no compatible con CPU {cpu.model_name}"

        if '9000' in cpu_model:
            if '9000' in mobo_chipset:
                return True, "Chipset compatible"
            return False, f"Chipset no compatible con CPU {cpu.model_name}"
        
        return True, "Compatibilidad de chipset asumida"

    def _validate_pcie_compatibility(self, mobo: ComponentInfo, gpu: ComponentInfo) -> Tuple[bool, str]:
        """Valida compatibilidad de slot PCIe entre GPU y motherboard"""
        gpu_interface = gpu.key_features.get('interface', '').lower()
        mobo_pcie_slots = mobo.key_features.get('pcie_slots', '').lower()
        mobo_pcie_slots2 = mobo.key_features.get('pcie_slots2', '').lower()
        
        if '4.0' in gpu_interface:
            if '4.0 x16' in mobo_pcie_slots2 or '5.0 x16' in mobo_pcie_slots:
                return True, "PCIe compatible"
        
        if '5.0' in gpu_interface:
            if '5.0 x16' in mobo_pcie_slots:
               return True, "PCIe compatible"
        
        return False, "PCIe no compatible"

    def _validate_size_compatibility(self, gpu: ComponentInfo, case: ComponentInfo) -> Tuple[bool, str]:
        """Valida que la GPU quepa en el gabinete"""
        gpu_length = gpu.key_features.get('length', '')
        case_max_gpu = case.key_features.get('max_gpu_length', '')
        
        try:
            # Extraer valores numéricos (asumiendo formato "300 mm")
            gpu_value = float(re.search(r'[\d.]+', gpu_length).group())
            case_value = float(re.search(r'[\d.]+', case_max_gpu).group())
            
            print(gpu_value)
            print(case_value)
            
            # Comparar valores (asumiendo misma unidad)
            if gpu_value > case_value:
                return False, f"GPU ({gpu_length}) más grande que espacio disponible en gabinete ({case_max_gpu})"
            
        except (AttributeError, ValueError):
            return True, "No se pudieron verificar dimensiones"

        return True, "Dimensiones compatibles"

    def _validate_ram_type_compatibility(self, mobo: ComponentInfo, ram: ComponentInfo) -> Tuple[bool, str]:
        """Valida que el tipo de RAM sea compatible con la motherboard"""
        ram_type = ram.key_features.get('ram_type_spped', '').upper()
        mobo_ram_types = mobo.key_features.get('ram_type_spped', '').upper()
        
        if not ram_type or not mobo_ram_types:
            return False, "Información de RAM no disponible"
        
        if 'DDR4' in ram_type and 'DDR4' in mobo_ram_types:
            return True, "Tipo de RAM compatible (DDR4)"
        
        if 'DDR5' in ram_type and 'DDR5' in mobo_ram_types:
            return True, "Tipo de RAM compatible (DDR5)"           
        
        return True, "Tipo de RAM compatible"

    def _validate_ram_speed_compatibility(self, mobo: ComponentInfo, ram: ComponentInfo) -> Tuple[bool, str]:
        """Valida que la velocidad de RAM sea compatible con la motherboard"""
        ram_speed = ram.key_features.get('ram_type_spped', '')
        mobo_speed = mobo.key_features.get('ram_type_spped', '')
     
        ram_speed_match = re.search(r'\b\d{4}\b', ram_speed)
        if not ram_speed_match:  return False, "No se encontró velocidad RAM válida"
        
        ram_speed = ram_speed_match.group()
        
        if ram_speed in mobo_speed: 
            return True, "Velocidad RAM compatible"
        
        return False, "Velocidad de RAM no compatible"

    def _validate_tdp_compatibility(self, cpu: ComponentInfo, cooler: ComponentInfo) -> Tuple[bool, str]:
        
        return True, "TDP compatible"

    def _validate_socket_support(self, cpu: ComponentInfo, cooler: ComponentInfo) -> Tuple[bool, str]:
        return True, "Socket soportado"

    def _validate_power_compatibility(self, psu: ComponentInfo, gpu: ComponentInfo) -> Tuple[bool, str]:
        
        return True, "Potencia compatible"

    def get_compatibility_report(self) -> str:
        """Genera un reporte detallado de compatibilidad"""
        issues = self.blackboard.get('compatibility_issues', [])
        
        if not issues:
            return "✅ Todos los componentes son compatibles entre sí"
        
        report = ["## Reporte Detallado de Compatibilidad", ""]
        
        # Agrupar por severidad
        critical_issues = [i for i in issues if i.severity == "critical"]
        warning_issues = [i for i in issues if i.severity == "warning"]
        
        if critical_issues:
            report.append("### ❌ Problemas Críticos de Compatibilidad")
            for issue in critical_issues:
                comp_a = f"{issue.component_a.type.value}: {issue.component_a.model_name}"
                comp_b = f"{issue.component_b.type.value}: {issue.component_b.model_name}"
                report.append(f"- **{comp_a}** y **{comp_b}**: {issue.reason}")
            report.append("")
        
        if warning_issues:
            report.append("### ⚠ Advertencias de Compatibilidad")
            for issue in warning_issues:
                comp_a = f"{issue.component_a.type.value}: {issue.component_a.model_name}"
                comp_b = f"{issue.component_b.type.value}: {issue.component_b.model_name}"
                report.append(f"- **{comp_a}** y **{comp_b}**: {issue.reason}")
            report.append("")
        
        # Resumen estadístico
        incompatible_pairs = set()
        for issue in issues:
            pair = tuple(sorted([issue.component_a.type.value, issue.component_b.type.value]))
            incompatible_pairs.add(pair)
        
        report.append("### Resumen de Compatibilidad")
        report.append(f"- **Problemas críticos**: {len(critical_issues)}")
        report.append(f"- **Advertencias**: {len(warning_issues)}")
        report.append(f"- **Pares de componentes incompatibles**: {len(incompatible_pairs)}")
        report.append("")
        
        report.append("### Recomendaciones:")
        if critical_issues:
            report.append("- Resuelva los problemas críticos antes de continuar")
        if warning_issues:
            report.append("- Revise las advertencias para posibles mejoras")
        report.append("- Considere alternativas para los componentes marcados")
        
        return "\n".join(report)