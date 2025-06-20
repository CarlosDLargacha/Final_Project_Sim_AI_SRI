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
    STORAGE = "Storage"
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
            EventType.COMPONENTS_PROPOSED,
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
                        'ram_slots': metadata.get('Memory_Number of Memory Slots', ''),
                        'pcie_slots': metadata.get('Expansion Slots_PCI Express 5.0 x16', ''),
                        'supported_gpu': metadata.get('Supported CPU_CPU Type', '')
                    }
                elif enum_type == ComponentType.GPU:
                    key_features = {
                        'length': metadata.get('Form Factor & Dimensions - Max GPU Length', ''),
                        'power': metadata.get('Details - Recommended PSU Wattage', ''),
                        'interface': metadata.get('Interface - InterfaceInterface', '')
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
        
        print(cpu_model)
        print(mobo_chipset)
        
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

    def _validate_pcie_compatibility(self, gpu: ComponentInfo, mobo: ComponentInfo) -> Tuple[bool, str]:
        """Valida compatibilidad de slot PCIe entre GPU y motherboard"""
        gpu_interface = gpu.full_metadata.get('price', '').lower()
        mobo_pcie_slots = mobo.key_features.get('pcie_slots', '0')
        
        #print('===================================================================================================')
        #print(gpu.key_features)
        #print(mobo.key_features)
        #print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        
        if 'pcie' not in gpu_interface:
            return True, "Interfaz GPU no es PCIe"
        
        if not mobo_pcie_slots.isdigit() or int(mobo_pcie_slots) < 1:
            return False, "Motherboard no tiene slots PCIe x16 disponibles"
        
        return True, "PCIe compatible"

    def _validate_size_compatibility(self, gpu: ComponentInfo, case: ComponentInfo) -> Tuple[bool, str]:
        """Valida que la GPU quepa en el gabinete"""
        gpu_length = gpu.key_features.get('length', '0 mm')
        case_max_gpu = case.key_features.get('max_gpu_length', '0 mm')
        
        try:
            # Extraer valores numéricos (asumiendo formato "300 mm")
            gpu_value = float(re.search(r'[\d.]+', gpu_length).group())
            case_value = float(re.search(r'[\d.]+', case_max_gpu).group())
            
            # Comparar valores (asumiendo misma unidad)
            if gpu_value > case_value:
                return False, f"GPU ({gpu_length}) más grande que espacio disponible en gabinete ({case_max_gpu})"
        except (AttributeError, ValueError):
            return True, "No se pudieron verificar dimensiones"
        
        return True, "Dimensiones compatibles"

    def _validate_ram_type_compatibility(self, ram: ComponentInfo, mobo: ComponentInfo) -> Tuple[bool, str]:
        """Valida que el tipo de RAM sea compatible con la motherboard"""
        ram_type = ram.key_features.get('type', '').upper()
        mobo_ram_types = mobo.key_features.get('ram_type', '').upper()
        
        if not ram_type or not mobo_ram_types:
            return True, "Información de RAM no disponible"
        
        if ram_type not in mobo_ram_types:
            return False, f"Tipo de RAM incompatible: {ram_type} vs soportado {mobo_ram_types}"
        
        return True, "Tipo de RAM compatible"

    def _validate_ram_speed_compatibility(self, ram: ComponentInfo, mobo: ComponentInfo) -> Tuple[bool, str]:
        """Valida que la velocidad de RAM sea compatible con la motherboard"""
        ram_speed = ram.key_features.get('speed', '0 MHz')
        mobo_max_speed = mobo.key_features.get('max_ram_speed', '0 MHz')
        
        try:
            ram_value = int(re.search(r'\d+', ram_speed).group())
            mobo_value = int(re.search(r'\d+', mobo_max_speed).group())
            
            if ram_value > mobo_value:
                return False, f"Velocidad RAM ({ram_speed}) excede soporte de motherboard ({mobo_max_speed})"
        except (AttributeError, ValueError):
            return True, "No se pudieron verificar velocidades"
        
        return True, "Velocidad RAM compatible"

    def _validate_tdp_compatibility(self, cpu: ComponentInfo, cooler: ComponentInfo) -> Tuple[bool, str]:
        """Valida que el cooler pueda manejar el TDP del CPU"""
        cpu_tdp = cpu.key_features.get('tdp', '0W')
        cooler_tdp = cooler.key_features.get('max_tdp', '0W')
        
        try:
            cpu_value = float(re.search(r'[\d.]+', cpu_tdp).group())
            cooler_value = float(re.search(r'[\d.]+', cooler_tdp).group())
            
            if cpu_value > cooler_value:
                return False, f"TDP CPU ({cpu_tdp}) excede capacidad de enfriamiento ({cooler_tdp})"
        except (AttributeError, ValueError):
            return True, "No se pudieron verificar valores TDP"
        
        return True, "TDP compatible"

    def _validate_socket_support(self, cpu: ComponentInfo, cooler: ComponentInfo) -> Tuple[bool, str]:
        """Valida que el cooler soporte el socket del CPU"""
        cpu_socket = cpu.key_features.get('socket', '')
        cooler_sockets = cooler.key_features.get('supported_sockets', [])
        
        if not cpu_socket or not cooler_sockets:
            return True, "Información de socket no disponible"
        
        if cpu_socket not in cooler_sockets:
            return False, f"Cooler no soporta socket CPU {cpu_socket}"
        
        return True, "Socket soportado"

    def _validate_power_compatibility(self, psu: ComponentInfo, gpu: ComponentInfo) -> Tuple[bool, str]:
        """Valida que la PSU tenga suficiente potencia para la GPU"""
        gpu_power = gpu.key_features.get('power', '0W')
        psu_wattage = psu.key_features.get('wattage', '0W')
        
        try:
            gpu_value = float(re.search(r'[\d.]+', gpu_power).group())
            psu_value = float(re.search(r'[\d.]+', psu_wattage).group())
            
            if gpu_value > psu_value * 0.8:  # 80% de la capacidad de la PSU
                return False, f"Requerimiento de GPU ({gpu_power}) excede capacidad de PSU ({psu_wattage})"
        except (AttributeError, ValueError):
            return True, "No se pudieron verificar valores de potencia"
        
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