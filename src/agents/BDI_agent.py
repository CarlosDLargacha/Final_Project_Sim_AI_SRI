from typing import Dict, Any, List
import json
from enum import Enum
import pandas as pd
from pydantic import BaseModel
from blackboard import Blackboard, EventType
from model.LLMClient import LLMClient
from agents.decorators import agent_error_handler
import re


class UseCase(Enum):
    GAMING = "gaming"
    VIDEO_EDITING = "video_editing"
    DATA_SCIENCE = "data_science"
    GENERAL = "general"
    SERVER = "server"
    CRYPTO_MINING = "crypto_mining"
    MACHINE_LEARNING = "machine_learning"
    WEB_DEVELOPMENT = "web_development"

    GAMING_VIDEO_EDITING = "gaming/video_editing"
    GAMING_DATA_SCIENCE = "gaming/data_science"
    VIDEO_EDITING_DATA_SCIENCE = "video_editing/data_science"
    ALL = "gaming/video_editing/data_science"


class HardwareRequirements(BaseModel):
    """Esquema estructurado de requisitos técnicos"""
    use_case: UseCase
    budget: Dict[str, float]  # {"min": 0, "max": 0}
    performance: Dict[str, Any]  # {"resolution": "4K", "fps": 60}
    aesthetics: Dict[str, Any]  # {"color": "black", "rgb": True}
    constraints: List[str]  # ["low_noise", "small_form_factor"]
    cpu: str 
    gpu: str
    storage: Dict[str, Any]
    ram: Dict[str, Any]

class BDIAgent:
    def __init__(self, llm_client: LLMClient, blackboard: Blackboard):
        """
        :param llm_client: Cliente para el modelo de lenguaje (OpenAI/Gemini)
        """
        
        self.blackboard = blackboard
        self.llm = llm_client
        self.current_beliefs = {}  # Creencias actuales del sistema
        self.user_desires = {}  # Deseos expresados por el usuario
        self.intentions = []  # Planes de acción generados
        
        
        self.blackboard.subscribe(
            EventType.USER_INPUT,
            self.extract_requirements
        )
        
        self.blackboard.subscribe(
            EventType.OPTIMIZATION_DONE,
            self.generate_user_response
        )
         
    @agent_error_handler
    def extract_requirements(self):
        """
        Proceso completo de extracción BDI:
        1. Análisis de texto para creencias (Beliefs)
        2. Identificación de deseos (Desires)
        3. Generación de intenciones (Intentions)
        """
        # Paso 1: Extraer información cruda con LLM

        cpu = pd.read_csv('src/data/component_specs/CPU_specs.csv')
        cpu_names = cpu['Model_Name'].unique().tolist()
        
        gpu = pd.read_csv('src/data/component_specs/GPU_specs.csv')
        gpu_names = cpu['Model_Name'].unique().tolist()

        raw_data = self._ask_llm(self.blackboard.get("user_input"), cpu_names, gpu_names)
        
        # Paso 2: Validar y normalizar
        try:
            requirements = self._validate_output(raw_data)
        except Exception as e:
            raw_data['use_case'] = "general"
            print(f"⚠️ Error: {str(e)}. Usando valores por defecto en use_case 'general'.")
            requirements = self._validate_output(raw_data)
        
        
        # Paso 3: Actualizar estados internos
        self._update_bdi_state(requirements)
        
        self.blackboard.update(
            section='user_requirements',
            data=requirements,
            agent_id='bdi_agent',
            notify=True  # Dispara EventType.REQUIREMENTS_UPDATED
        )

        print("[BDIAgent] Requerimientos extraíos")
        
    def _ask_llm(self, text: str, cpu_names, gpu_names) -> Dict[str, Any]:
        """Consulta al modelo de lenguaje para extracción estructurada"""
        
        sytem_prompt = f"""
            Eres un experto en hardware de computadoras. Extrae los siguientes datos del texto:
            
            Texto del usuario: "{text}" 
            """
            
        anser_prompt = """
            Devuelve SOLO un JSON con esta estructura:
            {{
                "use_case": "gaming/video_editing/data_science/crypto_mining/server/machine_learning/web_development/general (puede ser combinación como 'gaming/video_editing', defualt 'general')",
                "budget": {{
                    "min": número o None,
                    "max": número o None
                }},
                "performance": {{
                    "resolution": "1080p/1440p/4K",
                    "fps": número (si no es especificado en dependencia del use_case seleciona el mas usado en esa catogoría),
                    "software": ["nombres de programas"]
                }},
                "aesthetics": {{
                    "color": "string",
                    "rgb": boolean,
                    "window": boolean
                }},
                "cpu" : "1 cpu minima según el uso del caso (ej: "Intel Core i5-12400F") (valor obligatorio, defualt '')",
                "gpu" : "1 gpu minima según el uso del caso (ej: "NVIDIA RTX 3060") (valor obligatorio, default '')",
                "storage" : {
                    "prefer_ssd": boolean,       
                    "include_hdd": boolean,       
                    "capacity": "512GB/1TB/4TB (Capacidad mínima)",       
                    "performance": {
                        "read_speed": "3500MB/s (Velocidad mínima lectura para SSDs)" 
                    }
                },
                "ram" : {
                    "capacity" : "32GB (Capacidad mínima)",
                    "type": "DDR3/DDR4/DDR5 (Tipo específico requerido)",
                    "speed": "5600 (Velocidad mínima en MHz)"  
                }
                "constraints": ["lista de restricciones"]
            }} 

            Ejemplo para "Necesito PC para editar 4K bajo $1500 con RGB":
            {{
                "use_case": "video_editing",
                "budget": {{"min": null, "max": 1500}},
                "performance": {{"resolution": "4K", "fps": 60, "software": ["Premiere Pro"]}},
                "aesthetics": {{"color": null, "rgb": true}},
                "storage": {
                    "prefer_ssd": true,
                    "include_hdd": true,
                    "capacity": "1TB",
                    "performance": {
                        "read_speed": "3500MB/s"
                    }
                },
                "ram": {
                    "capacity": "32GB",
                    "type": "DDR4",
                    "speed": 3200
                },
                "constraints": []
            }} 
            """
        
        rules = f"""
            Reglas estrictas:
            1. Para use_case usar SOLO estas opciones o combinaciones:
            - gaming
            - video_editing
            - data_science
            - general
            - crypto_mining
            - server
            - machine_learning
            - web_development
            - gaming/video_editing
            - gaming/data_science
            - video_editing/data_science
            - gaming/video_editing/data_science
            2. Las combinaciones deben usar exactamente el formato 'tipo1/tipo2'
            3. Si el caso de uso no coincide con ninguno de los definidos, entonces usar "general"
            4. Estas son la listas de donde debes sacas los nombres de las cpu minini y gpu minima segun el prompt del usuario:
            - cpu : {cpu_names}
            - gpu : {gpu_names}
            """
        prompt = f"{sytem_prompt}\n{anser_prompt}\n{rules}"        
        response = self.llm.generate(prompt)
        
        try:
            return json.loads(response) 
        except json.JSONDecodeError:
            return self._safe_parse_json(response) 

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        """Extrae JSON de texto potencialmente mal formado"""
        try:
            json_str = re.search(r'\{.*\}', text, re.DOTALL).group()
            return json.loads(json_str)
        except (AttributeError, json.JSONDecodeError) as e:
            raise ValueError(f"Error parsing LLM output: {str(e)}")

    def _validate_output(self, raw_data: Dict) -> HardwareRequirements:
        """Valida y normaliza los datos extraídos"""
        
        if isinstance(raw_data, str):
             raw_data = self._safe_parse_json(raw_data)
                
        # Convertir budget a números
        if 'budget' in raw_data:
            raw_data['budget'] = {
                'min': self._parse_currency(raw_data['budget'].get('min'), True ),
                'max': self._parse_currency(raw_data['budget'].get('max'), False)
            }
        
        # Validar con Pydantic
        return HardwareRequirements(**raw_data)

    def _parse_currency(self, value: Any, isMin: bool) -> float:
        """Convierte valores monetarios a float"""
        if value is None:
            return float('-inf') if isMin else float('inf')
        if isinstance(value, str):
            return float(re.sub(r'[^\d.]', '', value))
        return float(value)

    def _update_bdi_state(self, requirements: HardwareRequirements):
        """Actualiza el estado interno BDI"""
        # Creencias (hechos técnicos confirmados)
        self.current_beliefs.update({
            'validated_requirements': requirements.dict(),
            'missing_fields': self._identify_missing_data(requirements)
        })
        
        # Deseos (preferencias subjetivas del usuario)
        self.user_desires = {
            'performance': requirements.performance,
            'aesthetics': requirements.aesthetics
        }
        
        # Intenciones (acciones a tomar)
        self.intentions = [
            "consultar_agentes_especializados",
            "verificar_compatibilidad",
            *(["solicitar_info_faltante"] if self.current_beliefs['missing_fields'] else [])
        ]

    def _identify_missing_data(self, requirements: HardwareRequirements) -> List[str]:
        """Identifica campos críticos faltantes"""
        missing = []
        
        if not requirements.use_case:
            missing.append("use_case")
        if not requirements.budget.get('max'):
            missing.append("budget.max")
        
        return missing

    def generate_clarification_questions(self) -> List[str]:
        """Genera preguntas para completar información faltante"""
        questions = {
            'use_case': "¿Para qué principal uso necesitas la computadora? (gaming, edición, programación...)",
            'budget.max': "¿Cuál es tu presupuesto máximo aproximado?",
            'performance.resolution': "¿Qué resolución necesitas para tu trabajo/juegos?",
            'aesthetics.color': "¿Tienes preferencia de color para los componentes?"
        }
        
        return [questions[field] for field in self.current_beliefs['missing_fields'] if field in questions]
    
    @agent_error_handler
    def generate_user_response(self):
        """
        Genera una respuesta en lenguaje natural basada en las configuraciones optimizadas
        y contextualizada con el estado interno BDI.
        """
        user_input = self.blackboard.get("user_input", {}).get("user_input", "")
        optimized_builds = self.blackboard.get("optimized_configs", [])
             
        if(optimized_builds):
            response = ""
            for i, build in enumerate(optimized_builds):
                
                response += f"{build.get('label', f'Build #{i}')}:\n\n" 
                for _, comps in build.items():
                    for comp_name, meta in comps.items():
                        response += comp_name + ":\n" + self._format_component_description(comp_name, meta) + "\n"
                    break                        
                response += "\n"
        else:
            response = "No se encontraron configuraciones que cumplan con los requisitos del usuario." 
        
        self.blackboard.update(
            section="user_response",
            data={"response": response},
            agent_id="bdi_agent",
            notify=True
        )
        
    def generate_user_response_llm(self):
        """
        Genera una respuesta en lenguaje natural basada en las configuraciones optimizadas
        y contextualizada con el estado interno BDI.
        """
        user_input = self.blackboard.get("user_input", {}).get("user_input", "")
        optimized_builds = self.blackboard.get("optimized_configs", [])
        
        # Construcción del prompt
        system_prompt = """
            Eres un experto en armado de computadoras personalizadas. Tu tarea es explicar de forma clara y profesional
            una o más configuraciones recomendadas para el usuario, basadas en sus necesidades y el estado interno del sistema.

            Tu respuesta debe:
            - Mencionar los componentes clave seleccionados (CPU, GPU, RAM, almacenamiento, etc.)
            - Justificar por qué se eligieron (rendimiento, eficiencia, estética, precio)
            - Indicar el precio total de la build
            - Mencionar cualquier ventaja técnica o compatibilidad relevante
            - Ser clara incluso para usuarios no expertos, pero sin perder rigor técnico
        """

        bdi_context = f"""
            === CONTEXTO INTERNO DEL SISTEMA (BDI) ===

            🧠 Creencias (Beliefs):
            - Requerimientos técnicos confirmados: {self.current_beliefs.get('validated_requirements')}
            - Campos faltantes en la solicitud: {self.current_beliefs.get('missing_fields')}

            🎯 Deseos (Desires):
            - Rendimiento deseado: {self.user_desires.get('performance')}
            - Estética deseada: {self.user_desires.get('aesthetics')}

            ✅ Intenciones ejecutadas (Intentions):
            - {', '.join(self.intentions)}

            ===========================================
        """

        prompt = f"""{system_prompt}

            Entrada original del usuario:
            \"\"\"{user_input}\"\"\"

            {bdi_context}

            Configuraciones optimizadas encontradas por ti (máximo 3):
            {self._format_optimized_builds_for_prompt(optimized_builds)}

            Responde de forma explicativa y clara, justificando las elecciones y destacando lo que cada build aporta. 
            
            PD: Recuerda que tu (el sistema) eres quien esta recomendando las builds, no el usuario. Por lo tanto explica cada una de las build.
            PD2: Ajustate solamente a las recomendaciones encontradas. Evita cualquier suposición o recomendación adicional que no esté basada en las builds optimizadas.
        """
        
        prompt = f"""{system_prompt}

            Entrada original del usuario:
            \"\"\"{user_input}\"\"\"

            {bdi_context}

            Configuraciones optimizadas encontradas por ti (máximo 3):
            {self._format_optimized_builds_for_prompt(optimized_builds)}

            Quisiera q me explicaras las caracteriscas cada una de las configuraciones optimizadas. Por ejejmplo:
            
            user_input: "Quiero una PC para gaming en 4K con presupuesto máximo de $1500. Prefiero NVIDIA para la GPU."
            
            response: Basado en tus requisitos de gaming con presupuesto de $1500, he encontrado estas opciones:

            **Opción 1** (Precio total: $1489.99):
            - **Procesador**: AMD Ryzen 7 7800X3D (8 núcleos, 4.2 GHz)
            - **Tarjeta gráfica**: NVIDIA RTX 4070 (12GB VRAM)
            - **Motherboard*: 32GB DDR5 5600MHz
            - **Puntos clave**: alto rendimiento, buen manejo térmico

            **Opción 2** (Precio total: $1350.50):
            - **Procesador**: Intel Core i5-13600K (14 núcleos, 3.5 GHz)
            - **Tarjeta gráfica**: AMD RX 7700 XT (12GB VRAM)
            - **Motherboard**: 32GB DDR5 5200MHz
            - **Puntos clave**: excelente relación precio-calidad
            
            PD: En caso de que no haya ninguna configuracion optimizada, simplemente responde que no se encontraron configuraciones que cumplan con los requisitos del usuario.
        """
        
        response = self.llm.generate(prompt)
        
        self.blackboard.update(
            section="user_response",
            data={"response": response},
            agent_id="bdi_agent",
            notify=True
        )
    
    def _format_component_description(self, comp_type, component: Dict) -> str:
        """
        Genera una descripción textual con las 5-7 características más relevantes de un componente,
        adaptándose dinámicamente a los campos disponibles en los metadatos.
        """
        meta = component.get("metadata", component)
        name = f"{meta.get('Model_Brand', '')} {meta.get('Model_Name', '')}".strip()
        price = meta.get("Price", meta.get("price", "N/A"))
        
        # Diccionario de campos relevantes por tipo de componente
        match comp_type:
            case "CPU": 
                relevant_fields = {
                    "Núcleos/Hilos": meta.get("Details_# of Cores# of Cores") or meta.get("Details_# of Cores"),
                    "Frecuencia Base": meta.get("Details_Operating FrequencyOperating Frequency") or meta.get("Details_Operating Frequency"),
                    "Frecuencia Turbo": meta.get("Details_Max Turbo FrequencyMax Turbo Frequency") or meta.get("Details_Max Turbo Frequency"),
                    "Socket": meta.get("Details_CPU Socket TypeCPU Socket Type") or meta.get("CPU Socket Type_CPU Socket Type"),
                    "Cache (L3)": meta.get("Details_L3 CacheL3 Cache") or meta.get("Details_L3 Cache"),
                    "TDP": meta.get("Details_Thermal Design PowerThermal Design Power") or meta.get("Details_Thermal Design Power"),
                    "Tecnología": meta.get("Details_Manufacturing TechManufacturing Tech") or meta.get("Details_Manufacturing Tech")
                }
            case "GPU": 
                relevant_fields = {
                    "Memoria": meta.get("Memory - Memory Size") or meta.get("Details_Memory Size"),
                    "Tipo Memoria": meta.get("Memory - Memory Type"),
                    "Interfaz": meta.get("Interface - InterfaceInterface") or meta.get("Interface - Interface"),
                    "TDP": meta.get("Details - Thermal Design PowerThermal Design Power") or meta.get("Details - Thermal Design Power"),
                    "Conectores": meta.get("Details - Power Connector"),
                    "Longitud": meta.get("Form Factor & Dimensions - Max GPU Length"),
                    "Puertos": self._extract_ports(meta)
                }
            case "Motherboard": 
                relevant_fields = {
                    "Socket": meta.get("Supported CPU_CPU Socket TypeCPU Socket Type"),
                    "Chipset": meta.get("Chipsets_ChipsetChipset"),
                    "Formato": meta.get("Physical Spec_Form Factor"),
                    "RAM Soporte": meta.get("Memory_Memory Standard"),
                    "Slots M.2": meta.get("Storage Devices_M.2"),
                    "Conectividad": self._extract_connectivity(meta),
                    "Características": meta.get("Features_Features")
                }
            case "HDD": 
                relevant_fields = {
                    "Capacidad": meta.get("Performance_Capacity"),
                    "Interfaz": meta.get("Performance_InterfaceInterface"),
                    "Velocidad (RPM)": meta.get("Performance_RPMRPM"),
                    "Caché": meta.get("Performance_CacheCache"),
                    "Tecnología": meta.get("Performance_Recording Technology"),
                    "Factor de Forma": meta.get("Dimensions_Form FactorForm Factor"),
                    "Latencia": meta.get("Performance_Average LatencyAverage Latency")
                }
            case "SSD": 
                relevant_fields = {
                    "Capacidad": meta.get("Details_Capacity"),
                    "Interfaz": meta.get("Details_Interface"),
                    "Protocolo": meta.get("Details_Protocol"),
                    "Lectura (MB/s)": meta.get("Performance_Max Sequential Read"),
                    "Escritura (MB/s)": meta.get("Performance_Max Sequential Write"),
                    "Factor de Forma": meta.get("Details_Form FactorForm Factor"),
                    "Durabilidad (TBW)": meta.get("Performance_Terabytes Written (TBW)")
                } 
            case "Case": 
                relevant_fields = {
                    "Tipo": meta.get("Details_TypeType"),
                    "Material": meta.get("Details_Case Material"),
                    "Compatibilidad": meta.get("Details_Motherboard CompatibilityMotherboard Compatibility"),
                    "Ventana Lateral": meta.get("Details_Side Panel WindowSide Panel Window"),
                    "Bahías 3.5\"": meta.get("Expansion_Internal 3.5\" Drive Bays"),
                    "Bahías 2.5\"": meta.get("Expansion_Internal 2.5\" Drive Bays"),
                    "Longitud Máx. GPU": meta.get("Dimensions & Weight_Max GPU Length"),
                    "Altura Máx. Cooler": meta.get("Dimensions & Weight_Max CPU Cooler Height"),
                    "Refrigeración": self._extract_cooling_info(meta),  # Método para combinar fan/radiator options
                    "Puertos Frontales": meta.get("Front Panel Ports_Front Ports"),
                    "Incluye Fuente": meta.get("Details_With Power Supply")
                } 
            case "Motherboard": 
                relevant_fields = {
                    "Socket": meta.get("Supported CPU_CPU Socket TypeCPU Socket Type"),
                    "Chipset": meta.get("Chipsets_ChipsetChipset"),
                    "Formato": meta.get("Physical Spec_Form Factor"),
                    "RAM": f"{meta.get('Memory_Number of Memory Slots', '?')}x {meta.get('Memory_Memory Standard', 'DDR?')}",
                    "Almacenamiento": f"{meta.get('Storage Devices_SATA 6Gb/sSATA 6Gb/s', '?')}x SATA, {meta.get('Storage Devices_M.2', '?')}x M.2",
                    "Slots PCIe": self._extract_pcie_slots(meta),  # Método para combinar slots PCIe
                    "Red": self._extract_network_info(meta),  # Combina LAN/Wi-Fi/Bluetooth
                    "Audio": meta.get("Onboard Audio_Audio ChipsetAudio Chipset"),
                    "Puertos Traseros": meta.get("Rear Panel Ports_Back I/O Ports"),
                    "RGB": meta.get("Physical Spec_LED Lighting")
                }
            case "PSU": 
                relevant_fields = {
                    "Potencia": meta.get("Details_Maximum PowerMaximum Power") or meta.get("Details_Maximum Power"),
                    "Eficiencia": meta.get("Details_EfficiencyEfficiency") or meta.get("Details_Efficiency"),
                    "Modular": meta.get("Details_ModularModular") or meta.get("Details_Modular"),
                    "Conectores CPU": meta.get("Details_Main ConnectorMain Connector") or meta.get("Details_Main Connector"),
                    "Conectores PCIe": meta.get("Details_PCI-Express ConnectorPCI-Express Connector"),
                    "Conectores SATA": meta.get("Details_SATA Power ConnectorSATA Power Connector"),
                    "Ventilador": meta.get("Details_FansFans") or meta.get("Details_Fans"),
                    "Protecciones": self._extract_protections(meta),  # Método para combinar protecciones
                    "Formato": meta.get("Details_Type"),
                    "MTBF": meta.get("Details_MTBFMTBF") or meta.get("Details_MTBF")
                } 
            case "RAM": 
                relevant_fields = {
                    "Capacidad": meta.get("Details_Capacity"),
                    "Tipo": meta.get("Details_Type"),
                    "Velocidad": meta.get("Details_SpeedSpeed") or meta.get("Details_Speed"),
                    "Latencia (CL)": meta.get("Details_CAS LatencyCAS Latency") or meta.get("Details_CAS Latency"),
                    "Timings": meta.get("Details_TimingTiming") or meta.get("Details_Timing"),
                    "Voltaje": meta.get("Details_VoltageVoltage") or meta.get("Details_Voltage"),
                    "Kit": meta.get("Details_Multi-channel Kit"),
                    "Perfil XMP/EXPO": meta.get("Details_BIOS/Performance ProfileBIOS/Performance Profile"),
                    "ECC/Registrada": self._extract_ecc_info(meta),  # Método para combinar ECC y Registrada
                    "RGB/Color": meta.get("Details_LED Color") or meta.get("Details_Color")
                }
        
        # Construcción de la descripción
        desc = f"**{name}** (${price})\n"
        
        for field_name in relevant_fields:
            field_value = relevant_fields[field_name]
            if field_value and field_value != "N/A":
                desc += f"- {field_name}: {field_value}\n"
        
        return desc

    def _extract_ports(self, meta: Dict) -> str:
        """Extrae información de puertos para GPUs"""
        ports = []
        for port_type in ["HDMI", "DisplayPort", "DVI"]:
            if meta.get(f"Ports - {port_type}{port_type}") or meta.get(f"Ports - {port_type}"):
                ports.append(port_type)
        return ", ".join(ports) if ports else "N/A"

    def _extract_connectivity(self, meta: Dict) -> str:
        """Extrae información de conectividad para motherboards"""
        connectivity = []
        if meta.get("Onboard LAN_Wireless LAN"): connectivity.append("Wi-Fi")
        if meta.get("Onboard LAN_Bluetooth"): connectivity.append("Bluetooth")
        if meta.get("Onboard LAN_Max LAN Speed"): connectivity.append(meta["Onboard LAN_Max LAN Speed"])
        return ", ".join(connectivity) if connectivity else "N/A"
    
    def _extract_cooling_info(self, meta: Dict) -> str:
        fan_options = meta.get("Cooling System_Fan Options", "")
        radiator_options = meta.get("Cooling System_Radiator Options", "")
        cooling_info = []
        
        if fan_options:
            cooling_info.append(f"Ventiladores: {fan_options}")
        if radiator_options:
            cooling_info.append(f"Radiadores: {radiator_options}")
        
        return "; ".join(cooling_info) if cooling_info else "No especificado"
    
    def _extract_pcie_slots(self, meta: Dict) -> str:
        pcie_slots = []
        for version in ["5.0", "4.0", "3.0"]:
            slot = meta.get(f"Expansion Slots_PCI Express {version} x16")
            if slot:
                pcie_slots.append(f"PCIe {version} x16: {slot}")
        return "; ".join(pcie_slots) if pcie_slots else "No especificado"
    
    def _extract_network_info(self, meta: Dict) -> str:
        network_info = []
        if meta.get("Onboard LAN_Max LAN Speed"):
            network_info.append(f"LAN: {meta['Onboard LAN_Max LAN Speed']}")
        if meta.get("Onboard LAN_Wireless LAN"):
            network_info.append(f"Wi-Fi: {meta['Onboard LAN_Wireless LAN']}")
        if meta.get("Onboard LAN_Bluetooth"):
            network_info.append(f"BT: {meta['Onboard LAN_Bluetooth']}")
        return ", ".join(network_info) if network_info else "No especificado"
    
    def _extract_protections(self, meta: Dict) -> str:
        protections = []
        protection_fields = [
            "Details_Protection",
            "Details_Over Voltage ProtectionOver Voltage Protection",
            "Details_Overload ProtectionOverload Protection"
        ]
        for field in protection_fields:
            if meta.get(field):
                protections.append(meta[field])
        return ", ".join(protections) if protections else "Estándar"
    
    def _extract_ecc_info(self, meta: Dict) -> str:
        ecc = meta.get("Details_ECCECC") or meta.get("Details_ECC")
        buffered = meta.get("Details_Buffered/RegisteredBuffered/Registered") or meta.get("Details_Buffered/Registered")
        info = []
        if ecc and ecc.lower() != "no":
            info.append("ECC Sí")
        if buffered and "unbuffered" not in buffered.lower():
            info.append(buffered)
        return ", ".join(info) if info else "No"
    