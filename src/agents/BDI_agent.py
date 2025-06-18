from typing import Dict, Any, List
import json
from enum import Enum
from pydantic import BaseModel
from blackboard import Blackboard, EventType
from model.LLMClient import LLMClient
from agents.decorators import agent_error_handler
import re

class UseCase(Enum):
    """Tipos de casos de uso soportados"""
    GAMING = "gaming"
    VIDEO_EDITING = "video_editing"
    DATA_SCIENCE = "data_science"
    GENERAL = "general"

class HardwareRequirements(BaseModel):
    """Esquema estructurado de requisitos técnicos"""
    use_case: UseCase
    budget: Dict[str, float]  # {"min": 0, "max": 0}
    performance: Dict[str, Any]  # {"resolution": "4K", "fps": 60}
    aesthetics: Dict[str, Any]  # {"color": "black", "rgb": True}
    constraints: List[str]  # ["low_noise", "small_form_factor"]

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
    def generate_user_response(self):
        """
        Genera una respuesta en lenguaje natural basada en las configuraciones optimizadas
        y contextualizada con el estado interno BDI.
        """
        user_input = self.blackboard.get("user_input", {}).get("user_input", "")
        optimized_builds = self.blackboard.get("optimized_configs", [])

        if not optimized_builds:
            print("[BDIAgent] No hay configuraciones optimizadas para generar respuesta.")
            return

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

        response = self.llm.generate(prompt)

        self.blackboard.update(
            section="user_response",
            data={"response": response},
            agent_id="bdi_agent",
            notify=True
        )

    def _format_optimized_builds_for_prompt(self, builds: List[Dict]) -> str:
        """Convierte las builds optimizadas a un texto amigable para LLM"""
        text = ""
        for i, build in enumerate(builds, 1):
            text += f"\n🛠️ Build #{i} - Precio Total: ${build['total_price']}\n"
            for comp_type, comp in build['components'].items():
                meta = comp.get("metadata", comp)
                name = f"{meta.get('Model_Brand', '')} {meta.get('Model_Name', '')}".strip()
                price = meta.get("Price", comp.get("price", "N/A"))
                
                highlights = []
                # CPU
                if comp_type == "CPU":
                    highlights = [
                        f"Núcleos: {meta.get('Details_# of Cores# of Cores', 'N/A')}",
                        f"Frecuencia: {meta.get('Details_Operating FrequencyOperating Frequency', 'N/A')}",
                        f"TDP: {meta.get('Details_Thermal Design PowerThermal Design Power', 'N/A')}"
                    ]
                # GPU
                elif comp_type == "GPU":
                    highlights = [
                        f"Memoria: {meta.get('Details_Memory Size', 'N/A')} GB",
                        f"Interfaz: {meta.get('Details_Interface', 'N/A')}"
                    ]
                # RAM
                elif comp_type == "RAM":
                    highlights = [
                        f"Capacidad: {meta.get('Details_Capacity', 'N/A')}",
                        f"Velocidad: {meta.get('Details_Speed', 'N/A')}"
                    ]
                # SSD
                elif comp_type == "SSD":
                    highlights = [
                        f"Capacidad: {meta.get('Details_Capacity', 'N/A')}",
                        f"Tipo: {meta.get('Details_Interface', 'N/A')}"
                    ]

                text += f"- {comp_type}: {name} (${price})\n"
                for h in highlights:
                    text += f"   - {h}\n"
            text += "\n"
        return text

    @agent_error_handler
    def extract_requirements(self):
        """
        Proceso completo de extracción BDI:
        1. Análisis de texto para creencias (Beliefs)
        2. Identificación de deseos (Desires)
        3. Generación de intenciones (Intentions)
        """
        # Paso 1: Extraer información cruda con LLM

        raw_data = self._ask_llm(self.blackboard.get("user_input"))
        
        # Paso 2: Validar y normalizar
        requirements = self._validate_output(raw_data)
        
        # Paso 3: Actualizar estados internos
        self._update_bdi_state(requirements)
        
        self.blackboard.update(
            section='user_requirements',
            data=requirements,
            agent_id='bdi_agent',
            notify=True  # Dispara EventType.REQUIREMENTS_UPDATED
        )
        
    def _ask_llm(self, text: str) -> Dict[str, Any]:
        """Consulta al modelo de lenguaje para extracción estructurada"""
        prompt = f"""
        Eres un experto en hardware de computadoras. Extrae los siguientes datos del texto:
        
        Texto del usuario: "{text}"

        Devuelve SOLO un JSON con esta estructura:
        {{
            "use_case": "gaming/video_editing/data_science/general",
            "budget": {{
                "min": número o null,
                "max": número o null
            }},
            "performance": {{
                "resolution": "1080p/1440p/4K",
                "fps": número (si no es especificado en dependencia del use_case seleciona el mas usado en esa catogoría),
                "software": ["nombres de programas"]
            }},
            "aesthetics": {{
                "color": "string",
                "rgb": boolean
            }},
            "constraints": ["lista de restricciones"]
        }}

        Ejemplo para "Necesito PC para editar 4K bajo $1500 con RGB":
        {{
            "use_case": "video_editing",
            "budget": {{"min": null, "max": 1500}},
            "performance": {{"resolution": "4K", "fps": 60, "software": ["Premiere Pro"]}},
            "aesthetics": {{"color": null, "rgb": true}},
            "constraints": []
        }}
        """
        
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
                'min': self._parse_currency(raw_data['budget'].get('min')),
                'max': self._parse_currency(raw_data['budget'].get('max'))
            }
        
        # Validar con Pydantic
        return HardwareRequirements(**raw_data)

    def _parse_currency(self, value: Any) -> float:
        """Convierte valores monetarios a float"""
        if value is None:
            return 0.0
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