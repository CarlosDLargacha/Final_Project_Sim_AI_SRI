import re
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity
from blackboard import Blackboard, EventType
from agents.BDI_agent import HardwareRequirements, UseCase

class MotherboardAgent:
    def __init__(self, vector_db: Dict[str, Any], compatibility_db: Dict[str, Any], blackboard: Blackboard):
        """
        :param vector_db: Base de datos vectorial de motherboards
        :param compatibility_db: Diccionario con datos de compatibilidad
        :param blackboard: Instancia compartida del blackboard
        """
        self.vector_db = vector_db
        self.blackboard = blackboard
        self.compatibility_db = compatibility_db
        self.embedding_model = vector_db['model']
        
        # Suscribirse a eventos relevantes
        self.blackboard.subscribe(
            EventType.COMPONENTS_PROPOSED,  # Esperar a que otros agentes propongan componentes
            self.process_requirements
        )

    def process_requirements(self):
        """Procesa los requisitos y componentes seleccionados para recomendar motherboards"""
        requirements = self.blackboard.get('user_requirements')
        if not requirements:
            return
        
        # Obtener componentes ya seleccionados/propuestos
        proposed_components = self.blackboard.get('component_proposals', {})
        
        # Generar embedding para los requisitos
        requirement_text = self._generate_requirement_text(requirements, proposed_components)
        requirement_embedding = self.embedding_model.encode([requirement_text])[0]
        
        # Calcular similitud con todas las motherboards
        similarities = cosine_similarity(
            [requirement_embedding],
            self.vector_db['embeddings']
        )[0]
        
        # Filtrar motherboards que cumplan con requisitos
        candidates = []
        for i, metadata in enumerate(self.vector_db['metadata']):
            # Verificar compatibilidad con componentes propuestos
            if not self._check_compatibility(metadata, proposed_components):
                continue
                
            # Verificar restricciones del usuario
            if not self._check_constraints(metadata, requirements.constraints):
                continue
            
            # Verificar presupuesto (hasta 20% del total para motherboard)
            try:
                price = float(metadata.get('Price', float('inf')))
            except (ValueError, TypeError):
                price = float('inf')
            
            max_mb_budget = requirements.budget.get('max', float('inf')) * 0.2
            if price > max_mb_budget:
                continue
            
            candidates.append({
                'metadata': metadata,
                'similarity': similarities[i],
                'price': price,
                'compatibility_score': self._calculate_compatibility_score(requirements, metadata, proposed_components)
            })
        
        # Ordenar por compatibilidad, similitud y precio
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (
                -x['compatibility_score'],
                -x['similarity'],
                x['price']
            )
        )
        
        # Proponer las mejores opciones (máximo 5)
        top_candidates = sorted_candidates[:5]
        
        # Actualizar el blackboard
        self.blackboard.update(
            section='component_proposals',
            data={'Motherboard': top_candidates},
            agent_id='motherboard_agent',
            notify=True
        )

    def _generate_requirement_text(self, requirements: HardwareRequirements, components: Dict) -> str:
        """Genera texto descriptivo de requisitos para embeddings"""
        text_parts = [
            f"Motherboard para {requirements.use_case.value}",
            "Requisitos:"
        ]
        
        # Añadir características de componentes seleccionados
        if 'CPU' in components:
            cpu_model = components['CPU'][0]['metadata'].get('Model_Name', 'CPU')
            text_parts.append(f"Compatibilidad con {cpu_model}")
        
        if 'GPU' in components:
            gpu_model = components['GPU'][0]['metadata'].get('Model_Name', 'GPU')
            text_parts.append(f"Ranura compatible con {gpu_model}")
        
        # Añadir requisitos específicos
        if requirements.use_case == UseCase.GAMING:
            text_parts.append("Optimizada para gaming")
        elif requirements.use_case == UseCase.VIDEO_EDITING:
            text_parts.append("Conectividad avanzada E/S")
        
        if requirements.constraints:
            text_parts.append(f"Restricciones: {', '.join(requirements.constraints)}")
        
        return ". ".join(text_parts)

    def _check_compatibility(self, mb_metadata: Dict, components: Dict) -> bool:
        """Verifica compatibilidad con componentes propuestos"""
        # Compatibilidad con CPU
        if 'CPU' in components:
            cpu_socket = components['CPU'][0]['metadata'].get('CPU Socket Type_CPU Socket Type', '')
            mb_socket = mb_metadata.get('CPU Socket Type_CPU Socket Type', '')
            if cpu_socket and mb_socket and cpu_socket != mb_socket:
                return False
        
        # Compatibilidad con RAM
        if 'RAM' in components:
            ram_type = components['RAM'][0]['metadata'].get('Memory - Memory Type', '')
            mb_ram_type = mb_metadata.get('Memory - Memory Type', '')
            if ram_type and mb_ram_type and ram_type != mb_ram_type:
                return False
        
        # Compatibilidad con tamaño de GPU
        if 'GPU' in components:
            gpu_length = components['GPU'][0]['metadata'].get('Form Factor & Dimensions - Max GPU Length', '0 mm')
            mb_max_gpu_length = mb_metadata.get('Form Factor & Dimensions - Max GPU Length', '0 mm')
            try:
                if float(gpu_length.split()[0]) > float(mb_max_gpu_length.split()[0]):
                    return False
            except (ValueError, IndexError):
                pass
        
        return True

    def _calculate_compatibility_score(self, requirements: HardwareRequirements,  mb_metadata: Dict, components: Dict) -> int:
        """Calcula puntaje de compatibilidad (mayor es mejor)"""
        score = 0
        
        # Puntos por compatibilidad exacta de socket
        if 'CPU' in components:
            cpu_socket = components['CPU'][0]['metadata'].get('CPU Socket Type_CPU Socket Type', '')
            mb_socket = mb_metadata.get('CPU Socket Type_CPU Socket Type', '')
            if cpu_socket and mb_socket and cpu_socket == mb_socket:
                score += 3
        
        # Puntos por slots de RAM adecuados
        if 'RAM' in components:
            ram_modules = int(components['RAM'][0]['metadata'].get('Memory - Modules', '1'))
            mb_ram_slots = int(mb_metadata.get('Memory - Slots', '0'))
            if mb_ram_slots >= ram_modules:
                score += 2
        
        # Puntos por conectividad adecuada
        if requirements.use_case == UseCase.VIDEO_EDITING:
            usb_ports = int(mb_metadata.get('Ports - USB', '0'))
            if usb_ports >= 4:
                score += 1
        
        return score

    def _check_constraints(self, metadata: Dict, constraints: List[str]) -> bool:
        """Verifica restricciones como tamaño, factor de forma, etc."""
        if 'small_form_factor' in constraints:
            form_factor = metadata.get('Form Factor & Dimensions - Form FactorForm Factor', '').lower()
            if form_factor not in ['mini-itx', 'microatx']:
                return False
        
        if 'wifi' in constraints:
            has_wifi = metadata.get('Networking - Wireless Networking Standard', '').lower()
            if 'wifi' not in has_wifi and '802.11' not in has_wifi:
                return False
        
        return True

    def get_recommendation_report(self, candidates: List[Dict]) -> str:
        """Genera un informe detallado de recomendaciones"""
        if not candidates:
            return "No se encontraron motherboards que cumplan con los requisitos"
        
        requirements = self.blackboard.get('user_requirements')
        use_case = requirements.use_case.value if requirements else "general"
        proposed_components = self.blackboard.get('component_proposals', {})
        
        report = [
            "## Recomendaciones de Motherboard",
            "",
            f"**Caso de uso:** {use_case.capitalize()}",
            f"**Presupuesto máximo:** ${requirements.budget.get('max', 'N/A')}",
            ""
        ]
        
        # Mostrar compatibilidad con componentes seleccionados
        if 'CPU' in proposed_components:
            cpu = proposed_components['CPU'][0]['metadata'].get('Model_Name', 'CPU')
            report.append(f"**CPU seleccionada:** {cpu}")
        
        if 'GPU' in proposed_components:
            gpu = proposed_components['GPU'][0]['metadata'].get('Model_Name', 'GPU')
            report.append(f"**GPU seleccionada:** {gpu}")
        
        report.append("")
        
        for i, candidate in enumerate(candidates[:3]):  # Mostrar top 3
            metadata = candidate['metadata']
            
            report.append(f"### Opción #{i+1}: {metadata.get('Model_Name', '')}")
            report.append("- **Especificaciones:**")
            report.append(f"  - **Socket:** {metadata.get('CPU Socket Type_CPU Socket Type', 'N/A')}")
            report.append(f"  - **Factor de forma:** {metadata.get('Form Factor & Dimensions - Form FactorForm Factor', 'N/A')}")
            report.append(f"  - **Slots RAM:** {metadata.get('Memory - Slots', 'N/A')}")
            report.append(f"  - **Tipo RAM:** {metadata.get('Memory - Memory Type', 'N/A')}")
            report.append("- **Conectividad:**")
            report.append(f"  - **USB:** {metadata.get('Ports - USB', 'N/A')}")
            report.append(f"  - **Red:** {metadata.get('Networking - Wireless Networking Standard', 'N/A')}")
            report.append("- **Compra:**")
            report.append(f"  - **Precio:** ${candidate['price']}")
            report.append(f"  - [Ver en Newegg]({metadata.get('URL', '')})")
            report.append("")
        
        report.append("### Notas importantes")
        report.append("- Verificar compatibilidad exacta con todos los componentes antes de comprar.")
        
        if 'small_form_factor' in (requirements.constraints or []):
            report.append("- Para builds pequeños, asegúrese que todos los componentes caben en el chasis.")
        
        return "\n".join(report)