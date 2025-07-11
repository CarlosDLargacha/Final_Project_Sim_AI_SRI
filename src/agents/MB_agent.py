import re
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity
from agents.decorators import agent_error_handler
from blackboard import Blackboard, EventType
from agents.BDI_agent import HardwareRequirements, UseCase

class MotherboardAgent:
    def __init__(self, vector_db: Dict[str, Any], blackboard: Blackboard):
        """
        :param vector_db: Base de datos vectorial de motherboards
        :param blackboard: Instancia compartida del blackboard
        """
        self.vector_db = vector_db
        self.blackboard = blackboard
        self.embedding_model = vector_db['model']
        
        # Suscribirse a eventos relevantes
        self.blackboard.subscribe(
            EventType.REQUIREMENTS_UPDATED,  
            self.process_requirements
        )

    @agent_error_handler
    def process_requirements(self):
        """Procesa los requisitos y componentes seleccionados para recomendar motherboards"""
        requirements = self.blackboard.get('user_requirements')
        if not requirements:
            return
        
        # Generar embedding para los requisitos
        requirement_text = self._generate_requirement_text(requirements)
        requirement_embedding = self.embedding_model.encode([requirement_text])[0]
        
        # Calcular similitud con todas las motherboards
        similarities = cosine_similarity(
            [requirement_embedding],
            self.vector_db['embeddings']
        )[0]
        
        # Filtrar motherboards que cumplan con requisitos
        candidates = []
        for i, metadata in enumerate(self.vector_db['metadata']):
            
            # Verificar restricciones del usuario
            if not self._check_constraints(metadata, requirements.constraints):
                continue
            
            try:
                price = float(metadata.get('Price', float('inf')))
            except (ValueError, TypeError):
                price = float('inf')
            
            max_mb_budget = requirements.budget.get('max', float('inf'))
            if price > max_mb_budget:
                continue
            
            candidates.append({
                'metadata': metadata,
                'similarity': similarities[i],
                'price': price,
            })
        
        # Ordenar por similitud y precio
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (
                -x['similarity'],
                x['price']
            )
        )
        
        top_candidates = sorted_candidates
        
        # Actualizar el blackboard
        self.blackboard.update(
            section='component_proposals',
            data={'Motherboard': top_candidates},
            agent_id='motherboard_agent',
            notify=True
        )

        print("[MBAgent] Componentes propuestos")

    def _generate_requirement_text(self, requirements: HardwareRequirements) -> str:
        """Genera texto descriptivo de requisitos para embeddings"""
        text_parts = [
            f"Motherboard para {requirements.use_case.value}",
            "Requisitos:"
        ]
        
        # Añadir requisitos específicos
        if requirements.use_case == UseCase.GAMING:
            text_parts.append("Optimizada para gaming")
        elif requirements.use_case == UseCase.VIDEO_EDITING:
            text_parts.append("Conectividad avanzada E/S")
        
        if requirements.constraints:
            text_parts.append(f"Restricciones: {', '.join(requirements.constraints)}")
        
        return ". ".join(text_parts)

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