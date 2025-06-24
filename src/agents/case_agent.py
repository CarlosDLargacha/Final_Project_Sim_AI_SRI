from typing import Dict, List, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from agents.decorators import agent_error_handler
from blackboard import Blackboard, EventType
from agents.BDI_agent import HardwareRequirements, UseCase

class CaseAgent:
    def __init__(self, vector_db: Dict[str, Any], blackboard: Blackboard):
        """
        :param vector_db: Base de datos vectorial de gabinetes
        :param blackboard: Instancia compartida del blackboard
        """
        self.vector_db = vector_db
        self.blackboard = blackboard
        self.embedding_model = vector_db['model']
        
        # Suscribirse a eventos del BDI
        self.blackboard.subscribe(
            EventType.REQUIREMENTS_UPDATED,
            self.process_requirements
        )

    @agent_error_handler
    def process_requirements(self):
        """Procesa los requisitos del usuario para recomendar gabinetes"""
        requirements = self.blackboard.get('user_requirements')
        if not requirements:
            return
        
        # Generar embedding para los requisitos
        requirement_text = self._generate_requirement_text(requirements)
        requirement_embedding = self.embedding_model.encode([requirement_text])[0]
        
        # Calcular similitud con todos los gabinetes
        similarities = cosine_similarity(
            [requirement_embedding],
            self.vector_db['embeddings']
        )[0]
        
        # Obtener componentes propuestos para verificar compatibilidad
        component_proposals = self.blackboard.get('component_proposals', {})
        
        # Filtrar y ordenar candidatos
        candidates = []
        for i, metadata in enumerate(self.vector_db['metadata']):
            # Verificar compatibilidad con componentes seleccionados
            if not self._check_components_compatibility(metadata, component_proposals):
                continue
                
            # Verificar presupuesto
            try:
                price = float(metadata.get('Price', float('inf')))
            except (ValueError, TypeError):
                price = float('inf')
            
            max_case_budget = requirements.budget.get('max', float('inf'))
            if price > max_case_budget:
                continue
            
            candidates.append({
                'metadata': metadata,
                'similarity': similarities[i],
                'price': price,
                'aesthetics_score': self._calculate_aesthetics_score(metadata, requirements)
            })
        
        # Ordenar por similitud, compatibilidad y estética
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (
                -x['similarity'],
                -x['aesthetics_score'],
                x['price']
            )
        )
        
        top_candidates = sorted_candidates
        
        # Actualizar el blackboard
        if top_candidates:
            self.blackboard.update(
                section='component_proposals',
                data={'Case': top_candidates},
                agent_id='case_agent',
                notify=True
            )

        print("[CaseAgent] Componentes propuestos")

    def _generate_requirement_text(self, requirements: HardwareRequirements) -> str:
        """Genera texto descriptivo de requisitos para embeddings"""
        text_parts = [
            f"Gabinete para sistema de computación {requirements.use_case.value}",
            "Requisitos principales:"
        ]
        
        # Factor de forma basado en restricciones o componentes
        if hasattr(requirements, 'constraints'):
            if 'small_form_factor' in requirements.constraints:
                text_parts.append("Factor de forma pequeño (mini-ITX/microATX)")
            elif 'mid_tower' in requirements.constraints:
                text_parts.append("Factor de forma medio (ATX)")
            elif 'full_tower' in requirements.constraints:
                text_parts.append("Factor de forma grande (E-ATX)")
        
        # Preferencias estéticas
        aesthetics = requirements.aesthetics if hasattr(requirements, 'aesthetics') else {}
        if aesthetics.get('rgb', False):
            text_parts.append("Iluminación RGB preferida")
        if aesthetics.get('color'):
            text_parts.append(f"Color preferido: {aesthetics['color']}")
        
        # Ventilación y refrigeración
        if requirements.use_case == UseCase.GAMING:
            text_parts.append("Buena ventilación para gaming")
        elif requirements.use_case == UseCase.VIDEO_EDITING:
            text_parts.append("Espacio para múltiples HDDs/SSDs")
        
        return ". ".join(text_parts)

    def _check_components_compatibility(self, case_metadata: Dict, components: Dict) -> bool:
        """Verifica que los componentes propuestos quepan en el gabinete"""
        # Verificar tamaño de motherboard
        mb_size = None
        if 'Motherboard' in components:
            mb_size = components['Motherboard'][0]['metadata'].get('Form Factor & Dimensions - Form FactorForm Factor', '').lower()
        
        case_supported_sizes = case_metadata.get('Motherboard Compatibility', '').lower()
        
        if mb_size and case_supported_sizes:
            if mb_size == 'atx' and 'atx' not in case_supported_sizes:
                return False
            elif mb_size == 'microatx' and 'microatx' not in case_supported_sizes and 'atx' not in case_supported_sizes:
                return False
            elif mb_size == 'mini-itx' and 'mini-itx' not in case_supported_sizes:
                return False
        
        # Verificar tamaño de GPU
        if 'GPU' in components:
            gpu_length = components['GPU'][0]['metadata'].get('Form Factor & Dimensions - Max GPU Length', '0 mm')
            case_max_gpu = case_metadata.get('Maximum Video Card Length', '0 mm')
            
            try:
                gpu_length_mm = float(gpu_length.split()[0])
                case_max_mm = float(case_max_gpu.split()[0])
                if gpu_length_mm > case_max_mm:
                    return False
            except (ValueError, IndexError):
                pass
        
        # Verificar altura de cooler CPU
        if 'CPU' in components and 'Cooler' in components:
            cooler_height = components['Cooler'][0]['metadata'].get('Height', '0 mm')
            case_cooler_height = case_metadata.get('Max CPU Cooler Height', '0 mm')
            
            try:
                cooler_h_mm = float(cooler_height.split()[0])
                case_h_mm = float(case_cooler_height.split()[0])
                if cooler_h_mm > case_h_mm:
                    return False
            except (ValueError, IndexError):
                pass
        
        return True

    def _calculate_aesthetics_score(self, case_metadata: Dict, requirements: HardwareRequirements) -> int:
        """Calcula puntaje estético basado en preferencias del usuario"""
        score = 0
        aesthetics = requirements.aesthetics if hasattr(requirements, 'aesthetics') else {}
        
        # Puntos por coincidencia de color
        if aesthetics.get('color'):
            case_color = case_metadata.get('Color', '').lower()
            if aesthetics['color'].lower() in case_color:
                score += 2
        
        # Puntos por iluminación RGB
        if aesthetics.get('rgb', False):
            case_features = case_metadata.get('Features', '').lower()
            if 'rgb' in case_features or 'led' in case_features:
                score += 1
        
        # Puntos por ventana lateral
        if aesthetics.get('window', False):
            case_features = case_metadata.get('Features', '').lower()
            if 'window' in case_features or 'transparent' in case_features:
                score += 1
        
        return score

    def get_recommendation_report(self, candidates: List[Dict]) -> str:
        """Genera un informe detallado de recomendaciones"""
        if not candidates:
            return "No se encontraron gabinetes que cumplan con los requisitos"
        
        requirements = self.blackboard.get('user_requirements')
        component_proposals = self.blackboard.get('component_proposals', {})
        
        report = [
            "## Recomendaciones de Gabinete",
            "",
            f"**Caso de uso:** {requirements.use_case.value.capitalize()}",
            ""
        ]
        
        # Mostrar compatibilidad con componentes si están disponibles
        if 'Motherboard' in component_proposals:
            mb_model = component_proposals['Motherboard'][0]['metadata'].get('Model_Name', '')
            report.append(f"**Compatibilidad verificada con placa madre:** {mb_model}")
        
        if 'GPU' in component_proposals:
            gpu_model = component_proposals['GPU'][0]['metadata'].get('Model_Name', '')
            report.append(f"**Compatibilidad verificada con GPU:** {gpu_model}")
        
        report.append("")
        
        for i, candidate in enumerate(candidates[:3]):  # Mostrar top 3
            metadata = candidate['metadata']
            
            report.append(f"### Opción #{i+1}: {metadata.get('Model_Name', '')}")
            report.append(f"- **Factor de forma:** {metadata.get('Type', 'N/A')}")
            report.append(f"- **Tamaños soportados:** {metadata.get('Motherboard Compatibility', 'N/A')}")
            report.append(f"- **Espacio para GPU:** {metadata.get('Maximum Video Card Length', 'N/A')}")
            report.append(f"- **Altura de cooler:** {metadata.get('Max CPU Cooler Height', 'N/A')}")
            report.append(f"- **Bahías:** {metadata.get('Drive Bays', 'N/A')}")
            report.append(f"- **Ventiladores incluidos:** {metadata.get('Fans Included', 'N/A')}")
            report.append(f"- **Color:** {metadata.get('Color', 'N/A')}")
            report.append(f"- **Precio:** ${candidate['price']}")
            report.append(f"- [Ver en Newegg]({metadata.get('URL', '')})")
            report.append("")
        
        # Añadir notas de compatibilidad
        report.append("### Consideraciones importantes:")
        report.append("- Verifique las dimensiones exactas antes de comprar")
        report.append("- Considere el flujo de aire y ventilación adicional")
        
        aesthetics = requirements.aesthetics if hasattr(requirements, 'aesthetics') else {}
        if aesthetics.get('rgb', False):
            report.append("- Los gabinetes con RGB pueden requerir controladores adicionales")
        
        return "\n".join(report)