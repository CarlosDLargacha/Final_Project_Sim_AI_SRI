from typing import Dict, List, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from blackboard import Blackboard, EventType
from agents.BDI_agent import HardwareRequirements, UseCase

class PSUAgent:
    def __init__(self, vector_db: Dict[str, Any], blackboard: Blackboard):
        """
        :param vector_db: Base de datos vectorial de fuentes de poder
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

    def process_requirements(self):
        """Procesa los requisitos del usuario para recomendar fuentes de poder"""
        requirements = self.blackboard.get('user_requirements')
        if not requirements:
            return
        
        # Generar embedding para los requisitos
        requirement_text = self._generate_requirement_text(requirements)
        requirement_embedding = self.embedding_model.encode([requirement_text])[0]
        
        # Calcular similitud con todas las PSUs
        similarities = cosine_similarity(
            [requirement_embedding],
            self.vector_db['embeddings']
        )[0]
        
        # Filtrar y ordenar candidatos
        candidates = []
        for i, metadata in enumerate(self.vector_db['metadata']):
            # Verificar presupuesto
            try:
                price = float(metadata.get('Price', float('inf')))
            except (ValueError, TypeError):
                price = float('inf')
            
            max_psu_budget = requirements.budget.get('max', float('inf')) 
            if price > max_psu_budget:
                continue
            
            candidates.append({
                'metadata': metadata,
                'similarity': similarities[i],
                'price': price
            })
        
        # Ordenar por similitud, certificación y precio
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (
                -x['similarity'],
                -self._get_certification_level(x['metadata']),
                x['price']
            )
        )
        
        # Proponer las mejores opciones (máximo 5)
        top_candidates = sorted_candidates[:5]
        
        # Actualizar el blackboard
        if top_candidates:
            self.blackboard.update(
                section='component_proposals',
                data={'PSU': top_candidates},
                agent_id='psu_agent',
                notify=True
            )

    def _generate_requirement_text(self, requirements: HardwareRequirements) -> str:
        """Genera texto descriptivo de requisitos para embeddings"""
        text_parts = [
            "Fuente de poder para sistema de computación",
            "Requisitos principales:"
        ]
        
        # Descripción según caso de uso
        case = requirements.use_case.value
        if case == "gaming":
            text_parts.append("Para sistema gaming de alto rendimiento con GPU potente")
        elif case == "video_editing":
            text_parts.append("Para estación de trabajo de edición de video profesional")
        elif case == "data_science":
            text_parts.append("Para estación de trabajo de ciencia de datos/análisis")
        elif case == "server":
            text_parts.append("Para servidor con alta disponibilidad")
        elif case == "crypto_mining":
            text_parts.append("Para sistema de minería de criptomonedas 24/7")
        elif case == "machine_learning":
            text_parts.append("Para sistema de entrenamiento de modelos de ML")
        elif case == "web_development":
            text_parts.append("Para estación de trabajo de desarrollo web")
        elif case == "gaming/video_editing":
            text_parts.append("Para sistema híbrido gaming/edición de video")
        elif case == "gaming/data_science":
            text_parts.append("Para sistema híbrido gaming/ciencia de datos")
        elif case == "video_editing/data_science":
            text_parts.append("Para estación de trabajo profesional edición/datos")
        elif case == "gaming/video_editing/data_science":
            text_parts.append("Para estación de trabajo multipropósito de alto rendimiento")
        else:  # GENERAL
            text_parts.append("Para sistema de uso general")
        
        # Requisitos de certificación basados en caso de uso
        if case in ["gaming", "video_editing", "data_science", "machine_learning",
                   "gaming/video_editing", "gaming/data_science", 
                   "video_editing/data_science", "gaming/video_editing/data_science"]:
            text_parts.append("Certificación 80 Plus Gold o superior recomendada")
        elif case in ["crypto_mining", "server"]:
            text_parts.append("Certificación 80 Plus Platinum/Titanium para eficiencia 24/7")
        
        # Restricciones de tamaño
        if hasattr(requirements, 'constraints'):
            if 'small_form_factor' in requirements.constraints:
                text_parts.append("Para gabinete pequeño (SFX/L)")
            if 'silent_operation' in requirements.constraints:
                text_parts.append("Operación silenciosa preferible")
            if 'high_reliability' in requirements.constraints:
                text_parts.append("Alta confiabilidad requerida")
        
        return ". ".join(text_parts)

    def _get_certification_level(self, metadata: Dict) -> int:
        """Asigna valor numérico a la certificación 80 Plus"""
        cert = metadata.get('Efficiency - Efficiency Certification', '').lower()
        if 'titanium' in cert:
            return 6
        elif 'platinum' in cert:
            return 5
        elif 'gold' in cert:
            return 4
        elif 'silver' in cert:
            return 3
        elif 'bronze' in cert:
            return 2
        elif 'white' in cert or '80 plus' in cert:
            return 1
        return 0

    def get_recommendation_report(self, candidates: List[Dict]) -> str:
        """Genera un informe detallado de recomendaciones"""
        if not candidates:
            return "No se encontraron fuentes de poder que cumplan con los requisitos"
        
        requirements = self.blackboard.get('user_requirements')
        
        report = [
            "## Recomendaciones de Fuente de Poder",
            "",
            f"**Caso de uso:** {requirements.use_case.value.capitalize()}",
            ""
        ]
        
        for i, candidate in enumerate(candidates[:3]):  # Mostrar top 3
            metadata = candidate['metadata']
            
            report.append(f"### Opción #{i+1}: {metadata.get('Model_Name', '')}")
            report.append(f"- **Potencia:** {metadata.get('Max Power Output', 'N/A')}")
            report.append(f"- **Certificación:** {metadata.get('Efficiency - Efficiency Certification', 'N/A')}")
            report.append(f"- **Modularidad:** {metadata.get('Cable Management - Cable Type', 'N/A')}")
            report.append(f"- **Conectores:**")
            report.append(f"  - PCIe: {metadata.get('Connectors - PCI Express Connector', 'N/A')}")
            report.append(f"  - CPU: {metadata.get('Connectors - EPS Connector', 'N/A')}")
            report.append(f"  - SATA: {metadata.get('Connectors - SATA Connector', 'N/A')}")
            report.append(f"- **Ventilador:** {metadata.get('Cooling - Fan Size', 'N/A')}")
            report.append(f"- **Precio:** ${candidate['price']}")
            report.append(f"- [Ver en Newegg]({metadata.get('URL', '')})")
            report.append("")
        
        # Añadir notas generales
        report.append("### Consideraciones importantes:")
        report.append("- La potencia debe ser suficiente para todos los componentes")
        report.append("- Las fuentes modulares facilitan la administración de cables")
        
        if hasattr(requirements, 'constraints') and 'small_form_factor' in requirements.constraints:
            report.append("- Para gabinetes pequeños, prefiera fuentes SFX/SFX-L")
        
        return "\n".join(report)