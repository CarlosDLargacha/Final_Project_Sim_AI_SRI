from typing import Dict, List, Any
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from agents.decorators import agent_error_handler
from blackboard import Blackboard, EventType
from agents.BDI_agent import HardwareRequirements

class RAMAgent:
    def __init__(self, vector_db: Dict[str, Any], blackboard: Blackboard):
        """
        :param vector_db: Base de datos vectorial de módulos RAM
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
        """Procesa los requisitos del usuario para recomendar módulos RAM"""
        requirements = self.blackboard.get('user_requirements')
        if not requirements:
            return
        
        # Obtener configuración de RAM de los requisitos del BDI
        ram_config = getattr(requirements, 'ram', {})
        # Generar embedding para los requisitos
        requirement_text = self._generate_requirement_text(requirements, ram_config)
        requirement_embedding = self.embedding_model.encode([requirement_text])[0]
        
        # Calcular similitud con todos los módulos RAM
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
            
            max_ram_budget = requirements.budget.get('max', float('inf'))
            if price > max_ram_budget:
                continue
            
            candidates.append({
                'metadata': metadata,
                'similarity': similarities[i],
                'price': price
            })

        # Ordenar por similitud, velocidad y latencia
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (
                -x['similarity'],
                -self._get_ram_speed(x['metadata']),
                self._get_ram_latency(x['metadata'])
            )
        )
        
        top_candidates = sorted_candidates
        
        # Actualizar el blackboard
        if top_candidates:
            self.blackboard.update(
                section='component_proposals',
                data={'RAM': top_candidates},
                agent_id='ram_agent',
                notify=True
            )

        print("[RAMAgent] Componentes propuestos")

    def _generate_requirement_text(self, requirements: HardwareRequirements, ram_config: Dict) -> str:
        """Genera texto descriptivo de requisitos para embeddings"""
        text_parts = [
            "Módulos de RAM para sistema de computación",
            "Requisitos principales:"
        ]
        
        # Capacidad total
        if ram_config.get('capacity'):
            text_parts.append(f"Capacidad total: {ram_config['capacity']}")
        
        # Tipo y velocidad
        if ram_config.get('type'):
            text_parts.append(f"Tipo: {ram_config['type']}")
        if ram_config.get('speed'):
            text_parts.append(f"Velocidad mínima: {ram_config['speed']} MHz")
        
        # Configuración de canales
        if ram_config.get('channels'):
            text_parts.append(f"Configuración: {ram_config['channels']}-channel")
        
        # Restricciones de perfil bajo
        if hasattr(requirements, 'constraints') and 'small_form_factor' in requirements.constraints:
            text_parts.append("Perfil bajo preferible")
        
        return ". ".join(text_parts)

    def _get_ram_speed(self, metadata: Dict) -> int:
        """Extrae la velocidad de la RAM en MHz"""
        speed_str = metadata.get('Memory - Speed', '0 MHz')
        try:
            return int(speed_str.split()[0])
        except (ValueError, IndexError):
            return 0

    def _get_ram_latency(self, metadata: Dict) -> int:
        """Extrae la latencia CAS de la RAM"""
        latency_str = metadata.get('Memory - CAS Latency', '0')
        try:
            return int(latency_str)
        except ValueError:
            return 99  # Valor alto para ordenar al final

    def _select_cheapest_per_model(self, candidates: List[Dict]) -> List[Dict]:
        """Selecciona la opción más barata por modelo de RAM"""
        model_map = {}
        
        for candidate in candidates:
            model_name = candidate['metadata'].get('Model_Name', '')
            normalized_name = self._normalize_ram_model(model_name)
            
            if normalized_name not in model_map or candidate['price'] < model_map[normalized_name]['price']:
                model_map[normalized_name] = candidate
        
        return list(model_map.values())

    def _normalize_ram_model(self, model_name: str) -> str:
        """Normaliza nombres de modelos de RAM para agrupación"""
        # Eliminar diferencias comunes en nombres de kits RAM
        model = model_name.lower()
        model = re.sub(r'(rgb|led|trident|vengeance|dominator|platinum|pro)\b', '', model)
        model = re.sub(r'\s+', ' ', model).strip()
        return model

    def get_recommendation_report(self, candidates: List[Dict]) -> str:
        """Genera un informe detallado de recomendaciones"""
        if not candidates:
            return "No se encontraron módulos RAM que cumplan con los requisitos"
        
        requirements = self.blackboard.get('user_requirements')
        ram_config = getattr(requirements, 'ram', {})
        
        report = [
            "## Recomendaciones de Memoria RAM",
            "",
            f"**Configuración deseada:**",
            f"- Capacidad: {ram_config.get('capacity', 'No especificada')}",
            f"- Tipo: {ram_config.get('type', 'DDR4/DDR5')}",
            f"- Velocidad: {ram_config.get('speed', 'No especificada')} MHz",
            ""
        ]
        
        # Mostrar compatibilidad con motherboard si está disponible
        mb_proposals = self.blackboard.get('component_proposals', {}).get('Motherboard', [])
        if mb_proposals:
            mb_model = mb_proposals[0]['metadata'].get('Model_Name', 'la placa madre seleccionada')
            report.append(f"**Compatibilidad verificada con:** {mb_model}")
            report.append("")
        
        for i, candidate in enumerate(candidates[:3]):  # Mostrar top 3
            metadata = candidate['metadata']
            
            report.append(f"### Opción #{i+1}: {metadata.get('Model_Name', '')}")
            report.append(f"- **Capacidad:** {metadata.get('Memory - Size', 'N/A')} ({metadata.get('Memory - Modules', '?')}x{metadata.get('Memory - Module Size', '?')})")
            report.append(f"- **Tipo:** {metadata.get('Memory - Memory Type', 'N/A')}")
            report.append(f"- **Velocidad:** {metadata.get('Memory - Speed', 'N/A')} (CL{metadata.get('Memory - CAS Latency', '?')})")
            report.append(f"- **Voltaje:** {metadata.get('Memory - Voltage', 'N/A')}")
            report.append(f"- **Precio:** ${candidate['price']}")
            report.append(f"- [Ver en Newegg]({metadata.get('URL', '')})")
            report.append("")
        
        # Añadir notas de configuración óptima
        report.append("### Notas de configuración:")
        report.append("- Para dual-channel, instale módulos en los slots recomendados por el manual de la placa madre")
        report.append("- Active XMP/DOCP en BIOS para alcanzar velocidades anunciadas")
        
        if hasattr(requirements, 'constraints') and 'small_form_factor' in requirements.constraints:
            report.append("- Verifique altura de los módulos si usa coolers de CPU grandes")
        
        return "\n".join(report)