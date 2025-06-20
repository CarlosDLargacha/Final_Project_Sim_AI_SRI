from typing import Dict, List, Any
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from blackboard import Blackboard, EventType
from bdi_agent import HardwareRequirements

class StorageType(Enum):
    SSD = "SSD"
    HDD = "HDD"

class StorageAgent:
    def __init__(self, ssd_vector_db: Dict[str, Any], hdd_vector_db: Dict[str, Any], blackboard: Blackboard):
        """
        :param ssd_vector_db: Base de datos vectorial de SSDs
        :param hdd_vector_db: Base de datos vectorial de HDDs
        :param blackboard: Instancia compartida del blackboard
        """
        self.ssd_db = ssd_vector_db
        self.hdd_db = hdd_vector_db
        self.blackboard = blackboard
        
        # Suscribirse a eventos del BDI
        self.blackboard.subscribe(
            EventType.REQUIREMENTS_UPDATED,
            self.process_requirements
        )

    def process_requirements(self):
        """Procesa los requisitos del usuario para recomendar almacenamiento"""
        requirements = self.blackboard.get('user_requirements')
        if not requirements:
            return
        
        # Determinar tipo de almacenamiento requerido (puede venir del BDI)
        storage_prefs = requirements.storage if hasattr(requirements, 'storage') else {}
        use_ssd = storage_prefs.get('prefer_ssd', True)
        use_hdd = storage_prefs.get('include_hdd', False)
        
        # Procesar cada tipo de almacenamiento por separado
        recommendations = {}
        
        if use_ssd:
            ssd_recs = self._recommend_storage(
                storage_type=StorageType.SSD,
                vector_db=self.ssd_db,
                requirements=requirements
            )
            if ssd_recs:
                recommendations['SSD'] = ssd_recs
        
        if use_hdd:
            hdd_recs = self._recommend_storage(
                storage_type=StorageType.HDD,
                vector_db=self.hdd_db,
                requirements=requirements
            )
            if hdd_recs:
                recommendations['HDD'] = hdd_recs
        
        # Actualizar el blackboard
        if recommendations:
            self.blackboard.update(
                section='component_proposals',
                data=recommendations,
                agent_id='storage_agent',
                notify=True
            )

    def _recommend_storage(self, storage_type: StorageType, vector_db: Dict[str, Any], requirements: HardwareRequirements) -> List[Dict]:
        """Recomienda unidades de almacenamiento de un tipo específico"""
        # Generar embedding para los requisitos
        requirement_text = self._generate_requirement_text(requirements, storage_type)
        requirement_embedding = vector_db['model'].encode([requirement_text])[0]
        
        # Calcular similitud con todos los items
        similarities = cosine_similarity(
            [requirement_embedding],
            vector_db['embeddings']
        )[0]
        
        # Filtrar y ordenar candidatos
        candidates = []
        for i, metadata in enumerate(vector_db['metadata']):
            # Verificar compatibilidad con motherboard si está disponible
            mb_proposals = self.blackboard.get('component_proposals', {}).get('Motherboard', [])
            if mb_proposals and not self._check_motherboard_compatibility(metadata, mb_proposals[0]['metadata']):
                continue
            
            # Verificar presupuesto
            try:
                price = float(metadata.get('Price', float('inf')))
            except (ValueError, TypeError):
                price = float('inf')
            
            max_storage_budget = requirements.budget.get('max', float('inf')) * 0.15  # 15% para almacenamiento
            if price > max_storage_budget:
                continue
            
            candidates.append({
                'metadata': metadata,
                'similarity': similarities[i],
                'price': price,
                'type': storage_type.value
            })
        
        # Ordenar por similitud y precio
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (-x['similarity'], x['price'])
        )
        
        return sorted_candidates[:3]  # Top 3 recomendaciones

    def _generate_requirement_text(self, requirements: HardwareRequirements, storage_type: StorageType) -> str:
        """Genera texto descriptivo de requisitos para embeddings"""
        text_parts = [
            f"{storage_type.value} para sistema de computación",
            f"Requisitos principales:"
        ]
        
        # Capacidad (si está especificada en los requisitos)
        if hasattr(requirements.storage, 'capacity'):
            text_parts.append(f"Capacidad mínima: {requirements.storage.capacity}")
        
        # Rendimiento (para SSDs)
        if storage_type == StorageType.SSD and hasattr(requirements.storage, 'performance'):
            perf = requirements.storage.performance
            text_parts.append(f"Velocidad lectura: {perf.get('read_speed', 'alta')}")
            text_parts.append(f"Velocidad escritura: {perf.get('write_speed', 'alta')}")
        
        # Restricciones físicas
        if hasattr(requirements, 'constraints'):
            if 'small_form_factor' in requirements.constraints:
                text_parts.append("Para sistema compacto")
        
        return ". ".join(text_parts)

    def _check_motherboard_compatibility(self, storage_metadata: Dict, mb_metadata: Dict) -> bool:
        """Verifica compatibilidad con la motherboard seleccionada"""
        # Verificar slots M.2 para SSDs
        if storage_metadata.get('Form Factor_Form Factor', '').lower() == 'm.2':
            m2_slots = int(mb_metadata.get('Storage - M.2 Slots', 0))
            if m2_slots < 1:
                return False
        
        # Verificar puertos SATA para HDDs/SSDs SATA
        if 'SATA' in storage_metadata.get('Interface_Interface', ''):
            sata_ports = int(mb_metadata.get('Storage - SATA Ports', 0))
            if sata_ports < 1:
                return False
        
        return True

    def get_recommendation_report(self, candidates: Dict[str, List[Dict]]) -> str:
        """Genera un informe detallado de recomendaciones"""
        if not candidates:
            return "No se encontraron opciones de almacenamiento que cumplan con los requisitos"
        
        report = ["## Recomendaciones de Almacenamiento", ""]
        
        for storage_type, items in candidates.items():
            report.append(f"### {storage_type}:")
            
            for i, item in enumerate(items[:2]):  # Mostrar top 2 por tipo
                metadata = item['metadata']
                
                report.append(f"#### Opción #{i+1}: {metadata.get('Model_Name', '')}")
                report.append(f"- **Capacidad:** {metadata.get('Capacity', 'N/A')}")
                
                if storage_type == "SSD":
                    report.append(f"- **Tipo:** {metadata.get('Form Factor_Form Factor', 'N/A')}")
                    report.append(f"- **Lectura/Escritura:** {metadata.get('Sequential Read_Sequential Read', 'N/A')}/{metadata.get('Sequential Write_Sequential Write', 'N/A')}")
                else:  # HDD
                    report.append(f"- **RPM:** {metadata.get('Spindle Speed_Spindle Speed', 'N/A')}")
                    report.append(f"- **Caché:** {metadata.get('Cache_Cache', 'N/A')}")
                
                report.append(f"- **Interfaz:** {metadata.get('Interface_Interface', 'N/A')}")
                report.append(f"- **Precio:** ${item['price']}")
                report.append(f"- [Ver en Newegg]({metadata.get('URL', '')})")
                report.append("")
        
        # Añadir notas sobre compatibilidad
        mb_proposals = self.blackboard.get('component_proposals', {}).get('Motherboard', [])
        if mb_proposals:
            mb_model = mb_proposals[0]['metadata'].get('Model_Name', 'la placa madre seleccionada')
            report.append("### Notas de compatibilidad:")
            report.append(f"- Verifique los slots disponibles en {mb_model}")
            report.append("- Algunas motherboards desactivan puertos SATA cuando se usan ciertos slots M.2")
        
        return "\n".join(report)