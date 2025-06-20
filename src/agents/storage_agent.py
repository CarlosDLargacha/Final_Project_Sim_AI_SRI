from typing import Dict, List, Any
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from blackboard import Blackboard, EventType
from agents.BDI_agent import HardwareRequirements

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
        """Recomienda unidades de almacenamiento que cumplan con la capacidad mínima requerida"""
        # 1. Obtener capacidad mínima requerida
        min_capacity = self._get_required_capacity(requirements)
        
        # 2. Generar embedding para los requisitos
        requirement_text = self._generate_requirement_text(requirements, storage_type)
        requirement_embedding = vector_db['model'].encode([requirement_text])[0]
        
        # 3. Calcular similitud con todos los items
        similarities = cosine_similarity(
            [requirement_embedding],
            vector_db['embeddings']
        )[0]
        
        # 4. Filtrar y ordenar candidatos
        candidates = []
        for i, metadata in enumerate(vector_db['metadata']):
            # Verificar capacidad mínima
            if min_capacity > 0:
                storage_cap = self._normalize_capacity(metadata.get('Capacity', '0GB'))
                if storage_cap < min_capacity:
                    continue
            
            # Verificar presupuesto
            try:
                price = float(metadata.get('Price', float('inf')))
            except (ValueError, TypeError):
                price = float('inf')
            
            max_storage_budget = requirements.budget.get('max', float('inf')) * 0.15
            if price > max_storage_budget:
                continue
            
            # Calcular puntaje de capacidad (mayor es mejor)
            capacity_score = 0
            if min_capacity > 0:
                storage_cap = self._normalize_capacity(metadata.get('Capacity', '0GB'))
                # Premiar capacidad cercana al mínimo requerido (evitar excesos)
                capacity_score = 1 - min(1, max(0, (storage_cap - min_capacity) / min_capacity))
            
            candidates.append({
                'metadata': metadata,
                'similarity': similarities[i],
                'price': price,
                'type': storage_type.value,
                'capacity': storage_cap,
                'capacity_score': capacity_score
            })
        
        # 5. Ordenar por similitud, capacidad y precio
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (
                -x['similarity'],
                -x['capacity_score'],  # Priorizar capacidades cercanas al mínimo
                x['price']
            )
        )
        
        return sorted_candidates

    def _get_required_capacity(self, requirements: HardwareRequirements) -> int:
        """Obtiene la capacidad mínima requerida en bytes (0 si no se especifica)"""
        if hasattr(requirements, 'storage') and hasattr(requirements.storage, 'capacity'):
            return self._normalize_capacity(requirements.storage.capacity)
        return 0

    def _normalize_capacity(self, capacity_str: str) -> int:
        """Convierte cualquier formato de capacidad a bytes"""
        # Limpieza y estandarización del texto
        capacity_str = str(capacity_str).strip().upper()
        
        # Extraer valor numérico
        try:
            num_part = ''.join(c for c in capacity_str if c.isdigit() or c == '.')
            value = float(num_part) if num_part else 0.0
        except ValueError:
            return 0
        
        # Convertir a bytes según unidad
        if 'TB' in capacity_str:
            return int(value * 1_000_000_000_000)  # 1 TB = 10^12 bytes
        elif 'GB' in capacity_str:
            return int(value * 1_000_000_000)  # 1 GB = 10^9 bytes
        elif 'MB' in capacity_str:
            return int(value * 1_000_000)  # 1 MB = 10^6 bytes
        elif 'KB' in capacity_str:
            return int(value * 1_000)  # 1 KB = 10^3 bytes
        return int(value)  # Asumir bytes si no se especifica unidad

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