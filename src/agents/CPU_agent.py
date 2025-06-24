import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Any, Tuple
from agents.BDI_agent import HardwareRequirements, UseCase
from agents.decorators import agent_error_handler
from blackboard import *

class CPUAgent:
    def __init__(self, vector_db: Dict[str, Any], cpu_scores_path: str, blackboard: Any):
        """
        :param vector_db: Base de datos vectorial de CPUs (resultado de CSVToEmbeddings)
        :param cpu_scores_path: Ruta al archivo JSON con los scores de CPUs
        :param blackboard: Instancia del blackboard para comunicación
        """
        self.vector_db = vector_db
        self.blackboard = blackboard
        self.cpu_scores = self._load_cpu_scores(cpu_scores_path)
        self.embedding_model = vector_db['model']
        
        # Suscribirse a eventos relevantes
        self.blackboard.subscribe(
            EventType.REQUIREMENTS_UPDATED,
            self.process_requirements
        )
    
    def _load_cpu_scores(self, path: str) -> Dict[str, Dict]:
        """Carga y normaliza los scores de CPUs"""
        with open(path, 'r') as f:
            raw_scores = json.load(f)
        
        # Crear un diccionario indexado por nombre normalizado de CPU
        scores_dict = {}
        for cpu in raw_scores['devices']:
            # Normalizar el nombre para mejorar la coincidencia
            normalized_name = self._normalize_cpu_name(cpu['name'])
            cpu['normalized_name'] = normalized_name
            scores_dict[normalized_name] = cpu
        
        return scores_dict
    
    def _normalize_cpu_name(self, name: str) -> str:
        """Normaliza los nombres de CPU para mejorar la coincidencia"""
        name = name.lower()
        # Eliminar términos irrelevantes
        name = re.sub(r'(processor|\(.*?\)|®|™)', '', name)
        # Reemplazar caracteres especiales
        name = re.sub(r'[^\w\s]', ' ', name)
        # Reducir espacios múltiples
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    
    def _find_matching_cpu(self, cpu_name: str) -> Dict:
        """Encuentra el score de CPU que mejor coincide con el nombre dado"""
        normalized = self._normalize_cpu_name(cpu_name)
        
        # Búsqueda exacta
        if normalized in self.cpu_scores:
            return self.cpu_scores[normalized]
        
        # Búsqueda aproximada
        for score_name in self.cpu_scores:
            if normalized in score_name or score_name in normalized:
                return self.cpu_scores[score_name]
        
        return None
    
    @agent_error_handler
    def process_requirements(self):
        """
        Proceso principal que filtra y recomienda CPUs basado en:
        - Requisitos de rendimiento mínimos
        - Presupuesto del usuario
        - Restricciones técnicas
        - Similitud semántica con los requisitos
        """
        # Obtener requisitos del usuario desde el blackboard
        requirements = self.blackboard.get('user_requirements')
        if not requirements:
            print("Advertencia: No se encontraron requisitos del usuario")
            return
        
        # 1. Determinar puntajes mínimos según el caso de uso
        min_scores = self._find_matching_cpu(requirements.cpu)
        
        
        # 2. Obtener presupuesto máximo (si no existe, usar infinito)
        max_budget = requirements.budget.get('max', float('inf'))
        cpu_budget_limit = max_budget
        
        # 3. Generar embedding para los requisitos
        requirement_text = self._generate_requirement_text(requirements, min_scores)
        requirement_embedding = self.embedding_model.encode([requirement_text])[0]
        
        # 4. Calcular similitud con todas las CPUs en la base vectorial
        similarities = cosine_similarity(
            [requirement_embedding],
            self.vector_db['embeddings']
        )[0]
        
        # 5. Filtrar y puntuar CPUs candidatas
        candidates = []
        for i, metadata in enumerate(self.vector_db['metadata']):
            # 5.1. Procesar precio (convertir de string a float)
            try:
                if isinstance(metadata.get('Price'), str):
                    # Eliminar símbolos de dólar y comas, y convertir a float
                    price_str = metadata.get('Price', '0').replace('$', '').replace(',', '').strip()
                    price = float(price_str) if price_str else float('inf')
                else:
                    price = float(metadata.get('Price', '0'))
            except (ValueError, TypeError) as e:
                print(f"Error al procesar precio {metadata.get('Price')}: {str(e)}")
                price = float('inf')
            
            # 5.2. Verificar presupuesto
            if price > cpu_budget_limit:
                continue
                
            # 5.3. Buscar puntajes de benchmark para esta CPU
            cpu_name = metadata.get('Model_Name', '')
            cpu_score = self._find_matching_cpu(cpu_name)
            
            if not cpu_score:
                continue
                
            # 5.4. Verificar requisitos mínimos de rendimiento
            if (cpu_score['score'] < min_scores['score'] or 
                cpu_score['multicore_score'] < min_scores['multicore_score']):
                continue
                
            # 5.5. Verificar compatibilidad con restricciones
            if not self._check_compatibility(metadata):
                continue
                
            # 5.6. Agregar a candidatos válidos
            candidates.append({
                'metadata': metadata,
                'similarity': similarities[i],
                'score': cpu_score,
                'price': price
            })
        
        # 6. Agrupar por modelo y seleccionar la opción más barata
        unique_candidates = self._select_cheapest_per_model(candidates)
        
        # 7. Ordenar por similitud y rendimiento
        sorted_candidates = sorted(
            unique_candidates,
            key=lambda x: (
                -x['similarity'],  # Mayor similitud primero
                -x['score']['score'],  # Mayor puntaje single-core
                -x['score']['multicore_score'],  # Mayor puntaje multi-core
                x['price']  # Menor precio
            )
        )
        
        # 8. Proponer las mejores opciones (máximo 5)
        top_candidates = sorted_candidates
        
        # 9. Actualizar el blackboard con las propuestas
        self.blackboard.update(
            section='component_proposals',
            data={'CPU': top_candidates},
            agent_id='cpu_agent',
            notify=True
        )

        print("[CPUAgent] Componentes propuestos")
    
    def _select_cheapest_per_model(self, candidates: List[Dict]) -> List[Dict]:
        """
        Filtra CPUs duplicadas (mismo modelo) y conserva solo la más barata.
        
        Args:
            candidates: Lista de CPUs candidatas
            
        Returns:
            List[Dict]: Lista filtrada sin duplicados
        """
        model_map = {}
        
        for candidate in candidates:
            # Crear clave única basada en marca y modelo
            model_key = f"{candidate['metadata'].get('Model_Brand', '')} {candidate['metadata'].get('Model_Name', '')}".strip().lower()
            
            # Si no tenemos este modelo o encontramos uno más barato
            if model_key not in model_map or candidate['price'] < model_map[model_key]['price']:
                model_map[model_key] = candidate
        
        return list(model_map.values())

    def _generate_requirement_text(self, requirements: HardwareRequirements, min_scores: Dict) -> str:
        """
        Genera texto descriptivo de requisitos para crear embeddings.
        
        Args:
            requirements: Requisitos del usuario
            min_scores: Puntajes mínimos requeridos
            
        Returns:
            str: Texto estructurado con los requisitos
        """
        text_parts = [
            f"CPU para {requirements.use_case.value}",
            f"Requisitos mínimos: Single-Core {min_scores.get('single_core', 'N/A')}, Multi-Core {min_scores.get('multi_core', 'N/A')}"
        ]
        
        # Detalles específicos por caso de uso
        if requirements.use_case == UseCase.GAMING:
            text_parts.append(f"Resolución: {requirements.performance.get('resolution', '1440p')}")
            text_parts.append(f"FPS objetivo: {requirements.performance.get('fps', 60)}")
        elif requirements.use_case == UseCase.VIDEO_EDITING:
            software = ", ".join(requirements.performance.get('software', []))
            if software:
                text_parts.append(f"Software: {software}")
        
        # Restricciones y preferencias
        if requirements.constraints:
            text_parts.append(f"Restricciones: {', '.join(requirements.constraints)}")
        if requirements.aesthetics:
            text_parts.append(f"Color: {requirements.aesthetics.get('color', 'cualquiera')}, RGB: {'Sí' if requirements.aesthetics.get('rgb', False) else 'No'}")
        
        return ". ".join(text_parts)
   
    def _check_compatibility(self, metadata: Dict) -> bool:
        """
        Verifica si el componente cumple con todas las restricciones del usuario.
        
        Args:
            metadata: Diccionario con los metadatos del componente a verificar
            
        Returns:
            bool: True si el componente es compatible, False si no cumple con alguna restricción
        """
        # Obtener los requisitos del usuario del blackboard
        requirements = self.blackboard.get('user_requirements')
        if not requirements or not requirements.constraints:
            return True
        
        # Verificar restricción de factor de forma pequeño
        if 'small_form_factor' in requirements.constraints:
            # Obtener el valor de TDP del metadata (ej: "125W")
            tdp_str = str(metadata.get('Details_Thermal Design PowerThermal Design Power', '0W'))
            
            try:
                # Extraer solo la parte numérica del TDP
                tdp_match = re.search(r'(\d+\.?\d*)', tdp_str)
                if tdp_match:
                    tdp_value = float(tdp_match.group(1))
                    # Para builds pequeños, limitamos a 65W máximo
                    if tdp_value > 65:
                        return False
                else:
                    # Si no podemos extraer el valor numérico, mejor excluir por precaución
                    return False
            except (ValueError, AttributeError) as e:
                print(f"Error al procesar TDP {tdp_str}: {str(e)}")
                return False
        
        # Verificar restricción de bajo ruido si existe
        if 'low_noise' in requirements.constraints:
            cooling = metadata.get('Details_Cooler', '').lower()
            if 'stock' in cooling or 'reference' in cooling:
                return False
        
        # Si pasó todas las verificaciones, el componente es compatible
        return True

    def get_recommendation_report(self, candidates: List[Dict]) -> str:
        """
        Genera un reporte detallado en formato Markdown con las mejores recomendaciones.
        
        Args:
            candidates: Lista de CPUs candidatas ya filtradas y ordenadas
            
        Returns:
            str: Reporte formateado en Markdown
        """
        if not candidates:
            return "## No se encontraron CPUs que cumplan con todos los requisitos\n\n" \
                "Por favor, considera ajustar tus requisitos o presupuesto."
        
        # Obtener requisitos para el contexto del reporte
        requirements = self.blackboard.get('user_requirements')
        use_case = requirements.use_case.value if requirements else "uso general"
        budget = requirements.budget.get('max', 'No especificado') if requirements else "No especificado"
        
        # Construir el reporte
        report = [
            "## Recomendaciones de CPU",
            "",
            f"**Caso de uso:** {use_case.capitalize()}",
            f"**Presupuesto máximo:** ${budget if isinstance(budget, (int, float)) else budget}",
            ""
        ]
        
        # Agregar información de cada recomendación
        for i, candidate in enumerate(candidates[:3]):  # Mostrar máximo 3 opciones
            metadata = candidate['metadata']
            score = candidate['score']
            
            # Encabezado de la opción
            report.append(f"### Opción #{i+1}: {metadata.get('Model_Brand', '')} {metadata.get('Model_Name', '')}")
            
            # Detalles técnicos
            report.append("- **Especificaciones:**")
            report.append(f"  - **Núcleos/Hilos:** {metadata.get('Details_# of Cores# of Cores', 'N/A')}")
            report.append(f"  - **Frecuencia base:** {metadata.get('Details_Operating FrequencyOperating Frequency', 'N/A')}")
            report.append(f"  - **TDP:** {metadata.get('Details_Thermal Design PowerThermal Design Power', 'N/A')}")
            
            # Puntajes de rendimiento
            report.append("- **Rendimiento:**")
            report.append(f"  - **Single-Core:** {score['score']} (más alto es mejor)")
            report.append(f"  - **Multi-Core:** {score['multicore_score']} (más alto es mejor)")
            
            # Precio y disponibilidad
            report.append("- **Compra:**")
            report.append(f"  - **Precio:** ${candidate['price']:.2f}")
            if metadata.get('URL'):
                report.append(f"  - [Ver en Newegg]({metadata['URL']})")
            
            report.append("")  # Línea en blanco entre opciones
        
        # Agregar notas finales
        report.append("### Notas importantes")
        report.append("- Los precios pueden variar y están sujetos a disponibilidad.")
        report.append("- El rendimiento real puede variar según la configuración del sistema.")
        report.append("- Para builds pequeños (SFF), se recomienda TDP ≤65W.")
        
        return "\n".join(report)