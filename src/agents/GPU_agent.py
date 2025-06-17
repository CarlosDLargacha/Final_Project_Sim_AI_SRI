from time import sleep
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity
from agents.decorators import agent_error_handler
from blackboard import Blackboard, EventType
from agents.BDI_agent import HardwareRequirements, UseCase
import re

class GPUAgent:
    def __init__(self, vector_db: Dict[str, Any], gpu_benchmarks_path: str, blackboard: Blackboard):
        """
        :param vector_db: Base de datos vectorial de GPUs
        :param gpu_benchmarks_path: Ruta al archivo CSV con benchmarks
        :param blackboard: Instancia compartida del blackboard
        """
        self.vector_db = vector_db
        self.blackboard = blackboard
        self.benchmarks = self._load_gpu_benchmarks(gpu_benchmarks_path)
        self.embedding_model = vector_db['model']
        
        # Suscribirse a eventos relevantes
        self.blackboard.subscribe(
            EventType.REQUIREMENTS_UPDATED,
            self.process_requirements
        )

    def _load_gpu_benchmarks(self, path: str) -> Dict[str, Dict]:
        """Carga y normaliza los benchmarks de GPU desde CSV"""
        df = pd.read_csv(path)
        
        df_numeric = df.select_dtypes(include=['int64', 'float64'])
        df[df_numeric.columns] = df_numeric.fillna(0)
        
        benchmarks_dict = {}
        
        for _, row in df.iterrows():
            gpu_name = self._normalize_gpu_name(row['gpuName'])
            benchmarks_dict[gpu_name] = {
                'G3Dmark': float(row['G3Dmark']),
                'G2Dmark': float(row['G2Dmark']),
                'price': float(row['price']),
                'gpuValue': float(row['gpuValue']),
                'TDP': int(row['TDP']),
                'powerPerformance': float(row['powerPerformance']),
                'category': row['category']
            }
        return benchmarks_dict

    def _normalize_gpu_name(self, name: str) -> str:
        """Normaliza nombres de GPU para mejor coincidencia"""
        name = name.lower()
        name = re.sub(r'(®|™|nvidia|geforce|radeon|amd|\s+)', '', name)
        return name.strip()

    @agent_error_handler
    def process_requirements(self):
        """Procesa los requisitos del usuario para recomendar GPUs"""
        requirements = self.blackboard.get('user_requirements')
        if not requirements:
            return
        
        # Determinar requisitos mínimos según caso de uso
        min_performance = self._get_min_performance(requirements.use_case, requirements.performance)
        
        # Generar embedding para los requisitos
        requirement_text = self._generate_requirement_text(requirements, min_performance)
        requirement_embedding = self.embedding_model.encode([requirement_text])[0]
        
        # Calcular similitud con todas las GPUs
        similarities = cosine_similarity(
            [requirement_embedding],
            self.vector_db['embeddings']
        )[0]
        
        # Filtrar GPUs que cumplan con requisitos
        candidates = []
        for i, metadata in enumerate(self.vector_db['metadata']):
            gpu_name = metadata.get('Model - Model', '')
            gpu_bench = self._find_matching_gpu(gpu_name)
            
            if not gpu_bench:
                continue
            
            #Verificar rendimiento mínimo
            if gpu_bench['G3Dmark'] < min_performance['G3Dmark']:
                continue
                
            # Verificar presupuesto (hasta 40% del total para GPU)
            try:
                price = float(metadata.get('price', float('inf')))
            except (ValueError, TypeError):
                price = float('inf')
            
            max_gpu_budget = requirements.budget.get('max', float('inf')) * 0.4
            if price > max_gpu_budget:
                continue
            
            # Verificar restricciones
            if not self._check_constraints(metadata, requirements.constraints):
                continue
            
            candidates.append({
                'metadata': metadata,
                'similarity': similarities[i],
                'benchmark': gpu_bench,
                'price': price
            })
        
        # Seleccionar la GPU más barata por modelo
        unique_candidates = self._select_cheapest_per_model(candidates)
        
        # Ordenar por similitud, rendimiento y valor
        sorted_candidates = sorted(
            unique_candidates,
            key=lambda x: (
                -x['similarity'],
                -x['benchmark']['G3Dmark'],
                -x['benchmark']['gpuValue']  # Considerar valor precio/rendimiento
            )
        )
        
        # Proponer las mejores opciones (máximo 5)
        top_candidates = sorted_candidates[:5]
        
        # Actualizar el blackboard
        self.blackboard.update(
            section='component_proposals',
            data={'GPU': top_candidates},
            agent_id='gpu_agent',
            notify=True
        )

    def _get_min_performance(self, use_case: UseCase, performance: Dict) -> Dict[str, float]:
        """Determina los requisitos mínimos de rendimiento"""
        # Valores base para 1080p @ 60fps
        min_perf = {
            'G3Dmark': 10000,
            'TDP': 300  # Límite por defecto
        }
        
        if use_case == UseCase.GAMING:
            resolution = performance.get('resolution', '1080p')
            fps = performance.get('fps', 60)
            
            # Ajustar según resolución
            if resolution == '1440p':
                min_perf['G3Dmark'] = 15000
            elif resolution == '4K':
                min_perf['G3Dmark'] = 20000
            
            # Ajustar según FPS
            min_perf['G3Dmark'] *= (fps / 60)  # Escalar linealmente
            
        elif use_case == UseCase.VIDEO_EDITING:
            min_perf['G3Dmark'] = 18000  # Buen rendimiento para edición
        
        elif use_case == UseCase.DATA_SCIENCE:
            min_perf['G3Dmark'] = 12000  # Depende más de VRAM y CUDA cores
            
        return min_perf

    def _generate_requirement_text(self, requirements: HardwareRequirements, min_perf: Dict) -> str:
        """Genera texto descriptivo de requisitos para embeddings"""
        perf = requirements.performance
        text_parts = [
            f"GPU para {requirements.use_case.value}",
            f"Requisitos mínimos: G3Dmark {min_perf['G3Dmark']}"
        ]
        
        if requirements.use_case == UseCase.GAMING:
            text_parts.append(f"Resolución: {perf.get('resolution', '1080p')}")
            text_parts.append(f"FPS objetivo: {perf.get('fps', 60)}")
        elif requirements.use_case == UseCase.VIDEO_EDITING:
            text_parts.append(f"Software: {', '.join(perf.get('software', []))}")
        
        if requirements.constraints:
            text_parts.append(f"Restricciones: {', '.join(requirements.constraints)}")
        
        return ". ".join(text_parts)

    def _find_matching_gpu(self, gpu_name: str) -> Dict:
        """Encuentra el benchmark que mejor coincide con el nombre de GPU"""
        normalized = self._normalize_gpu_name(gpu_name)
        
        # Búsqueda exacta primero
        if normalized in self.benchmarks:
            return self.benchmarks[normalized]
        
        # Búsqueda parcial
        for benchmark_name in self.benchmarks:
            if benchmark_name in normalized or normalized in benchmark_name:
                return self.benchmarks[benchmark_name]
        
        return None

    def _select_cheapest_per_model(self, candidates: List[Dict]) -> List[Dict]:
        """Selecciona la opción más barata por modelo de GPU"""
        model_map = {}
        
        for candidate in candidates:
            model_name = self._normalize_gpu_name(candidate['metadata'].get('Model_Name', ''))
            
            if model_name not in model_map or candidate['price'] < model_map[model_name]['price']:
                model_map[model_name] = candidate
        
        return list(model_map.values())

    def _check_constraints(self, metadata: Dict, constraints: List[str]) -> bool:
        """Verifica restricciones como tamaño, consumo, etc."""
        if 'small_form_factor' in constraints:
            length = metadata.get('Form Factor & Dimensions - Max GPU Length', '0 mm')
            try:
                length_mm = float(length.split()[0])
                if length_mm > 250:  # Límite para builds pequeños
                    return False
            except (ValueError, IndexError):
                pass
            
            tdp = metadata.get('Details_Thermal Design PowerThermal Design Power', '0W')
            try:
                tdp_value = float(tdp[:-1]) if tdp.endswith('W') else float(tdp)
                if tdp_value > 200:  # Límite de consumo para SFF
                    return False
            except ValueError:
                pass
        
        return True

    def get_recommendation_report(self, candidates: List[Dict]) -> str:
        """Genera un informe detallado de recomendaciones"""
        if not candidates:
            return "No se encontraron GPUs que cumplan con los requisitos"
        
        requirements = self.blackboard.get('user_requirements')
        use_case = requirements.use_case.value if requirements else "general"
        
        report = [
            "## Recomendaciones de GPU",
            "",
            f"**Caso de uso:** {use_case.capitalize()}",
            f"**Presupuesto máximo:** ${requirements.budget.get('max', 'N/A')}",
            ""
        ]
        
        for i, candidate in enumerate(candidates[:3]):  # Mostrar top 3
            metadata = candidate['metadata']
            bench = candidate['benchmark']
            
            report.append(f"### Opción #{i+1}: {metadata.get('Model_Name', '')}")
            report.append("- **Especificaciones:**")
            report.append(f"  - **VRAM:** {metadata.get('Memory - Memory Size', 'N/A')}")
            report.append(f"  - **Tipo Memoria:** {metadata.get('Memory - Memory Type', 'N/A')}")
            report.append(f"  - **TDP:** {metadata.get('Details_Thermal Design PowerThermal Design Power', 'N/A')}")
            report.append("- **Rendimiento:**")
            report.append(f"  - **G3Dmark:** {bench['G3Dmark']} (más alto es mejor)")
            report.append(f"  - **Valor:** {bench['gpuValue']:.2f} puntos por dólar")
            report.append("- **Compra:**")
            report.append(f"  - **Precio:** ${candidate['price']}")
            report.append(f"  - [Ver en Newegg]({metadata.get('URL', '')})")
            report.append("")
        
        report.append("### Notas importantes")
        report.append("- Los precios pueden variar y están sujetos a disponibilidad.")
        report.append("- El rendimiento real puede variar según la configuración del sistema.")
        
        if 'small_form_factor' in (requirements.constraints or []):
            report.append("- Para builds pequeños (SFF), se recomienda GPUs <250mm y TDP ≤200W.")
        
        return "\n".join(report)
