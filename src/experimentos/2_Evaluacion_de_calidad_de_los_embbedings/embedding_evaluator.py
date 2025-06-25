import sys
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
import pandas as pd
from collections import defaultdict

# Añadir el directorio src al path de Python
src_path = str(Path(__file__).parent.parent.parent)  # Sube 3 niveles desde la ubicación actual
sys.path.append(src_path)

from blackboard import Blackboard
from model.LLMClient import GeminiClient, LLMClient
from agents.CPU_agent import CPUAgent
from agents.GPU_agent import GPUAgent
from agents.MB_agent import MotherboardAgent
from agents.storage_agent import StorageAgent
from agents.RAM_agent import RAMAgent
from agents.PSU_agent import PSUAgent
from agents.case_agent import CaseAgent


class EmbeddingEvaluator:
    def __init__(self, processor):
        """Inicializa con todos los embeddings cargados"""
        self.embeddings_db = {
            'CPU': processor.load_embeddings('CPU'),
            'GPU': processor.load_embeddings('GPU'),
            'MB': processor.load_embeddings('Motherboard'),
            'RAM': processor.load_embeddings('RAM'),
            'PSU': processor.load_embeddings('PSU'),
            'Case': processor.load_embeddings('case')
        }

        blackboard = Blackboard()


        self.agents = {
        'CPU': CPUAgent(
            vector_db=self.embeddings_db["CPU"],
            cpu_scores_path='src/data/benchmarks/CPU_benchmarks.json',
            blackboard=blackboard
        ),
        'GPU': GPUAgent(
            vector_db=self.embeddings_db["GPU"],
            gpu_benchmarks_path='src/data/benchmarks/GPU_benchmarks_v7.csv',
            blackboard=blackboard
        ),
        'MB': MotherboardAgent(
            vector_db=self.embeddings_db["MB"],
            blackboard=blackboard
        ),
        'RAM': RAMAgent(
            vector_db=self.embeddings_db["RAM"],
            blackboard=blackboard
        ),
        'PSU': PSUAgent(
            vector_db=self.embeddings_db["PSU"],
            blackboard=blackboard
        ),
        'Case': CaseAgent(
            vector_db=self.embeddings_db["Case"],
            blackboard=blackboard
        )
    }
  

    def evaluate_all_queries(self, quality_embedding_dict: Dict, top_k: int = 15) -> pd.DataFrame:
        """Evalúa todas las consultas contra todos los componentes"""
        results = []
        
        for query, (requirements, expected_components) in quality_embedding_dict.items():
            for component_type in expected_components.keys():
                # Generar embedding de la consulta
                query_embedding = self._generate_query_embedding(component_type, requirements)
                
                # Recuperar componentes más similares
                top_items = self._retrieve_similar_components(component_type, query_embedding, top_k)
                
                # Calcular métricas
                expected = expected_components[component_type]
                precision, recall = self._calculate_metrics(top_items, expected)
                
                results.append({
                    'query': query,
                    'component_type': component_type,
                    'top_k': top_k,
                    'precision': precision,
                    'recall': recall,
                    'retrieved': top_items,
                    'expected': expected
                })
        
        return pd.DataFrame(results)

    def _generate_query_embedding(self, component_type: str, requirements) -> np.ndarray:
        """Genera el embedding para la consulta usando el agente correspondiente"""
        agent = self.agents[component_type]
        
        # Cada agente tiene su propia implementación de _generate_requirement_text
        if component_type == 'CPU':
            min_scores = {'score': 0, 'multicore_score': 0}  # Valores dummy para prueba
            text = agent._generate_requirement_text(requirements, min_scores)
        elif component_type == 'GPU':
            min_perf = {'G3Dmark': 0}  # Valor dummy
            text = agent._generate_requirement_text(requirements, min_perf)
        elif component_type == 'RAM':
            ram_config = requirements.ram
            text = agent._generate_requirement_text(requirements, ram_config)
        elif component_type == 'Storage':
            # Determinar si es SSD o HDD
            storage_type = 'SSD' if requirements.storage['prefer_ssd'] else 'HDD'
            text = agent._generate_requirement_text(requirements, storage_type)
        else:
            text = agent._generate_requirement_text(requirements)
            
        return self.embeddings_db[component_type]['model'].encode([text])[0]

    def _retrieve_similar_components(self, component_type: str, query_embedding: np.ndarray, top_k: int) -> List[str]:
        """Recupera los componentes más similares usando embeddings"""
        db = self.embeddings_db[component_type]
        similarities = cosine_similarity([query_embedding], db['embeddings'])[0]
        
        # Obtener los índices de los top_k más similares
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Devolver los nombres de los modelos
        return [db['metadata'][i]['Model_Name'] for i in top_indices]

    def _calculate_metrics(self, retrieved: List[str], expected: List[str]) -> tuple:
        """Calcula precisión y recall para los resultados recuperados"""
        # Consideramos una coincidencia si el modelo esperado está contenido en el recuperado
        matches = sum(1 for item in retrieved if any(exp in item for exp in expected))
        precision = matches / len(retrieved)
        recall = matches / min(len(expected), len(retrieved))
        return precision, recall

    def generate_report(self, results_df: pd.DataFrame) -> Dict[str, Dict]:
        """Genera un reporte resumido por tipo de componente"""
        report = defaultdict(dict)
        
        for component_type in results_df['component_type'].unique():
            component_results = results_df[results_df['component_type'] == component_type]
            
            avg_precision = component_results['precision'].mean()
            avg_recall = component_results['recall'].mean()
            
            # Ejemplos de fallos
            failures = []
            for _, row in component_results[component_results['precision'] < 1.0].iterrows():
                failures.append({
                    'query': row['query'],
                    'expected': row['expected'],
                    'retrieved': row['retrieved']
                })
            
            report[component_type] = {
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'num_queries': len(component_results),
                'failure_examples': failures[:3]  # Mostrar solo 3 ejemplos
            }
        
        return dict(report)
    
    def generate_text_report(self, results_df: pd.DataFrame, output_file: str = "embedding_quality_report.txt"):
        """Genera un reporte detallado en formato TXT"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== EVALUACIÓN DE CALIDAD DE EMBEDDINGS ===\n\n")
            
            # Agrupar por query
            for query, group in results_df.groupby('query'):
                f.write(f"QUERY: {query}\n")
                f.write("-" * 80 + "\n")
                
                for _, row in group.iterrows():
                    f.write(f"\nComponente: {row['component_type']}\n")
                    
                    # Resultados obtenidos
                    # f.write("  Top resultados recuperados:\n")
                    for i, item in enumerate(row['retrieved'], 1):
                        f.write(f"    {i}. {item}\n")
                    
                    # Resultados esperados
                    # f.write("\n  Resultados esperados:\n")
                    # for exp in row['expected']:
                    #     f.write(f"    {exp}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")

