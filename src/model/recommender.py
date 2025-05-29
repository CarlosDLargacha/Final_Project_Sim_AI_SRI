import numpy as np
from typing import List, Dict, Any
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderSystem:
    def __init__(self, vector_db, csp):
        """
        :param embedding_model: Modelo de embeddings pre-cargado
        :param vector_db_embeddings: Array numpy con todos los embeddings (n x d)
        :param metadata: Lista de diccionarios con metadatos de cada componente
        """
        self.embedding_model = vector_db["model"]
        self.embeddings = vector_db["embeddings"]
        self.metadata = vector_db["metadata"]
        self.csp = csp

    def recommend(self, user_query: str, constraints: Dict[str, Any], top_k: int = 5) -> List[Dict]:
        """
        Proceso completo de recomendación:
        1. Transforma la consulta a embedding
        2. Busca componentes similares
        3. Aplica restricciones CSP
        4. Devuelve mejores resultados
        """
        # Paso 1: Embedding de la consulta
        query_embedding = self.embedding_model.encode([user_query])[0]
        
        # Paso 2: Búsqueda semántica inicial
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        candidate_indices = np.argsort(similarities)[-100:][::-1]  # Top 100 candidatos
        
        # Paso 3: Preparar datos para CSP
        candidate_vectors = self.embeddings[candidate_indices]
        candidate_metadata = [self.metadata[i] for i in candidate_indices]
        
        # Paso 4: Aplicar restricciones técnicas
        valid_indices = self.csp.solve(candidate_vectors, candidate_metadata, constraints)
        
        # Paso 5: Ordenar por score combinado (similitud + cumplimiento)
        results = sorted(
            valid_indices,
            key=lambda x: (-x['similarity'], x['csp_score']))
        
        return results[:top_k]
    
