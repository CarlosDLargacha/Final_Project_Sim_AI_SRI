import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderSystem:
    def __init__(self, vector_dbs: dict):
        """
        :param vector_dbs: Diccionario de bases de datos vectoriales por tipo
                          Ej: {'CPU': cpu_db, 'GPU': gpu_db}
        """
        self.vector_dbs = vector_dbs

    def recommend(self, user_query: str, component_type: str = None, 
                 top_k: int = 5, min_price: float = None, max_price: float = None) -> list:
        if not component_type:
            component_type = self._infer_component_type(user_query)
            if not component_type:
                return []

        db = self.vector_dbs.get(component_type)
        if not db:
            return []

        # Embedding de la consulta
        query_embed = db['model'].encode([user_query.lower()])
        
        # Calcular similitudes
        sim_matrix = cosine_similarity(query_embed, db['embeddings'])
        sim_scores = sim_matrix[0]
        
        # Filtrar por precio si se especifica
        metadata = db['metadata']
        valid_indices = []
        for idx, score in enumerate(sim_scores):
            item = metadata[idx]
            price_ok = True
            if min_price is not None and float(item.get('Price', 0)) < min_price:
                price_ok = False
            if max_price is not None and float(item.get('Price', float('inf'))) > max_price:
                price_ok = False
            if price_ok:
                valid_indices.append((idx, score))
        
        # Ordenar y seleccionar top_k
        valid_indices.sort(key=lambda x: -x[1])
        top_indices = [idx for idx, _ in valid_indices[:top_k]]
        
        # Formatear resultados
        results = []
        for idx in top_indices:
            item = metadata[idx].copy()
            item['similarity_score'] = float(sim_scores[idx])
            
            item['purchase_link'] = item.get('URL', '#')
            if item['purchase_link'] != 'N/A':
               item['purchase_link'] = f"[Comprar en Newegg]({item['purchase_link']})"
               
            results.append(item)
        
        return results

    def _infer_component_type(self, query: str) -> str:
        query_lower = query.lower()
        type_keywords = {
            'CPU': ['cpu', 'procesador', 'core', 'ryzen', 'intel'],
            'GPU': ['gpu', 'tarjeta gráfica', 'nvidia', 'radeon', 'rtx', 'gtx'],
            'RAM': ['ram', 'memoria', 'ddr'],
            'Motherboard': ['motherboard', 'placa base', 'placa madre']
        }
        
        for comp_type, keywords in type_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return comp_type
        return None