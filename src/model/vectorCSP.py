from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorCSP:
    def __init__(self, field_mapping: Dict[str, int], embeddings):
        """
        :param field_mapping: Mapeo de nombres de campo a índices en el metadata
                             Ej: {'price': 'precio', 'tdp': 'Thermal Design Power'}
        """
        self.field_mapping = field_mapping
        self.embeddings = embeddings
        self.constraint_handlers = {
            'min': lambda val, thresh: val >= thresh,
            'max': lambda val, thresh: val <= thresh,
            'eq': lambda val, target: val == target,
            'in': lambda val, options: val in options
        }

    def solve(self, vectors: np.ndarray, metadata: List[Dict], 
             constraints: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """
        Filtra componentes basado en restricciones técnicas
        
        :param vectors: Vectores de embeddings (no usados directamente aquí)
        :param metadata: Metadatos de los componentes candidatos
        :param constraints: Restricciones en formato:
                           {'field': {'type': 'min|max|eq|in', 'value': x}}
        :return: Lista de metadatos enriquecidos con scores
        """
        valid_components = []
        
        for i, component in enumerate(metadata):
            valid = True
            csp_score = 0
            
            for field, constraint in constraints.items():
                # Obtener el valor real del campo en el componente
                field_name = self.field_mapping.get(field, field)
                field_value = component.get(field_name)
                
                if field_value is None:
                    continue
                
                # Aplicar la restricción
                handler = self.constraint_handlers.get(constraint['type'])
                if handler:
                    try:
                        if not handler(field_value, constraint['value']):
                            valid = False
                            break
                        csp_score += 1  # Premiar por cada restricción cumplida
                    except (TypeError, ValueError):
                        continue
            
            if valid:
                component['similarity'] = float(cosine_similarity(
                    [vectors[i]], 
                    [self.embeddings[i]]
                )[0][0])
                component['csp_score'] = csp_score
                valid_components.append(component)
        
        return valid_components
    
    def _handle_min_constraint(self, values: np.ndarray, threshold: float) -> np.ndarray:
        return values >= threshold
    
    def _handle_max_constraint(self, values: np.ndarray, threshold: float) -> np.ndarray:
        return values <= threshold
    
    def _handle_eq_constraint(self, values: np.ndarray, target: Any) -> np.ndarray:
        return values == target
    
    def _handle_in_constraint(self, values: np.ndarray, options: List[Any]) -> np.ndarray:
        return np.isin(values, options)
    
    def _handle_compat_constraint(self, values: np.ndarray, required: Dict[str, Any]) -> np.ndarray:
        """Maneja restricciones de compatibilidad entre componentes"""
        # Implementación específica según tu estructura de compatibilidad
        # Esto asume que el campo de compatibilidad es un dict codificado en el vector
        return np.array([self._check_compatibility(v, required) for v in values])
    
    def _check_compatibility(self, compat_value: float, required: Dict[str, Any]) -> bool:
        """Lógica específica para verificar compatibilidad"""
        # Aquí necesitarías decodificar el valor del vector a tu estructura de compatibilidad
        # Ejemplo simplificado:
        return True  # Implementar lógica real
    
    
    