import random
from typing import List, Dict, Tuple
import pandas as pd

class CompatibilityExperiment:
    def __init__(self, agents: Dict, component_dbs: Dict):
        self.agents = agents
        self.component_dbs = component_dbs
        self.compatibility_agent = agents['comp']
        self.results = []
        
    def run_test_cases(self, test_cases: List[Tuple[str, str, str, str]]):
        """
        Ejecuta los casos de prueba de compatibilidad
        
        Args:
            test_cases: Lista de tuplas (tipo_comp1, modelo_comp1, tipo_comp2, modelo_comp2, esperado)
                       donde 'esperado' es True si deberían ser incompatibles
        """
        for case in test_cases:
            comp1_type, comp1_model, comp2_type, comp2_model, should_be_incompatible = case
            
            # Simular propuestas de componentes
            self._simulate_component_proposal(comp1_type, comp1_model)
            self._simulate_component_proposal(comp2_type, comp2_model)
            
            # Activar verificación de compatibilidad
            self.compatibility_agent.check_compatibility()
            
            # Obtener resultados
            issues = self.agents['blackboard'].get('compatibility_issues', [])
            detected = self._was_incompatibility_detected(issues, comp1_type, comp1_model, comp2_type, comp2_model)
            
            # Registrar resultado
            result = {
                'component1': f"{comp1_type}:{comp1_model}",
                'component2': f"{comp2_type}:{comp2_model}",
                'expected_incompatible': should_be_incompatible,
                'detected_incompatible': detected,
                'correct': (detected == should_be_incompatible)
            }
            self.results.append(result)
            
            # Resetear blackboard para el próximo caso
            self.agents['blackboard'].reset()
    
    def _simulate_component_proposal(self, comp_type: str, model_name: str):
        """Simula la propuesta de un componente al blackboard"""
        agent = self.agents.get(comp_type.lower())
        if not agent:
            raise ValueError(f"No hay agente para el tipo {comp_type}")
            
        # Buscar el componente específico o uno aleatorio
        metadata = self._find_component_metadata(comp_type, model_name)
        
        # Crear estructura de propuesta
        proposal = {
            'metadata': metadata,
            'similarity': 1.0,  # Máxima similitud ya que es el componente que queremos
            'price': float(metadata.get('Price', 0))
        }
        
        # Actualizar blackboard
        self.agents['blackboard'].update(
            section='component_proposals',
            data={comp_type: [proposal]},
            agent_id=f'{comp_type.lower()}_agent',
            notify=False  # No notificar aún para evitar activación prematura
        )
    
    def _find_component_metadata(self, comp_type: str, model_name: str) -> Dict:
        """Busca metadata de un componente específico o devuelve uno aleatorio"""
        db = self.component_dbs.get(comp_type.lower())
        if not db:
            raise ValueError(f"No hay DB para el tipo {comp_type}")
            
        # Buscar componente exacto
        for item in db['metadata']:
            if item.get('Model_Name', '').lower() == model_name.lower():
                return item
                
        # Si no se encuentra, seleccionar aleatorio
        return random.choice(db['metadata'])
    
    def _was_incompatibility_detected(self, issues, comp1_type, comp1_model, comp2_type, comp2_model) -> bool:
        """Verifica si se detectó la incompatibilidad específica"""
        for issue in issues:
            comp_a_match = (issue.component_a.type.value == comp1_type and 
                          issue.component_a.model_name.lower() == comp1_model.lower())
            comp_b_match = (issue.component_b.type.value == comp2_type and 
                          issue.component_b.model_name.lower() == comp2_model.lower())
            
            if comp_a_match and comp_b_match:
                return True
                
            # Verificar también en orden inverso
            comp_a_match = (issue.component_a.type.value == comp2_type and 
                          issue.component_a.model_name.lower() == comp2_model.lower())
            comp_b_match = (issue.component_b.type.value == comp1_type and 
                          issue.component_b.model_name.lower() == comp1_model.lower())
            
            if comp_a_match and comp_b_match:
                return True
                
        return False
    
    def get_results_df(self) -> pd.DataFrame:
        """Devuelve los resultados como DataFrame para análisis"""
        return pd.DataFrame(self.results)
    
    def calculate_metrics(self) -> Dict:
        """Calcula métricas de rendimiento"""
        df = self.get_results_df()
        if df.empty:
            return {}
            
        total = len(df)
        correct = df['correct'].sum()
        accuracy = correct / total
        
        true_positives = ((df['expected_incompatible'] == True) & 
                         (df['detected_incompatible'] == True)).sum()
        false_negatives = ((df['expected_incompatible'] == True) & 
                          (df['detected_incompatible'] == False)).sum()
        
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'true_positives': true_positives,
            'false_negatives': false_negatives,
            'total_cases': total
        }