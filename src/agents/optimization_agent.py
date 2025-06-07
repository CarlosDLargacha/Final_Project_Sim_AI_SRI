from agents.base_agent import BaseAgent
from blackboard import Blackboard

class OptimizationAgent(BaseAgent):
    """
    Agente que ajusta recomendaciones para cumplir objetivos globales
    """
    def __init__(self, blackboard: Blackboard):
        super().__init__(blackboard)
        self.strategies = {
            'budget': self.optimize_budget,
            'performance': self.optimize_performance,
            'thermal': self.optimize_thermal
        }

    def evaluate(self, strategy: str = 'balanced'):
        """Aplica estrategia de optimización seleccionada"""
        pass

    def calculate_bottlenecks(self) -> dict:
        """Identifica cuellos de botella en la configuración"""
        pass