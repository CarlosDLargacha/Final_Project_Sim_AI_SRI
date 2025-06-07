from agents.base_agent import BaseAgent
from blackboard import Blackboard

class CompatibilityAgent(BaseAgent):
    """
    Agente que verifica restricciones técnicas entre componentes
    """
    def __init__(self, blackboard: Blackboard):
        super().__init__(blackboard)
        self.rules_engine = CompatibilityRulesEngine()

    def evaluate(self):
        """Verifica todas las reglas de compatibilidad"""
        pass

    def get_upgrade_recommendations(self, component: dict) -> list:
        """Sugiere alternativas compatibles para upgrades"""
        pass

class CompatibilityRulesEngine:
    """
    Motor de reglas técnicas independiente
    """
    def __init__(self):
        self.rules = {
            'cpu_mb_socket': self.check_socket,
            'psu_wattage': self.check_power
        }

    def check_socket(self, cpu: dict, mb: dict) -> bool:
        pass

    def check_power(self, components: list) -> bool:
        pass