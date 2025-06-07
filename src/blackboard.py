from agents.base_agent import BaseAgent
from typing import Any

class Blackboard:
    """
    Pizarra central que coordina la comunicación entre agentes
    Almacena el estado global del sistema y datos compartidos
    """
    def __init__(self):
        self.state = {
            'user_requirements': None,
            'current_components': {},
            'compatibility_issues': [],
            'budget_allocation': {}
        }
        self.subscribers = []  # Agentes registrados

    def update(self, section: str, data: Any):
        """Actualiza una sección específica del estado"""
        pass

    def subscribe(self, agent: 'BaseAgent'):
        """Registra un agente para recibir actualizaciones"""
        pass

    def notify(self, section: str):
        """Notifica a agentes sobre cambios en una sección"""
        pass