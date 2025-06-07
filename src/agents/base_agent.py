from abc import ABC, abstractmethod
from blackboard import Blackboard

class BaseAgent(ABC):
    """
    Clase abstracta base para todos los agentes especializados
    """
    def __init__(self, blackboard: Blackboard):
        self.blackboard = blackboard
        self.knowledge_base = None
        self.component_type = None  # Ej: 'CPU', 'GPU'

    @abstractmethod
    def evaluate(self):
        """Método principal para ejecutar la lógica del agente"""
        pass

    @abstractmethod
    def update_knowledge(self, new_data: dict):
        """Actualiza la base de conocimiento del agente"""
        pass