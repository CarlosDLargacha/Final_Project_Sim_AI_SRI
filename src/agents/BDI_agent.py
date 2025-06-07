from agents.base_agent import BaseAgent
from blackboard import Blackboard

class BDIAgent(BaseAgent):
    """
    Agente que procesa lenguaje natural y extrae requisitos técnicos
    """
    def __init__(self, blackboard: Blackboard, nlp_model):
        super().__init__(blackboard)
        self.nlp_model = nlp_model
        self.requirements_schema = {
            'use_case': str,
            'budget': dict,
            'performance': dict,
            'aesthetics': dict
        }

    def parse_user_input(self, text: str) -> dict:
        """Transforma input de usuario a requisitos estructurados"""
        pass

    def request_missing_info(self, missing_fields: list) -> dict:
        """Genera preguntas para completar información faltante"""
        pass