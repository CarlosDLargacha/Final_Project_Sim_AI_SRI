from agents.base_agent import BaseAgent
from blackboard import Blackboard

class ComponentAgentFactory:
    """
    Factory para crear agentes especializados según tipo de componente
    """
    @staticmethod
    def create_agent(component_type: str, blackboard: Blackboard) -> BaseAgent:
        if component_type == 'CPU':
            return CPUAgent(blackboard)
        elif component_type == 'GPU':
            return GPUAgent(blackboard)
        # ... otros componentes
        else:
            raise ValueError(f"Tipo de agente no soportado: {component_type}")

class CPUAgent(BaseAgent):
    """Agente especializado en procesadores"""
    def __init__(self, blackboard: Blackboard):
        super().__init__(blackboard)
        self.component_type = 'CPU'
        self.metrics = ['clock_speed', 'cores', 'tdp', 'socket']

    def evaluate(self):
        """Lógica específica para recomendación de CPUs"""
        pass

class GPUAgent(BaseAgent):
    """Agente especializado en tarjetas gráficas"""
    def __init__(self, blackboard: Blackboard):
        super().__init__(blackboard)
        self.component_type = 'GPU'
        self.metrics = ['vram', 'cuda_cores', 'memory_bandwidth']
    
    def evaluate(self):
        """Lógica específica para recomendación de GPUs"""
        pass