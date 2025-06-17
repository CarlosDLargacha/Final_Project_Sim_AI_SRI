# test_blackboard_integration.py
import json
from time import sleep
from agents.BDI_agent import BDIAgent, HardwareRequirements
from agents.CPU_agent import CPUAgent
from agents.GPU_agent import GPUAgent
from agents.optimization_agent import OptimizationAgent
from blackboard import Blackboard, EventType
from model.vectorDB import CSVToEmbeddings
from model.LLMClient import GeminiClient

class User:
    def __init__(self, blackboard: Blackboard):
        
        self.blackboard = blackboard
        self.agents_proposed = 0
                
        self.blackboard.subscribe(event_type=EventType.COMPONENTS_PROPOSED, callback=self.fake_compability_check)
        self.blackboard.subscribe(event_type=EventType.OPTIMIZATION_DONE, callback=self.on_log)
        
    def fake_compability_check(self):
        
        self.agents_proposed += 1
        
        if(self.agents_proposed >= 2):
            self.blackboard.update(
                'compatibility_issues',
                {
                    'compatible': True,
                    'message': 'Todos los componentes son compatibles.'
                },
                'user_agent'
            )
    
    def on_log(self):
        
        print("\n----AGENTS_LOG-----------------------------------\n")
        
        for i, log in enumerate(self.blackboard.audit_log):
            print(f"Log {i}: {log}")
            print()
            
        print("\n-----------------------------------------------------------------\n")
            
    def make_request(self, user_input: str):
        """
        Simula la entrada del usuario.
        En un escenario real, esto podría ser reemplazado por una interfaz de usuario.
        """
        
        self.blackboard.update('user_input', {'user_input': user_input}, 'user_agent')
        
def run_test_scenario():
    
    # Cargar embeddings
    processor = CSVToEmbeddings()

    # Inicializar Blackboard
    blackboard = Blackboard()
    
    agents = {
        'bdi': BDIAgent(
            llm_client=GeminiClient(),
            blackboard=blackboard
        ),
        'cpu': CPUAgent(
            vector_db= processor.load_embeddings('CPU'),
            cpu_scores_path='src/data/benchmarks/CPU_benchmarks.json',
            blackboard=blackboard
        ),
        'gpu': GPUAgent(
            vector_db=processor.load_embeddings('GPU'),
            gpu_benchmarks_path='src/data/benchmarks/GPU_benchmarks_v7.csv',
            blackboard=blackboard
        ),
        'opt': OptimizationAgent(
            blackboard=blackboard,
            agents_proposal_number=2
        )
    }
    
    user_agent = User(blackboard=blackboard)
    user_agent.make_request("Quiero una PC para gaming en 4K con presupuesto máximo de $1500. Prefiero NVIDIA para la GPU.")
    
    sleep(3000)

if __name__ == "__main__":
    run_test_scenario()