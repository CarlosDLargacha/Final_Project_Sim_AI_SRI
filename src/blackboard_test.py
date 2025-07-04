import json
from time import sleep
from agents.BDI_agent import BDIAgent, HardwareRequirements
from agents.CPU_agent import CPUAgent
from agents.GPU_agent import GPUAgent
from agents.MB_agent import MotherboardAgent
from agents.storage_agent import StorageAgent
from agents.RAM_agent import RAMAgent
from agents.PSU_agent import PSUAgent
from agents.case_agent import CaseAgent
from agents.compatibility_agent import CompatibilityAgent
from agents.optimization_agent import OptimizationAgent
from blackboard import Blackboard, EventType
from model.vectorDB import CSVToEmbeddings
from model.LLMClient import GeminiClient

class User:
    def __init__(self, blackboard: Blackboard):
        
        self.blackboard = blackboard
                
        self.blackboard.subscribe(event_type=EventType.USER_RESPONSE, callback=self.on_log)
    
    def on_log(self):
        
        proposol = self.blackboard.get_consolidated_components()

        r = f"Querie: {self.blackboard.get('user_input')['user_input']}"
        
        r += "\n\nPropuestas: "
        for comp_name, prop in proposol.items():
            r+= f"\n{comp_name}: {len(prop)}"
        
        r += f"\n\nResponse: \n\n{self.blackboard.get('user_response')['response']}\n"
        
        print(r)
            
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
    blackboard = Blackboard(7)
    
    cpu_db = processor.load_embeddings('CPU')
    gpu_db = processor.load_embeddings('GPU')
    mb_db = processor.load_embeddings('Motherboard')
    hdd_db = processor.load_embeddings('HDD')
    ssd_db = processor.load_embeddings('SSD')
    ram_db = processor.load_embeddings('RAM')
    psu_db = processor.load_embeddings('PSU')
    case_db = processor.load_embeddings('case')
    
    agents = {
        'bdi': BDIAgent(
            llm_client=GeminiClient(),
            blackboard=blackboard
        ),
        'cpu': CPUAgent(
            vector_db=cpu_db ,
            cpu_scores_path='src/data/benchmarks/CPU_benchmarks.json',
            blackboard=blackboard
        ),
        'gpu': GPUAgent(
            vector_db=gpu_db,
            gpu_benchmarks_path='src/data/benchmarks/GPU_benchmarks_v7.csv',
            blackboard=blackboard
        ),
        'mb': MotherboardAgent(
            vector_db=mb_db,
            blackboard=blackboard
        ),
        'storage': StorageAgent(
            ssd_vector_db=ssd_db,
            hdd_vector_db=hdd_db,
            blackboard=blackboard
        ),
        'ram': RAMAgent(
            vector_db=ram_db,
            blackboard=blackboard
        ),
        'psu': PSUAgent(
            vector_db=psu_db,
            blackboard=blackboard
        ),
        'case': CaseAgent(
            vector_db=case_db, 
            blackboard=blackboard
        ),
        'comp': CompatibilityAgent(
            blackboard=blackboard,
        ),
        'opt': OptimizationAgent(
            blackboard=blackboard,
        )
    }
    
    user_agent = User(blackboard=blackboard)
    user_agent.make_request("PC familiar económica para oficina y Netflix")
    
    sleep(600)

if __name__ == "__main__":
    run_test_scenario()