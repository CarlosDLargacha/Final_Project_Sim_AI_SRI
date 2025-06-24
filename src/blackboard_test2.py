# test_blackboard_integration.py
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
        self.agents_proposed = 0
        self.request = [{}]
        
                
        self.blackboard.subscribe(event_type=EventType.USER_RESPONSE, callback=self.on_log)
    
    def on_log(self):
        
        proposol = self.blackboard.get_consolidated_components()
        r = "\n----------------------------------------------------------------------------------\n\n"
        
        r += f"Querie: {self.blackboard.get('user_input')['user_input']}"
        
        r += "\n\nPropuestas: "
        for comp_name, prop in proposol.items():
            r+= f"\n{comp_name}: {len(prop)}"
        
        r += f"\n\nResponse: \n\n{self.blackboard.get('user_response')['response']}\n"
        
        with open('experiments.txt', 'a') as file:
            file.write(r)
        
        #print(r)
            
    def make_request(self, user_input: str):
        """
        Simula la entrada del usuario.
        En un escenario real, esto podría ser reemplazado por una interfaz de usuario.
        """
        self.blackboard.update('user_input', {'user_input': user_input}, 'user_agent')  

# Cargar embeddings
processor = CSVToEmbeddings()

cpu_db = processor.load_embeddings('CPU')
gpu_db = processor.load_embeddings('GPU')
mb_db = processor.load_embeddings('Motherboard')
hdd_db = processor.load_embeddings('HDD')
ssd_db = processor.load_embeddings('SSD')
ram_db = processor.load_embeddings('RAM')
psu_db = processor.load_embeddings('PSU')
case_db = processor.load_embeddings('case')


def run_test_scenario(querie):
    
    # Inicializar Blackboard
    blackboard = Blackboard(7)
    
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
    user_agent.make_request(querie)
    
    while(True):
        if(blackboard.get('user_response')):
            sleep(3)
            break

if __name__ == "__main__":
    user_queries = [
        # Gaming
        # "Quiero una PC para gaming en 4K con presupuesto máximo de $1500. Prefiero NVIDIA para la GPU.",
        # "Necesito una PC para jugar en 1440p a 144fps con presupuesto de $1200",
        # "Armar setup gamer económico para 1080p que corra Fortnite a 120fps",
        # "PC high-end para 4K/120Hz en juegos AAA, sin límite de presupuesto",
        # "Recomienda componentes para streaming y gaming simultáneo (presupuesto $2000)",
        # "Setup compacto para gaming en LAN parties (max $1000, preferencia AMD)",
        # "PC para esports (Valorant, CS2) que alcance 240fps estables",
        # "Configuración VR-ready para Half-Life Alyx (presupuesto $1500)",
        # "Build silenciosa para gaming nocturno (sin RGB, max $1300)",
        # "PC futurista con mucho RGB para juegos en 1440p ($1800)",
        # "Recomendación para upgrade de GPU manteniendo mi Ryzen 5 3600",
        
        # # Diseño/Edición
        # "Workstation para edición 4K en Premiere (budget $2500)",
        # "PC económica para diseño gráfico (Photoshop/Illustrator)",
        # "Configuración para renderizado 3D en Blender (sin límite de precio)",
        # "Build optimizada para After Effects con previews fluidas",
        # "Estación de trabajo para ingeniería CAD (Autodesk, SolidWorks)",
        # "PC para producción musical con baja latencia (presupuesto $1500)",
        # "Setup para streamer que haga diseño en vivo (multitarea intensiva)",
        "Recomendación para animación 2D/3D (medio alcance, $1700)",
        "PC para arquitectura con renders en tiempo real (Enscape, V-Ray)",
        "Workstation móvil para fotógrafo profesional (ITX compacta)",
        
        # Usos Generales/Especializados
        "PC familiar económica para oficina y Netflix",
        "Build para programación y desarrollo web (multimonitor)",
        "Configuración para minería de criptomonedas (eficiencia energética)",
        "Servidor doméstico para Plex/NAS (bajo consumo, 24/7)",
        "Setup para trading algorítmico (múltiples pantallas, baja latencia)",
        "PC todo-en-uno para estudio universitario (presupuesto ajustado)",
        "Build para machine learning local (GPU con mucho VRAM)",
        "Configuración silenciosa para biblioteca/home office",
        "PC para emulación retro (hasta PS3/Xbox 360)",
        "Recomendación para upgrade priorizando SSD y RAM"
    ]

    for i in user_queries:
        run_test_scenario(i)